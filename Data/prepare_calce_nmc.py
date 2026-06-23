from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROFILE_ORDER = ("BJDST", "DST", "US06", "FUDS")
TEMP_ORDER = (0.0, 25.0, 45.0)
MANUSCRIPT_INITIAL_SOC_PERCENT = 80.0
REQUIRED_OUTPUT_COLUMNS = [
    "time_s",
    "V_raw",
    "I_raw",
    "T",
    "SOC_percent",
    "V_corr_raw",
    "V_corr_raw_ema50",
    "V_corr_raw_dev_ema50",
    "V_corr_raw_ema200",
    "V_corr_raw_dev_ema200",
    "V_corr_raw_ema800",
    "V_corr_raw_dev_ema800",
    "I_raw_ema50",
    "I_raw_dev_ema50",
    "I_raw_ema200",
    "I_raw_dev_ema200",
    "absI_ema50",
    "absI_dev_ema50",
    "absI_ema200",
    "absI_dev_ema200",
    "profile",
    "temperature_C",
    "source_file",
]


@dataclass
class R0Estimate:
    temperature_C: float
    r0_ohm: float
    n_events: int
    source: str


def norm_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(columns: list[object], aliases: list[str]) -> str | None:
    normalized = {norm_col(c): str(c) for c in columns}
    for alias in aliases:
        hit = normalized.get(norm_col(alias))
        if hit is not None:
            return hit
    return None


def infer_profile(path: Path) -> str | None:
    upper = path.stem.upper()
    for profile in PROFILE_ORDER:
        if profile in upper:
            return profile
    return None


def infer_temperature(path: Path) -> float | None:
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*C", path.stem, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def infer_initial_soc_percent(path: Path) -> float | None:
    stem = path.stem.upper()
    profile_pattern = "|".join(PROFILE_ORDER)
    patterns = [
        rf"(?:{profile_pattern})[^A-Z0-9]+(?P<soc>\d+(?:\.\d+)?)\s*(?:SOC|PCT|PERCENT)?(?:$|[^A-Z0-9])",
        r"(?P<soc>\d+(?:\.\d+)?)\s*SOC(?:$|[^A-Z0-9])",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            value = float(match.group("soc"))
            if 0.0 <= value <= 100.0:
                return value
    return None


def format_soc_for_filename(soc_percent: float) -> str:
    text = f"{float(soc_percent):g}"
    return text.replace(".", "p")


def expected_manuscript_records() -> set[tuple[float, str, float]]:
    return {
        (float(temp), str(profile), MANUSCRIPT_INITIAL_SOC_PERCENT)
        for temp in TEMP_ORDER
        for profile in PROFILE_ORDER
    }


def observed_records(frames: list[pd.DataFrame]) -> set[tuple[float, str, float]]:
    out = set()
    for frame in frames:
        temp = float(frame["temperature_C"].iloc[0])
        profile = str(frame["profile"].iloc[0])
        soc = round(float(frame.attrs.get("initial_soc_percent", MANUSCRIPT_INITIAL_SOC_PERCENT)), 6)
        out.add((temp, profile, soc))
    return out


def validate_manuscript_dynamic_set(frames: list[pd.DataFrame]) -> None:
    missing = sorted(expected_manuscript_records() - observed_records(frames))
    if not missing:
        return
    formatted = "\n".join(f"  *{int(temp)}C_{profile}_{format_soc_for_filename(soc)}SOC.xls*" for temp, profile, soc in missing)
    raise ValueError(
        "Missing required manuscript dynamic-profile files. "
        "Place all 0/25/45 C BJDST/DST/US06/FUDS 80SOC files in Data/raw_dynamic/ "
        "or pass --allow-incomplete for a partial conversion.\n"
        f"{formatted}"
    )


def read_first_measurement_sheet(path: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        if frame.empty:
            continue
        cols = list(frame.columns)
        has_v = find_col(cols, ["Voltage(V)", "Voltage", "V", "mV"]) is not None
        has_i = find_col(cols, ["Current(A)", "Current", "I", "mA"]) is not None
        if has_v and has_i:
            return frame
    raise ValueError(f"No measurement-like sheet found in {path}")


def standardize_frame(path: Path, default_temperature: float | None = None, default_profile: str | None = None) -> pd.DataFrame:
    raw = read_first_measurement_sheet(path)
    cols = list(raw.columns)
    time_col = find_col(cols, ["Test_Time(s)", "Step_Time(s)", "Duration (sec)", "time_s", "Time"])
    voltage_col = find_col(cols, ["Voltage(V)", "Voltage", "V"])
    current_col = find_col(cols, ["Current(A)", "Current", "I"])
    mv_col = find_col(cols, ["mV"])
    ma_col = find_col(cols, ["mA"])
    temp_col = find_col(cols, ["Temperature(C)", "Temperature", "Temp", "T", "TempLabel"])
    charge_col = find_col(cols, ["Charge_Capacity(Ah)", "Charge Capacity(Ah)", "Charge_Capacity"])
    discharge_col = find_col(cols, ["Discharge_Capacity(Ah)", "Discharge Capacity(Ah)", "Discharge_Capacity"])

    if time_col is None:
        time = np.arange(len(raw), dtype=float)
    else:
        time = pd.to_numeric(raw[time_col], errors="coerce").to_numpy(float)
        if not np.isfinite(time).any():
            time = np.arange(len(raw), dtype=float)
    time = time - np.nanmin(time)

    if voltage_col is not None:
        voltage = pd.to_numeric(raw[voltage_col], errors="coerce").to_numpy(float)
    elif mv_col is not None:
        voltage = pd.to_numeric(raw[mv_col], errors="coerce").to_numpy(float) / 1000.0
    else:
        raise ValueError(f"Voltage column not found in {path}")

    if current_col is not None:
        current = pd.to_numeric(raw[current_col], errors="coerce").to_numpy(float)
    elif ma_col is not None:
        current = pd.to_numeric(raw[ma_col], errors="coerce").to_numpy(float) / 1000.0
    else:
        raise ValueError(f"Current column not found in {path}")

    temp = infer_temperature(path) if default_temperature is None else default_temperature
    if temp is None and temp_col is not None:
        tvals = pd.to_numeric(raw[temp_col], errors="coerce").dropna()
        temp = float(tvals.median()) if not tvals.empty else np.nan
    if temp is None:
        temp = np.nan

    profile = infer_profile(path) if default_profile is None else default_profile
    if profile is None:
        profile = "UNKNOWN"

    out = pd.DataFrame(
        {
            "time_s": time,
            "V_raw": voltage,
            "I_raw": current,
            "T": float(temp),
            "profile": profile,
            "temperature_C": float(temp) if np.isfinite(temp) else np.nan,
            "source_file": path.name,
        }
    )
    if charge_col is not None:
        out["Charge_Capacity_Ah"] = pd.to_numeric(raw[charge_col], errors="coerce")
    if discharge_col is not None:
        out["Discharge_Capacity_Ah"] = pd.to_numeric(raw[discharge_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "V_raw", "I_raw"]).copy()
    out = out.sort_values("time_s").drop_duplicates("time_s").reset_index(drop=True)
    return out


def infer_capacity_ah(reference_dir: Path, fallback: float = 2.0) -> float:
    for path in sorted(reference_dir.glob("*Initial*capacity*.xls*")):
        try:
            frame = standardize_frame(path, default_profile="CAPACITY")
        except Exception:
            continue
        if "Discharge_Capacity_Ah" in frame:
            cap = pd.to_numeric(frame["Discharge_Capacity_Ah"], errors="coerce").max()
            if np.isfinite(cap) and cap > 0:
                return float(cap)
    return fallback


def compute_soc_percent(frame: pd.DataFrame, capacity_ah: float, initial_soc_percent: float) -> pd.Series:
    if "Discharge_Capacity_Ah" in frame and frame["Discharge_Capacity_Ah"].notna().any():
        discharged = pd.to_numeric(frame["Discharge_Capacity_Ah"], errors="coerce").ffill().fillna(0.0)
        discharged = discharged - float(discharged.iloc[0])
    else:
        time = frame["time_s"].to_numpy(float)
        current = frame["I_raw"].to_numpy(float)
        dt = np.diff(time, prepend=time[0])
        dt = np.where(np.isfinite(dt) & (dt > 0), dt, np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0)
        discharged = np.cumsum(np.maximum(-current, 0.0) * dt / 3600.0)
    soc = initial_soc_percent - 100.0 * np.asarray(discharged, dtype=float) / capacity_ah
    return pd.Series(np.clip(soc, 0.0, 100.0), index=frame.index)


def causal_ema(values: np.ndarray, span: float) -> np.ndarray:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(values, dtype=float)
    prev = float(values[0])
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            value = prev
        prev = alpha * float(value) + (1.0 - alpha) * prev
        out[idx] = prev
    return out


def causal_time_ema(values: np.ndarray, time_s: np.ndarray, tau_s: float) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    prev = float(values[0])
    last_t = float(time_s[0])
    for idx, (t, value) in enumerate(zip(time_s, values, strict=True)):
        dt = max(float(t) - last_t, 0.0)
        alpha = 1.0 - np.exp(-dt / tau_s) if tau_s > 0 else 1.0
        if not np.isfinite(value):
            value = prev
        prev = alpha * float(value) + (1.0 - alpha) * prev
        out[idx] = prev
        last_t = float(t)
    return out


def estimate_r0(frame: pd.DataFrame, min_delta_i: float = 0.5, window: int = 3) -> R0Estimate:
    current = frame["I_raw"].to_numpy(float)
    voltage = frame["V_raw"].to_numpy(float)
    jumps = np.where(np.abs(np.diff(current, prepend=current[0])) >= min_delta_i)[0]
    vals = []
    for idx in jumps:
        lo0 = max(0, idx - window)
        lo1 = max(0, idx)
        hi0 = min(len(frame), idx + 1)
        hi1 = min(len(frame), idx + 1 + window)
        if lo1 <= lo0 or hi1 <= hi0:
            continue
        di = np.nanmean(current[hi0:hi1]) - np.nanmean(current[lo0:lo1])
        dv = np.nanmean(voltage[hi0:hi1]) - np.nanmean(voltage[lo0:lo1])
        if np.isfinite(di) and abs(di) >= min_delta_i and np.isfinite(dv):
            vals.append(abs(dv / di))
    if vals:
        r0 = float(np.nanmedian(vals))
        return R0Estimate(float(frame["temperature_C"].iloc[0]), r0, len(vals), "current-step median |dV/dI|")
    return R0Estimate(float(frame["temperature_C"].iloc[0]), 0.05, 0, "fallback 0.05 ohm")


def add_features(frame: pd.DataFrame, r0_ohm: float) -> pd.DataFrame:
    out = frame.copy()
    v_ohmic = out["V_raw"].to_numpy(float) - out["I_raw"].to_numpy(float) * r0_ohm
    out["V_corr_raw"] = causal_time_ema(v_ohmic, out["time_s"].to_numpy(float), tau_s=120.0)
    for span in (50, 200, 800):
        ema = causal_ema(out["V_corr_raw"].to_numpy(float), span)
        out[f"V_corr_raw_ema{span}"] = ema
        out[f"V_corr_raw_dev_ema{span}"] = out["V_corr_raw"].to_numpy(float) - ema
    for span in (50, 200):
        ema_i = causal_ema(out["I_raw"].to_numpy(float), span)
        out[f"I_raw_ema{span}"] = ema_i
        out[f"I_raw_dev_ema{span}"] = out["I_raw"].to_numpy(float) - ema_i
        ema_abs = causal_ema(np.abs(out["I_raw"].to_numpy(float)), span)
        out[f"absI_ema{span}"] = ema_abs
        out[f"absI_dev_ema{span}"] = np.abs(out["I_raw"].to_numpy(float)) - ema_abs
    return out


def prepare_dataset(
    raw_dir: Path,
    reference_dir: Path,
    out_dir: Path,
    initial_soc_percent: float,
    capacity_ah: float | None,
    allow_incomplete: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    capacity = capacity_ah if capacity_ah is not None else infer_capacity_ah(reference_dir)
    frames = []
    for path in sorted(raw_dir.rglob("*.xls*")):
        frame = standardize_frame(path)
        if frame["profile"].iloc[0] == "UNKNOWN" or not np.isfinite(frame["temperature_C"].iloc[0]):
            print(f"SKIP metadata not inferred: {path.name}")
            continue
        filename_soc = infer_initial_soc_percent(path)
        frame.attrs["initial_soc_percent"] = filename_soc if filename_soc is not None else initial_soc_percent
        frame.attrs["initial_soc_source"] = "filename" if filename_soc is not None else "cli_default"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No dynamic profile Excel files found in {raw_dir}")
    if not allow_incomplete:
        validate_manuscript_dynamic_set(frames)

    r0_rows = []
    for frame in frames:
        est = estimate_r0(frame)
        r0_rows.append(est.__dict__)
    r0_table = pd.DataFrame(r0_rows)
    r0_by_temp = r0_table.groupby("temperature_C")["r0_ohm"].median().to_dict()
    key_counts: dict[tuple[float, str], int] = {}
    for frame in frames:
        key = (float(frame["temperature_C"].iloc[0]), str(frame["profile"].iloc[0]))
        key_counts[key] = key_counts.get(key, 0) + 1

    manifest_rows = []
    for frame in frames:
        temp = float(frame["temperature_C"].iloc[0])
        profile = str(frame["profile"].iloc[0])
        r0 = float(r0_by_temp.get(temp, 0.05))
        start_soc = float(frame.attrs.get("initial_soc_percent", initial_soc_percent))
        start_soc_source = str(frame.attrs.get("initial_soc_source", "cli_default"))
        frame["SOC_percent"] = compute_soc_percent(frame, capacity, start_soc)
        frame = add_features(frame, r0)
        frame = frame[REQUIRED_OUTPUT_COLUMNS]
        key = (temp, profile)
        soc_suffix = f"_{format_soc_for_filename(start_soc)}SOC" if key_counts.get(key, 0) > 1 else ""
        out_file = out_dir / f"NMC_{int(temp)}C_{profile}{soc_suffix}.csv"
        frame.to_csv(out_file, index=False)
        manifest_rows.append(
            {
                "file": out_file.name,
                "raw_source_file": str(frame["source_file"].iloc[0]),
                "profile": profile,
                "temperature_C": temp,
                "n_samples": len(frame),
                "duration_s": float(frame["time_s"].max() - frame["time_s"].min()),
                "capacity_Ah_used": capacity,
                "initial_soc_percent": start_soc,
                "initial_soc_source": start_soc_source,
                "r0_ohm_used": r0,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "processed_manifest.csv", index=False)
    r0_table.to_csv(out_dir / "r0_estimates.csv", index=False)
    print(f"Wrote processed CSV files to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CALCE NMC Excel files for CEMA-TCN training.")
    parser.add_argument("--raw-dir", default=(Path(__file__).resolve().parent / "raw_dynamic").as_posix())
    parser.add_argument("--reference-dir", default=(Path(__file__).resolve().parent / "raw_reference").as_posix())
    parser.add_argument("--out-dir", default=(Path(__file__).resolve().parent / "processed").as_posix())
    parser.add_argument("--initial-soc-percent", type=float, default=80.0)
    parser.add_argument("--capacity-ah", type=float, default=None)
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow partial dynamic-profile conversion instead of requiring the full 12-file manuscript 80SOC set.")
    args = parser.parse_args()
    prepare_dataset(Path(args.raw_dir), Path(args.reference_dir), Path(args.out_dir), args.initial_soc_percent, args.capacity_ah, args.allow_incomplete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
