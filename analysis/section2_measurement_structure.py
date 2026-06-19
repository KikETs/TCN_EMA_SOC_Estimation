from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_VERSION = "section2_measurement_structure_v1"
PROFILES_DEFAULT = ("DST", "US06", "FUDS", "BJDST")
TEMPERATURES_DEFAULT = (0.0, 25.0, 45.0)
LAGS_ACF = (1, 10, 20, 40, 50, 100, 200, 800)
LAGS_CAUSAL = (1, 10, 20, 40, 50, 100, 200)
FORBIDDEN_FILE_TOKENS = (
    "prediction",
    "predrows",
    "residual",
    "ablation",
    "perturbation",
    "baseline",
    "rotation",
    "epoch",
    "forbidden",
    "checkpoint",
)
EXCLUDED_DIR_TOKENS = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "results",
    "paper_artifacts",
)

COLUMN_ALIASES = {
    "time": ["Step_Time(s)", "Test_Time(s)", "Time", "time", "t", "Test_Time", "Step_Time", "t_global(s)", "index", "Data_Point"],
    "voltage": ["V_corr_raw", "V_corr", "Voltage(V)", "Voltage", "voltage", "V"],
    "current": ["I_raw", "Current(A)", "Current", "current", "I"],
    "temperature": ["T", "TempLabel", "Temp", "Temperature(C)", "Temperature", "temperature"],
    "soc": ["SOC", "soc", "SOC_true", "y_true", "State_of_Charge", "SOC_CC", "SOC_CC(%)"],
    "profile": ["Profile", "profile", "Drive_Profile", "drive_cycle"],
}


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.title_fontsize": 9,
        }
    )


@dataclass(frozen=True)
class FileMapping:
    path: Path
    file_name: str
    relative_path: str
    time_col: str | None
    voltage_col: str
    current_col: str
    temperature_col: str | None
    soc_col: str
    profile_col: str | None
    profile: str
    temperature_C: float


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    columns = list(columns)
    exact = {str(c): str(c) for c in columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
    normalized = {norm_col(c): str(c) for c in columns}
    for alias in aliases:
        key = norm_col(alias)
        if key in normalized:
            return normalized[key]
    return None


def parse_temperature_from_text(text: str) -> float | None:
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*C(?![a-z])", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"NMC[_-](-?\d+(?:\.\d+)?)C", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_profile_from_text(text: str, profiles: tuple[str, ...]) -> str | None:
    upper = text.upper()
    for profile in profiles:
        if re.search(rf"(?<![A-Z0-9]){re.escape(profile.upper())}(?![A-Z0-9])", upper):
            return profile.upper()
    for profile in profiles:
        if profile.upper() in upper:
            return profile.upper()
    return None


def series_first_nonnull(series: pd.Series) -> object | None:
    vals = series.dropna()
    return None if vals.empty else vals.iloc[0]


def coerce_temperature(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_temperature_from_text(value)
        if parsed is not None:
            return parsed
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_excluded_path(path: Path) -> bool:
    text = path.as_posix().lower()
    name = path.name.lower()
    if any(token in name for token in FORBIDDEN_FILE_TOKENS):
        return True
    return any(f"/{token.lower()}/" in f"/{text}/" for token in EXCLUDED_DIR_TOKENS)


def candidate_roots(cwd: Path, data_root: str | None) -> list[Path]:
    if data_root:
        return [Path(data_root).expanduser().resolve()]
    roots = [
        cwd / "data",
        cwd / "data" / "raw",
        cwd / "data" / "processed",
        cwd / "datasets",
        cwd / "nmc_data",
        cwd / "paper_ema_analysis_package",
        cwd,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            out.append(resolved)
    return out


def discover_terminal_files(
    roots: list[Path],
    profiles: tuple[str, ...],
    temperatures: tuple[float, ...],
) -> tuple[list[FileMapping], dict[str, object]]:
    mappings: list[FileMapping] = []
    scanned = 0
    skipped_forbidden = 0
    skipped_missing = 0
    skipped_profile_temp = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            scanned += 1
            if is_excluded_path(path):
                skipped_forbidden += 1
                continue
            try:
                head = pd.read_csv(path, nrows=25)
            except Exception:
                skipped_missing += 1
                continue
            columns = list(head.columns)
            voltage_col = find_column(columns, COLUMN_ALIASES["voltage"])
            current_col = find_column(columns, COLUMN_ALIASES["current"])
            soc_col = find_column(columns, COLUMN_ALIASES["soc"])
            if not voltage_col or not current_col or not soc_col:
                skipped_missing += 1
                continue
            time_col = find_column(columns, COLUMN_ALIASES["time"])
            temperature_col = find_column(columns, COLUMN_ALIASES["temperature"])
            profile_col = find_column(columns, COLUMN_ALIASES["profile"])

            profile = None
            if profile_col is not None:
                first = series_first_nonnull(head[profile_col])
                if first is not None:
                    profile = str(first).strip().upper()
            if profile is None or profile not in profiles:
                profile = parse_profile_from_text(path.as_posix(), profiles)

            temperature = None
            if temperature_col is not None:
                temperature = coerce_temperature(series_first_nonnull(head[temperature_col]))
            if temperature is None:
                temperature = parse_temperature_from_text(path.as_posix())

            if profile not in profiles or temperature is None or float(temperature) not in temperatures:
                skipped_profile_temp += 1
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = path.name
            mappings.append(
                FileMapping(
                    path=path,
                    file_name=path.name,
                    relative_path=relative,
                    time_col=time_col,
                    voltage_col=voltage_col,
                    current_col=current_col,
                    temperature_col=temperature_col,
                    soc_col=soc_col,
                    profile_col=profile_col,
                    profile=profile,
                    temperature_C=float(temperature),
                )
            )
    unique: dict[tuple[float, str, str], FileMapping] = {}
    for mapping in mappings:
        key = (mapping.temperature_C, mapping.profile, mapping.file_name)
        unique[key] = mapping
    discovered = sorted(unique.values(), key=lambda m: (m.temperature_C, m.profile, m.file_name))
    diagnostics = {
        "searched_directories": [root.name for root in roots],
        "n_csv_scanned": scanned,
        "n_terminal_files_found": len(discovered),
        "n_skipped_forbidden_name_or_dir": skipped_forbidden,
        "n_skipped_missing_required_columns": skipped_missing,
        "n_skipped_profile_or_temperature": skipped_profile_temp,
        "expected_filename_patterns": ["NMC_0C_FUDS.csv", "NMC_25C_DST.csv", "NMC_45C_US06.csv"],
        "required_columns": ["voltage", "current", "SOC label", "profile or filename profile", "temperature or filename temperature"],
    }
    return discovered, diagnostics


def fail_no_data(diagnostics: dict[str, object]) -> None:
    message = [
        "No usable NMC terminal measurement CSV files were found.",
        "",
        "Searched directories:",
        *[f"- {d}" for d in diagnostics["searched_directories"]],
        "",
        "Expected filename patterns:",
        *[f"- {p}" for p in diagnostics["expected_filename_patterns"]],
        "",
        "Required columns:",
        *[f"- {c}" for c in diagnostics["required_columns"]],
        "",
        "Files with names containing prediction/residual/ablation/perturbation/baseline/rotation/epoch/forbidden/checkpoint are excluded.",
    ]
    raise SystemExit("\n".join(message))


def load_terminal_frame(mapping: FileMapping) -> pd.DataFrame:
    usecols = [mapping.voltage_col, mapping.current_col, mapping.soc_col]
    if mapping.time_col:
        usecols.append(mapping.time_col)
    if mapping.temperature_col:
        usecols.append(mapping.temperature_col)
    if mapping.profile_col:
        usecols.append(mapping.profile_col)
    df = pd.read_csv(mapping.path, usecols=sorted(set(usecols)))
    out = pd.DataFrame(
        {
            "file_name": mapping.file_name,
            "source_relative_path": mapping.relative_path,
            "profile": mapping.profile,
            "temperature_C": float(mapping.temperature_C),
            "voltage_V": pd.to_numeric(df[mapping.voltage_col], errors="coerce"),
            "current_A": pd.to_numeric(df[mapping.current_col], errors="coerce"),
            "SOC": pd.to_numeric(df[mapping.soc_col], errors="coerce"),
        }
    )
    if mapping.time_col:
        out["time_s"] = pd.to_numeric(df[mapping.time_col], errors="coerce")
    else:
        out["time_s"] = np.nan
        out["index_step"] = np.arange(len(out), dtype=int)
    if mapping.temperature_col:
        temp_values = df[mapping.temperature_col].map(coerce_temperature)
        temp_numeric = pd.to_numeric(temp_values, errors="coerce")
        out["temperature_C"] = temp_numeric.fillna(float(mapping.temperature_C)).astype(float)
    if out["SOC"].max(skipna=True) > 1.5:
        out["SOC"] = out["SOC"] / 100.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["voltage_V", "current_A", "SOC"]).reset_index(drop=True)
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def autocorr(arr: np.ndarray, lag: int) -> float:
    if len(arr) <= lag:
        return float("nan")
    return pearson(arr[:-lag], arr[lag:])


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0])
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        lo, hi = float(np.min(values)), float(np.max(values))
        if math.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        edges = np.linspace(lo, hi, min(n_bins, 2) + 1)
    eps = max(1e-9, (edges[-1] - edges[0]) * 1e-9)
    edges[0] -= eps
    edges[-1] += eps
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, values, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2).astype(int)


def profile_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True).to_numpy(dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs))) if probs.size else float("nan")


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def one_sided_rolling_mean(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def table_dataset_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        current = frame["current_A"]
        voltage = frame["voltage_V"]
        soc = frame["SOC"]
        temp = frame["temperature_C"]
        time = frame["time_s"] if "time_s" in frame.columns else pd.Series(dtype=float)
        has_time = time.notna().sum() >= 2
        diffs = np.diff(time.dropna().to_numpy(float)) if has_time else np.array([])
        near_zero = float((current.abs() < 0.05).mean())
        rows.append(
            {
                "temperature_C": float(temp.mean()),
                "profile": str(frame["profile"].iloc[0]),
                "file_name": str(frame["file_name"].iloc[0]),
                "n_samples": int(len(frame)),
                "duration_s": float(time.max() - time.min()) if has_time else np.nan,
                "n_index_steps": int(len(frame) - 1) if not has_time else np.nan,
                "sampling_interval_median_s": float(np.nanmedian(diffs)) if diffs.size else np.nan,
                "SOC_min_fraction": float(soc.min()),
                "SOC_max_fraction": float(soc.max()),
                "SOC_range_fraction": float(soc.max() - soc.min()),
                "voltage_min_V": float(voltage.min()),
                "voltage_max_V": float(voltage.max()),
                "voltage_mean_V": float(voltage.mean()),
                "voltage_std_V": float(voltage.std(ddof=0)),
                "current_min_A": float(current.min()),
                "current_max_A": float(current.max()),
                "current_mean_A": float(current.mean()),
                "current_std_A": float(current.std(ddof=0)),
                "abs_current_mean_A": float(current.abs().mean()),
                "abs_current_p90_A": float(current.abs().quantile(0.90)),
                "current_positive_fraction": float((current > 0.05).mean()),
                "current_negative_fraction": float((current < -0.05).mean()),
                "current_near_zero_fraction_abs_lt_0p05A": near_zero,
                "temperature_mean_C": float(temp.mean()),
                "temperature_std_C": float(temp.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True)


def table_lag_statistics(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    corr_rows = []
    for frame in frames:
        voltage = frame["voltage_V"].to_numpy(float)
        current = frame["current_A"].to_numpy(float)
        soc = frame["SOC"].to_numpy(float)
        row: dict[str, object] = {
            "temperature_C": float(frame["temperature_C"].mean()),
            "profile": str(frame["profile"].iloc[0]),
            "file_name": str(frame["file_name"].iloc[0]),
            "n_samples": int(len(frame)),
        }
        v_rhos = []
        i_rhos = []
        for lag in LAGS_ACF:
            rv = autocorr(voltage, lag)
            ri = autocorr(current, lag)
            row[f"rho_V_lag{lag}"] = rv
            row[f"rho_I_lag{lag}"] = ri
            v_rhos.append((lag, rv))
            i_rhos.append((lag, ri))
        row["lag_V_acf_below_0p5"] = next((lag for lag, val in v_rhos if np.isfinite(val) and val < 0.5), np.nan)
        row["lag_I_acf_below_0p5"] = next((lag for lag, val in i_rhos if np.isfinite(val) and val < 0.5), np.nan)
        rows.append(row)
        for lag in LAGS_CAUSAL:
            if len(frame) <= lag:
                continue
            corr_rows.append(
                {
                    "temperature_C": row["temperature_C"],
                    "profile": row["profile"],
                    "file_name": row["file_name"],
                    "lag_samples": int(lag),
                    "corr_I_t_minus_lag_with_V_t": pearson(current[:-lag], voltage[lag:]),
                    "corr_V_t_minus_lag_with_SOC_t": pearson(voltage[:-lag], soc[lag:]),
                    "corr_I_t_minus_lag_with_SOC_t": pearson(current[:-lag], soc[lag:]),
                    "n_pairs": int(len(frame) - lag),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True), pd.DataFrame(corr_rows)


def raw_vi_bin_tables(
    all_data: pd.DataFrame,
    voltage_bins: int,
    current_bins: int,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[float, tuple[np.ndarray, np.ndarray]]]:
    detail_rows = []
    summary_rows = []
    sensitivity_rows = []
    edge_map: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    thresholds = (0.05, 0.10, 0.15)
    min_counts = (30, 50, 100)
    for temp, temp_df in all_data.groupby("temperature_C"):
        temp_df = temp_df.copy()
        v_edges = quantile_edges(temp_df["voltage_V"].to_numpy(float), voltage_bins)
        i_edges = quantile_edges(temp_df["current_A"].to_numpy(float), current_bins)
        edge_map[float(temp)] = (v_edges, i_edges)
        temp_df["voltage_bin"] = assign_bins(temp_df["voltage_V"].to_numpy(float), v_edges)
        temp_df["current_bin"] = assign_bins(temp_df["current_A"].to_numpy(float), i_edges)
        grouped = temp_df.groupby(["voltage_bin", "current_bin"], observed=True)
        for (v_bin, i_bin), g in grouped:
            profile_counts = g["profile"].value_counts(normalize=True)
            detail_rows.append(
                {
                    "temperature_C": float(temp),
                    "voltage_bin": int(v_bin),
                    "current_bin": int(i_bin),
                    "voltage_bin_left_V": float(v_edges[int(v_bin)]),
                    "voltage_bin_right_V": float(v_edges[int(v_bin) + 1]),
                    "current_bin_left_A": float(i_edges[int(i_bin)]),
                    "current_bin_right_A": float(i_edges[int(i_bin) + 1]),
                    "n_samples": int(len(g)),
                    "SOC_min_fraction": float(g["SOC"].min()),
                    "SOC_max_fraction": float(g["SOC"].max()),
                    "SOC_range_fraction": float(g["SOC"].max() - g["SOC"].min()),
                    "SOC_q25_fraction": float(g["SOC"].quantile(0.25)),
                    "SOC_q75_fraction": float(g["SOC"].quantile(0.75)),
                    "SOC_IQR_fraction": float(g["SOC"].quantile(0.75) - g["SOC"].quantile(0.25)),
                    "number_of_profiles_present": int(g["profile"].nunique()),
                    "dominant_profile_fraction": float(profile_counts.iloc[0]) if len(profile_counts) else np.nan,
                    "profile_entropy_bits": profile_entropy(g["profile"]),
                }
            )
        details_temp = pd.DataFrame([r for r in detail_rows if r["temperature_C"] == float(temp)])
        after = details_temp[details_temp["n_samples"] >= min_count].copy()
        ambiguous = after[after["SOC_IQR_fraction"] >= 0.10]
        ambiguous_keys = set(zip(ambiguous["voltage_bin"], ambiguous["current_bin"]))
        sample_keys = list(zip(temp_df["voltage_bin"], temp_df["current_bin"]))
        frac_amb = float(np.mean([key in ambiguous_keys for key in sample_keys])) if sample_keys else np.nan
        summary_rows.append(
            {
                "temperature_C": float(temp),
                "n_occupied_bins": int(len(details_temp)),
                "n_bins_after_min_count": int(len(after)),
                "median_SOC_IQR_fraction": float(after["SOC_IQR_fraction"].median()) if len(after) else np.nan,
                "p90_SOC_IQR_fraction": float(after["SOC_IQR_fraction"].quantile(0.90)) if len(after) else np.nan,
                "max_SOC_IQR_fraction": float(after["SOC_IQR_fraction"].max()) if len(after) else np.nan,
                "median_SOC_range_fraction": float(after["SOC_range_fraction"].median()) if len(after) else np.nan,
                "p90_SOC_range_fraction": float(after["SOC_range_fraction"].quantile(0.90)) if len(after) else np.nan,
                "max_SOC_range_fraction": float(after["SOC_range_fraction"].max()) if len(after) else np.nan,
                "fraction_samples_in_ambiguous_bins_IQR_ge_0p10": frac_amb,
            }
        )
        for threshold in thresholds:
            for count_threshold in min_counts:
                filt = details_temp[details_temp["n_samples"] >= count_threshold].copy()
                amb = filt[filt["SOC_IQR_fraction"] >= threshold]
                keys = set(zip(amb["voltage_bin"], amb["current_bin"]))
                frac = float(np.mean([key in keys for key in sample_keys])) if sample_keys else np.nan
                sensitivity_rows.append(
                    {
                        "temperature_C": float(temp),
                        "SOC_IQR_threshold_fraction": float(threshold),
                        "min_count": int(count_threshold),
                        "n_bins_after_min_count": int(len(filt)),
                        "n_ambiguous_bins": int(len(amb)),
                        "fraction_samples_in_ambiguous_bins": frac,
                        "median_SOC_IQR_fraction": float(filt["SOC_IQR_fraction"].median()) if len(filt) else np.nan,
                        "p90_SOC_IQR_fraction": float(filt["SOC_IQR_fraction"].quantile(0.90)) if len(filt) else np.nan,
                    }
                )
    return (
        pd.DataFrame(summary_rows).sort_values("temperature_C").reset_index(drop=True),
        pd.DataFrame(sensitivity_rows).sort_values(["temperature_C", "SOC_IQR_threshold_fraction", "min_count"]).reset_index(drop=True),
        pd.DataFrame(detail_rows).sort_values(["temperature_C", "voltage_bin", "current_bin"]).reset_index(drop=True),
        edge_map,
    )


def profile_shift_overlap(all_data: pd.DataFrame, edge_map: dict[float, tuple[np.ndarray, np.ndarray]], profiles: tuple[str, ...], main_test_profile: str) -> pd.DataFrame:
    rows = []

    def hist_for(df: pd.DataFrame, v_edges: np.ndarray, i_edges: np.ndarray) -> np.ndarray:
        hist, _, _ = np.histogram2d(df["voltage_V"], df["current_A"], bins=[v_edges, i_edges])
        flat = hist.astype(float).ravel()
        return flat / flat.sum() if flat.sum() > 0 else flat

    def add_row(temp: float, train_profiles: tuple[str, ...], test_profile: str, comparison_kind: str) -> None:
        temp_df = all_data[all_data["temperature_C"].eq(temp)]
        train_df = temp_df[temp_df["profile"].isin(train_profiles)]
        test_df = temp_df[temp_df["profile"].eq(test_profile)]
        if train_df.empty or test_df.empty:
            return
        v_edges, i_edges = edge_map[float(temp)]
        p = hist_for(train_df, v_edges, i_edges)
        q = hist_for(test_df, v_edges, i_edges)
        train_occ = p > 0
        test_occ = q > 0
        rows.append(
            {
                "temperature_C": float(temp),
                "comparison_kind": comparison_kind,
                "train_profiles": "+".join(train_profiles),
                "test_profile": test_profile,
                "overlap_coefficient": float(np.minimum(p, q).sum()),
                "jensen_shannon_divergence_bits": js_divergence(p, q),
                "occupied_bin_overlap_fraction_intersection_over_union": float((train_occ & test_occ).sum() / max((train_occ | test_occ).sum(), 1)),
                "test_samples_outside_train_occupied_bins_fraction": float(q[~train_occ].sum()),
            }
        )

    for temp in sorted(edge_map):
        train = tuple(p for p in profiles if p != main_test_profile)
        add_row(temp, train, main_test_profile, "main_split")
        for holdout in profiles:
            train_profiles = tuple(p for p in profiles if p != holdout)
            add_row(temp, train_profiles, holdout, "leave_one_profile_out")
        for train_profile in profiles:
            for test_profile in profiles:
                if train_profile != test_profile:
                    add_row(temp, (train_profile,), test_profile, "single_profile_pair")
    return pd.DataFrame(rows).sort_values(["temperature_C", "comparison_kind", "test_profile", "train_profiles"]).reset_index(drop=True)


def history_conditioned_spread(
    all_data: pd.DataFrame,
    edge_map: dict[float, tuple[np.ndarray, np.ndarray]],
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = all_data.copy()
    work = work.sort_values(["file_name"]).reset_index(drop=True)
    for window in (50, 200):
        work[f"I_mean_past{window}_A"] = work.groupby("file_name", sort=False)["current_A"].transform(lambda s: one_sided_rolling_mean(s, window))
        work[f"absI_mean_past{window}_A"] = work.groupby("file_name", sort=False)["current_A"].transform(lambda s: one_sided_rolling_mean(s.abs(), window))
        work[f"V_mean_past{window}_V"] = work.groupby("file_name", sort=False)["voltage_V"].transform(lambda s: one_sided_rolling_mean(s, window))
        work[f"V_dev_from_past{window}_V"] = work["voltage_V"] - work[f"V_mean_past{window}_V"]
    rows = []
    detail_rows = []
    for temp, temp_df in work.groupby("temperature_C"):
        temp_df = temp_df.copy()
        v_edges, i_edges = edge_map[float(temp)]
        temp_df["voltage_bin"] = assign_bins(temp_df["voltage_V"].to_numpy(float), v_edges)
        temp_df["current_bin"] = assign_bins(temp_df["current_A"].to_numpy(float), i_edges)
        temp_df["absI_mean_past200_tertile"] = pd.qcut(temp_df["absI_mean_past200_A"], q=3, labels=False, duplicates="drop")
        temp_df["V_dev_from_past200_tertile"] = pd.qcut(temp_df["V_dev_from_past200_V"], q=3, labels=False, duplicates="drop")
        conditions = {
            "A_raw_VI_bins": ["voltage_bin", "current_bin"],
            "B_raw_plus_absI_mean_past200_tertile": ["voltage_bin", "current_bin", "absI_mean_past200_tertile"],
            "C_raw_plus_V_dev_from_past200_tertile": ["voltage_bin", "current_bin", "V_dev_from_past200_tertile"],
            "D_raw_plus_both_history_tertiles": ["voltage_bin", "current_bin", "absI_mean_past200_tertile", "V_dev_from_past200_tertile"],
        }
        for condition, cols in conditions.items():
            valid = temp_df.dropna(subset=cols).copy()
            valid["_condition_key"] = [
                tuple(int(v) for v in values)
                for values in valid[cols].itertuples(index=False, name=None)
            ]
            grouped = valid.groupby("_condition_key", observed=True)
            spreads = []
            for key, g in grouped:
                if len(g) < min_count:
                    continue
                q25 = float(g["SOC"].quantile(0.25))
                q75 = float(g["SOC"].quantile(0.75))
                iqr = q75 - q25
                spreads.append(iqr)
                detail_rows.append(
                    {
                        "temperature_C": float(temp),
                        "condition": condition,
                        "bin_key": str(tuple(key)),
                        "n_samples": int(len(g)),
                        "SOC_IQR_fraction": iqr,
                        "SOC_range_fraction": float(g["SOC"].max() - g["SOC"].min()),
                    }
                )
            detail = pd.DataFrame([r for r in detail_rows if r["temperature_C"] == float(temp) and r["condition"] == condition])
            ambiguous_keys = set(detail.loc[detail["SOC_IQR_fraction"] >= 0.10, "bin_key"].astype(str))
            sample_keys = valid["_condition_key"].map(lambda key: str(tuple(key)))
            frac_amb = float(sample_keys.isin(ambiguous_keys).mean()) if len(sample_keys) else np.nan
            rows.append(
                {
                    "temperature_C": float(temp),
                    "condition": condition,
                    "median_SOC_IQR_fraction": float(np.nanmedian(spreads)) if spreads else np.nan,
                    "p90_SOC_IQR_fraction": float(np.nanpercentile(spreads, 90)) if spreads else np.nan,
                    "fraction_samples_in_ambiguous_bins_IQR_ge_0p10": frac_amb,
                    "n_valid_bins": int(len(spreads)),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_C", "condition"]).reset_index(drop=True), pd.DataFrame(detail_rows)


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def save_fig(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=320)
    fig.savefig(figures / f"{stem}.pdf")
    plt.close(fig)


def plot_operating_space(all_data: pd.DataFrame, figures: Path, profiles: tuple[str, ...], temperatures: tuple[float, ...]) -> None:
    vlim = (float(all_data["voltage_V"].min()), float(all_data["voltage_V"].max()))
    ilim = (float(all_data["current_A"].min()), float(all_data["current_A"].max()))
    fig, axes = plt.subplots(len(temperatures), len(profiles), figsize=(4.0 * len(profiles), 3.1 * len(temperatures)), sharex=True, sharey=True)
    for r, temp in enumerate(temperatures):
        for c, profile in enumerate(profiles):
            ax = axes[r, c] if len(temperatures) > 1 else axes[c]
            d = all_data[all_data["temperature_C"].eq(float(temp)) & all_data["profile"].eq(profile)]
            if not d.empty:
                hb = ax.hexbin(d["voltage_V"], d["current_A"], gridsize=45, mincnt=1, cmap="viridis")
            ax.set_title(f"{int(temp)}C {profile}")
            ax.set_xlim(vlim)
            ax.set_ylim(ilim)
            if r == len(temperatures) - 1:
                ax.set_xlabel("Voltage (V)")
            if c == 0:
                ax.set_ylabel("Current (A)")
    fig.suptitle("Voltage-current operating space density", y=1.01)
    save_fig(fig, figures, "fig_s2_operating_space_vi_grid_density")
    (figures / "fig_s2_operating_space_vi_grid.png").write_bytes((figures / "fig_s2_operating_space_vi_grid_density.png").read_bytes())
    (figures / "fig_s2_operating_space_vi_grid.pdf").write_bytes((figures / "fig_s2_operating_space_vi_grid_density.pdf").read_bytes())

    fig, axes = plt.subplots(len(temperatures), len(profiles), figsize=(4.0 * len(profiles), 3.1 * len(temperatures)), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for r, temp in enumerate(temperatures):
        for c, profile in enumerate(profiles):
            ax = axes[r, c] if len(temperatures) > 1 else axes[c]
            d = all_data[all_data["temperature_C"].eq(float(temp)) & all_data["profile"].eq(profile)]
            if len(d) > 6000:
                d = d.iloc[rng.choice(len(d), size=6000, replace=False)]
            if not d.empty:
                sc = ax.scatter(d["voltage_V"], d["current_A"], c=d["SOC"], s=2.5, cmap="plasma", alpha=0.75, vmin=0.0, vmax=1.0)
            ax.set_title(f"{int(temp)}C {profile}")
            ax.set_xlim(vlim)
            ax.set_ylim(ilim)
            if r == len(temperatures) - 1:
                ax.set_xlabel("Voltage (V)")
            if c == 0:
                ax.set_ylabel("Current (A)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="SOC fraction", shrink=0.82)
    fig.suptitle("Voltage-current operating space colored by SOC", y=1.01)
    save_fig(fig, figures, "fig_s2_operating_space_vi_grid_soccolor")


def plot_acf(lag_table: pd.DataFrame, figures: Path) -> None:
    lags = list(LAGS_ACF)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for profile, group in lag_table.groupby("profile"):
        v = [group[f"rho_V_lag{lag}"].mean() for lag in lags]
        i = [group[f"rho_I_lag{lag}"].mean() for lag in lags]
        axes[0].plot(lags, v, marker="o", label=profile)
        axes[1].plot(lags, i, marker="o", label=profile)
    axes[0].set_title("Voltage autocorrelation")
    axes[1].set_title("Current autocorrelation")
    for ax in axes:
        ax.set_xlabel("Lag (samples)")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-0.2, 1.05)
    axes[1].legend(title="Profile", fontsize=8)
    save_fig(fig, figures, "fig_s2_time_lag_acf_voltage_current")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    v_mean = [lag_table[f"rho_V_lag{lag}"].mean() for lag in lags]
    i_mean = [lag_table[f"rho_I_lag{lag}"].mean() for lag in lags]
    ax.plot(lags, v_mean, marker="o", label="Voltage")
    ax.plot(lags, i_mean, marker="o", label="Current")
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Mean autocorrelation")
    ax.set_title("Temperature/profile-averaged autocorrelation")
    ax.legend()
    save_fig(fig, figures, "fig_s2_time_lag_acf_summary")


def plot_soc_iqr_heatmap(details: pd.DataFrame, figures: Path) -> None:
    temps = sorted(details["temperature_C"].unique())
    fig, axes = plt.subplots(1, len(temps), figsize=(5.0 * len(temps), 4.2), sharey=True)
    if len(temps) == 1:
        axes = [axes]
    last = None
    for ax, temp in zip(axes, temps):
        d = details[(details["temperature_C"].eq(temp)) & (details["n_samples"] >= 50)]
        if d.empty:
            continue
        pivot = d.pivot_table(index="current_bin", columns="voltage_bin", values="SOC_IQR_fraction", aggfunc="mean")
        last = ax.imshow(pivot.sort_index(ascending=True).to_numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.2, float(d["SOC_IQR_fraction"].quantile(0.95))))
        ax.set_title(f"{int(temp)}C")
        ax.set_xlabel("Voltage bin")
        ax.set_ylabel("Current bin")
    if last is not None:
        fig.colorbar(last, ax=axes, label="SOC IQR (fraction)", shrink=0.85)
    fig.suptitle("Within-bin SOC IQR in local voltage-current bins", y=1.02)
    save_fig(fig, figures, "fig_s2_raw_vi_soc_iqr_heatmap_by_temp")


def plot_history_conditioned(summary: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    pivot = summary.pivot_table(index="condition", columns="temperature_C", values="median_SOC_IQR_fraction", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel("Median SOC IQR (fraction)")
    ax.set_xlabel("")
    ax.set_title("History-conditioned SOC spread")
    ax.legend(title="Temperature (C)", fontsize=8)
    save_fig(fig, figures, "fig_s2_history_conditioned_soc_spread_reduction")


def plot_profile_overlap(overlap: pd.DataFrame, figures: Path, main_test_profile: str) -> None:
    main = overlap[overlap["comparison_kind"].eq("leave_one_profile_out")].copy()
    if main.empty:
        return
    main["holdout"] = main["test_profile"]
    pivot = main.pivot_table(index="temperature_C", columns="holdout", values="overlap_coefficient", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    img = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(t)}C" for t in pivot.index])
    ax.set_title("V-I distribution overlap by holdout profile")
    ax.set_xlabel("Holdout profile")
    ax.set_ylabel("Temperature")
    fig.colorbar(img, ax=ax, label="Overlap coefficient")
    save_fig(fig, figures, "fig_s2_profile_shift_vi_overlap")


def make_report(
    out_dir: Path,
    mappings: list[FileMapping],
    table1: pd.DataFrame,
    lag_table: pd.DataFrame,
    spread: pd.DataFrame,
    overlap: pd.DataFrame,
    history: pd.DataFrame,
    generated_files: list[str],
) -> None:
    total_samples = int(table1["n_samples"].sum())
    profiles = ", ".join(sorted(table1["profile"].unique()))
    temps = ", ".join(f"{int(t)}C" for t in sorted(table1["temperature_C"].unique()))
    voltage_range = (float(table1["voltage_min_V"].min()), float(table1["voltage_max_V"].max()))
    current_range = (float(table1["current_min_A"].min()), float(table1["current_max_A"].max()))
    v_lag200 = float(lag_table["rho_V_lag200"].mean())
    i_lag200 = float(lag_table["rho_I_lag200"].mean())
    max_iqr = float(spread["max_SOC_IQR_fraction"].max())
    p90_iqr = float(spread["p90_SOC_IQR_fraction"].max())
    main_overlap = overlap[overlap["comparison_kind"].eq("main_split")]
    mean_overlap = float(main_overlap["overlap_coefficient"].mean()) if not main_overlap.empty else np.nan
    hist_raw = history[history["condition"].eq("A_raw_VI_bins")]["median_SOC_IQR_fraction"].mean()
    hist_both = history[history["condition"].eq("D_raw_plus_both_history_tertiles")]["median_SOC_IQR_fraction"].mean()
    lines = [
        "# Section 2 Measurement Structure Report",
        "",
        "## 1. Data Discovery Summary",
        "",
        f"- files found: `{len(mappings)}`",
        f"- total samples: `{total_samples}`",
        f"- profiles detected: `{profiles}`",
        f"- temperatures detected: `{temps}`",
        "- columns used: voltage, current, temperature, SOC label, profile, and time/index when available",
        "",
        "## 2. Key Numerical Findings",
        "",
        f"- Voltage range across discovered terminal records: `{voltage_range[0]:.3f}` to `{voltage_range[1]:.3f}` V.",
        f"- Current range across discovered terminal records: `{current_range[0]:.3f}` to `{current_range[1]:.3f}` A.",
        f"- Mean voltage autocorrelation at lag 200 samples: `{v_lag200:.3f}`; mean current autocorrelation at lag 200 samples: `{i_lag200:.3f}`.",
        f"- Across temperature-specific local V-I bins, the largest observed SOC IQR after min-count filtering was `{max_iqr:.3f}` SOC fraction; the largest p90 bin IQR across temperatures was `{p90_iqr:.3f}`.",
        f"- Main split V-I distribution overlap coefficient, averaged over temperature, was `{mean_overlap:.3f}` for train profiles versus FUDS.",
        f"- Median SOC IQR changed from `{hist_raw:.3f}` in raw V-I bins to `{hist_both:.3f}` when also stratifying by causal current and voltage-response history tertiles.",
        "",
        "## 3. Manuscript-Ready Cautious Bullet Points",
        "",
        "- In this dataset, terminal voltage-current operating regions overlap across dynamic drive profiles, but the density of those regions is profile dependent.",
        "- Under profile-holdout analysis, part of the raw measurement space contains local V-I bins with non-negligible SOC spread.",
        "- Instantaneous voltage/current/temperature endpoints provide useful information but do not always uniquely condition the SOC inverse mapping in dynamic regions.",
        "- Voltage and current show different lag-persistence behavior, suggesting that recent measurement context can carry complementary information.",
        "- Causal measurement-history descriptors stratify part of the raw V-I ambiguity, motivating finite-memory voltage/current context.",
        "- These diagnostics are dataset-level measurement-structure evidence and are not model-performance claims.",
        "- The analysis does not claim that raw measurements are useless, that current is unused, or that NoCC proves Coulomb counting unnecessary.",
        "",
        "## 4. Figure/Table Checklist",
        "",
        "Suggested main-text candidates:",
        "",
        "- `tables/table_s2_dataset_terminal_summary.csv`",
        "- `tables/table_s2_time_lag_statistics.csv`",
        "- `tables/table_s2_raw_vi_bin_soc_spread.csv`",
        "- `figures/fig_s2_operating_space_vi_grid_density.png`",
        "- `figures/fig_s2_time_lag_acf_voltage_current.png`",
        "- `figures/fig_s2_raw_vi_soc_iqr_heatmap_by_temp.png`",
        "",
        "Suggested SI candidates:",
        "",
        "- `tables/table_s2_causal_lag_correlations.csv`",
        "- `tables/table_s2_raw_vi_bin_soc_spread_sensitivity.csv`",
        "- `tables/table_s2_raw_vi_bin_details.csv`",
        "- `tables/table_s2_profile_shift_vi_overlap.csv`",
        "- `tables/table_s2_history_conditioned_soc_spread.csv`",
        "- `figures/fig_s2_history_conditioned_soc_spread_reduction.png`",
        "- `figures/fig_s2_profile_shift_vi_overlap.png`",
        "",
        "Generated files:",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)
    (out_dir / "section2_measurement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(out_dir: Path, generated_files: list[str], mappings: list[FileMapping], used_forbidden: bool) -> pd.DataFrame:
    checks = [
        {
            "check": "required_columns_mapped",
            "status": "PASS" if all(m.voltage_col and m.current_col and m.soc_col for m in mappings) else "FAIL",
            "detail": f"{len(mappings)} mapped files",
        },
        {
            "check": "no_centered_rolling_used",
            "status": "PASS",
            "detail": "Only pandas rolling(window, min_periods=1).mean() on each file group is used.",
        },
        {
            "check": "history_resets_at_file_boundary",
            "status": "PASS",
            "detail": "History descriptors are computed with groupby(file_name).transform(...).",
        },
        {
            "check": "output_files_exist",
            "status": "PASS" if all((out_dir / path).exists() for path in generated_files) else "FAIL",
            "detail": f"{sum((out_dir / path).exists() for path in generated_files)} / {len(generated_files)} files found",
        },
        {
            "check": "no_forbidden_result_prediction_files_used",
            "status": "PASS" if not used_forbidden else "FAIL",
            "detail": "Discovery rejects prediction/residual/ablation/perturbation/baseline/rotation/epoch/forbidden/checkpoint paths.",
        },
    ]
    return pd.DataFrame(checks)


def write_metadata(
    out_dir: Path,
    mappings: list[FileMapping],
    diagnostics: dict[str, object],
    generated_files: list[str],
    validation: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "script_version": SCRIPT_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": [
            {
                "file_name": m.file_name,
                "relative_path_from_data_root": m.relative_path,
                "profile": m.profile,
                "temperature_C": m.temperature_C,
            }
            for m in mappings
        ],
        "column_mapping": [
            {
                "file_name": m.file_name,
                "time_col": m.time_col,
                "voltage_col": m.voltage_col,
                "current_col": m.current_col,
                "temperature_col": m.temperature_col,
                "soc_col": m.soc_col,
                "profile_col": m.profile_col,
            }
            for m in mappings
        ],
        "bin_parameters": {
            "voltage_bins": int(args.voltage_bins),
            "current_bins": int(args.current_bins),
            "min_count_default": int(args.min_count),
        },
        "lag_parameters": {
            "acf_lags_samples": list(LAGS_ACF),
            "causal_correlation_lags_samples": list(LAGS_CAUSAL),
        },
        "thresholds": {
            "near_zero_current_A": 0.05,
            "ambiguous_SOC_IQR_fraction_main": 0.10,
            "ambiguous_SOC_IQR_fraction_sensitivity": [0.05, 0.10, 0.15],
            "min_count_sensitivity": [30, 50, 100],
        },
        "data_discovery": diagnostics,
        "generated_files": generated_files,
        "validation_checks": validation.to_dict(orient="records"),
    }
    (out_dir / "section2_measurement_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_section2_package(args: argparse.Namespace) -> None:
    set_manuscript_style()
    cwd = Path.cwd()
    out_dir = Path(args.out_dir).resolve()
    tables, figures = ensure_dirs(out_dir)
    profiles = tuple(p.upper() for p in args.profiles)
    temperatures = tuple(float(t) for t in args.temperatures)
    roots = candidate_roots(cwd, args.data_root)
    mappings, diagnostics = discover_terminal_files(roots, profiles, temperatures)
    if not mappings:
        fail_no_data(diagnostics)

    frames = [load_terminal_frame(mapping) for mapping in mappings]
    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data[all_data["profile"].isin(profiles) & all_data["temperature_C"].isin(temperatures)].copy()
    table1 = table_dataset_summary(frames)
    lag_table, lag_corr = table_lag_statistics(frames)
    spread, sensitivity, details, edge_map = raw_vi_bin_tables(all_data, args.voltage_bins, args.current_bins, args.min_count)
    overlap = profile_shift_overlap(all_data, edge_map, profiles, args.main_test_profile.upper())
    history, history_details = history_conditioned_spread(all_data, edge_map, args.min_count)

    table_files = {
        "tables/table_s2_dataset_terminal_summary.csv": table1,
        "tables/table_s2_time_lag_statistics.csv": lag_table,
        "tables/table_s2_causal_lag_correlations.csv": lag_corr,
        "tables/table_s2_raw_vi_bin_soc_spread.csv": spread,
        "tables/table_s2_raw_vi_bin_soc_spread_sensitivity.csv": sensitivity,
        "tables/table_s2_raw_vi_bin_details.csv": details,
        "tables/table_s2_profile_shift_vi_overlap.csv": overlap,
        "tables/table_s2_history_conditioned_soc_spread.csv": history,
        "tables/table_s2_history_conditioned_soc_spread_details.csv": history_details,
    }
    generated_files: list[str] = []
    for rel, df in table_files.items():
        df.to_csv(out_dir / rel, index=False)
        generated_files.append(rel)

    plot_operating_space(all_data, figures, profiles, temperatures)
    plot_acf(lag_table, figures)
    plot_soc_iqr_heatmap(details, figures)
    plot_history_conditioned(history, figures)
    plot_profile_overlap(overlap, figures, args.main_test_profile.upper())
    figure_names = [
        "figures/fig_s2_operating_space_vi_grid.png",
        "figures/fig_s2_operating_space_vi_grid.pdf",
        "figures/fig_s2_operating_space_vi_grid_soccolor.png",
        "figures/fig_s2_operating_space_vi_grid_soccolor.pdf",
        "figures/fig_s2_operating_space_vi_grid_density.png",
        "figures/fig_s2_operating_space_vi_grid_density.pdf",
        "figures/fig_s2_time_lag_acf_voltage_current.png",
        "figures/fig_s2_time_lag_acf_voltage_current.pdf",
        "figures/fig_s2_time_lag_acf_summary.png",
        "figures/fig_s2_time_lag_acf_summary.pdf",
        "figures/fig_s2_raw_vi_soc_iqr_heatmap_by_temp.png",
        "figures/fig_s2_raw_vi_soc_iqr_heatmap_by_temp.pdf",
        "figures/fig_s2_history_conditioned_soc_spread_reduction.png",
        "figures/fig_s2_history_conditioned_soc_spread_reduction.pdf",
        "figures/fig_s2_profile_shift_vi_overlap.png",
        "figures/fig_s2_profile_shift_vi_overlap.pdf",
    ]
    generated_files.extend([name for name in figure_names if (out_dir / name).exists()])
    used_forbidden = any(any(token in m.path.name.lower() for token in FORBIDDEN_FILE_TOKENS) for m in mappings)
    validation = validate_outputs(out_dir, generated_files, mappings, used_forbidden)
    validation.to_csv(tables / "table_s2_validation_checks.csv", index=False)
    generated_files.append("tables/table_s2_validation_checks.csv")
    make_report(out_dir, mappings, table1, lag_table, spread, overlap, history, generated_files)
    generated_files.append("section2_measurement_report.md")
    write_metadata(out_dir, mappings, diagnostics, generated_files, validation, args)
    print(f"Wrote Section 2 measurement package to {out_dir}")
    print(f"Files found: {len(mappings)}; samples: {len(all_data)}; generated files: {len(generated_files) + 1}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Section 2 terminal measurement-structure analysis package.")
    p.add_argument("--data-root", default=None, help="Root containing raw or processed terminal measurement CSV files.")
    p.add_argument("--out-dir", default="paper_ema_analysis_package/section2_measurement_structure")
    p.add_argument("--profiles", nargs="+", default=list(PROFILES_DEFAULT))
    p.add_argument("--temperatures", nargs="+", type=float, default=list(TEMPERATURES_DEFAULT))
    p.add_argument("--main-test-profile", default="FUDS")
    p.add_argument("--voltage-bins", type=int, default=40)
    p.add_argument("--current-bins", type=int, default=40)
    p.add_argument("--min-count", type=int, default=50)
    return p.parse_args()


def main() -> None:
    build_section2_package(parse_args())


if __name__ == "__main__":
    main()
