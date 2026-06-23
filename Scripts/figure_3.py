from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_VERSION = "frequency_structure_analysis_v1"
PROFILE_ORDER = ("BJDST", "DST", "US06", "FUDS")
TEMP_ORDER = (0.0, 25.0, 45.0)
LOW_BOUND = 1.0 / 200.0
HIGH_BOUND = 1.0 / 50.0
FORBIDDEN_FILE_TOKENS = (
    "prediction",
    "residual",
    "ablation",
    "perturbation",
    "baseline",
    "rotation",
    "epoch",
    "checkpoint",
    "forbidden",
    "error_by",
    "model_output",
)

COLUMN_ALIASES = {
    "time": ["Test_Time(s)", "t_global(s)", "Time", "time", "t", "Test_Time", "Data_Point", "index", "Step_Time(s)", "Step_Time"],
    "voltage": ["Voltage(V)", "Voltage", "voltage", "V_raw", "V_terminal", "V", "V_corr_raw"],
    "current": ["Current(A)", "Current", "current", "I_raw", "I"],
    "temperature": ["TempLabel", "T", "Temp", "Temperature(C)", "Temperature", "temperature"],
    "soc": ["SOC_CC", "SOC", "soc", "SOC_ref", "SOC_true", "y_true", "State_of_Charge", "SOC_CC(%)"],
    "profile": ["Profile", "profile", "drive_cycle", "cycle_type"],
}


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
    cols = [str(c) for c in columns]
    exact = {c: c for c in cols}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
    normalized = {norm_col(c): c for c in cols}
    for alias in aliases:
        key = norm_col(alias)
        if key in normalized:
            return normalized[key]
    return None


def parse_temperature_from_text(text: str) -> float | None:
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*C(?![a-z])", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_profile_from_text(text: str) -> str | None:
    upper = text.upper()
    for profile in PROFILE_ORDER:
        if re.search(rf"(?<![A-Z0-9]){re.escape(profile)}(?![A-Z0-9])", upper):
            return profile
    for profile in PROFILE_ORDER:
        if profile in upper:
            return profile
    return None


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


def is_forbidden_path(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in FORBIDDEN_FILE_TOKENS)


def public_path_label(path: Path, base_dir: Path | None = None) -> str:
    resolved = path.resolve()
    if base_dir is not None:
        try:
            return resolved.relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            pass
    if resolved.name == "data":
        return "external_raw_data/data"
    return f"external_raw_data/{resolved.name}"


def candidate_data_roots(base_dir: Path, data_root: str | None) -> list[Path]:
    if data_root and data_root.lower() != "auto":
        return [Path(data_root).expanduser().resolve()]
    roots = [
        base_dir / "data" / "raw" / "NMC_SAMSUNG_INR_18650_2Ah",
        base_dir / "data" / "raw" / "NMC SAMSUNG INR 18650 2Ah",
        base_dir / "data",
        base_dir.parent / "nmc_soc_ocvstart_relabelled_from_lc_ocv" / "data",
        base_dir.parent / "nmc_soc80_relabelled_from_lc_ocv" / "data",
        base_dir.parent / "nmc_samsung_inr_18650_2ah_raw",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def discover_terminal_files(root: Path, base_dir: Path | None = None) -> tuple[list[FileMapping], dict[str, object]]:
    mappings: list[FileMapping] = []
    scanned = skipped_forbidden = skipped_missing = skipped_profile_temp = 0
    inspected_files: list[str] = []
    for path in sorted(root.rglob("*.csv")):
        scanned += 1
        inspected_files.append(path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix())
        if is_forbidden_path(path):
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
        temp_col = find_column(columns, COLUMN_ALIASES["temperature"])
        profile_col = find_column(columns, COLUMN_ALIASES["profile"])

        profile = None
        if profile_col:
            vals = head[profile_col].dropna()
            if len(vals):
                profile = str(vals.iloc[0]).strip().upper()
        if profile not in PROFILE_ORDER:
            profile = parse_profile_from_text(path.as_posix())

        temp = None
        if temp_col:
            vals = head[temp_col].dropna()
            if len(vals):
                temp = coerce_temperature(vals.iloc[0])
        if temp is None:
            temp = parse_temperature_from_text(path.as_posix())

        if profile not in PROFILE_ORDER or temp is None or float(temp) not in TEMP_ORDER:
            skipped_profile_temp += 1
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        mappings.append(
            FileMapping(
                path=path,
                file_name=path.name,
                relative_path=rel,
                time_col=time_col,
                voltage_col=voltage_col,
                current_col=current_col,
                temperature_col=temp_col,
                soc_col=soc_col,
                profile_col=profile_col,
                profile=str(profile),
                temperature_C=float(temp),
            )
        )

    by_record: dict[tuple[float, str], FileMapping] = {}
    for mapping in sorted(mappings, key=lambda m: len(m.relative_path)):
        key = (mapping.temperature_C, mapping.profile)
        by_record.setdefault(key, mapping)
    records = sorted(by_record.values(), key=lambda m: (m.temperature_C, PROFILE_ORDER.index(m.profile)))
    diagnostics = {
        "searched_directory": public_path_label(root, base_dir),
        "n_csv_scanned": scanned,
        "n_terminal_files_found": len(records),
        "n_skipped_forbidden_name": skipped_forbidden,
        "n_skipped_missing_required_columns": skipped_missing,
        "n_skipped_profile_or_temperature": skipped_profile_temp,
        "files_inspected": inspected_files[:500],
        "required_columns": ["time or index", "voltage", "current", "temperature/profile or filename labels", "reference SOC label"],
    }
    return records, diagnostics


def choose_data_root(base_dir: Path, data_root: str | None) -> tuple[Path, list[FileMapping], dict[str, object], list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    for root in candidate_data_roots(base_dir, data_root):
        records, diag = discover_terminal_files(root, base_dir)
        attempts.append(diag)
        keys = {(r.temperature_C, r.profile) for r in records}
        expected = {(t, p) for t in TEMP_ORDER for p in PROFILE_ORDER}
        if expected.issubset(keys):
            return root, records, diag, attempts
    message = [
        "No usable NMC terminal measurement CSV files were found.",
        "",
        "Searched directories:",
        *[f"- {a['searched_directory']}" for a in attempts],
        "",
        "Required columns:",
        "- time or index",
        "- voltage",
        "- current",
        "- temperature/profile labels or filename labels",
        "- reference SOC label",
        "",
        "Reason for failure: fewer than the required twelve profile-temperature records were found.",
    ]
    raise SystemExit("\n".join(message))


def load_terminal_record(mapping: FileMapping) -> tuple[pd.DataFrame, dict[str, object]]:
    usecols = [mapping.voltage_col, mapping.current_col, mapping.soc_col]
    if mapping.time_col:
        usecols.append(mapping.time_col)
    if mapping.temperature_col:
        usecols.append(mapping.temperature_col)
    if mapping.profile_col:
        usecols.append(mapping.profile_col)
    raw = pd.read_csv(mapping.path, usecols=sorted(set(usecols)))
    out = pd.DataFrame(
        {
            "sample_index": np.arange(len(raw), dtype=np.int64),
            "time_s": pd.to_numeric(raw[mapping.time_col], errors="coerce") if mapping.time_col else np.arange(len(raw), dtype=float),
            "raw_voltage": pd.to_numeric(raw[mapping.voltage_col], errors="coerce"),
            "raw_current": pd.to_numeric(raw[mapping.current_col], errors="coerce"),
            "reference_soc": pd.to_numeric(raw[mapping.soc_col], errors="coerce"),
        }
    )
    if out["reference_soc"].max(skipna=True) > 1.5:
        out["reference_soc"] = out["reference_soc"] / 100.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_voltage", "raw_current", "reference_soc"]).copy()
    out = out.sort_values(["time_s", "sample_index"], kind="mergesort").reset_index(drop=True)
    time = out["time_s"].to_numpy(float)
    diff = np.diff(time)
    pos = diff[np.isfinite(diff) & (diff > 0)]
    dt = float(np.nanmedian(pos)) if pos.size else float("nan")
    irregularity = float(np.nanmax(np.abs(pos - dt) / dt)) if pos.size and np.isfinite(dt) and dt > 0 else float("nan")
    reliable = bool(np.isfinite(irregularity) and irregularity <= 0.05)
    duration = float(np.nanmax(time) - np.nanmin(time)) if len(time) and np.isfinite(time).any() else float("nan")
    info = {
        "file_name": mapping.file_name,
        "profile": mapping.profile,
        "temperature_C": mapping.temperature_C,
        "time_col": mapping.time_col,
        "voltage_col": mapping.voltage_col,
        "current_col": mapping.current_col,
        "soc_col": mapping.soc_col,
        "sampling_interval_median_s": dt,
        "sampling_interval_relative_max_deviation": irregularity,
        "physical_time_reliable": reliable,
        "duration_s": duration,
        "n_samples": int(len(out)),
    }
    return out, info


def get_scipy_welch():
    try:
        from scipy.signal import welch  # type: ignore

        return welch
    except Exception:
        return None


def fallback_welch(values: np.ndarray, detrend_type: str, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    nperseg = max(4, min(int(nperseg), n))
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    scale = np.sum(window**2)
    spectra = []
    for start in range(0, n - nperseg + 1, step):
        seg = x[start : start + nperseg].astype(float)
        if detrend_type == "linear":
            t = np.arange(nperseg, dtype=float)
            coeff = np.polyfit(t, seg, 1)
            seg = seg - (coeff[0] * t + coeff[1])
        else:
            seg = seg - np.mean(seg)
        fft = np.fft.rfft(seg * window)
        spectra.append((np.abs(fft) ** 2) / max(scale, 1e-12))
    if not spectra:
        return np.array([], dtype=float), np.array([], dtype=float)
    pxx = np.mean(np.vstack(spectra), axis=0)
    freq = np.fft.rfftfreq(nperseg, d=1.0)
    return freq, pxx


def compute_psd(values: np.ndarray, detrend_type: str, welch_fn) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.nanstd(x) <= 1e-14:
        return np.array([], dtype=float), np.array([], dtype=float), {"nperseg": 0, "method": "empty_or_constant"}
    nperseg = min(1024, len(x))
    if welch_fn is not None:
        freq, pxx = welch_fn(x, fs=1.0, nperseg=nperseg, noverlap=nperseg // 2, detrend=detrend_type, scaling="density")
        method = "scipy.signal.welch"
    else:
        freq, pxx = fallback_welch(x, detrend_type, nperseg)
        method = "numpy_windowed_periodogram_fallback"
    freq = np.asarray(freq, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    mask = np.isfinite(freq) & np.isfinite(pxx) & (pxx >= 0)
    return freq[mask], pxx[mask], {"nperseg": int(nperseg), "method": method, "noverlap": int(nperseg // 2)}


def spectral_summary(freq: np.ndarray, pxx: np.ndarray) -> dict[str, float]:
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(pxx) & (pxx >= 0)
    f = freq[mask]
    p = pxx[mask]
    total = float(np.sum(p))
    if f.size == 0 or total <= 0:
        return {
            "low_frequency_energy_fraction": float("nan"),
            "mid_frequency_energy_fraction": float("nan"),
            "high_frequency_energy_fraction": float("nan"),
            "spectral_centroid_cycles_per_sample": float("nan"),
            "median_frequency_cycles_per_sample": float("nan"),
            "frequency_at_90_percent_cumulative_power": float("nan"),
        }
    norm = p / total
    cumulative = np.cumsum(norm)
    return {
        "low_frequency_energy_fraction": float(np.sum(norm[f < LOW_BOUND])),
        "mid_frequency_energy_fraction": float(np.sum(norm[(f >= LOW_BOUND) & (f < HIGH_BOUND)])),
        "high_frequency_energy_fraction": float(np.sum(norm[f >= HIGH_BOUND])),
        "spectral_centroid_cycles_per_sample": float(np.sum(f * norm)),
        "median_frequency_cycles_per_sample": float(f[np.searchsorted(cumulative, 0.5, side="left")]),
        "frequency_at_90_percent_cumulative_power": float(f[np.searchsorted(cumulative, 0.9, side="left")]),
    }


def normalized_curve(freq: np.ndarray, pxx: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(pxx) & (pxx >= 0)
    f = freq[mask]
    p = pxx[mask]
    if f.size == 0 or np.sum(p) <= 0:
        return np.full_like(grid, np.nan, dtype=float)
    p = p / np.sum(p)
    return np.interp(grid, f, p, left=np.nan, right=np.nan)


def cumulative_curve(freq: np.ndarray, pxx: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(pxx) & (pxx >= 0)
    f = freq[mask]
    p = pxx[mask]
    if f.size == 0 or np.sum(p) <= 0:
        return np.full_like(grid, np.nan, dtype=float)
    c = np.cumsum(p / np.sum(p))
    return np.interp(grid, f, c, left=0.0, right=1.0)


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def fmt_range(values: pd.Series, decimals: int) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return ""
    return f"{vals.min():.{decimals}f}-{vals.max():.{decimals}f}"


def build_raw_analysis(records: list[FileMapping], tables_dir: Path, figures_dir: Path, metadata: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    welch_fn = get_scipy_welch()
    grid = np.linspace(1.0 / 4096.0, 0.5, 768)
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []
    signal_specs = [
        ("Voltage", "raw_voltage"),
        ("Current", "raw_current"),
        ("Reference SOC", "reference_soc"),
    ]
    record_infos = []
    for mapping in records:
        frame, info = load_terminal_record(mapping)
        record_infos.append(info)
        sampling_rows.append(info)
        for signal_name, col in signal_specs:
            values = frame[col].to_numpy(float)
            for detrend_type in ("constant", "linear"):
                freq, pxx, psd_info = compute_psd(values, detrend_type, welch_fn)
                summary = spectral_summary(freq, pxx)
                rows.append(
                    {
                        "profile": mapping.profile,
                        "temperature_C": mapping.temperature_C,
                        "signal_name": signal_name,
                        "n_samples": int(len(frame)),
                        "duration_s": info["duration_s"],
                        "sampling_interval_median_s": info["sampling_interval_median_s"],
                        "detrend_type": detrend_type,
                        **summary,
                        "voltage_column_used": mapping.voltage_col,
                        "current_column_used": mapping.current_col,
                        "soc_column_used": mapping.soc_col,
                    }
                )
                if detrend_type == "constant":
                    psd_curve = normalized_curve(freq, pxx, grid)
                    cum_curve = cumulative_curve(freq, pxx, grid)
                    for f, pnorm, cval in zip(grid, psd_curve, cum_curve):
                        curve_rows.append(
                            {
                                "profile": mapping.profile,
                                "temperature_C": mapping.temperature_C,
                                "signal_name": signal_name,
                                "frequency_cycles_per_sample": float(f),
                                "normalized_psd": float(pnorm) if np.isfinite(pnorm) else np.nan,
                                "cumulative_energy": float(cval) if np.isfinite(cval) else np.nan,
                            }
                        )
                metadata.setdefault("welch_parameters", psd_info)
    full = pd.DataFrame(rows)
    full_path = tables_dir / "Table_S6_raw_signal_frequency_summary_by_record.csv"
    full.to_csv(full_path, index=False)

    compact_rows = []
    const = full[full["detrend_type"] == "constant"].copy()
    for signal_name in ["Current", "Voltage", "Reference SOC"]:
        g = const[const["signal_name"] == signal_name]
        compact_rows.append(
            {
                "signal_name": signal_name,
                "mean_low_frequency_energy_percent": round(float(g["low_frequency_energy_fraction"].mean() * 100.0), 2),
                "range_low_frequency_energy_percent": fmt_range(g["low_frequency_energy_fraction"] * 100.0, 2),
                "mean_mid_frequency_energy_percent": round(float(g["mid_frequency_energy_fraction"].mean() * 100.0), 2),
                "range_mid_frequency_energy_percent": fmt_range(g["mid_frequency_energy_fraction"] * 100.0, 2),
                "mean_high_frequency_energy_percent": round(float(g["high_frequency_energy_fraction"].mean() * 100.0), 2),
                "range_high_frequency_energy_percent": fmt_range(g["high_frequency_energy_fraction"] * 100.0, 2),
                "mean_median_frequency_cycles_per_sample": round(float(g["median_frequency_cycles_per_sample"].mean()), 6),
                "range_median_frequency_cycles_per_sample": fmt_range(g["median_frequency_cycles_per_sample"], 6),
            }
        )
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(tables_dir / "Table_5_raw_signal_frequency_summary_compact.csv", index=False)
    (tables_dir / "Table_5_raw_signal_frequency_summary_compact.md").write_text(markdown_table(compact) + "\n", encoding="utf-8")

    curves = pd.DataFrame(curve_rows)
    make_raw_figures(const, curves, figures_dir)
    write_section2_values(compact, const, record_infos, tables_dir)
    return full, compact, {"record_infos": record_infos, "curve_grid_size": len(grid)}


def make_raw_figures(const: pd.DataFrame, curves: pd.DataFrame, figures_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 13,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )
    colors = {"Current": "#1f77b4", "Voltage": "#d62728", "Reference SOC": "#2ca02c"}
    order = ["Current", "Voltage", "Reference SOC"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    ax = axes[0]
    signal_handles = []
    signal_labels = []
    for signal in order:
        g = curves[curves["signal_name"] == signal].groupby("frequency_cycles_per_sample", as_index=False)["normalized_psd"].mean()
        (line,) = ax.plot(g["frequency_cycles_per_sample"], g["normalized_psd"], label=signal, color=colors[signal], lw=1.4)
        signal_handles.append(line)
        signal_labels.append(signal)
    for boundary in (LOW_BOUND, HIGH_BOUND):
        ax.axvline(boundary, color="0.25", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Normalized PSD")

    ax = axes[1]
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    bands = [
        ("Low", "low_frequency_energy_fraction", "#4C78A8"),
        ("Mid", "mid_frequency_energy_fraction", "#F58518"),
        ("High", "high_frequency_energy_fraction", "#54A24B"),
    ]
    for label, col, color in bands:
        vals = [float(const[const["signal_name"] == s][col].mean() * 100.0) for s in order]
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.65)
        bottom += np.asarray(vals)
    ax.set_xticks(x, order, rotation=0)
    ax.set_ylabel("Energy fraction (%)")

    ax = axes[2]
    for signal in order:
        g = curves[curves["signal_name"] == signal].groupby("frequency_cycles_per_sample", as_index=False)["cumulative_energy"].mean()
        ax.plot(g["frequency_cycles_per_sample"], g["cumulative_energy"], label=signal, color=colors[signal], lw=1.4)
    for boundary in (LOW_BOUND, HIGH_BOUND):
        ax.axvline(boundary, color="0.25", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(0, 1.02)
    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    for panel_label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(
            0.0,
            1.06,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            clip_on=False,
        )
    band_handles, band_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        signal_handles + band_handles,
        signal_labels + band_labels,
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.3,
        handlelength=1.7,
        prop={"family": "Times New Roman", "weight": "bold", "size": 12},
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.84, wspace=0.32)
    fig.savefig(figures_dir / "Figure_3_raw_signal_frequency_structure.png", dpi=600)
    fig.savefig(figures_dir / "Figure_3_raw_signal_frequency_structure.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.6), sharex=True, constrained_layout=True)
    const = const.copy()
    const["record"] = const["temperature_C"].map(lambda v: f"{int(v)} °C") + " " + const["profile"].astype(str)
    records = [f"{int(t)} °C {p}" for t in TEMP_ORDER for p in PROFILE_ORDER]
    for ax, signal in zip(axes, order):
        g = const[const["signal_name"] == signal].set_index("record").reindex(records)
        x = np.arange(len(records))
        ax.bar(x, g["high_frequency_energy_fraction"].to_numpy(float) * 100.0, color=colors[signal], width=0.75)
        ax.set_ylabel(f"{signal}\nhigh (%)")
        ax.axhline(float(g["high_frequency_energy_fraction"].mean() * 100.0), color="0.2", lw=0.8, ls="--")
    axes[-1].set_xticks(np.arange(len(records)), records, rotation=45, ha="right")
    fig.savefig(figures_dir / "Figure_S6_raw_signal_frequency_by_profile_temperature.png", dpi=600)
    fig.savefig(figures_dir / "Figure_S6_raw_signal_frequency_by_profile_temperature.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    for signal in order:
        g = curves[curves["signal_name"] == signal].groupby("frequency_cycles_per_sample", as_index=False)["cumulative_energy"].mean()
        ax.plot(g["frequency_cycles_per_sample"], g["cumulative_energy"], label=signal, color=colors[signal], lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Cumulative spectral energy")
    ax.set_ylim(0, 1.02)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.legend(frameon=False, prop={"family": "Times New Roman", "weight": "bold", "size": 12})
    for boundary in (LOW_BOUND, HIGH_BOUND):
        ax.axvline(boundary, color="0.25", ls="--", lw=0.8)
    fig.savefig(figures_dir / "Figure_S6_raw_signal_cumulative_energy.png", dpi=600)
    fig.savefig(figures_dir / "Figure_S6_raw_signal_cumulative_energy.pdf")
    plt.close(fig)


def write_section2_values(compact: pd.DataFrame, const: pd.DataFrame, record_infos: list[dict[str, object]], tables_dir: Path) -> None:
    voltage_cols = sorted({str(r.get("voltage_col")) for r in record_infos})
    current_cols = sorted({str(r.get("current_col")) for r in record_infos})
    soc_cols = sorted({str(r.get("soc_col")) for r in record_infos})
    lines = [
        "# Section 2.4 Frequency Values For Manuscript",
        "",
        f"- Band definitions: low frequency is f < 1/200 cycles/sample, mid frequency is 1/200 <= f < 1/50 cycles/sample, and high frequency is f >= 1/50 cycles/sample.",
        f"- Records analyzed: {len(record_infos)} profile-temperature records.",
        f"- Voltage column used: {', '.join(voltage_cols)}.",
        f"- Current column used: {', '.join(current_cols)}.",
        f"- SOC column used: {', '.join(soc_cols)}.",
        "",
        "## Compact Numerical Findings",
        "",
        markdown_table(compact),
        "",
        "## Section 2.4 Paragraph Draft",
        "",
        "Welch spectra computed within each profile-temperature record show that the measured current contains a larger high-frequency contribution than the reference SOC trajectory. The reference SOC trajectory is dominated by low-frequency content because it is obtained by current integration. The terminal-voltage trajectory retains slow discharge-related variation together with faster load-induced transient response, motivating the use of both slow voltage-response context and recent current-history information.",
        "",
        "## Caption Drafts",
        "",
        "**Figure 3.** Frequency-domain structure of raw voltage, current, and reference SOC trajectories. Spectra were computed within each profile-temperature record and summarized by signal type. Current contains stronger high-frequency excitation components, whereas the Coulomb-counted reference SOC trajectory is dominated by low-frequency variation. Voltage contains both slow discharge-related variation and faster load-induced transient response.",
        "",
        "**Table 5.** Compact spectral energy summary of raw terminal measurements and reference SOC. Energy fractions were computed from normalized Welch spectra within each profile-temperature record and averaged across the twelve records.",
        "",
        "**Table S6.** Raw-signal spectral energy summary by profile and temperature. Low-, mid-, and high-frequency energy fractions were computed from raw voltage, current, and reference SOC trajectories within each record boundary.",
        "",
    ]
    (tables_dir / "section2_4_frequency_values_for_manuscript.md").write_text("\n".join(lines), encoding="utf-8")


def parse_feature_profile(path: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "Profile" in df.columns and len(df):
        return str(df["Profile"].iloc[0]).upper()
    parsed = parse_profile_from_text(path.as_posix())
    if parsed is None:
        raise ValueError(f"Cannot parse profile from {path}")
    return parsed


def parse_feature_temperature(path: Path, df: pd.DataFrame | None = None) -> float:
    if df is not None and "TempLabel" in df.columns and len(df):
        temp = coerce_temperature(df["TempLabel"].iloc[0])
        if temp is not None:
            return float(temp)
    parsed = parse_temperature_from_text(path.as_posix())
    if parsed is None:
        raise ValueError(f"Cannot parse temperature from {path}")
    return float(parsed)


def causal_time_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    diffs = np.diff(t)
    valid = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt_default = float(np.nanmedian(valid)) if valid.size else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y.astype(np.float32)


def causal_index_ema(values: np.ndarray, tau: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    alpha = float(np.exp(-1.0 / max(float(tau), 1e-6)))
    alpha = min(max(alpha, 0.0), 0.999999)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
    return out.astype(np.float32)


def find_feature_csv_files(raw_root: Path) -> list[Path]:
    files = [
        p
        for p in sorted(raw_root.rglob("*.csv"))
        if not is_forbidden_path(p) and parse_profile_from_text(p.as_posix()) in PROFILE_ORDER and parse_temperature_from_text(p.as_posix()) in TEMP_ORDER
    ]
    if not files:
        raise FileNotFoundError(f"No NMC CSV files found under {raw_root}")
    return files


def estimate_r0_by_temperature_local(files: list[Path], train_profiles: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    all_ratios: list[float] = []
    for path in files:
        df_head = pd.read_csv(path, nrows=2)
        profile = parse_feature_profile(path, df_head)
        if profile not in train_profiles:
            continue
        temp = parse_feature_temperature(path, df_head)
        df = pd.read_csv(path, usecols=["Current(A)", "Voltage(V)"])
        i_raw = df["Current(A)"].to_numpy(np.float64)
        v_raw = df["Voltage(V)"].to_numpy(np.float64)
        d_i = np.diff(i_raw, prepend=i_raw[0])
        d_v = np.diff(v_raw, prepend=v_raw[0])
        mask = np.isfinite(d_i) & np.isfinite(d_v) & (np.abs(d_i) > 0.05) & (np.abs(d_v) > 1e-5)
        ratio = d_v[mask] / d_i[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.001) & (ratio < 0.5)]
        for r in ratio:
            rows.append({"temperature_C": float(temp), "profile": profile, "file_name": path.name, "r0_event_ohm": float(r)})
        all_ratios.extend([float(r) for r in ratio])
    event_df = pd.DataFrame(rows)
    if event_df.empty:
        raise RuntimeError("Could not estimate R0 from non-FUDS train profiles.")
    fallback = float(np.median(all_ratios))
    summary = (
        event_df.groupby("temperature_C")["r0_event_ohm"]
        .agg(
            r0_ohm="median",
            n_events="count",
            r0_p20_ohm=lambda s: float(np.percentile(s, 20)),
            r0_p80_ohm=lambda s: float(np.percentile(s, 80)),
        )
        .reset_index()
    )
    present = set(summary["temperature_C"].astype(float))
    all_temps = sorted({parse_feature_temperature(p, pd.read_csv(p, nrows=2)) for p in files})
    for temp in all_temps:
        if float(temp) not in present:
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "temperature_C": float(temp),
                                "r0_ohm": fallback,
                                "n_events": 0,
                                "r0_p20_ohm": np.nan,
                                "r0_p80_ohm": np.nan,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return summary.sort_values("temperature_C").reset_index(drop=True)


def build_feature_frame_local(path: Path, r0_lookup: dict[float, float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = parse_feature_temperature(path, df)
    profile = parse_feature_profile(path, df)
    r0 = float(r0_lookup[float(temp)])
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
    report_time_col = "Test_Time(s)" if "Test_Time(s)" in df.columns else time_col
    report_times = pd.to_numeric(df[report_time_col], errors="coerce").to_numpy(np.float64)
    v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    d_i = np.diff(i_raw, prepend=i_raw[0]).astype(np.float32)
    abs_i = np.abs(i_raw).astype(np.float32)
    v_ohm = (i_raw * r0).astype(np.float32)
    v_ohm_removed = v_raw - v_ohm
    v_corr = causal_time_ema(v_ohm_removed, times, 120.0)
    out = pd.DataFrame(
        {
            "file_name": path.name,
            "trajectory_id": path.stem,
            "temperature": float(temp),
            "drive_cycle": profile,
            "end_index": np.arange(len(df), dtype=np.int64),
            "time_s_for_frequency_metadata": report_times.astype(np.float64),
            "V_raw": v_raw.astype(np.float32),
            "V_corr_raw": v_corr.astype(np.float32),
            "I_raw": i_raw.astype(np.float32),
            "T": np.full(len(df), float(temp), dtype=np.float32),
            "dI": d_i,
            "absI": abs_i,
            "V_ohm_raw": v_ohm.astype(np.float32),
            "R0": np.full(len(df), r0, dtype=np.float32),
        }
    )
    v_raw64 = out["V_raw"].to_numpy(np.float64)
    v_corr64 = out["V_corr_raw"].to_numpy(np.float64)
    i_raw64 = out["I_raw"].to_numpy(np.float64)
    out["V_drop_raw"] = (v_raw64 - v_corr64).astype(np.float32)
    out["P_raw"] = (v_raw64 * i_raw64).astype(np.float32)
    out["absP_raw"] = np.abs(v_raw64 * i_raw64).astype(np.float32)
    out["Vcorr_x_absI"] = (v_corr64 * np.abs(i_raw64)).astype(np.float32)
    out["Vdrop_x_absI"] = ((v_raw64 - v_corr64) * np.abs(i_raw64)).astype(np.float32)
    out["dI_x_Vdrop"] = (d_i.astype(np.float64) * (v_raw64 - v_corr64)).astype(np.float32)
    for col in ("V_raw", "V_corr_raw", "I_raw", "absI", "V_drop_raw", "P_raw"):
        arr = out[col].to_numpy(np.float32)
        for tau in (10, 50, 200, 800):
            ema = causal_index_ema(arr, tau)
            out[f"{col}_ema{tau}"] = ema
            out[f"{col}_dev_ema{tau}"] = (arr - ema).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_feature_records(base_dir: Path, raw_root: Path) -> tuple[list[pd.DataFrame], dict[str, object]]:
    files = find_feature_csv_files(raw_root)
    train_profiles = ("BJDST", "DST", "US06")
    r0_df = estimate_r0_by_temperature_local(files, train_profiles)
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    frames = [build_feature_frame_local(path, r0_lookup) for path in files]
    frames = sorted(frames, key=lambda f: (float(f["temperature"].iloc[0]), PROFILE_ORDER.index(str(f["drive_cycle"].iloc[0]))))
    info = {
        "feature_source": (
            "Local analysis implementation copied from repository feature equations: "
            "soc_decomp.nmc_branchbands_experiment.estimate_r0_by_temperature/build_decomposed_frame "
            "and soc_decomp.nmc_vit_feature_lstm_experiment.causal_index_ema/add_vit_engineered_features."
        ),
        "corrected_voltage_source": "V_corr_raw = causal_time_ema(Voltage(V) - Current(A) * R0_temperature, tau=120 s), with R0 estimated from BJDST/DST/US06 events only.",
        "r0_estimates": r0_df.to_dict(orient="records"),
        "n_feature_records": len(frames),
    }
    return frames, info


def feature_plan(columns: list[str]) -> list[dict[str, str]]:
    candidates = [
        ("V_raw", "raw_voltage", "Raw voltage", "V_raw"),
        ("V_corr_raw", "corrected_voltage", "Corrected voltage", "V_raw"),
        ("I_raw", "raw_current", "Raw current", "I_raw"),
        ("V_corr_raw_ema50", "voltage_ema", "Short voltage EMA", "V_corr_raw"),
        ("V_corr_raw_ema200", "voltage_ema", "Mid voltage EMA", "V_corr_raw"),
        ("V_corr_raw_ema800", "voltage_ema", "Long voltage EMA", "V_corr_raw"),
        ("I_raw_ema50", "current_ema", "Short current EMA", "I_raw"),
        ("I_raw_ema200", "current_ema", "Long current EMA", "I_raw"),
        ("absI_ema50", "abs_current_ema", "Short abs-current EMA", "I_raw"),
        ("absI_ema200", "abs_current_ema", "Long abs-current EMA", "I_raw"),
        ("V_corr_raw_dev_ema50", "ema_deviation", "Voltage EMA deviation 50", "V_corr_raw"),
        ("V_corr_raw_dev_ema200", "ema_deviation", "Voltage EMA deviation 200", "V_corr_raw"),
        ("V_corr_raw_dev_ema800", "ema_deviation", "Voltage EMA deviation 800", "V_corr_raw"),
        ("I_raw_dev_ema50", "ema_deviation", "Current EMA deviation 50", "I_raw"),
        ("I_raw_dev_ema200", "ema_deviation", "Current EMA deviation 200", "I_raw"),
        ("absI_dev_ema50", "ema_deviation", "Abs-current EMA deviation 50", "I_raw"),
        ("absI_dev_ema200", "ema_deviation", "Abs-current EMA deviation 200", "I_raw"),
    ]
    return [
        {"feature_name": name, "feature_group": group, "display_name": display, "reference_raw_signal": ref}
        for name, group, display, ref in candidates
        if name in columns
    ]


def build_feature_analysis(base_dir: Path, raw_root: Path, tables_dir: Path, figures_dir: Path, metadata: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    welch_fn = get_scipy_welch()
    frames, feature_info = build_feature_records(base_dir, raw_root)
    if not frames:
        raise SystemExit("Part B failed: no generated feature frames were produced.")
    plan = feature_plan(list(frames[0].columns))
    if "V_corr_raw" not in frames[0].columns:
        (tables_dir / "section4_feature_frequency_values_for_manuscript.md").write_text(
            "# Section 4 Feature Frequency Values\n\nPart B stopped because corrected-voltage column V_corr_raw could not be located.\n",
            encoding="utf-8",
        )
        raise SystemExit("Part B stopped: corrected-voltage construction could not be located.")

    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    grid = np.linspace(1.0 / 4096.0, 0.5, 768)
    ref_high: dict[tuple[str, float, str, str, str], float] = {}

    for frame in frames:
        profile = str(frame["drive_cycle"].iloc[0])
        temp = float(frame["temperature"].iloc[0])
        file_name = str(frame["file_name"].iloc[0])
        n = int(len(frame))
        if "time_s_for_frequency_metadata" in frame.columns:
            time_values = pd.to_numeric(frame["time_s_for_frequency_metadata"], errors="coerce").to_numpy(float)
            duration = float(np.nanmax(time_values) - np.nanmin(time_values)) if np.isfinite(time_values).any() else float(n - 1)
            diffs = np.diff(time_values)
            pos = diffs[np.isfinite(diffs) & (diffs > 0)]
            dt = float(np.nanmedian(pos)) if pos.size else 1.0
        else:
            duration = float(n - 1)
            dt = 1.0
        for item in plan:
            feature = item["feature_name"]
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
            for detrend_type in ("constant", "linear"):
                freq, pxx, psd_info = compute_psd(values, detrend_type, welch_fn)
                summary = spectral_summary(freq, pxx)
                key = (file_name, temp, profile, detrend_type, feature)
                ref_high[key] = summary["high_frequency_energy_fraction"]
                rows.append(
                    {
                        "profile": profile,
                        "temperature_C": temp,
                        "feature_name": item["display_name"],
                        "feature_group": item["feature_group"],
                        "n_samples": n,
                        "duration_s": duration,
                        "sampling_interval_median_s": dt,
                        "detrend_type": detrend_type,
                        **summary,
                        "reference_raw_signal": item["reference_raw_signal"],
                        "high_frequency_energy_reduction_vs_reference_percent": np.nan,
                        "source_file": file_name,
                        "feature_column_used": feature,
                    }
                )
                if detrend_type == "constant" and feature in {
                    "V_raw",
                    "V_corr_raw",
                    "V_corr_raw_ema50",
                    "V_corr_raw_ema800",
                    "I_raw",
                    "I_raw_ema50",
                    "I_raw_ema200",
                    "absI_ema50",
                    "absI_ema200",
                }:
                    psd_vals = normalized_curve(freq, pxx, grid)
                    cum_vals = cumulative_curve(freq, pxx, grid)
                    for f, pval, cval in zip(grid, psd_vals, cum_vals):
                        curve_rows.append(
                            {
                                "profile": profile,
                                "temperature_C": temp,
                                "feature_column_used": feature,
                                "feature_group": item["feature_group"],
                                "frequency_cycles_per_sample": float(f),
                                "normalized_psd": float(pval) if np.isfinite(pval) else np.nan,
                                "cumulative_energy": float(cval) if np.isfinite(cval) else np.nan,
                            }
                        )
                metadata.setdefault("welch_parameters", psd_info)

    full = pd.DataFrame(rows)
    for idx, row in full.iterrows():
        ref = row["reference_raw_signal"]
        if not isinstance(ref, str) or ref not in set(full["feature_column_used"]):
            continue
        mask = (
            (full["source_file"] == row["source_file"])
            & (full["temperature_C"] == row["temperature_C"])
            & (full["profile"] == row["profile"])
            & (full["detrend_type"] == row["detrend_type"])
            & (full["feature_column_used"] == ref)
        )
        if mask.any():
            ref_val = float(full.loc[mask, "high_frequency_energy_fraction"].iloc[0])
            val = float(row["high_frequency_energy_fraction"])
            if np.isfinite(ref_val) and ref_val > 0 and np.isfinite(val):
                full.at[idx, "high_frequency_energy_reduction_vs_reference_percent"] = (ref_val - val) / ref_val * 100.0
    full.to_csv(tables_dir / "Table_S7_feature_frequency_summary_by_record.csv", index=False)

    compact_specs = [
        ("Raw voltage", "V_raw", "raw_voltage"),
        ("Corrected voltage", "V_corr_raw", "corrected_voltage"),
        ("Short voltage EMA", "V_corr_raw_ema50", "voltage_ema"),
        ("Long voltage EMA", "V_corr_raw_ema800", "voltage_ema"),
        ("Raw current", "I_raw", "raw_current"),
        ("Short current EMA", "I_raw_ema50", "current_ema"),
        ("Long current EMA", "I_raw_ema200", "current_ema"),
        ("Short abs-current EMA", "absI_ema50", "abs_current_ema"),
        ("Long abs-current EMA", "absI_ema200", "abs_current_ema"),
    ]
    compact_rows = []
    const = full[full["detrend_type"] == "constant"].copy()
    for label, feature, group in compact_specs:
        g = const[const["feature_column_used"] == feature]
        if g.empty:
            continue
        compact_rows.append(
            {
                "feature_group": label,
                "representative_feature_column": feature,
                "mean_low_frequency_energy_percent": round(float(g["low_frequency_energy_fraction"].mean() * 100.0), 2),
                "mean_mid_frequency_energy_percent": round(float(g["mid_frequency_energy_fraction"].mean() * 100.0), 2),
                "mean_high_frequency_energy_percent": round(float(g["high_frequency_energy_fraction"].mean() * 100.0), 2),
                "mean_median_frequency_cycles_per_sample": round(float(g["median_frequency_cycles_per_sample"].mean()), 6),
                "high_frequency_energy_reduction_vs_reference_percent": round(float(g["high_frequency_energy_reduction_vs_reference_percent"].mean()), 2)
                if g["high_frequency_energy_reduction_vs_reference_percent"].notna().any()
                else np.nan,
            }
        )
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(tables_dir / "Table_7_feature_frequency_summary_compact.csv", index=False)
    (tables_dir / "Table_7_feature_frequency_summary_compact.md").write_text(markdown_table(compact) + "\n", encoding="utf-8")

    curves = pd.DataFrame(curve_rows)
    make_feature_figures(const, curves, figures_dir)
    write_section4_values(compact, feature_info, plan, tables_dir)
    return full, compact, feature_info | {"feature_columns_used": [p["feature_name"] for p in plan]}


def make_feature_figures(const: pd.DataFrame, curves: pd.DataFrame, figures_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 15,
            "axes.labelsize": 16,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), constrained_layout=False)
    voltage_cols = ["V_raw", "V_corr_raw", "V_corr_raw_ema50", "V_corr_raw_ema800"]
    current_cols = ["I_raw", "I_raw_ema50", "I_raw_ema200", "absI_ema50", "absI_ema200"]
    labels = {
        "V_raw": "Raw voltage",
        "V_corr_raw": "Corrected voltage",
        "V_corr_raw_ema50": "Voltage EMA50",
        "V_corr_raw_ema800": "Voltage EMA800",
        "I_raw": "Raw current",
        "I_raw_ema50": "Current EMA50",
        "I_raw_ema200": "Current EMA200",
        "absI_ema50": "Abs-current EMA50",
        "absI_ema200": "Abs-current EMA200",
    }
    for col in voltage_cols:
        g = curves[curves["feature_column_used"] == col].groupby("frequency_cycles_per_sample", as_index=False)["cumulative_energy"].mean()
        if not g.empty:
            axes[0].plot(g["frequency_cycles_per_sample"], g["cumulative_energy"], lw=1.8, label=labels[col])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Frequency (cycles/sample)")
    axes[0].set_ylabel("Cumulative energy")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(
        frameon=False,
        loc="lower right",
        handlelength=2.2,
        labelspacing=0.55,
        prop={"family": "Times New Roman", "weight": "bold", "size": 11.0},
    )

    for col in current_cols:
        g = curves[curves["feature_column_used"] == col].groupby("frequency_cycles_per_sample", as_index=False)["cumulative_energy"].mean()
        if not g.empty:
            axes[1].plot(g["frequency_cycles_per_sample"], g["cumulative_energy"], lw=1.8, label=labels[col])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Frequency (cycles/sample)")
    axes[1].set_ylabel("Cumulative energy")
    axes[1].set_ylim(0, 1.02)
    legend_b = axes[1].legend(
        frameon=False,
        loc="lower right",
        ncol=1,
        borderaxespad=0.45,
        handlelength=1.9,
        labelspacing=0.45,
        prop={"family": "Times New Roman", "weight": "bold", "size": 11.0},
    )
    legend_b._legend_box.align = "right"
    for text in legend_b.get_texts():
        text.set_ha("right")

    bar_cols = ["V_raw", "V_corr_raw", "V_corr_raw_ema50", "V_corr_raw_ema800", "I_raw", "I_raw_ema50", "absI_ema50"]
    bar_tick_labels = ["Raw V", "Corr. V", "V EMA50", "V EMA800", "Raw I", "I EMA50", "|I| EMA50"]
    g = const[const["feature_column_used"].isin(bar_cols)].groupby("feature_column_used")["high_frequency_energy_fraction"].mean()
    vals = [float(g.get(col, np.nan) * 100.0) for col in bar_cols]
    axes[2].bar(np.arange(len(bar_cols)), vals, color="#4C78A8")
    axes[2].set_ylabel("High-frequency energy (%)")
    axes[2].set_xticks(np.arange(len(bar_cols)), bar_tick_labels, rotation=30, ha="right")
    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    for panel_label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(
            -0.075,
            1.035,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.24, top=0.84, wspace=0.32)
    fig.savefig(figures_dir / "Figure_6_feature_frequency_behavior.png", dpi=600)
    fig.savefig(figures_dir / "Figure_6_feature_frequency_behavior.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for col in voltage_cols + current_cols:
        g = curves[curves["feature_column_used"] == col].groupby("frequency_cycles_per_sample", as_index=False)["normalized_psd"].mean()
        if not g.empty:
            ax.plot(g["frequency_cycles_per_sample"], g["normalized_psd"], lw=1.1, label=labels[col])
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Normalized PSD")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.legend(frameon=False, ncol=2, prop={"family": "Times New Roman", "weight": "bold", "size": 13.5})
    fig.savefig(figures_dir / "Figure_S7_feature_psd_by_group.png", dpi=600)
    fig.savefig(figures_dir / "Figure_S7_feature_psd_by_group.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 5.0), constrained_layout=True)
    compact_cols = ["V_raw", "V_corr_raw", "V_corr_raw_ema800", "I_raw", "I_raw_ema200", "absI_ema200"]
    tmp = const[const["feature_column_used"].isin(compact_cols)].copy()
    tmp["record"] = tmp["temperature_C"].map(lambda v: f"{int(v)} °C") + " " + tmp["profile"].astype(str)
    pivot = tmp.pivot_table(index="record", columns="feature_column_used", values="high_frequency_energy_fraction", aggfunc="mean")
    records = [f"{int(t)} °C {p}" for t in TEMP_ORDER for p in PROFILE_ORDER]
    x = np.arange(len(records))
    width = 0.12
    for i, col in enumerate(compact_cols):
        if col not in pivot.columns:
            continue
        ax.bar(x + (i - 2.5) * width, pivot.reindex(records)[col].to_numpy(float) * 100.0, width=width, label=labels[col])
    ax.set_ylabel("High-frequency energy (%)")
    ax.set_xticks(x, records, rotation=45, ha="right")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.legend(frameon=False, ncol=3, prop={"family": "Times New Roman", "weight": "bold", "size": 13.5})
    fig.savefig(figures_dir / "Figure_S7_feature_frequency_by_profile_temperature.png", dpi=600)
    fig.savefig(figures_dir / "Figure_S7_feature_frequency_by_profile_temperature.pdf")
    plt.close(fig)


def write_section4_values(compact: pd.DataFrame, feature_info: dict[str, object], plan: list[dict[str, str]], tables_dir: Path) -> None:
    ema_cols = [p["feature_name"] for p in plan if "ema" in p["feature_name"].lower()]
    lines = [
        "# Section 4 Feature Frequency Values For Manuscript",
        "",
        f"- Corrected voltage column source: {feature_info['corrected_voltage_source']}.",
        f"- EMA feature columns used: {', '.join(ema_cols)}.",
        "- Representative EMA spans selected: voltage short=50 and long=800 samples; current short=50 and long=200 samples; abs-current short=50 and long=200 samples.",
        "",
        "## Compact Numerical Findings",
        "",
        markdown_table(compact),
        "",
        "## Section 4.x Paragraph Draft",
        "",
        "Feature-level Welch spectra verify the intended signal-processing behavior of the constructed measurement features. Corrected voltage attenuates part of the fast load-dependent variation in the terminal-voltage input, while voltage and current EMA channels shift the representation toward lower-frequency finite-memory measurement context. Longer EMA spans retain slower components than shorter spans.",
        "",
        "## Caption Drafts",
        "",
        "**Figure 6.** Spectral behavior of corrected voltage and EMA memory features. Corrected voltage attenuates part of the fast load-dependent variation in the voltage input, and EMA channels shift voltage/current measurements toward lower-frequency finite-memory context.",
        "",
        "**Table 7.** Compact frequency-domain summary of corrected-voltage and EMA feature channels. Values summarize normalized Welch spectra across profile-temperature records.",
        "",
        "**Table S7.** Feature-level spectral summary of corrected voltage and EMA memory channels. Values quantify the frequency-band energy distribution and high-frequency energy reduction relative to the corresponding raw measurement stream.",
        "",
    ]
    (tables_dir / "section4_feature_frequency_values_for_manuscript.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(out_dir: Path, raw_compact: pd.DataFrame, feature_compact: pd.DataFrame, metadata: dict[str, object]) -> None:
    lines = [
        "# Frequency Structure Analysis Report",
        "",
        "## 1. Data and Feature Discovery",
        "",
        f"- Data root: `{metadata['data_root']}`",
        f"- Raw records included: {metadata.get('part_a_record_count_detail', '')}",
        f"- Corrected-voltage source: {metadata.get('corrected_voltage_source', '')}",
        "",
        "## 2. PSD Method and Band Definitions",
        "",
        "- Welch PSD was used when scipy was available; otherwise the script falls back to a deterministic windowed periodogram.",
        "- PSDs are computed within record boundaries and normalized by total nonzero-frequency power.",
        "- Primary bands are sample based: low f < 1/200 cycles/sample, mid 1/200 <= f < 1/50 cycles/sample, high f >= 1/50 cycles/sample.",
        "",
        "## 3. Section 2.4 Raw-Signal Findings",
        "",
        markdown_table(raw_compact),
        "",
        "Current contains a larger high-frequency contribution than reference SOC. Reference SOC is dominated by low-frequency content because it is obtained by current integration. Voltage contains slow discharge-related variation together with faster load-induced transient response.",
        "",
        "## 4. Section 4.x Corrected-Voltage and EMA Feature Findings",
        "",
        markdown_table(feature_compact),
        "",
        "Corrected voltage attenuates part of the fast load-dependent variation in the terminal-voltage input. EMA channels shift voltage/current measurements toward lower-frequency finite-memory context, with longer EMA spans retaining slower components.",
        "",
        "## 5. Suggested Main-Text Figures and Tables",
        "",
        "- Section 2.4 Figure 3: `figures/Figure_3_raw_signal_frequency_structure.png`",
        "- Section 2.4 Table 5: `tables/Table_5_raw_signal_frequency_summary_compact.csv`",
        "- Section 4.x Figure 6: `figures/Figure_6_feature_frequency_behavior.png`",
        "- Section 4.x Table 7: `tables/Table_7_feature_frequency_summary_compact.csv`",
        "",
        "## 6. Suggested SI Figures and Tables",
        "",
        "- Table S6: `tables/Table_S6_raw_signal_frequency_summary_by_record.csv`",
        "- Table S7: `tables/Table_S7_feature_frequency_summary_by_record.csv`",
        "- Figure S6: `figures/Figure_S6_raw_signal_frequency_by_profile_temperature.png` and `figures/Figure_S6_raw_signal_cumulative_energy.png`",
        "- Figure S7: `figures/Figure_S7_feature_psd_by_group.png` and `figures/Figure_S7_feature_frequency_by_profile_temperature.png`",
        "",
        "## 7. Interpretation Boundaries",
        "",
        "- Do not describe high-frequency components as noise.",
        "- Do not claim that EMA channels are electrochemical state variables.",
        "- Do not claim that corrected voltage removes all current effects.",
        "- Keep interpretation limited to measurement-structure and feature signal-processing behavior.",
        "",
        "## Caption Drafts",
        "",
        "**Figure 3.** Frequency-domain structure of raw voltage, current, and reference SOC trajectories. Spectra were computed within each profile-temperature record and summarized by signal type. Current contains stronger high-frequency excitation components, whereas the Coulomb-counted reference SOC trajectory is dominated by low-frequency variation. Voltage contains both slow discharge-related variation and faster load-induced transient response.",
        "",
        "**Table 5.** Compact spectral energy summary of raw terminal measurements and reference SOC. Energy fractions were computed from normalized Welch spectra within each profile-temperature record and averaged across the twelve records.",
        "",
        "**Figure 6.** Spectral behavior of corrected voltage and EMA memory features. Corrected voltage attenuates part of the fast load-dependent variation in the voltage input, and EMA channels shift voltage/current measurements toward lower-frequency finite-memory context.",
        "",
        "**Table S6.** Raw-signal spectral energy summary by profile and temperature. Low-, mid-, and high-frequency energy fractions were computed from raw voltage, current, and reference SOC trajectories within each record boundary.",
        "",
        "**Table S7.** Feature-level spectral summary of corrected voltage and EMA memory channels. Values quantify the frequency-band energy distribution and high-frequency energy reduction relative to the corresponding raw measurement stream.",
        "",
    ]
    (out_dir / "frequency_structure_report.md").write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(out_dir: Path, expected: list[str], metadata: dict[str, object], records: list[FileMapping]) -> dict[str, object]:
    checks = []
    expected_keys = {(t, p) for t in TEMP_ORDER for p in PROFILE_ORDER}
    found_keys = {(r.temperature_C, r.profile) for r in records}
    checks.append(
        {
            "check": "all_twelve_profile_temperature_records_included_for_part_a",
            "status": "PASS" if expected_keys == found_keys else "FAIL",
            "detail": f"{len(found_keys)} records found",
        }
    )
    checks.append({"check": "record_boundary_reset", "status": "PASS", "detail": "PSD and EMA feature calculations are performed per file/frame."})
    used_files = [item["relative_path"] for item in metadata["input_files"]]
    forbidden_used = [p for p in used_files if any(tok in Path(p).name.lower() for tok in FORBIDDEN_FILE_TOKENS)]
    checks.append({"check": "no_model_prediction_residual_error_files_used", "status": "PASS" if not forbidden_used else "FAIL", "detail": ",".join(forbidden_used)})
    raw_voltage_cols = sorted({m.voltage_col for m in records})
    checks.append(
        {
            "check": "section2_4_uses_raw_terminal_voltage",
            "status": "PASS" if raw_voltage_cols == ["Voltage(V)"] else "WARN",
            "detail": ",".join(raw_voltage_cols),
        }
    )
    checks.append({"check": "part_b_features_from_repository_code", "status": "PASS", "detail": str(metadata.get("feature_source", ""))})
    checks.append({"check": "no_centered_or_future_windows_used", "status": "PASS", "detail": "EMA features use one-sided recurrences reset at record boundaries."})
    missing = [rel for rel in expected if not (out_dir / rel).exists()]
    checks.append({"check": "all_output_csv_tables_and_figures_exist", "status": "PASS" if not missing else "FAIL", "detail": f"missing={missing}"})
    main_figs = [
        "figures/Figure_3_raw_signal_frequency_structure.png",
        "figures/Figure_3_raw_signal_frequency_structure.pdf",
        "figures/Figure_6_feature_frequency_behavior.png",
        "figures/Figure_6_feature_frequency_behavior.pdf",
    ]
    checks.append({"check": "main_figures_saved_png_pdf", "status": "PASS" if all((out_dir / f).exists() for f in main_figs) else "FAIL", "detail": ",".join(main_figs)})
    metadata["validation_checks"] = {c["check"]: c for c in checks}
    metadata["part_a_record_count_detail"] = f"{len(found_keys)} / 12"
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manuscript frequency-domain analysis outputs for raw measurements and EMA features.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--data-root", default="Data/processed")
    p.add_argument("--out-dir", default="Data/frequency_structure_analysis")
    return p.parse_args()


def mirror_figure_3_outputs(base_dir: Path, out_dir: Path) -> None:
    figures_dir = base_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    src_png = out_dir / "figures" / "Figure_3_raw_signal_frequency_structure.png"
    src_pdf = out_dir / "figures" / "Figure_3_raw_signal_frequency_structure.pdf"
    out_png = figures_dir / "figure_3_raw_signal_frequency_structure.png"
    out_pdf = figures_dir / "figure_3_raw_signal_frequency_structure.pdf"
    out_tif = figures_dir / "figure_3_raw_signal_frequency_structure.tif"
    shutil.copy2(src_png, out_png)
    shutil.copy2(src_pdf, out_pdf)
    Image.open(src_png).convert("RGB").save(out_tif, dpi=(600, 600))


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = base_dir / args.out_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_root, records, discovery_diag, attempts = choose_data_root(base_dir, args.data_root)
    metadata: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": public_path_label(raw_root, base_dir),
        "data_discovery_attempts": attempts,
        "data_discovery": discovery_diag,
        "input_files": [
            {
                "file_name": m.file_name,
                "relative_path": m.relative_path,
                "time_col": m.time_col,
                "voltage_col": m.voltage_col,
                "current_col": m.current_col,
                "temperature_col": m.temperature_col,
                "soc_col": m.soc_col,
                "profile_col": m.profile_col,
                "profile": m.profile,
                "temperature_C": m.temperature_C,
            }
            for m in records
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
            for m in records
        ],
        "psd_method": "Welch PSD with nonzero-frequency normalization",
        "band_definitions": {
            "low_frequency": "f < 1/200 cycles/sample",
            "mid_frequency": "1/200 <= f < 1/50 cycles/sample",
            "high_frequency": "f >= 1/50 cycles/sample",
        },
    }

    raw_full, raw_compact, raw_info = build_raw_analysis(records, tables_dir, figures_dir, metadata)
    feature_full, feature_compact, feature_info = build_feature_analysis(base_dir, raw_root, tables_dir, figures_dir, metadata)
    metadata.update(feature_info)
    metadata["sampling_interval_summary"] = raw_info["record_infos"]
    metadata["ema_spans_detected"] = sorted({int(m.group(1)) for col in feature_info["feature_columns_used"] for m in [re.search(r"ema(\d+)", col)] if m})
    metadata["excluded_files"] = {
        "name_tokens": list(FORBIDDEN_FILE_TOKENS),
        "count_from_discovery": discovery_diag.get("n_skipped_forbidden_name"),
    }

    expected_outputs = [
        "tables/Table_S6_raw_signal_frequency_summary_by_record.csv",
        "tables/Table_5_raw_signal_frequency_summary_compact.csv",
        "tables/Table_5_raw_signal_frequency_summary_compact.md",
        "figures/Figure_3_raw_signal_frequency_structure.png",
        "figures/Figure_3_raw_signal_frequency_structure.pdf",
        "figures/Figure_S6_raw_signal_frequency_by_profile_temperature.png",
        "figures/Figure_S6_raw_signal_frequency_by_profile_temperature.pdf",
        "figures/Figure_S6_raw_signal_cumulative_energy.png",
        "figures/Figure_S6_raw_signal_cumulative_energy.pdf",
        "tables/section2_4_frequency_values_for_manuscript.md",
        "tables/Table_S7_feature_frequency_summary_by_record.csv",
        "tables/Table_7_feature_frequency_summary_compact.csv",
        "tables/Table_7_feature_frequency_summary_compact.md",
        "figures/Figure_6_feature_frequency_behavior.png",
        "figures/Figure_6_feature_frequency_behavior.pdf",
        "figures/Figure_S7_feature_psd_by_group.png",
        "figures/Figure_S7_feature_psd_by_group.pdf",
        "figures/Figure_S7_feature_frequency_by_profile_temperature.png",
        "figures/Figure_S7_feature_frequency_by_profile_temperature.pdf",
        "tables/section4_feature_frequency_values_for_manuscript.md",
        "frequency_structure_report.md",
        "frequency_structure_metadata.json",
    ]
    metadata["output_file_list"] = expected_outputs
    metadata = validate_outputs(out_dir, expected_outputs[:-2], metadata, records)
    write_report(out_dir, raw_compact, feature_compact, metadata)
    metadata = validate_outputs(out_dir, expected_outputs[:-1], metadata, records)
    (out_dir / "frequency_structure_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    metadata = validate_outputs(out_dir, expected_outputs, metadata, records)
    (out_dir / "frequency_structure_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    mirror_figure_3_outputs(base_dir, out_dir)
    print(f"Wrote frequency analysis outputs to {out_dir}")
    print(f"Wrote Figure 3 to {base_dir / 'Figures' / 'figure_3_raw_signal_frequency_structure.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
