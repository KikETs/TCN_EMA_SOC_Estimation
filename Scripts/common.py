from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
FIGURES = ROOT / "Figures"
TABLES = ROOT / "Tables"

PROFILE_ORDER = ("BJDST", "DST", "US06", "FUDS")
TEMP_ORDER = (0.0, 25.0, 45.0)
TEMP_COLORS = {0.0: "#1f77b4", 25.0: "#2ca02c", 45.0: "#d62728"}


def manuscript_style(font_size: float = 12.0, bold: bool = True) -> None:
    weight = "bold" if bold else "normal"
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": weight,
            "axes.labelweight": weight,
            "axes.titleweight": weight,
            "axes.linewidth": 1.0,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    ensure_dirs()
    for ext in ("png", "tif", "pdf"):
        kwargs = {"dpi": dpi} if ext in {"png", "tif"} else {}
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)


def save_table(df: pd.DataFrame, stem: str, digits: int = 4) -> None:
    ensure_dirs()
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(digits)
    out.to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(out.to_markdown(index=False), encoding="utf-8")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def norm_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(columns: list[str], aliases: list[str]) -> str | None:
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


def read_record(temp: float, profile: str) -> pd.DataFrame:
    temp_i = int(temp)
    path = DATA / "processed" / f"{temp_i}C" / f"NMC_{temp_i}C_{profile}.csv"
    if not path.exists():
        path = DATA / "processed" / f"NMC_{temp_i}C_{profile}.csv"
    require(path)
    return pd.read_csv(path)


def standardize_record(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    time_col = find_col(cols, ["Test_Time(s)", "t_global(s)", "Step_Time(s)", "Time", "Data_Point"])
    cycle_time_col = find_col(cols, ["Step_Time(s)", "Step_Time", "Time"])
    voltage_col = find_col(cols, ["Voltage(V)", "Voltage", "V_raw", "V"])
    current_col = find_col(cols, ["Current(A)", "Current", "I_raw", "I"])
    soc_col = find_col(cols, ["SOC_CC", "SOC_CC(%)", "SOC", "SOC_fraction", "y_true"])
    temp_col = find_col(cols, ["TempLabel", "temperature_C", "T"])
    missing = [name for name, col in {"time": time_col, "voltage": voltage_col, "current": current_col, "soc": soc_col}.items() if col is None]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = pd.DataFrame(
        {
            "time_s": pd.to_numeric(df[time_col], errors="coerce"),
            "cycle_time_s": pd.to_numeric(df[cycle_time_col or time_col], errors="coerce"),
            "V_raw": pd.to_numeric(df[voltage_col], errors="coerce"),
            "I_raw": pd.to_numeric(df[current_col], errors="coerce"),
            "SOC": pd.to_numeric(df[soc_col], errors="coerce"),
        }
    )
    if temp_col:
        out["T"] = df[temp_col]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "V_raw", "I_raw", "SOC"]).reset_index(drop=True)
    if out["SOC"].max() <= 1.5:
        out["SOC_percent"] = out["SOC"] * 100.0
    else:
        out["SOC_percent"] = out["SOC"]
    out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])
    out["cycle_time_s"] = out["cycle_time_s"] - float(out["cycle_time_s"].iloc[0])
    return out


def first_current_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    t = frame["cycle_time_s"].to_numpy(float)
    resets = np.where(np.diff(t) < -1e-9)[0]
    return frame.iloc[: int(resets[0]) + 1].copy() if resets.size else frame.copy()


def simple_psd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.nanmean(x)
    if x.size < 8:
        raise ValueError("Too few finite samples for PSD.")
    win = np.hanning(x.size)
    spectrum = np.fft.rfft(x * win)
    power = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(x.size, d=1.0)
    keep = freq > 0
    freq = freq[keep]
    power = power[keep]
    total = power.sum()
    if total > 0:
        power = power / total
    return freq, power


def cumulative_energy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq, p = simple_psd(values)
    return freq, np.cumsum(p)


def high_energy_fraction(values: np.ndarray, high_bound: float = 1.0 / 50.0) -> float:
    freq, p = simple_psd(values)
    return float(p[freq >= high_bound].sum() * 100.0)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04, size: float = 16.0) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=size, fontweight="bold", clip_on=False)


def format_corr_voltage_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "V_corr_raw" not in out.columns:
        # fallback for already processed records without CEMA channels
        out["V_corr_raw"] = out["V_raw"]
    return out


def mean_min_max(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    return f"{values.mean():.3f} ({values.min():.3f}-{values.max():.3f})"


def temp_label(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    return f"{int(v)} °C" if math.isclose(v, int(v)) else f"{v:g} °C"
