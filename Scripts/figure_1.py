from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROFILE_ORDER = ("BJDST", "DST", "US06", "FUDS")
PROFILE_PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")
TEMP_ORDER = (0.0, 25.0, 45.0)
TEMP_COLORS = {
    0.0: "#1f77b4",
    25.0: "#2ca02c",
    45.0: "#d62728",
}
TEMP_ALPHA = {
    0.0: 0.95,
    25.0: 0.52,
    45.0: 0.58,
}

COLUMN_ALIASES = {
    "cycle_time": ["Step_Time(s)", "Step_Time", "step_time", "Time", "time", "t"],
    "trajectory_time": ["Test_Time(s)", "Test_Time", "t_global(s)", "Data_Point", "index"],
    "voltage": ["Voltage(V)", "Voltage", "voltage", "V_corr_raw", "V_corr", "V"],
    "current": ["Current(A)", "Current", "current", "I_raw", "I"],
    "soc": ["SOC_CC", "SOC_CC(%)", "SOC", "soc", "SOC_true", "State_of_Charge"],
    "profile": ["Profile", "profile", "drive_cycle"],
    "temperature": ["TempLabel", "T", "Temp", "Temperature(C)", "Temperature", "temperature"],
}


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns: list[str], aliases: list[str]) -> str | None:
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


def parse_temp_from_text(text: str) -> float | None:
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*C(?![a-z])", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_profile_from_text(text: str) -> str | None:
    upper = text.upper()
    for profile in PROFILE_ORDER:
        if re.search(rf"(?<![A-Z0-9]){profile}(?![A-Z0-9])", upper):
            return profile
    return None


def coerce_temp(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_temp_from_text(value)
        if parsed is not None:
            return parsed
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_files(data_root: Path) -> dict[tuple[float, str], Path]:
    found: dict[tuple[float, str], Path] = {}
    for path in sorted(data_root.rglob("*.csv")):
        lower = path.name.lower()
        if any(token in lower for token in ("prediction", "residual", "ablation", "perturbation", "baseline", "rotation", "checkpoint", "forbidden")):
            continue
        profile = parse_profile_from_text(path.as_posix())
        temp = parse_temp_from_text(path.as_posix())
        if profile in PROFILE_ORDER and temp in TEMP_ORDER:
            found[(float(temp), str(profile))] = path
    return found


def load_terminal_frame(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=5)
    columns = list(head.columns)
    cycle_time_col = find_column(columns, COLUMN_ALIASES["cycle_time"])
    trajectory_time_col = find_column(columns, COLUMN_ALIASES["trajectory_time"])
    voltage_col = find_column(columns, COLUMN_ALIASES["voltage"])
    current_col = find_column(columns, COLUMN_ALIASES["current"])
    soc_col = find_column(columns, COLUMN_ALIASES["soc"])
    profile_col = find_column(columns, COLUMN_ALIASES["profile"])
    temp_col = find_column(columns, COLUMN_ALIASES["temperature"])
    required = {"cycle_time": cycle_time_col or trajectory_time_col, "voltage": voltage_col, "current": current_col, "soc": soc_col}
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    usecols = sorted({c for c in (cycle_time_col, trajectory_time_col, voltage_col, current_col, soc_col, profile_col, temp_col) if c is not None})
    raw = pd.read_csv(path, usecols=usecols)
    cycle_source = cycle_time_col or trajectory_time_col
    trajectory_source = trajectory_time_col or cycle_time_col
    out = pd.DataFrame(
        {
            "cycle_time_s": pd.to_numeric(raw[cycle_source], errors="coerce"),
            "trajectory_time_s": pd.to_numeric(raw[trajectory_source], errors="coerce"),
            "voltage_V": pd.to_numeric(raw[voltage_col], errors="coerce"),
            "current_A": pd.to_numeric(raw[current_col], errors="coerce"),
            "soc": pd.to_numeric(raw[soc_col], errors="coerce"),
        }
    )
    if profile_col:
        out["profile"] = raw[profile_col].astype(str)
    if temp_col:
        out["temperature_C"] = raw[temp_col].map(coerce_temp)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["cycle_time_s", "trajectory_time_s", "voltage_V", "current_A", "soc"]).reset_index(drop=True)
    if out["soc"].max() <= 1.5:
        out["soc_percent"] = out["soc"] * 100.0
    else:
        out["soc_percent"] = out["soc"]
    out["cycle_time_s"] = out["cycle_time_s"] - float(out["cycle_time_s"].iloc[0])
    out["trajectory_time_s"] = out["trajectory_time_s"] - float(out["trajectory_time_s"].iloc[0])
    return out


def one_current_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    time = frame["cycle_time_s"].to_numpy(float)
    resets = np.where(np.diff(time) < -1e-9)[0]
    if resets.size:
        return frame.iloc[: int(resets[0]) + 1].copy()
    return frame.copy()


def make_figure(data_root: Path, out_path: Path, current_temp: float) -> None:
    files = discover_files(data_root)
    missing = [(temp, profile) for temp in TEMP_ORDER for profile in PROFILE_ORDER if (temp, profile) not in files]
    if missing:
        raise FileNotFoundError(f"Missing temperature/profile CSV files: {missing}")

    frames = {(temp, profile): load_terminal_frame(path) for (temp, profile), path in files.items()}

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 0.8,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11.0,
        }
    )

    fig, axes = plt.subplots(
        3,
        len(PROFILE_ORDER),
        figsize=(11.2, 6.6),
        constrained_layout=False,
        sharex=False,
    )
    row_labels = ("Current (A)", "Voltage (V)", "SOC (%)")

    for col, profile in enumerate(PROFILE_ORDER):
        representative_key = (current_temp, profile)
        if representative_key not in frames:
            representative_key = (TEMP_ORDER[0], profile)
        current_frame = one_current_cycle(frames[representative_key])
        axes[0, col].plot(current_frame["cycle_time_s"], current_frame["current_A"], color="black", lw=0.8)
        axes[0, col].text(
            0.0,
            1.12,
            PROFILE_PANEL_LABELS[col],
            transform=axes[0, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=16.0,
            fontweight="bold",
            clip_on=False,
        )

        for temp in TEMP_ORDER:
            frame = frames[(temp, profile)]
            label = f"{int(temp)}°C"
            axes[1, col].plot(
                frame["trajectory_time_s"],
                frame["voltage_V"],
                color=TEMP_COLORS[temp],
                alpha=TEMP_ALPHA[temp],
                lw=0.9,
                label=label,
            )
            axes[2, col].plot(
                frame["trajectory_time_s"],
                frame["soc_percent"],
                color=TEMP_COLORS[temp],
                alpha=TEMP_ALPHA[temp],
                lw=0.9,
                label=label,
            )

        for row in range(3):
            ax = axes[row, col]
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel(row_labels[row])
        axes[2, col].set_ylim(-2, 90)

    handles, labels = axes[1, -1].get_legend_handles_labels()
    fig.supxlabel("Trajectory time (s)", y=0.075, fontsize=12.0)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 0.012))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.88, wspace=0.23, hspace=0.24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg")
    fig.savefig(out_path.with_suffix(".png"), dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(out_path) as img:
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, "white")
            background.paste(img, mask=img.getchannel("A"))
            background.save(out_path, compression="tiff_lzw", dpi=(600, 600))
        elif img.mode != "RGB":
            img.convert("RGB").save(out_path, compression="tiff_lzw", dpi=(600, 600))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot manuscript Figure 1 terminal current, voltage, and SOC trajectories.")
    p.add_argument("--data-root", default="Data/processed")
    p.add_argument(
        "--out",
        default="Figures/figure_1_current_voltage_soc_by_profile.tif",
    )
    p.add_argument("--current-temp", type=float, default=25.0, help="Temperature record used for the representative current cycle.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(Path(args.data_root).expanduser().resolve(), Path(args.out), float(args.current_temp))


if __name__ == "__main__":
    main()
