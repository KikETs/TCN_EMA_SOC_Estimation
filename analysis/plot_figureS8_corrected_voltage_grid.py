from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_frequency_structure_analysis import (  # noqa: E402
    build_feature_records,
    candidate_data_roots,
)


OUT_PNG = REPO_ROOT / "paper_artifacts" / "figures" / "figS8_corrected_voltage_profile_temperature_grid.png"
OUT_PDF = REPO_ROOT / "paper_artifacts" / "figures" / "figS8_corrected_voltage_profile_temperature_grid.pdf"
TEMP_ORDER = (0.0, 25.0, 45.0)
PROFILE_ORDER = ("DST", "US06", "BJDST", "FUDS")
REQUIRED_COLUMNS = ("V_raw", "V_corr_raw", "time_s_for_frequency_metadata", "drive_cycle", "temperature")


def set_si_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )


def format_temp(temp: float) -> str:
    return f"{int(temp) if float(temp).is_integer() else temp:g} \u00b0C"


def choose_feature_root(data_root: str | None) -> tuple[Path, list[pd.DataFrame], dict[str, object], list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    expected = {(float(t), p) for t in TEMP_ORDER for p in PROFILE_ORDER}
    best: tuple[Path, list[pd.DataFrame], dict[str, object], set[tuple[float, str]]] | None = None

    for root in candidate_data_roots(REPO_ROOT, data_root):
        try:
            frames, info = build_feature_records(REPO_ROOT, root)
        except Exception as exc:  # pragma: no cover - diagnostic search path.
            attempts.append({"root": root.as_posix(), "status": "failed", "message": str(exc)})
            continue

        valid_frames = []
        for frame in frames:
            if all(col in frame.columns for col in REQUIRED_COLUMNS):
                valid_frames.append(frame)
        keys = {
            (float(frame["temperature"].iloc[0]), str(frame["drive_cycle"].iloc[0]))
            for frame in valid_frames
            if len(frame)
        }
        attempts.append(
            {
                "root": root.as_posix(),
                "status": "ok",
                "n_records": len(valid_frames),
                "n_requested_records": len(keys & expected),
                "available_conditions": sorted([f"{int(t)}C {p}" for t, p in keys]),
            }
        )
        if best is None or len(keys & expected) > len(best[3] & expected):
            best = (root, valid_frames, info, keys)
        if expected.issubset(keys):
            return root, valid_frames, info, attempts

    if best is None:
        detail = "\n".join(f"- {a['root']}: {a['status']} {a.get('message', '')}" for a in attempts)
        raise FileNotFoundError(f"No usable feature records were found.\n{detail}")
    return best[0], best[1], best[2], attempts


def frame_key(frame: pd.DataFrame) -> tuple[float, str]:
    return float(frame["temperature"].iloc[0]), str(frame["drive_cycle"].iloc[0])


def add_plot_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    values = pd.to_numeric(out["time_s_for_frequency_metadata"], errors="coerce").to_numpy(float)
    if len(values) == len(out) and np.isfinite(values).any():
        first = values[np.where(np.isfinite(values))[0][0]]
        out["_x"] = values - first
        return out
    out["_x"] = np.arange(len(out), dtype=float)
    return out


def decimate(frame: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = int(np.ceil(len(frame) / max_points))
    return frame.iloc[::step].copy()


def global_voltage_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    values = []
    for frame in frames:
        values.append(pd.to_numeric(frame["V_raw"], errors="coerce").to_numpy(float))
        values.append(pd.to_numeric(frame["V_corr_raw"], errors="coerce").to_numpy(float))
    merged = np.concatenate(values)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return 2.4, 4.2
    lo = float(np.nanpercentile(merged, 0.2))
    hi = float(np.nanpercentile(merged, 99.8))
    pad = max((hi - lo) * 0.04, 0.03)
    return lo - pad, hi + pad


def plot_grid(frames: list[pd.DataFrame], out_png: Path, out_pdf: Path) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    set_si_style()
    by_key = {frame_key(frame): add_plot_time(frame) for frame in frames}
    expected = [(temp, profile) for temp in TEMP_ORDER for profile in PROFILE_ORDER]
    included = [key for key in expected if key in by_key]
    missing = [key for key in expected if key not in by_key]
    y_limits = global_voltage_limits([by_key[key] for key in included]) if included else (2.4, 4.2)

    fig, axes = plt.subplots(len(TEMP_ORDER), len(PROFILE_ORDER), figsize=(10.8, 6.9), constrained_layout=False)
    raw_color = "#4A4A4A"
    corr_color = "#2F6FAE"
    legend_handles = None

    for row, temp in enumerate(TEMP_ORDER):
        for col, profile in enumerate(PROFILE_ORDER):
            ax = axes[row, col]
            key = (temp, profile)
            if key not in by_key:
                ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes, color="0.45")
                ax.set_ylim(*y_limits)
            else:
                frame = decimate(by_key[key])
                line_raw = ax.plot(frame["_x"], frame["V_raw"], color=raw_color, linewidth=0.55, alpha=0.68, label="Raw voltage")[0]
                line_corr = ax.plot(frame["_x"], frame["V_corr_raw"], color=corr_color, linewidth=0.85, label="Corrected voltage")[0]
                if legend_handles is None:
                    legend_handles = [line_raw, line_corr]
                ax.set_ylim(*y_limits)
                ax.margins(x=0.01)
            ax.text(
                0.03,
                0.94,
                f"{profile}, {format_temp(temp)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row < len(TEMP_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["Raw voltage", "Corrected voltage"],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.supxlabel("Time (s)", y=0.035)
    fig.supylabel("Voltage (V)", x=0.018)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.085, top=0.925, wspace=0.09, hspace=0.34)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)
    return included, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw and corrected voltage across profile-temperature conditions.")
    parser.add_argument("--data-root", default="auto", help="Raw NMC data root, or 'auto' to search known local locations.")
    parser.add_argument("--out-png", default=OUT_PNG.as_posix())
    parser.add_argument("--out-pdf", default=OUT_PDF.as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = None if args.data_root == "auto" else args.data_root
    raw_root, frames, feature_info, attempts = choose_feature_root(data_root)
    included, missing = plot_grid(frames, Path(args.out_png), Path(args.out_pdf))

    print("Corrected-voltage source")
    print(f"- {feature_info.get('corrected_voltage_source', 'not reported')}")
    print(f"- raw_root: {raw_root}")
    print()
    print("Included profile-temperature conditions")
    for temp, profile in included:
        print(f"- {profile}, {format_temp(temp)}")
    if missing:
        print()
        print("Unavailable profile-temperature conditions")
        for temp, profile in missing:
            print(f"- {profile}, {format_temp(temp)}")
    print()
    print("Data-root search summary")
    for attempt in attempts:
        print(f"- {attempt['root']}: {attempt['status']}, requested_records={attempt.get('n_requested_records', 0)}")
    print()
    print(f"Wrote {Path(args.out_png)}")
    print(f"Wrote {Path(args.out_pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
