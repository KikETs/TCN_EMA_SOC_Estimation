from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plot_figure7_corrected_voltage_behavior as fig7  # noqa: E402


OUT_PNG = REPO_ROOT / "Figures" / "figure_aux_voltage_current_abs_current_behavior.png"
OUT_PDF = REPO_ROOT / "Figures" / "figure_aux_voltage_current_abs_current_behavior.pdf"
OUT_SVG = REPO_ROOT / "Figures" / "figure_aux_voltage_current_abs_current_behavior.svg"
REQUIRED_COLUMNS = [
    "V_corr_raw",
    "V_corr_raw_ema50",
    "V_corr_raw_ema200",
    "V_corr_raw_ema800",
    "I_raw",
    "I_raw_ema50",
    "I_raw_ema200",
    "absI_ema50",
    "absI_ema200",
]


def set_compact_panel_style() -> None:
    fig7.set_manuscript_style()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.default": "bf",
            "svg.fonttype": "none",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.labelsize": 7.9,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.4,
        }
    )


def set_ylim_with_lower_legend_space(ax: plt.Axes, values: list[object], bottom_fraction: float = 0.30) -> None:
    arrays = [np.asarray(value, dtype=float) for value in values]
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()])
    if finite.size == 0:
        return
    ymin = float(finite.min())
    ymax = float(finite.max())
    yrange = ymax - ymin
    if yrange <= 0:
        yrange = max(abs(ymax), 1.0) * 0.05
    ax.set_ylim(ymin - bottom_fraction * yrange, ymax + 0.08 * yrange)


def plot_figure(
    frame,
    selected,
    zoom: tuple[int, int, float],
    xlabel: str,
    out_png: Path,
    out_pdf: Path,
    out_svg: Path,
) -> None:
    set_compact_panel_style()
    start, end, _ = zoom
    zoom_frame = frame.iloc[start:end].copy()
    x_zoom = zoom_frame["_x"]

    fig, axes = plt.subplots(3, 1, figsize=(3.55, 2.75), constrained_layout=False, sharex=True)
    raw_color = "#4A4A4A"
    corr_color = "#2F6FAE"
    ema50_color = "#4C8C6B"
    ema200_color = "#C27A2C"
    ema800_color = "#6E5A9A"

    axes[0].plot(x_zoom, zoom_frame["V_corr_raw"], color=corr_color, linewidth=1.0, label=r"$\mathbf{V}_{\mathbf{t}}^{\mathregular{corr}}$")
    axes[0].plot(
        x_zoom,
        zoom_frame["V_corr_raw_ema50"],
        color=ema50_color,
        linewidth=0.95,
        label=r"$\mathbf{m}^{(50)}$",
    )
    axes[0].plot(
        x_zoom,
        zoom_frame["V_corr_raw_ema200"],
        color=ema200_color,
        linewidth=0.95,
        label=r"$\mathbf{m}^{(200)}$",
    )
    axes[0].plot(
        x_zoom,
        zoom_frame["V_corr_raw_ema800"],
        color=ema800_color,
        linewidth=0.95,
        label=r"$\mathbf{m}^{(800)}$",
    )
    axes[1].plot(x_zoom, zoom_frame["I_raw"], color=raw_color, linewidth=0.85, label=r"$\mathbf{I}_{\mathbf{t}}$")
    axes[1].plot(x_zoom, zoom_frame["I_raw_ema50"], color=ema50_color, linewidth=0.95, label=r"$\mathbf{m}^{(50)}$")
    axes[1].plot(x_zoom, zoom_frame["I_raw_ema200"], color=ema200_color, linewidth=0.95, label=r"$\mathbf{m}^{(200)}$")

    axes[2].plot(x_zoom, zoom_frame["absI_ema50"], color=ema50_color, linewidth=0.95, label=r"$\mathbf{m}^{(50)}$")
    axes[2].plot(x_zoom, zoom_frame["absI_ema200"], color=ema200_color, linewidth=0.95, label=r"$\mathbf{m}^{(200)}$")

    ylabels = ("Voltage (V)", "Current (A)", "|Current| (A)")
    for ax, ylabel in zip(axes, ylabels, strict=True):
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            handlelength=1.45,
            labelspacing=0.45,
            borderaxespad=0.15,
            prop={"family": "Times New Roman", "size": 7.4, "weight": "bold"},
        )
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    set_ylim_with_lower_legend_space(
        axes[0],
        [
            zoom_frame["V_corr_raw"],
            zoom_frame["V_corr_raw_ema50"],
            zoom_frame["V_corr_raw_ema200"],
            zoom_frame["V_corr_raw_ema800"],
        ],
        bottom_fraction=0.08,
    )
    set_ylim_with_lower_legend_space(
        axes[1],
        [zoom_frame["I_raw"], zoom_frame["I_raw_ema50"], zoom_frame["I_raw_ema200"]],
        bottom_fraction=0.08,
    )
    set_ylim_with_lower_legend_space(
        axes[2],
        [zoom_frame["absI_ema50"], zoom_frame["absI_ema200"]],
        bottom_fraction=0.08,
    )

    axes[2].set_xlabel(xlabel, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.74, bottom=0.14, top=0.985, hspace=0.22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def selected_profile_record(profile: str, temperature: float, file_name: str | None = None) -> dict[str, object]:
    temp_token = int(temperature) if float(temperature).is_integer() else temperature
    trajectory_id = Path(file_name).stem if file_name else f"NMC_{temp_token:g}C_{profile.upper()}"
    return {
        "drive_cycle": profile.upper(),
        "temperature": float(temperature),
        "trajectory_id": trajectory_id,
        "file_name": file_name or f"{trajectory_id}.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot zoomed voltage, current, and absolute-current EMA behavior.")
    parser.add_argument("--profile", default="DST", help="Drive profile to plot.")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature in Celsius.")
    parser.add_argument("--file-name", default=None, help="Explicit raw CSV file name, e.g. NMC_25C_DST.csv.")
    parser.add_argument("--raw-root", action="append", default=[], help="Additional raw NMC root containing NMC CSV files.")
    parser.add_argument("--out-png", default=OUT_PNG.as_posix())
    parser.add_argument("--out-pdf", default=OUT_PDF.as_posix())
    parser.add_argument("--out-svg", default=OUT_SVG.as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = selected_profile_record(args.profile, args.temperature, args.file_name)
    raw_roots = fig7.candidate_raw_roots(args.raw_root)
    if not raw_roots:
        raise FileNotFoundError("No raw NMC roots found. Set G4_RAW_ROOTS or pass --raw-root.")

    fig7.REQUIRED_COLUMNS = REQUIRED_COLUMNS
    frame, raw_root = fig7.load_selected_feature_frame(selected, raw_roots)
    frame, xlabel = fig7.attach_time_axis(frame, raw_root, str(selected["file_name"]))
    zoom = fig7.choose_zoom_interval(frame)
    plot_figure(frame, selected, zoom, xlabel, Path(args.out_png), Path(args.out_pdf), Path(args.out_svg))

    start, end, score = zoom
    x0 = float(frame["_x"].iloc[start])
    x1 = float(frame["_x"].iloc[end - 1])
    print("Selected trajectory metadata")
    print("- selection_rule: selected requested profile/temperature raw trajectory")
    print(f"- profile: {selected['drive_cycle']}")
    print(f"- temperature: {fig7.format_temp(float(selected['temperature']))}")
    print(f"- trajectory_id: {selected['trajectory_id']}")
    print(f"- file_name: {selected['file_name']}")
    print(f"- raw_root: {raw_root}")
    print()
    print("Selected zoom interval")
    print("- zoom_rule: max rolling std of V_raw - V_corr_raw over a deterministic fixed-length window")
    print(f"- start_row: {start}")
    print(f"- end_row_exclusive: {end}")
    print(f"- x_start: {x0:.3f}")
    print(f"- x_end: {x1:.3f}")
    print(f"- fluctuation_score_V: {score:.6f}")
    print()
    print(f"Wrote {Path(args.out_png)}")
    print(f"Wrote {Path(args.out_pdf)}")
    print(f"Wrote {Path(args.out_svg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
