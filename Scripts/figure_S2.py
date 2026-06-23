from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "Data" / "figures_source" / "representative_cema_traces.csv"
DEFAULT_OUT_PNG = REPO_ROOT / "Figures" / "figure_S2_representative_cema_channels.png"
DEFAULT_OUT_PDF = REPO_ROOT / "Figures" / "figure_S2_representative_cema_channels.pdf"
DEFAULT_OUT_SVG = REPO_ROOT / "Figures" / "figure_S2_representative_cema_channels.svg"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.1,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.default": "bf",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def padded_ylim(values: list[pd.Series | np.ndarray], bottom_pad: float = 0.08, top_pad: float = 0.28) -> tuple[float, float]:
    finite_parts = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return 0.0, 1.0
    merged = np.concatenate(finite_parts)
    ymin = float(merged.min())
    ymax = float(merged.max())
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0) * 0.05
    return ymin - bottom_pad * span, ymax + top_pad * span


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.1, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontfamily="Times New Roman",
        fontweight="bold",
    )


def bold_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontweight("bold")


def plot_figure(frame: pd.DataFrame, out_png: Path, out_pdf: Path, out_svg: Path) -> None:
    set_style()

    x = frame["time_s"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.8), sharex=True, constrained_layout=False)

    colors = {
        "raw": "#4A4A4A",
        "ema50": "#0072BD",
        "ema200": "#D55E00",
        "ema800": "#009E73",
        "soc": "#6E6E6E",
    }

    axes[0].plot(x, frame["V_corr_raw"], color=colors["raw"], linewidth=1.1, label=r"$\mathbf{V}_{t}^{\mathbf{corr}}$")
    axes[0].plot(
        x,
        frame["V_corr_raw_ema50"],
        color=colors["ema50"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(50)}(\mathbf{V}_{t}^{\mathbf{corr}})$",
    )
    axes[0].plot(
        x,
        frame["V_corr_raw_ema200"],
        color=colors["ema200"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(200)}(\mathbf{V}_{t}^{\mathbf{corr}})$",
    )
    axes[0].plot(
        x,
        frame["V_corr_raw_ema800"],
        color=colors["ema800"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(800)}(\mathbf{V}_{t}^{\mathbf{corr}})$",
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_ylim(
        *padded_ylim(
            [
                frame["V_corr_raw"],
                frame["V_corr_raw_ema50"],
                frame["V_corr_raw_ema200"],
                frame["V_corr_raw_ema800"],
            ],
            bottom_pad=0.08,
            top_pad=0.24,
        )
    )

    axes[1].plot(x, frame["I_raw"], color=colors["raw"], linewidth=0.9, label=r"$\mathbf{I}_{t}$")
    axes[1].plot(
        x,
        frame["I_raw_ema50"],
        color=colors["ema50"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(50)}(\mathbf{I}_{t})$",
    )
    axes[1].plot(
        x,
        frame["I_raw_ema200"],
        color=colors["ema200"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(200)}(\mathbf{I}_{t})$",
    )
    axes[1].set_ylabel("Current (A)")
    axes[1].set_ylim(*padded_ylim([frame["I_raw"], frame["I_raw_ema50"], frame["I_raw_ema200"]], top_pad=0.24))

    axes[2].plot(
        x,
        frame["absI_ema50"],
        color=colors["ema50"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(50)}(|\mathbf{I}_{t}|)$",
    )
    axes[2].plot(
        x,
        frame["absI_ema200"],
        color=colors["ema200"],
        linewidth=1.2,
        label=r"$\mathbf{m}^{(200)}(|\mathbf{I}_{t}|)$",
    )
    axes[2].set_ylabel(r"$|\mathbf{I}_{t}|$ EMA (A)")
    axes[2].set_ylim(*padded_ylim([frame["absI_ema50"], frame["absI_ema200"]], bottom_pad=0.12, top_pad=0.55))

    ax_soc = axes[2].twinx()
    ax_soc.plot(x, frame["SOC_fraction"] * 100.0, color=colors["soc"], linewidth=1.3, label=r"$\mathbf{SOC}$")
    ax_soc.set_ylabel("SOC (%SOC)")
    ax_soc.spines["top"].set_visible(False)
    ax_soc.spines["right"].set_linewidth(1.1)
    ax_soc.tick_params(axis="y", width=1.1, length=5)
    for label in ax_soc.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")
    ax_soc.set_ylim(*padded_ylim([frame["SOC_fraction"] * 100.0], bottom_pad=0.10, top_pad=0.55))

    for ax, panel in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        style_axis(ax)
        add_panel_label(ax, panel)

    for ax in axes[:2]:
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            ncol=4 if ax is axes[0] else 3,
            handlelength=1.7,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
        bold_legend(legend)

    handles_left, labels_left = axes[2].get_legend_handles_labels()
    handles_right, labels_right = ax_soc.get_legend_handles_labels()
    legend = axes[2].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper right",
        frameon=False,
        ncol=3,
        handlelength=1.7,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    bold_legend(legend)

    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(float(x.min()), float(x.max()))

    for ax in axes:
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
    ax_soc.yaxis.label.set_fontweight("bold")

    fig.subplots_adjust(left=0.105, right=0.89, top=0.95, bottom=0.09, hspace=0.30)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    Image.open(out_png).convert("RGB").save(out_png.with_suffix(".tif"), dpi=(600, 600))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure S2 representative CEMA traces.")
    parser.add_argument("--input", default=DEFAULT_INPUT.as_posix())
    parser.add_argument("--time-max", type=float, default=1200.0)
    parser.add_argument("--out-png", default=DEFAULT_OUT_PNG.as_posix())
    parser.add_argument("--out-pdf", default=DEFAULT_OUT_PDF.as_posix())
    parser.add_argument("--out-svg", default=DEFAULT_OUT_SVG.as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = [
        "time_s",
        "V_corr_raw",
        "V_corr_raw_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_ema800",
        "I_raw",
        "I_raw_ema50",
        "I_raw_ema200",
        "absI_ema50",
        "absI_ema200",
        "SOC_fraction",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.loc[frame["time_s"] <= args.time_max, required].dropna().copy()
    if frame.empty:
        raise ValueError("No rows remain after time filtering.")

    plot_figure(frame, Path(args.out_png), Path(args.out_pdf), Path(args.out_svg))
    print(f"Wrote {Path(args.out_png)}")
    print(f"Wrote {Path(args.out_pdf)}")
    print(f"Wrote {Path(args.out_svg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
