from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FREQ_ROOT = REPO_ROOT / "Data" / "frequency_structure_analysis"
DEFAULT_RAW_TABLE = FREQ_ROOT / "tables" / "Table_S6_raw_signal_frequency_summary_by_record.csv"
DEFAULT_FEATURE_TABLE = FREQ_ROOT / "tables" / "Table_S7_feature_frequency_summary_by_record.csv"
DEFAULT_OUT_PNG = REPO_ROOT / "Figures" / "figure_S3_profile_temperature_spectral_summary.png"
DEFAULT_OUT_PDF = REPO_ROOT / "Figures" / "figure_S3_profile_temperature_spectral_summary.pdf"

PROFILE_ORDER = ("BJDST", "DST", "US06", "FUDS")
TEMP_ORDER = (0.0, 25.0, 45.0)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.default": "bf",
        }
    )


def record_labels() -> list[str]:
    return [f"{int(temp)} °C {profile}" for temp in TEMP_ORDER for profile in PROFILE_ORDER]


def style_ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", width=1.0, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")


def plot_figure(raw_table: Path, feature_table: Path, out_png: Path, out_pdf: Path) -> None:
    set_style()
    raw = pd.read_csv(raw_table)
    feature = pd.read_csv(feature_table)
    raw = raw[raw["detrend_type"] == "constant"].copy()
    feature = feature[feature["detrend_type"] == "constant"].copy()

    labels = record_labels()
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(9.6, 12.0), constrained_layout=False)
    grid = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.0, 1.0, 0.28, 1.85], hspace=0.13)
    raw_axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    feature_ax = fig.add_subplot(grid[4, 0])

    raw_specs = [
        ("Current", "#1f77b4", "Current\nhigh (%)"),
        ("Voltage", "#d62728", "Voltage\nhigh (%)"),
        ("Reference SOC", "#2ca02c", "Reference SOC\nhigh (%)"),
    ]
    for idx, (ax, (signal, color, ylabel)) in enumerate(zip(raw_axes, raw_specs, strict=True)):
        frame = raw[raw["signal_name"] == signal].copy()
        frame["record"] = frame["temperature_C"].map(lambda value: f"{int(value)} °C") + " " + frame["profile"].astype(str)
        values = frame.set_index("record").reindex(labels)["high_frequency_energy_fraction"].to_numpy(float) * 100.0
        ax.bar(x, values, color=color, width=0.75)
        ax.axhline(float(np.nanmean(values)), color="0.25", linewidth=0.9, linestyle="--")
        ax.set_ylabel(ylabel)
        ymax = float(np.nanmax(values))
        ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
        style_ticks(ax)
        if idx < len(raw_axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks(x, labels, rotation=45, ha="right")

    compact_cols = ["V_raw", "V_corr_raw", "V_corr_raw_ema800", "I_raw", "I_raw_ema200", "absI_ema200"]
    feature_labels = {
        "V_raw": r"$\mathbf{V}_{t}$",
        "V_corr_raw": r"$\mathbf{V}_{t}^{\mathbf{corr}}$",
        "V_corr_raw_ema800": r"$\mathbf{m}^{(800)}(\mathbf{V}_{t}^{\mathbf{corr}})$",
        "I_raw": r"$\mathbf{I}_{t}$",
        "I_raw_ema200": r"$\mathbf{m}^{(200)}(\mathbf{I}_{t})$",
        "absI_ema200": r"$\mathbf{m}^{(200)}(|\mathbf{I}_{t}|)$",
    }
    colors = {
        "V_raw": "#1f77b4",
        "V_corr_raw": "#ff7f0e",
        "V_corr_raw_ema800": "#2ca02c",
        "I_raw": "#d62728",
        "I_raw_ema200": "#9467bd",
        "absI_ema200": "#8c564b",
    }
    tmp = feature[feature["feature_column_used"].isin(compact_cols)].copy()
    tmp["record"] = tmp["temperature_C"].map(lambda value: f"{int(value)} °C") + " " + tmp["profile"].astype(str)
    pivot = tmp.pivot_table(index="record", columns="feature_column_used", values="high_frequency_energy_fraction", aggfunc="mean")
    width = 0.12
    feature_max = 0.0
    for i, col in enumerate(compact_cols):
        values = pivot.reindex(labels)[col].to_numpy(float) * 100.0
        feature_max = max(feature_max, float(np.nanmax(values)))
        feature_ax.bar(
            x + (i - (len(compact_cols) - 1) / 2.0) * width,
            values,
            width=width,
            color=colors[col],
            label=feature_labels[col],
        )

    feature_ax.set_ylabel("High-frequency energy (%)")
    feature_ax.set_xticks(x, labels, rotation=45, ha="right")
    feature_ax.set_ylim(0.0, max(95.0, feature_max * 1.22))
    style_ticks(feature_ax)
    legend = feature_ax.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        columnspacing=1.25,
        handlelength=1.9,
        borderaxespad=0.2,
        prop={"family": "Times New Roman", "weight": "bold", "size": 11},
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontweight("bold")

    fig.subplots_adjust(left=0.095, right=0.985, top=0.965, bottom=0.08)
    fig.text(
        0.014,
        min(0.99, raw_axes[0].get_position().y1 + 0.02),
        "(a)",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        fontfamily="Times New Roman",
    )
    fig.text(
        0.014,
        min(0.99, feature_ax.get_position().y1 + 0.02),
        "(b)",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        fontfamily="Times New Roman",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_png.with_suffix(".tif"), dpi=600)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined high-frequency energy panels.")
    parser.add_argument("--raw-table", default=DEFAULT_RAW_TABLE.as_posix())
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE.as_posix())
    parser.add_argument("--out-png", default=DEFAULT_OUT_PNG.as_posix())
    parser.add_argument("--out-pdf", default=DEFAULT_OUT_PDF.as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_figure(Path(args.raw_table), Path(args.feature_table), Path(args.out_png), Path(args.out_pdf))
    print(f"Wrote {Path(args.out_png)}")
    print(f"Wrote {Path(args.out_pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
