from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "paper_artifacts" / "source_metrics" / "region_error_reduction_g0_g4.csv"
OUT_PNG = REPO_ROOT / "paper_artifacts" / "figures" / "fig8_region_error_reduction.png"
OUT_PDF = REPO_ROOT / "paper_artifacts" / "figures" / "fig8_region_error_reduction.pdf"

ORDER = [
    ("SOC band", "Low SOC"),
    ("SOC band", "Mid SOC"),
    ("SOC band", "High SOC"),
    ("Recent absolute-current history", "Low"),
    ("Recent absolute-current history", "High"),
    ("Voltage-response deviation", "Low"),
    ("Voltage-response deviation", "High"),
    ("Local V-I ambiguity", "Non-ambiguous bins"),
    ("Local V-I ambiguity", "Ambiguous bins"),
]

REGION_LABEL_X = {
    "SOC band": -0.22,
    "Recent absolute-current history": -0.235,
    "Voltage-response deviation": -0.235,
    "Local V-I ambiguity": -0.205,
}
ROW_LABEL_X = -0.015


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
        }
    )


def load_ordered_table(path: Path) -> pd.DataFrame:
    required = {
        "region_definition",
        "group",
        "G0_MAE",
        "G4_MAE",
        "delta_MAE_G4_minus_G0",
        "relative_change",
        "n_windows",
        "seed_count_G0",
        "seed_count_G4",
    }
    df = pd.read_csv(path)
    missing_cols = sorted(required.difference(df.columns))
    if missing_cols:
        raise ValueError(f"{path} is missing required columns: {missing_cols}")

    order_index = {key: idx for idx, key in enumerate(ORDER)}
    present = set(zip(df["region_definition"], df["group"]))
    missing_rows = [key for key in ORDER if key not in present]
    if missing_rows:
        raise ValueError(f"{path} is missing ordered region rows: {missing_rows}")

    df = df.copy()
    df["_order"] = [order_index[(row.region_definition, row.group)] for row in df.itertuples(index=False)]
    return df.sort_values("_order").reset_index(drop=True)


def y_positions(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, float]]]:
    positions: list[float] = []
    region_centers: list[tuple[str, float]] = []
    y = 0.0
    for region, group in df.groupby("region_definition", sort=False):
        start = y
        for _ in range(len(group)):
            positions.append(y)
            y += 1.0
        end = y - 1.0
        region_centers.append((region, (start + end) / 2.0))
        if len(positions) < len(df):
            y += 0.45
    return np.array(positions), region_centers


def display_group_label(group: str) -> str:
    if group == "Non-ambiguous bins":
        return "Non-ambiguous\nbins"
    if group == "Ambiguous bins":
        return "Ambiguous\nbins"
    return group


def display_region_label(region: str) -> str:
    if region == "Recent absolute-current history":
        return "Recent absolute-\ncurrent history"
    if region == "Voltage-response deviation":
        return "Voltage-response\ndeviation"
    if region == "Local V-I ambiguity":
        return "Local V-I\nambiguity"
    return region


def plot(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    set_manuscript_style()
    y, region_centers = y_positions(df)
    bar_h = 0.32
    g0 = df["G0_MAE"].to_numpy(float)
    g4 = df["G4_MAE"].to_numpy(float)
    delta = df["delta_MAE_G4_minus_G0"].to_numpy(float)
    xmax = float(max(g0.max(), g4.max()))
    annotation_x = xmax + 0.08

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.barh(y - bar_h / 2.0, g0, height=bar_h, color="#8A8A8A", edgecolor="black", linewidth=0.35, label="G0 raw")
    ax.barh(y + bar_h / 2.0, g4, height=bar_h, color="#3B6EA8", edgecolor="black", linewidth=0.35, label="G4 raw+EMA")

    for yi, value in zip(y, delta):
        ax.text(annotation_x, yi, f"\u0394MAE={value:+.3f}", va="center", ha="left", fontsize=8.5)

    for region, center in region_centers:
        x_offset = REGION_LABEL_X.get(region, -0.20)
        ax.text(
            x_offset,
            center,
            display_region_label(region),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            linespacing=1.05,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(y))
    for yi, group in zip(y, df["group"]):
        ax.text(
            ROW_LABEL_X,
            yi,
            display_group_label(str(group)),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.set_xlabel("MAE (%SOC)")
    ax.set_xlim(0.0, annotation_x + 0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.64, 0.995))
    fig.subplots_adjust(left=0.39, right=0.97, bottom=0.12, top=0.91)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    df = load_ordered_table(INPUT_CSV)
    plot(df, OUT_PNG, OUT_PDF)
    summary = df[["region_definition", "group", "G0_MAE", "G4_MAE", "delta_MAE_G4_minus_G0"]].copy()
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
