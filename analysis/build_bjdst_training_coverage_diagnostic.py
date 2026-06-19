from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from section2_measurement_structure import (
    assign_bins,
    candidate_roots,
    discover_terminal_files,
    fail_no_data,
    js_divergence,
    load_terminal_frame,
    quantile_edges,
    raw_vi_bin_tables,
)


PROFILES = ("DST", "US06", "BJDST", "FUDS")
TEMPERATURES = (0.0, 25.0, 45.0)
TRAINING_SETS = {
    "DST+US06": ("DST", "US06"),
    "DST+US06+BJDST": ("DST", "US06", "BJDST"),
}
TEST_PROFILE = "FUDS"


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def hist2d_fraction(df: pd.DataFrame, v_edges: np.ndarray, i_edges: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(df["voltage_V"], df["current_A"], bins=[v_edges, i_edges])
    flat = hist.astype(float).ravel()
    return flat / flat.sum() if flat.sum() > 0 else flat


def hist1d_fraction(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float), bins=edges)
    hist = hist.astype(float)
    return hist / hist.sum() if hist.sum() > 0 else hist


def format_range(lo: float, hi: float, digits: int = 3) -> str:
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def load_data(
    data_root: str | None,
    voltage_bins: int,
    current_bins: int,
) -> tuple[pd.DataFrame, dict[float, tuple[np.ndarray, np.ndarray]]]:
    roots = candidate_roots(Path.cwd(), data_root)
    mappings, diagnostics = discover_terminal_files(roots, PROFILES, TEMPERATURES)
    if not mappings:
        fail_no_data(diagnostics)

    frames = []
    for mapping in mappings:
        frame = load_terminal_frame(mapping)
        frame["_row_in_file"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data[all_data["profile"].isin(PROFILES) & all_data["temperature_C"].isin(TEMPERATURES)].copy()
    all_data = all_data.sort_values(["file_name", "_row_in_file"]).reset_index(drop=True)
    all_data["absI_mean_past200_A"] = all_data.groupby("file_name", sort=False)["current_A"].transform(
        lambda s: s.abs().rolling(window=200, min_periods=1).mean()
    )

    _, _, _, edge_map = raw_vi_bin_tables(all_data, voltage_bins=voltage_bins, current_bins=current_bins, min_count=50)
    return all_data, edge_map


def compute_rows(all_data: pd.DataFrame, edge_map: dict[float, tuple[np.ndarray, np.ndarray]], history_bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for temp in TEMPERATURES:
        temp_df = all_data[all_data["temperature_C"].eq(temp)].copy()
        test_df = temp_df[temp_df["profile"].eq(TEST_PROFILE)].copy()
        v_edges, i_edges = edge_map[float(temp)]
        h_edges = quantile_edges(temp_df["absI_mean_past200_A"].to_numpy(float), history_bins)
        for label, train_profiles in TRAINING_SETS.items():
            train_df = temp_df[temp_df["profile"].isin(train_profiles)].copy()
            p_vi = hist2d_fraction(train_df, v_edges, i_edges)
            q_vi = hist2d_fraction(test_df, v_edges, i_edges)
            p_h = hist1d_fraction(train_df["absI_mean_past200_A"], h_edges)
            q_h = hist1d_fraction(test_df["absI_mean_past200_A"], h_edges)
            train_occ = p_vi > 0

            rows.append(
                {
                    "temperature_C": f"{int(temp)}",
                    "training_profiles": label,
                    "heldout_profile": TEST_PROFILE,
                    "vi_overlap_coefficient": float(np.minimum(p_vi, q_vi).sum()),
                    "vi_jensen_shannon_divergence_bits": js_divergence(p_vi, q_vi),
                    "fuds_samples_outside_train_vi_bins_fraction": float(q_vi[~train_occ].sum()),
                    "train_occupied_vi_bin_count": int(train_occ.sum()),
                    "train_voltage_range_V": format_range(float(train_df["voltage_V"].min()), float(train_df["voltage_V"].max())),
                    "train_current_range_A": format_range(float(train_df["current_A"].min()), float(train_df["current_A"].max())),
                    "absI_mean_past200_overlap_coefficient": float(np.minimum(p_h, q_h).sum()),
                    "absI_mean_past200_jensen_shannon_divergence_bits": js_divergence(p_h, q_h),
                }
            )

    detail = pd.DataFrame(rows)
    avg_rows: list[dict[str, object]] = []
    numeric_cols = [
        "vi_overlap_coefficient",
        "vi_jensen_shannon_divergence_bits",
        "fuds_samples_outside_train_vi_bins_fraction",
        "train_occupied_vi_bin_count",
        "absI_mean_past200_overlap_coefficient",
        "absI_mean_past200_jensen_shannon_divergence_bits",
    ]
    for label, train_profiles in TRAINING_SETS.items():
        sub = detail[detail["training_profiles"].eq(label)]
        pooled_train = all_data[all_data["profile"].isin(train_profiles)]
        avg = {
            "temperature_C": "Average",
            "training_profiles": label,
            "heldout_profile": TEST_PROFILE,
            "train_voltage_range_V": format_range(float(pooled_train["voltage_V"].min()), float(pooled_train["voltage_V"].max())),
            "train_current_range_A": format_range(float(pooled_train["current_A"].min()), float(pooled_train["current_A"].max())),
        }
        for col in numeric_cols:
            avg[col] = float(sub[col].mean())
        avg_rows.append(avg)

    out = pd.concat([detail, pd.DataFrame(avg_rows)], ignore_index=True)
    rounded = out.copy()
    for col in rounded.select_dtypes(include=[float]).columns:
        rounded[col] = rounded[col].round(4)
    rounded["train_occupied_vi_bin_count"] = rounded["train_occupied_vi_bin_count"].round(1)
    return rounded


def plot_figure(table: pd.DataFrame, out_png: Path) -> None:
    labels = ["0", "25", "45", "Average"]
    x = np.arange(len(labels))
    width = 0.36
    colors = {"DST+US06": "#4C78A8", "DST+US06+BJDST": "#F58518"}
    metrics = [
        ("vi_overlap_coefficient", "V-I overlap coefficient", False),
        ("vi_jensen_shannon_divergence_bits", "V-I JSD (bits)", False),
        ("fuds_samples_outside_train_vi_bins_fraction", "FUDS outside train bins (%)", True),
        ("train_occupied_vi_bin_count", "Training occupied V-I bins", False),
        ("absI_mean_past200_overlap_coefficient", "Recent-current overlap", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4))
    axes_flat = axes.ravel()
    for ax, (metric, ylabel, as_percent) in zip(axes_flat, metrics):
        for offset, (name, color) in zip((-width / 2, width / 2), colors.items()):
            sub = table[table["training_profiles"].eq(name)].set_index("temperature_C").loc[labels]
            values = sub[metric].to_numpy(float)
            if as_percent:
                values = values * 100.0
            ax.bar(x + offset, values, width=width, label=name, color=color, edgecolor="black", linewidth=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(["0C", "25C", "45C", "Avg."])
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
    axes_flat[-1].axis("off")
    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def write_summary(table: pd.DataFrame, out_md: Path) -> None:
    avg = table[table["temperature_C"].eq("Average")].set_index("training_profiles")
    vi_a = avg.loc["DST+US06", "vi_overlap_coefficient"]
    vi_b = avg.loc["DST+US06+BJDST", "vi_overlap_coefficient"]
    occ_a = avg.loc["DST+US06", "train_occupied_vi_bin_count"]
    occ_b = avg.loc["DST+US06+BJDST", "train_occupied_vi_bin_count"]
    out_a = avg.loc["DST+US06", "fuds_samples_outside_train_vi_bins_fraction"] * 100.0
    out_b = avg.loc["DST+US06+BJDST", "fuds_samples_outside_train_vi_bins_fraction"] * 100.0
    h_a = avg.loc["DST+US06", "absI_mean_past200_overlap_coefficient"]
    h_b = avg.loc["DST+US06+BJDST", "absI_mean_past200_overlap_coefficient"]
    text = f"""# BJDST Training-Coverage Rationale Summary

- BJDST broadens the non-FUDS training-side measurement coverage: adding BJDST increased the mean occupied V-I bin count from `{occ_a:.1f}` to `{occ_b:.1f}` and changed the FUDS outside-bin fraction from `{out_a:.2f}%` to `{out_b:.2f}%` under the same temperature-wise binning.
- The main split preserves FUDS as the held-out profile: the diagnostic compares `DST+US06` and `DST+US06+BJDST` training-side measurement distributions against FUDS without using model predictions, residuals, checkpoints, or FUDS-derived training decisions.
- This reduces dependence on a deliberately narrower two-profile training set by adding a non-FUDS drive pattern with additional measurement-history coverage; the causal `absI_mean_past200` overlap changed from `{h_a:.3f}` to `{h_b:.3f}`, while the average V-I histogram overlap changed from `{vi_a:.3f}` to `{vi_b:.3f}`, so the diagnostic should be interpreted as coverage broadening rather than a guarantee of improved generalization.
"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BJDST training-side coverage diagnostic for FUDS holdout.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default="paper_ema_analysis_package/section2_measurement_structure")
    parser.add_argument("--voltage-bins", type=int, default=40)
    parser.add_argument("--current-bins", type=int, default=40)
    parser.add_argument("--history-bins", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    set_manuscript_style()
    all_data, edge_map = load_data(args.data_root, args.voltage_bins, args.current_bins)
    table = compute_rows(all_data, edge_map, args.history_bins)
    table_path = tables / "Table_S5_training_coverage_with_without_BJDST.csv"
    fig_path = figures / "Figure_S5_training_coverage_with_without_BJDST.png"
    md_path = tables / "bjdst_training_rationale_summary.md"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    plot_figure(table, fig_path)
    write_summary(table, md_path)
    print(f"Wrote {table_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
