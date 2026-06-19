from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLES = [
    "table1_main_g4_fuds_results",
    "table2_feature_ablation",
    "table3_ema_group_sweep",
    "table4_profile_rotation",
    "table5_epoch_sweep",
    "table6_model_class_baselines",
    "appendix_forbidden_reference",
]

FIGURES = [
    "fig1_main_g4_fuds_mae_by_temp",
    "fig2_feature_ablation_by_temp",
    "fig3_ema_group_sweep_by_temp",
    "fig4_profile_rotation_by_temp",
    "fig5_epoch_sweep",
    "fig6_ema_perturbation_importance",
    "fig7_model_class_baselines",
    "fig_appendix_ema_correlation",
]


def set_manuscript_figure_style() -> None:
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


def fmt_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt_value(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_tex(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    out.append(" & ".join(cols) + " \\\\")
    out.append("\\hline")
    for _, row in df.iterrows():
        out.append(" & ".join(fmt_value(row[c]) for c in cols) + " \\\\")
    out.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(out)


def save_table(df: pd.DataFrame, stem: str, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / f"{stem}.csv", index=False)
    (table_dir / f"{stem}.md").write_text(to_markdown(df), encoding="utf-8")
    (table_dir / f"{stem}.tex").write_text(to_tex(df), encoding="utf-8")


def numeric_temp(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def pivot_temp(df: pd.DataFrame, index_cols: list[str], value_col: str, temp_col: str) -> pd.DataFrame:
    work = df.copy()
    work[temp_col] = numeric_temp(work, temp_col)
    work = work.dropna(subset=[temp_col])
    piv = work.pivot_table(index=index_cols, columns=temp_col, values=value_col, aggfunc="mean").reset_index()
    piv.columns = [f"{int(c)}C_MAE_pct" if isinstance(c, (float, np.floating)) else str(c) for c in piv.columns]
    temp_cols = [c for c in piv.columns if c.endswith("C_MAE_pct")]
    if temp_cols:
        piv["tempmean_MAE_pct"] = piv[temp_cols].mean(axis=1)
        piv["worst_temp_MAE_pct"] = piv[temp_cols].max(axis=1)
    return piv


def build_model_class_table(baselines: pd.DataFrame) -> pd.DataFrame:
    temp_rows = baselines[baselines["temperature_C"].astype(str).ne("ALL")].copy()
    temp_rows["temperature_C"] = numeric_temp(temp_rows, "temperature_C")
    temp_rows = temp_rows.dropna(subset=["temperature_C"])
    index_cols = ["model_id", "description", "approx_trainable_params"]
    out = temp_rows.pivot_table(index=index_cols, columns="temperature_C", values="MAE_pct", aggfunc="mean").reset_index()
    out.columns = [f"{int(c)}C_MAE_pct" if isinstance(c, (float, np.floating)) else str(c) for c in out.columns]
    temp_mae_cols = [c for c in out.columns if c.endswith("C_MAE_pct")]
    out["temperature_mean_MAE_pct"] = out[temp_mae_cols].mean(axis=1)
    out["worst_temp_MAE_pct"] = out[temp_mae_cols].max(axis=1)

    for metric, out_col in [("RMSE_pct", "temperature_mean_RMSE_pct"), ("MaxAE_pct", "temperature_mean_MaxAE_pct")]:
        if metric in temp_rows.columns:
            metric_mean = temp_rows.groupby(index_cols, as_index=False)[metric].mean().rename(columns={metric: out_col})
            out = out.merge(metric_mean, on=index_cols, how="left")
        else:
            out[out_col] = np.nan
    return out


def bar_by_temp(df: pd.DataFrame, label_col: str, temp_col: str, value_col: str, title: str, out_stem: str, fig_dir: Path) -> None:
    work = df.copy()
    work[temp_col] = numeric_temp(work, temp_col)
    work = work.dropna(subset=[temp_col])
    if work.empty:
        return
    pivot = work.pivot_table(index=label_col, columns=temp_col, values=value_col, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(9, 4.8), width=0.82)
    ax.set_ylabel("MAE (%)")
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{out_stem}.png", dpi=180)
    plt.savefig(fig_dir / f"{out_stem}.pdf")
    plt.close()


def build_tables(source: Path, table_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    by_temp = pd.read_csv(source / "g4_seed_reproduction_by_temp.csv")
    summary = pd.read_csv(source / "g4_seed_reproduction_summary.csv")
    t1_temp = (
        by_temp.groupby("temperature", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            MAE_pct_mean=("mae_pct", "mean"),
            MAE_pct_std=("mae_pct", "std"),
            RMSE_pct_mean=("rmse_pct", "mean"),
            MaxAE_pct_mean=("maxae_pct", "mean"),
        )
        .sort_values("temperature")
    )
    agg = summary[summary["metric_name"].eq("g4_seed_aggregate_tempmean_mae_pct")]
    if not agg.empty:
        tempmean_rmse = float(t1_temp["RMSE_pct_mean"].mean())
        tempmean_maxae = float(t1_temp["MaxAE_pct_mean"].mean())
        t1_temp = pd.concat(
            [
                t1_temp,
                pd.DataFrame(
                    [
                        {
                            "temperature": "ALL_tempmean",
                            "seed_count": int(agg["done_seed_count"].iloc[0]),
                            "MAE_pct_mean": float(agg["mean_tempmean_mae_pct"].iloc[0]),
                            "MAE_pct_std": float(agg["std_tempmean_mae_pct"].iloc[0]),
                            "RMSE_pct_mean": tempmean_rmse,
                            "MaxAE_pct_mean": tempmean_maxae,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    tables[TABLES[0]] = t1_temp

    ablation = pd.read_csv(source / "feature_ablation_reanalysis.csv")
    t2 = pivot_temp(
        ablation[ablation["temperature"].astype(str).ne("ALL")],
        ["ablation_group", "feature_set", "input_dim", "evidence_scope"],
        "metric_value",
        "temperature",
    ).sort_values(["tempmean_MAE_pct", "ablation_group"])
    tables[TABLES[1]] = t2

    tau = pd.read_csv(source / "ema_tau_group_sweep.csv")
    t3 = pivot_temp(tau, ["group_id", "description", "feature_set", "input_dim"], "MAE_pct", "temperature_C").sort_values(
        ["tempmean_MAE_pct", "group_id"]
    )
    tables[TABLES[2]] = t3

    rot_temp = pd.read_csv(source / "g4_profile_rotation_by_seed_temp.csv")
    t4 = pivot_temp(rot_temp, ["rotation_id", "train_profiles_rotation", "test_profile_rotation"], "MAE_pct", "temperature_C")
    rot_summary = pd.read_csv(source / "g4_profile_rotation_summary.csv")
    rot_agg = (
        rot_summary.groupby(["rotation_id", "train_profiles_rotation", "test_profile_rotation"], as_index=False)
        .agg(seed_count=("seed", "nunique"), reported_tempmean_MAE_pct=("tempmean_MAE_pct", "mean"), reported_worst_temp_MAE_pct=("worst_temp_MAE_pct", "mean"))
    )
    t4 = t4.merge(rot_agg, on=["rotation_id", "train_profiles_rotation", "test_profile_rotation"], how="left")
    tables[TABLES[3]] = t4

    epoch = pd.read_csv(source / "g4_epoch_sweep.csv")
    t5 = pivot_temp(epoch, ["epoch"], "MAE_pct", "temperature_C").sort_values("epoch")
    tables[TABLES[4]] = t5

    baselines = pd.read_csv(source / "g4_model_class_baselines.csv")
    t6 = build_model_class_table(baselines).sort_values(["temperature_mean_MAE_pct", "model_id"])
    tables[TABLES[5]] = t6

    forbidden = pd.read_csv(source / "forbidden_reference_baselines.csv")
    tapp = pivot_temp(forbidden, ["baseline", "model", "feature_columns", "run_status"], "MAE_pct", "temperature_C").sort_values(
        ["baseline"]
    )
    tables[TABLES[6]] = tapp

    for stem, df in tables.items():
        save_table(df, stem, table_dir)
    return tables


def build_figures(source: Path, fig_dir: Path) -> None:
    set_manuscript_figure_style()
    by_temp = pd.read_csv(source / "g4_seed_reproduction_by_temp.csv")
    fig1 = by_temp.groupby("temperature", as_index=False).agg(MAE_pct=("mae_pct", "mean"), MAE_std=("mae_pct", "std"))
    ax = fig1.plot(x="temperature", y="MAE_pct", yerr="MAE_std", kind="bar", legend=False, figsize=(6, 4), capsize=4)
    ax.set_ylabel("MAE (%)")
    ax.set_xlabel("Temperature (C)")
    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig1_main_g4_fuds_mae_by_temp.png", dpi=180)
    plt.savefig(fig_dir / "fig1_main_g4_fuds_mae_by_temp.pdf")
    plt.close()

    ablation = pd.read_csv(source / "feature_ablation_reanalysis.csv")
    bar_by_temp(ablation, "ablation_group", "temperature", "metric_value", "Feature Ablation MAE", "fig2_feature_ablation_by_temp", fig_dir)

    tau = pd.read_csv(source / "ema_tau_group_sweep.csv")
    bar_by_temp(tau, "group_id", "temperature_C", "MAE_pct", "EMA Group Sweep MAE", "fig3_ema_group_sweep_by_temp", fig_dir)

    rot = pd.read_csv(source / "g4_profile_rotation_by_seed_temp.csv")
    rot["rotation_label"] = rot["rotation_id"] + ": " + rot["test_profile_rotation"]
    bar_by_temp(rot, "rotation_label", "temperature_C", "MAE_pct", "Profile Rotation MAE", "fig4_profile_rotation_by_temp", fig_dir)

    epoch = pd.read_csv(source / "g4_epoch_sweep.csv")
    epoch["temperature_C"] = numeric_temp(epoch, "temperature_C")
    e_piv = epoch.dropna(subset=["temperature_C"]).pivot_table(index="epoch", columns="temperature_C", values="MAE_pct", aggfunc="mean")
    ax = e_piv.plot(figsize=(7, 4.2), marker="o")
    ax.set_ylabel("MAE (%)")
    ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_epoch_sweep.png", dpi=180)
    plt.savefig(fig_dir / "fig5_epoch_sweep.pdf")
    plt.close()

    pert = pd.read_csv(source / "ema_perturbation_importance.csv")
    pert_all = pert[pert["temperature_C"].astype(str).eq("ALL")].copy()
    if pert_all.empty:
        pert_all = pert.copy()
    pert_all = pert_all.sort_values("delta_MAE_vs_P0_pct")
    ax = pert_all.plot(x="perturbation", y="delta_MAE_vs_P0_pct", kind="barh", legend=False, figsize=(8, 5))
    ax.set_xlabel("Delta MAE vs unperturbed (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_ema_perturbation_importance.png", dpi=180)
    plt.savefig(fig_dir / "fig6_ema_perturbation_importance.pdf")
    plt.close()

    baselines = pd.read_csv(source / "g4_model_class_baselines.csv")
    bar_by_temp(baselines, "model_id", "temperature_C", "MAE_pct", "Model Class Baselines", "fig7_model_class_baselines", fig_dir)

    corr = pd.read_csv(source / "ema_vs_cumulative_correlation.csv")
    corr = corr[corr["reference"].astype(str).str.contains("cumulative|SOC_physical|progress", case=False, na=False)].copy()
    corr["abs_pearson_r"] = corr["pearson_r"].abs()
    top = corr.sort_values("abs_pearson_r", ascending=False).head(20)
    ax = top.plot(x="ema_feature", y="pearson_r", kind="bar", figsize=(10, 4.8), legend=False)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=75)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_appendix_ema_correlation.png", dpi=180)
    plt.savefig(fig_dir / "fig_appendix_ema_correlation.pdf")
    plt.close()


def write_caption_index(artifact_root: Path) -> None:
    lines = ["# Paper Artifact Captions", ""]
    captions = {
        "fig1_main_g4_fuds_mae_by_temp": "Frozen G4 FUDS MAE across 0C, 25C, and 45C, averaged over completed seeds.",
        "fig2_feature_ablation_by_temp": "Minimal feature ablation showing the contribution of derivative and EMA groups.",
        "fig3_ema_group_sweep_by_temp": "EMA group sweep used to interpret which causal memory groups are useful.",
        "fig4_profile_rotation_by_temp": "Profile-rotation diagnostic for FUDS and DST holdouts.",
        "fig5_epoch_sweep": "Diagnostic checkpoint sweep; the paper candidate remains the frozen epoch-160 setting.",
        "fig5_representative_fuds_soc_trajectory": "Temperature-wise representative FUDS profile-holdout SOC estimation trajectories of the frozen G4 CEMA-TCN model. For each temperature, the displayed seed trajectory is selected as the FUDS test trajectory whose trajectory-level MAE is closest to the corresponding three-seed temperature-mean MAE. The top row compares predicted and ground-truth SOC, and the bottom row shows the corresponding absolute error.",
        "fig6_ema_perturbation_importance": "Inference-only perturbation diagnostic for G4 EMA channels.",
        "fig7_model_class_baselines": "Model-class baselines compared under the same G4 feature protocol.",
        "fig7_corrected_voltage_behavior": "Time-domain behavior of corrected voltage and voltage EMA features on a representative held-out FUDS trajectory. (a) Raw terminal voltage and corrected voltage over the full trajectory. (b) Zoomed view of the same segment, showing attenuation of fast load-dependent variation in the corrected voltage. (c) Corrected voltage together with representative voltage EMA channels (`ema50`, `ema200`, and `ema800`), illustrating progressively slower finite-memory coordinates.",
        "fig8_region_error_reduction": "Regional error reduction from G0 to G4 across SOC bands, recent absolute-current-history regions, voltage-response-deviation regions, and local V-I ambiguity groups. Negative \u0394MAE indicates that the causal EMA representation reduces error relative to the raw corrected-voltage/current/temperature input.",
        "figS8_corrected_voltage_profile_temperature_grid": "Raw and corrected voltage trajectories across profile-temperature conditions. Each panel compares raw terminal voltage and corrected voltage for one profile-temperature record. Across profiles and temperatures, corrected voltage exhibits a smoother response than the raw terminal voltage, providing a more stable voltage-response coordinate for causal SOC estimation.",
        "fig_appendix_ema_correlation": "Appendix diagnostic showing EMA correlation caveats with forbidden references.",
    }
    for stem in TABLES:
        lines.append(f"- `{stem}`: generated as CSV, Markdown, and TeX.")
    lines.append("")
    for stem, caption in captions.items():
        lines.append(f"- `{stem}`: {caption}")
    (artifact_root / "manuscript_snippets" / "captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build paper tables and figures from small source metric CSVs.")
    p.add_argument("--base-dir", default=".")
    args = p.parse_args()
    base_dir = Path(args.base_dir).resolve()
    artifact_root = base_dir / "paper_artifacts"
    source = artifact_root / "source_metrics"
    table_dir = artifact_root / "tables"
    fig_dir = artifact_root / "figures"
    missing = [p.name for p in [source / "g4_seed_reproduction_by_temp.csv", source / "feature_ablation_reanalysis.csv"] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source metric files: " + ", ".join(missing))
    build_tables(source, table_dir)
    build_figures(source, fig_dir)
    write_caption_index(artifact_root)
    print(f"Generated {len(TABLES)} table bundles and {len(FIGURES)} figure pairs under {artifact_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
