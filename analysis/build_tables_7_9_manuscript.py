from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from analysis.build_frequency_structure_analysis import (
    build_feature_frame_local,
    estimate_r0_by_temperature_local,
    find_feature_csv_files,
    parse_feature_profile,
)


FEATURE_ROWS = [
    ("G0", "paper_g0_raw", "Corrected voltage + current + temperature", 3),
    ("G1", "paper_g1_derivatives", "G0 + local derivatives/excitation", 8),
    ("G4", "paper_g4_all_ema", "G0 + voltage/current/absolute-current EMA memory", 17),
    ("G6", "paper_g6_full23", "G4 + derivative/excitation terms", 23),
    ("G7", "paper_g7_no_current_ema", "G6 without current/absolute-current EMA", 15),
    ("G8", "paper_g8_no_voltage_ema", "G6 without voltage EMA", 17),
]

TABLE7_COLUMNS = [
    "feature_set",
    "input_role",
    "input_dim",
    "mae_0C",
    "mae_25C",
    "mae_45C",
    "temp_mean_mae",
    "worst_temp_mae",
]

PRETTY_TABLE7 = {
    "feature_set": "Feature set",
    "input_role": "Input role",
    "input_dim": "Input dim.",
    "mae_0C": "0 °C MAE",
    "mae_25C": "25 °C MAE",
    "mae_45C": "45 °C MAE",
    "temp_mean_mae": "Temp-mean MAE",
    "worst_temp_mae": "Worst-temp MAE",
}


def fmt_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def to_markdown(df: pd.DataFrame, pretty: dict[str, str] | None = None) -> str:
    work = df.rename(columns=pretty or {})
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt_value(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_tex(df: pd.DataFrame, pretty: dict[str, str] | None = None) -> str:
    work = df.rename(columns=pretty or {})
    cols = list(work.columns)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for _, row in work.iterrows():
        lines.append(" & ".join(fmt_value(row[c]) for c in cols) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def save_table(df: pd.DataFrame, stem: str, table_dir: Path, pretty: dict[str, str] | None = None) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / f"{stem}.csv", index=False)
    (table_dir / f"{stem}.md").write_text(to_markdown(df, pretty), encoding="utf-8")
    (table_dir / f"{stem}.tex").write_text(to_tex(df, pretty), encoding="utf-8")


def parse_results_dirs(values: list[str], base_dir: Path) -> list[Path]:
    out: list[Path] = []
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if path.exists():
            out.append(path)
    return out


def selected_test_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    test = df[df["split"].astype(str).eq("test")].copy()
    selected = test[test["variant"].astype(str).str.contains("selected", na=False)].copy()
    if selected.empty:
        selected = test.copy()
    selected["source_file"] = path.name
    return selected


def build_feature_ablation_by_seed(results_dirs: list[Path], source_dir: Path) -> pd.DataFrame:
    latest_by_seed = ROOT / "output" / "revision_risk_hardening" / "tables" / "main_fuds_by_seed.csv"
    if latest_by_seed.exists():
        latest = pd.read_csv(latest_by_seed)
        meta = {feature_set: (group, role, dim) for group, feature_set, role, dim in FEATURE_ROWS}
        rows: list[dict[str, object]] = []
        temp_rows = latest[latest["temperature_C"].astype(str).isin(["0.0", "25.0", "45.0"])].copy()
        temp_rows["temperature_C"] = pd.to_numeric(temp_rows["temperature_C"], errors="coerce")
        for _, r in temp_rows.iterrows():
            feature_set = str(r["feature_set_id"])
            if feature_set not in meta:
                continue
            group, role, dim = meta[feature_set]
            rows.append(
                {
                    "model_name": "anchor_residual_tcn",
                    "config_id": "paperdef_featabl_seed012_e160_latest",
                    "seed": int(r["seed"]),
                    "split_id": "train_0_25_45_DST_US06_BJDST__test_FUDS_0_25_45__fixed_epoch160",
                    "train_profiles": "DST,US06,BJDST",
                    "test_profile": "FUDS",
                    "temperature": float(r["temperature_C"]),
                    "metric_name": "feature_ablation_mae_pct",
                    "metric_value": float(r["MAE"]),
                    "notes": f"latest feature ablation 3-seed reproduction: {role}",
                    "ablation_group": group,
                    "feature_set": feature_set,
                    "input_dim": dim,
                    "result_prefix": f"paperdef_featabl_{feature_set}_seed012_e160",
                    "run_status": "DONE",
                    "evidence_scope": "3-seed reproduction",
                }
            )
        out = pd.DataFrame(rows)
        have = out.groupby("ablation_group")["seed"].agg(lambda s: set(int(v) for v in s)).to_dict()
        for group, *_ in FEATURE_ROWS:
            if have.get(group) != {0, 1, 2}:
                raise RuntimeError(f"{group} latest feature ablation rows are incomplete: {sorted(have.get(group, set()))}")
        tempmean_rows = []
        for (group, feature_set, seed), sdf in out.groupby(["ablation_group", "feature_set", "seed"]):
            temps = sdf[sdf["temperature"].isin([0.0, 25.0, 45.0])]
            if len(temps) != 3:
                continue
            first = temps.iloc[0].to_dict()
            first["temperature"] = "ALL"
            first["metric_name"] = "feature_ablation_tempmean_mae_pct"
            first["metric_value"] = float(temps["metric_value"].mean())
            tempmean_rows.append(first)
        if tempmean_rows:
            out = pd.concat([out, pd.DataFrame(tempmean_rows)], ignore_index=True)
        return out.sort_values(["ablation_group", "seed", "temperature"], key=lambda s: s.astype(str)).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    g4 = pd.read_csv(source_dir / "g4_seed_reproduction_by_temp.csv")
    for _, r in g4.iterrows():
        rows.append(
            {
                "model_name": r.get("model_name", "anchor_residual_tcn"),
                "config_id": r.get("config_id", "paperema_g4_candidate_frozen_e160"),
                "seed": int(r["seed"]),
                "split_id": r.get("split_id", "train_DST_US06_BJDST_test_FUDS"),
                "train_profiles": "DST,US06,BJDST",
                "test_profile": "FUDS",
                "temperature": float(r["temperature"]),
                "metric_name": "feature_ablation_mae_pct",
                "metric_value": float(r["mae_pct"]),
                "notes": "frozen G4 3-seed reproduction",
                "ablation_group": "G4",
                "feature_set": "paper_g4_all_ema",
                "input_dim": 17,
                "result_prefix": "paperema_g4_candidate_frozen_e160",
                "run_status": "DONE",
                "evidence_scope": "3-seed reproduction",
            }
        )
    meta = {feature_set: (group, role, dim) for group, feature_set, role, dim in FEATURE_ROWS}
    for group, feature_set, role, dim in FEATURE_ROWS:
        if group == "G4":
            continue
        seen: set[tuple[int, float]] = set()
        pattern = f"paperdef_featabl_{feature_set}_seed*_e160_by_temperature.csv"
        for root in results_dirs:
            for path in sorted(root.glob(pattern)):
                for _, r in selected_test_rows(path).iterrows():
                    temp = float(r["temperature_C"])
                    seed = int(r["seed"])
                    key = (seed, temp)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "model_name": "anchor_residual_tcn",
                            "config_id": "nmc_alltemp25_strict_nocc_W08A12_alltemps_noval_w0_0p8_anchor_l12a01_e160",
                            "seed": seed,
                            "split_id": "train_0_25_45_DST_US06_BJDST__test_FUDS_0_25_45__fixed_epoch160",
                            "train_profiles": "DST,US06,BJDST",
                            "test_profile": "FUDS",
                            "temperature": temp,
                            "metric_name": "feature_ablation_mae_pct",
                            "metric_value": float(r["MAE_pct"]),
                            "notes": f"feature ablation 3-seed reproduction: {role}",
                            "ablation_group": group,
                            "feature_set": feature_set,
                            "input_dim": dim,
                            "result_prefix": path.stem.replace("_by_temperature", ""),
                            "run_status": "DONE",
                            "evidence_scope": "3-seed reproduction",
                        }
                    )
        have = {seed for seed, _ in seen}
        if have != {0, 1, 2}:
            missing = sorted({0, 1, 2} - have)
            raise RuntimeError(f"{group}/{feature_set} missing seeds for Table 7: {missing}")
    out = pd.DataFrame(rows)
    tempmean_rows = []
    for (group, feature_set, seed), sdf in out.groupby(["ablation_group", "feature_set", "seed"]):
        temps = sdf[sdf["temperature"].isin([0.0, 25.0, 45.0])]
        if len(temps) != 3:
            continue
        first = temps.iloc[0].to_dict()
        first["temperature"] = "ALL"
        first["metric_name"] = "feature_ablation_tempmean_mae_pct"
        first["metric_value"] = float(temps["metric_value"].mean())
        tempmean_rows.append(first)
    if tempmean_rows:
        out = pd.concat([out, pd.DataFrame(tempmean_rows)], ignore_index=True)
    return out.sort_values(["ablation_group", "seed", "temperature"], key=lambda s: s.astype(str)).reset_index(drop=True)


def build_table7(ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, feature_set, role, dim in FEATURE_ROWS:
        temp_df = ablation[
            ablation["ablation_group"].eq(group)
            & ablation["temperature"].astype(str).isin(["0.0", "25.0", "45.0"])
        ].copy()
        by_temp = temp_df.groupby("temperature")["metric_value"].mean()
        values = {
            0.0: float(by_temp.loc[0.0]),
            25.0: float(by_temp.loc[25.0]),
            45.0: float(by_temp.loc[45.0]),
        }
        rows.append(
            {
                "feature_set": group,
                "input_role": role,
                "input_dim": dim,
                "mae_0C": values[0.0],
                "mae_25C": values[25.0],
                "mae_45C": values[45.0],
                "temp_mean_mae": float(np.mean(list(values.values()))),
                "worst_temp_mae": float(np.max(list(values.values()))),
            }
        )
    return pd.DataFrame(rows, columns=TABLE7_COLUMNS)


def build_table8(base_dir: Path) -> pd.DataFrame:
    src = base_dir / "paper_ema_analysis_package" / "frequency_structure_analysis" / "tables" / "Table_7_feature_frequency_summary_compact.csv"
    freq = pd.read_csv(src)
    wanted = [
        ("Raw voltage", "V_raw", "Raw voltage"),
        ("Corrected voltage", "V_corr_raw", "Corrected voltage"),
        ("Short voltage EMA", "V_corr_raw_ema50", "Short voltage EMA"),
        ("Long voltage EMA", "V_corr_raw_ema800", "Long voltage EMA"),
        ("Raw current", "I_raw", "Raw current"),
        ("Short current EMA", "I_raw_ema50", "Short current EMA"),
        ("Absolute-current EMA", "absI_ema50", "Short abs-current EMA"),
    ]
    rows = []
    for group, channel, source_group in wanted:
        row = freq[freq["representative_feature_column"].eq(channel)]
        if row.empty:
            row = freq[freq["feature_group"].eq(source_group)]
        if row.empty:
            raise RuntimeError(f"Missing frequency row for {group}/{channel}")
        r = row.iloc[0]
        rows.append(
            {
                "feature_group": group,
                "representative_channel": channel,
                "low_frequency_energy_percent": float(r["mean_low_frequency_energy_percent"]),
                "mid_frequency_energy_percent": float(r["mean_mid_frequency_energy_percent"]),
                "high_frequency_energy_percent": float(r["mean_high_frequency_energy_percent"]),
                "median_frequency": float(r["mean_median_frequency_cycles_per_sample"]),
                "high_frequency_reduction_vs_raw_reference_percent": float(
                    r["high_frequency_energy_reduction_vs_reference_percent"]
                ),
            }
        )
    return pd.DataFrame(rows)


def seed_from_prediction_path(path: Path) -> int:
    matches = re.findall(r"_seed(\d+)_sel", path.name)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"seed(\d+)", path.name)
    if not matches:
        raise RuntimeError(f"Could not parse seed from {path.name}")
    return int(matches[-1])


def load_predictions(results_dirs: list[Path], feature_label: str) -> pd.DataFrame:
    patterns = {
        "G0": ["paperdef_featabl_paper_g0_raw_seed*_e160_seed*_sel160_*_test_prediction_rows.csv.gz"],
        "G4": [
            "paperdef_featabl_paper_g4_all_ema_seed012_e160_seed*_sel160_*_test_prediction_rows.csv.gz",
        ],
    }
    frames = []
    seen_paths: set[Path] = set()
    for root in results_dirs:
        for pattern in patterns[feature_label]:
            for path in sorted(root.glob(pattern)):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                df = pd.read_csv(path)
                df["seed"] = seed_from_prediction_path(path)
                df["feature_set"] = feature_label
                frames.append(df)
    if not frames:
        raise RuntimeError(f"No prediction rows found for {feature_label}")
    out = pd.concat(frames, ignore_index=True)
    have = set(out["seed"].astype(int).unique())
    if have != {0, 1, 2}:
        raise RuntimeError(f"{feature_label} prediction rows missing seeds: {sorted({0, 1, 2} - have)}")
    return out


def load_fuds_descriptors(raw_root: Path) -> pd.DataFrame:
    files = find_feature_csv_files(raw_root)
    r0 = estimate_r0_by_temperature_local(files, ("DST", "US06", "BJDST"))
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0.iterrows()}
    frames = []
    for path in files:
        head = pd.read_csv(path, nrows=2)
        if parse_feature_profile(path, head) != "FUDS":
            continue
        feat = build_feature_frame_local(path, r0_lookup)
        raw = pd.read_csv(path, usecols=["SOC_CC"])
        feat["SOC_CC"] = pd.to_numeric(raw["SOC_CC"], errors="coerce").to_numpy(np.float64)[: len(feat)]
        frames.append(feat)
    desc = pd.concat(frames, ignore_index=True)
    desc["soc_band"] = pd.cut(
        desc["SOC_CC"].astype(float),
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=["Low SOC", "Mid SOC", "High SOC"],
    ).astype(str)
    desc["absI_history_group"] = np.where(
        desc["absI_ema200"] <= desc["absI_ema200"].median(),
        "Low",
        "High",
    )
    response = desc["V_corr_raw_dev_ema200"].abs()
    desc["voltage_response_group"] = np.where(response <= response.median(), "Low", "High")
    desc["voltage_bin"] = pd.qcut(desc["V_raw"], q=40, labels=False, duplicates="drop")
    desc["current_bin"] = pd.qcut(desc["I_raw"], q=40, labels=False, duplicates="drop")
    amb = (
        desc.groupby(["temperature", "voltage_bin", "current_bin"])["SOC_CC"]
        .agg(n="count", q25=lambda s: float(np.nanpercentile(s, 25)), q75=lambda s: float(np.nanpercentile(s, 75)))
        .reset_index()
    )
    amb["soc_iqr"] = amb["q75"] - amb["q25"]
    amb["local_vi_ambiguity_group"] = np.where(
        (amb["n"] >= 20) & (amb["soc_iqr"] >= 0.10),
        "Ambiguous bins",
        "Non-ambiguous bins",
    )
    desc = desc.merge(
        amb[["temperature", "voltage_bin", "current_bin", "local_vi_ambiguity_group"]],
        on=["temperature", "voltage_bin", "current_bin"],
        how="left",
    )
    desc["local_vi_ambiguity_group"] = desc["local_vi_ambiguity_group"].fillna("Non-ambiguous bins")
    keep = [
        "trajectory_id",
        "end_index",
        "soc_band",
        "absI_history_group",
        "voltage_response_group",
        "local_vi_ambiguity_group",
    ]
    return desc[keep].drop_duplicates(["trajectory_id", "end_index"])


def summarize_region(pred: pd.DataFrame, desc: pd.DataFrame, region_col: str, group_order: list[str]) -> pd.DataFrame:
    merged = pred.merge(desc, on=["trajectory_id", "end_index"], how="left")
    if merged[region_col].isna().any():
        missing = int(merged[region_col].isna().sum())
        raise RuntimeError(f"Missing {region_col} descriptor for {missing} prediction rows")
    rows = []
    for group in group_order:
        g = merged[merged[region_col].astype(str).eq(group)]
        if g.empty:
            continue
        seed_mae = g.groupby(["feature_set", "seed"])["abs_error"].mean().mul(100.0).reset_index(name="MAE")
        by_feature = seed_mae.groupby("feature_set")["MAE"].mean()
        if {"G0", "G4"}.issubset(set(by_feature.index)):
            g0 = float(by_feature.loc["G0"])
            g4 = float(by_feature.loc["G4"])
            delta = g4 - g0
            rows.append(
                {
                    "region_definition": region_col,
                    "group": group,
                    "G0_MAE": g0,
                    "G4_MAE": g4,
                    "delta_MAE_G4_minus_G0": delta,
                    "relative_change": 100.0 * delta / g0 if g0 != 0 else np.nan,
                    "n_windows": int(g.groupby(["trajectory_id", "end_index"]).ngroups),
                    "seed_count_G0": int(seed_mae[seed_mae["feature_set"].eq("G0")]["seed"].nunique()),
                    "seed_count_G4": int(seed_mae[seed_mae["feature_set"].eq("G4")]["seed"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def build_table9(results_dirs: list[Path], raw_root: Path) -> pd.DataFrame:
    desc = load_fuds_descriptors(raw_root)
    pred = pd.concat([load_predictions(results_dirs, "G0"), load_predictions(results_dirs, "G4")], ignore_index=True)
    parts = [
        summarize_region(pred, desc, "soc_band", ["Low SOC", "Mid SOC", "High SOC"]).assign(
            region_definition="SOC band"
        ),
        summarize_region(pred, desc, "absI_history_group", ["Low", "High"]).assign(
            region_definition="Recent absolute-current history"
        ),
        summarize_region(pred, desc, "voltage_response_group", ["Low", "High"]).assign(
            region_definition="Voltage-response deviation"
        ),
        summarize_region(pred, desc, "local_vi_ambiguity_group", ["Non-ambiguous bins", "Ambiguous bins"]).assign(
            region_definition="Local V-I ambiguity"
        ),
    ]
    out = pd.concat(parts, ignore_index=True)
    return out[
        [
            "region_definition",
            "group",
            "G0_MAE",
            "G4_MAE",
            "delta_MAE_G4_minus_G0",
            "relative_change",
            "n_windows",
            "seed_count_G0",
            "seed_count_G4",
        ]
    ]


def write_filled_snippet(table7: pd.DataFrame, table8: pd.DataFrame, table9: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Filled Manuscript Tables 7-9",
        "",
        "## Table 7. Feature-set ablation under the FUDS profile-holdout protocol.",
        "",
        to_markdown(table7, PRETTY_TABLE7),
        "## Table 8. Spectral energy distribution of representative measurement and EMA channels.",
        "",
        to_markdown(table8),
        "## Table 9. Error reduction by SOC, current-history, and voltage-response regions.",
        "",
        to_markdown(table9),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build manuscript-ready Tables 7-9 from frozen metrics and diagnostics.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default="data/raw/NMC_SAMSUNG_INR_18650_2Ah")
    p.add_argument("--results-dir", action="append", default=[])
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    source_dir = base_dir / "paper_artifacts" / "source_metrics"
    table_dir = base_dir / "paper_artifacts" / "tables"
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = (base_dir / raw_root).resolve()
    default_results = [
        base_dir / "nmc_goal_vcorr_it_train_dst_selector_results",
        base_dir / "feature_ablation_runs" / "nmc_goal_vcorr_it_train_dst_selector_results",
    ]
    results_dirs = [p for p in default_results if p.exists()]
    results_dirs.extend(parse_results_dirs(args.results_dir, base_dir))
    if not results_dirs:
        raise RuntimeError("No result directories found. Pass --results-dir.")

    ablation = build_feature_ablation_by_seed(results_dirs, source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(source_dir / "feature_ablation_reanalysis.csv", index=False)
    ablation[ablation["temperature"].astype(str).isin(["0.0", "25.0", "45.0"])].to_csv(
        source_dir / "feature_ablation_by_seed_temp.csv",
        index=False,
    )
    table7 = build_table7(ablation)
    table7.to_csv(source_dir / "feature_ablation_3seed_summary.csv", index=False)

    table8 = build_table8(base_dir)
    table9 = build_table9(results_dirs, raw_root)
    table9.to_csv(source_dir / "region_error_reduction_g0_g4.csv", index=False)

    save_table(table7, "table7_feature_set_ablation_fuds_holdout", table_dir, PRETTY_TABLE7)
    save_table(table8, "table8_spectral_energy_distribution", table_dir)
    save_table(table9, "table9_region_error_reduction", table_dir)
    write_filled_snippet(
        table7,
        table8,
        table9,
        base_dir / "paper_artifacts" / "manuscript_snippets" / "tables_7_9_filled.md",
    )
    print(f"Wrote Tables 7-9 under {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
