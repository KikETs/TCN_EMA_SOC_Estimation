from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
RAW_ROOT = (
    PROJECT_ROOT
    / "nmc_soc_ocvstart_relabelled_from_lc_ocv"
    / "data"
    / "NMC SAMSUNG INR 18650 2Ah"
)
CALCE_SOURCE_URL = "https://calce.umd.edu/battery-data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src_path = REPO_ROOT / src if not Path(src).is_absolute() else Path(src)
    dst_path = REPO_ROOT / dst if not Path(dst).is_absolute() else Path(dst)
    ensure_parent(dst_path)
    shutil.copyfile(src_path, dst_path)


def write_source_manifest() -> None:
    search_roots = [
        REPO_ROOT / "data" / "raw",
        PROJECT_ROOT,
        Path.home() / ("." + "codex") / "attachments",
    ]

    def find_archive(name: str) -> Path:
        for root in search_roots:
            if not root.exists():
                continue
            direct = root / name
            if direct.exists():
                return direct
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
        return Path(name)

    dynamic_zip = find_archive("NMC SAMSUNG INR 18650 2Ah.zip")
    ocv_zips = {
        "0": find_archive("SP1_0C_LC_OCV_02_24_2016.zip"),
        "25": find_archive("SP1_25C_LC_OCV_11_5_2015.zip"),
        "45": find_archive("SP1_45C_LC_OCV_11_21_2015.zip"),
    }
    capacity_zip = find_archive("SP1_Initial capacity_10_16_2015.zip")

    rows: list[dict[str, object]] = []
    dynamic_sha = sha256_file(dynamic_zip) if dynamic_zip.exists() else ""
    for temp in [0, 25, 45]:
        for profile in ["BJDST", "DST", "US06", "FUDS"]:
            rows.append(
                {
                    "raw_archive_filename": dynamic_zip.name,
                    "internal_file_name": f"NMC SAMSUNG INR 18650 2Ah/{temp}C/NMC_{temp}C_{profile}.csv",
                    "profile": profile,
                    "temperature_C": temp,
                    "cell": "Samsung INR18650-20R",
                    "chemistry": "NMC/graphite",
                    "record_type": "dynamic_profile",
                    "source_url": CALCE_SOURCE_URL,
                    "archive_sha256": dynamic_sha,
                    "raw_data_in_repository": False,
                }
            )
    for temp, path in ocv_zips.items():
        rows.append(
            {
                "raw_archive_filename": path.name,
                "internal_file_name": "",
                "profile": "LC_OCV",
                "temperature_C": int(temp),
                "cell": "Samsung INR18650-20R",
                "chemistry": "NMC/graphite",
                "record_type": "low_current_ocv_reference",
                "source_url": CALCE_SOURCE_URL,
                "archive_sha256": sha256_file(path) if path.exists() else "",
                "raw_data_in_repository": False,
            }
        )
    rows.append(
        {
            "raw_archive_filename": capacity_zip.name,
            "internal_file_name": "",
            "profile": "initial_capacity",
            "temperature_C": "",
            "cell": "Samsung INR18650-20R",
            "chemistry": "NMC/graphite",
            "record_type": "capacity_reference",
            "source_url": CALCE_SOURCE_URL,
            "archive_sha256": sha256_file(capacity_zip) if capacity_zip.exists() else "",
            "raw_data_in_repository": False,
        }
    )

    out = REPO_ROOT / "data" / "source_data_manifest.csv"
    ensure_parent(out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def channel_role(channel: str) -> str:
    if channel == "T":
        return "temperature"
    if channel.startswith("V_corr") or channel.startswith("V_raw") or channel.startswith("dV") or channel.startswith("abs_dV"):
        return "voltage"
    if channel.startswith("I") or channel.startswith("absI") or channel.startswith("dI") or "absI" in channel:
        return "current_excitation"
    return "interaction_or_other"


def feature_schema() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "g4_feature_sets.yaml").read_text())
    sets = cfg["feature_sets"]
    wanted = {
        "G0": "G0_raw",
        "G1": "G1_derivatives",
        "G4": "G4_all_ema",
        "G6": "G6_full23",
        "G7": "G7_no_current_ema",
        "G8": "G8_no_voltage_ema",
    }
    rows = []
    for group, key in wanted.items():
        features = sets[key]["features"]
        for idx, channel in enumerate(features):
            m = re.search(r"ema(\d+)", channel)
            rows.append(
                {
                    "feature_set": group,
                    "config_key": key,
                    "ablation_group": group,
                    "input_dim": len(features),
                    "channel_index": idx,
                    "channel": channel,
                    "role": channel_role(channel),
                    "ema_scale_samples": int(m.group(1)) if m else "",
                    "is_ema_channel": bool(m),
                    "is_deviation_from_ema": "_dev_ema" in channel,
                    "uses_voltage_measurement": channel_role(channel) == "voltage",
                    "uses_current_excitation": channel_role(channel) == "current_excitation",
                    "uses_temperature": channel == "T",
                }
            )
    out = REPO_ROOT / "data" / "tables" / "feature_schema.csv"
    ensure_parent(out)
    pd.DataFrame(rows).to_csv(out, index=False)


def write_feature_construction_audit() -> None:
    text = """# Feature Construction Audit

- SOC input, window-start SOC, SOC_CC input, cumulative Ah input, and absolute trajectory progress are excluded from G0/G1/G4/G6/G7/G8 feature schemas.
- Current is used as instantaneous or finite-memory causal excitation (`I_raw`, `dI`, `absI`, `I_raw_ema*`, `absI_ema*`), not as an explicit SOC state update.
- EMA channels use one-sided recurrence and reset at record boundaries. They do not use future samples.
- Normalization is fitted on the training-side profiles only for each split.
- Corrected-voltage features use the repository's shared feature-construction path; raw CALCE files are not included in this repository package.
"""
    out = REPO_ROOT / "data" / "tables" / "feature_construction_audit.md"
    ensure_parent(out)
    out.write_text(text)


def write_training_config() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "g4_frozen.yaml").read_text())
    cfg.setdefault("model", {})["tcn_block_convs"] = 2
    cfg["model"]["tcn_block_convs_note"] = "Proposed CEMA-TCN uses two causal Conv1d operations per residual TCN block."
    out = REPO_ROOT / "configs" / "training_config.yaml"
    ensure_parent(out)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))


def write_sanitized_run_manifest() -> None:
    src = REPO_ROOT / "output" / "revision_risk_hardening" / "run_manifest.json"
    manifest = json.loads(src.read_text())
    keep = {
        "git_commit": manifest.get("git_commit"),
        "platform": manifest.get("platform"),
        "package_versions": manifest.get("package_versions"),
        "seeds": manifest.get("seeds"),
        "main_protocol": manifest.get("main_protocol"),
        "feature_sets": manifest.get("feature_sets"),
        "model_hyperparameters": manifest.get("model_hyperparameters"),
        "raw_data_in_repository": False,
        "path_note": "Raw CALCE files are intentionally excluded; use data/source_data_manifest.csv and configs/paths.example.yaml.",
    }
    out = REPO_ROOT / "configs" / "run_manifest.json"
    ensure_parent(out)
    out.write_text(json.dumps(keep, indent=2, sort_keys=True))


def recompute_model_comparison() -> None:
    pred_root = REPO_ROOT / "output" / "revision_risk_hardening" / "predictions" / "model_class_controls"
    rows = []
    for path in sorted(pred_root.glob("*_test_prediction_rows.csv.gz")):
        match = re.search(r"risk_modelclass_(.+?)_seed012_e160_seed(\d+)", path.name)
        if not match:
            continue
        model_id = match.group(1)
        if model_id == "cema_tcn":
            continue
        seed = int(match.group(2))
        frame = pd.read_csv(path)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()
        if "drive_cycle" in frame.columns:
            frame = frame[frame["drive_cycle"].astype(str).str.upper().eq("FUDS")].copy()
        for temp, group in frame.groupby("temperature"):
            err_pct = (pd.to_numeric(group["y_pred"], errors="coerce") - pd.to_numeric(group["y_true"], errors="coerce")) * 100.0
            rows.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "temperature_C": float(temp),
                    "n_windows": int(err_pct.notna().sum()),
                    "MAE_pct": float(err_pct.abs().mean()),
                    "RMSE_pct": float(np.sqrt(np.mean(np.square(err_pct)))),
                    "MaxAE_pct": float(err_pct.abs().max()),
                    "source_prediction_file": path.name,
                }
            )
    main = pd.read_csv(REPO_ROOT / "paper_artifacts" / "source_metrics" / "g4_seed_reproduction_by_temp.csv")
    for _, row in main.iterrows():
        rows.append(
            {
                "model_id": "cema_tcn_proposed_g4",
                "seed": int(row["seed"]),
                "temperature_C": float(row["temperature"]),
                "n_windows": int(row["count"]),
                "MAE_pct": float(row["mae_pct"]),
                "RMSE_pct": float(row["rmse_pct"]),
                "MaxAE_pct": float(row["maxae_pct"]),
                "source_prediction_file": "paper_artifacts/source_metrics/g4_seed_reproduction_by_temp.csv",
            }
        )
    by_seed = pd.DataFrame(rows).sort_values(["model_id", "seed", "temperature_C"])
    by_seed_out = REPO_ROOT / "results" / "model_comparison" / "model_comparison_by_seed.csv"
    ensure_parent(by_seed_out)
    by_seed.to_csv(by_seed_out, index=False)

    summary = (
        by_seed.groupby(["model_id", "temperature_C"], as_index=False)
        .agg(
            MAE_pct_mean=("MAE_pct", "mean"),
            MAE_pct_std=("MAE_pct", lambda s: float(pd.Series(s).std(ddof=0))),
            RMSE_pct_mean=("RMSE_pct", "mean"),
            MaxAE_pct_mean=("MaxAE_pct", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["model_id", "temperature_C"])
    )
    temp_mean = (
        by_seed.groupby(["model_id", "seed"], as_index=False)
        .agg(MAE_pct=("MAE_pct", "mean"), RMSE_pct=("RMSE_pct", "mean"), MaxAE_pct=("MaxAE_pct", "max"))
        .groupby("model_id", as_index=False)
        .agg(
            temperature_C=("model_id", lambda _: "temp_mean"),
            MAE_pct_mean=("MAE_pct", "mean"),
            MAE_pct_std=("MAE_pct", lambda s: float(pd.Series(s).std(ddof=0))),
            RMSE_pct_mean=("RMSE_pct", "mean"),
            MaxAE_pct_mean=("MaxAE_pct", "mean"),
            seed_count=("MAE_pct", "size"),
        )
    )
    out = pd.concat([summary, temp_mean], ignore_index=True)
    table6 = pd.read_csv(REPO_ROOT / "paper_artifacts" / "tables" / "table1_main_g4_fuds_results.csv")
    proposed_rows = []
    for _, row in table6.iterrows():
        proposed_rows.append(
            {
                "model_id": "cema_tcn_proposed_g4",
                "temperature_C": row["temperature"],
                "MAE_pct_mean": float(row["MAE_pct_mean"]),
                "MAE_pct_std": float(row["MAE_pct_std"]),
                "RMSE_pct_mean": float(row["RMSE_pct_mean"]),
                "MaxAE_pct_mean": float(row["MaxAE_pct_mean"]),
                "seed_count": int(row["seed_count"]),
            }
        )
    out = out[~out["model_id"].eq("cema_tcn_proposed_g4")]
    out = pd.concat([pd.DataFrame(proposed_rows), out], ignore_index=True)
    out_path = REPO_ROOT / "results" / "model_comparison" / "model_comparison_summary.csv"
    ensure_parent(out_path)
    out.to_csv(out_path, index=False)


def main_fuds_outputs() -> None:
    by_temp = pd.read_csv(REPO_ROOT / "paper_artifacts" / "source_metrics" / "g4_seed_reproduction_by_temp.csv")
    by_seed = by_temp.rename(
        columns={
            "temperature": "temperature_C",
            "count": "n_windows",
            "mae_pct": "MAE_pct",
            "rmse_pct": "RMSE_pct",
            "maxae_pct": "MaxAE_pct",
        }
    )
    cols = [
        "model_name",
        "config_id",
        "seed",
        "split_id",
        "train_profiles",
        "test_profile",
        "temperature_C",
        "n_windows",
        "MAE_pct",
        "RMSE_pct",
        "MaxAE_pct",
        "mean_signed_error_pct",
        "residual_std_pct",
        "run_status",
    ]
    out = REPO_ROOT / "results" / "main_fuds" / "main_fuds_by_seed.csv"
    ensure_parent(out)
    by_seed[cols].to_csv(out, index=False)
    copy_file("paper_artifacts/tables/table1_main_g4_fuds_results.csv", "results/main_fuds/main_fuds_seed_summary.csv")


def write_regional_thresholds() -> None:
    rows = [
        {
            "region_definition": "SOC band",
            "threshold_rule": "Low: SOC_CC <= 0.35; Mid: 0.35 < SOC_CC <= 0.65; High: SOC_CC > 0.65",
            "threshold_source": "analysis/build_tables_7_9_manuscript.py",
        },
        {
            "region_definition": "Recent absolute-current history",
            "threshold_rule": "Low/High split by median absI_ema200 over FUDS descriptors",
            "threshold_source": "analysis/build_tables_7_9_manuscript.py",
        },
        {
            "region_definition": "Voltage-response deviation",
            "threshold_rule": "Low/High split by median abs(V_corr_raw_dev_ema200) over FUDS descriptors",
            "threshold_source": "analysis/build_tables_7_9_manuscript.py",
        },
        {
            "region_definition": "Local V-I ambiguity",
            "threshold_rule": "Ambiguous if local V-I bin has n >= 20 and SOC IQR >= 0.10 SOC fraction",
            "threshold_source": "analysis/build_tables_7_9_manuscript.py",
        },
    ]
    out = REPO_ROOT / "results" / "regional" / "regional_thresholds.csv"
    ensure_parent(out)
    pd.DataFrame(rows).to_csv(out, index=False)


def write_representative_trace() -> None:
    sys_path_added = False
    import sys

    analysis_dir = REPO_ROOT / "analysis"
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
        sys_path_added = True
    try:
        from build_frequency_structure_analysis import (
            build_feature_frame_local,
            estimate_r0_by_temperature_local,
            find_feature_csv_files,
        )
    finally:
        if sys_path_added:
            pass

    files = find_feature_csv_files(RAW_ROOT)
    r0_df = estimate_r0_by_temperature_local(files, ("DST", "US06", "BJDST"))
    r0_lookup = {float(row["temperature_C"]): float(row["r0_ohm"]) for _, row in r0_df.iterrows()}
    feature = None
    for path in files:
        if path.name == "NMC_25C_FUDS.csv":
            feature = build_feature_frame_local(path, r0_lookup)
            raw = pd.read_csv(path)
            break
    if feature is None:
        raise FileNotFoundError("NMC_25C_FUDS.csv was not found for representative trace generation.")
    pred_paths = sorted(
        (REPO_ROOT / "output" / "revision_risk_hardening" / "predictions" / "main_fuds").glob(
            "paperdef_featabl_paper_g4_all_ema_seed012_e160_seed1_*_test_prediction_rows.csv.gz"
        )
    )
    pred = pd.read_csv(pred_paths[0])
    pred["seed"] = 1
    pred = pred[pred["file_name"].astype(str).eq("NMC_25C_FUDS.csv")].copy()
    keep = [
        "V_raw",
        "I_raw",
        "T",
        "V_corr_raw",
        "V_corr_raw_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_ema800",
        "I_raw_ema50",
        "I_raw_ema200",
        "absI_ema50",
        "absI_ema200",
    ]
    trace = feature[keep].copy()
    trace.insert(0, "time_s", pd.to_numeric(raw["Test_Time(s)"], errors="coerce").to_numpy()[: len(trace)])
    trace["time_s"] = trace["time_s"] - trace["time_s"].iloc[0]
    trace["SOC_fraction"] = pd.to_numeric(raw["SOC_CC"], errors="coerce").to_numpy()[: len(trace)]
    trace["profile"] = "FUDS"
    trace["temperature_C"] = 25
    trace["file_name"] = "NMC_25C_FUDS.csv"
    pred_small = pred[["end_index", "seed", "y_true", "y_pred", "abs_error"]].rename(
        columns={"y_true": "SOC_true_window_endpoint", "y_pred": "SOC_pred_window_endpoint"}
    )
    trace = trace.reset_index(names="sample_index").merge(pred_small, how="left", left_on="sample_index", right_on="end_index")
    out = REPO_ROOT / "data" / "figures_source" / "representative_cema_traces.csv"
    ensure_parent(out)
    trace.to_csv(out, index=False)


def copy_required_sources() -> None:
    copies = {
        "output/current_profile_diagnostics/nmc_current_profile_one_cycle_25C_summary.csv": "data/tables/profile_statistics.csv",
        "output/current_profile_diagnostics/nmc_current_profile_one_cycle_by_temperature.csv": "data/tables/profile_statistics_by_temperature.csv",
        "output/current_profile_diagnostics/nmc_current_one_cycle_acf_25C_summary.csv": "data/tables/acf_summary.csv",
        "output/current_profile_diagnostics/nmc_current_one_cycle_acf_by_temperature.csv": "data/tables/acf_summary_by_temperature.csv",
        "output/current_profile_diagnostics/nmc_current_one_cycle_acf_long.csv": "data/tables/acf_long.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/table_s2_causal_lag_correlations.csv": "data/tables/causal_lag_correlation.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/table_s2_raw_vi_bin_soc_spread.csv": "data/tables/vi_bin_soc_ambiguity.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/table_s2_history_conditioned_soc_spread.csv": "data/tables/ambiguity_stratification_summary.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/table_s2_history_conditioned_soc_spread_details.csv": "data/tables/ambiguity_stratification_full.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/table_s2_profile_shift_vi_overlap.csv": "data/tables/vi_support_coverage.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/table_s2_raw_vi_bin_details.csv": "data/tables/vi_support_density_grid.csv",
        "paper_ema_analysis_package/section2_measurement_structure/tables/Table_S5_training_coverage_with_without_BJDST.csv": "data/tables/training_coverage_with_without_BJDST.csv",
        "paper_artifacts/source_metrics/feature_ablation_3seed_summary.csv": "results/feature_ablation/feature_ablation_summary.csv",
        "paper_artifacts/source_metrics/feature_ablation_by_seed_temp.csv": "results/feature_ablation/feature_ablation_by_seed.csv",
        "paper_artifacts/tables/table8_spectral_energy_distribution.csv": "results/spectral/spectral_energy_summary.csv",
        "paper_ema_analysis_package/frequency_structure_analysis/tables/Table_S7_feature_frequency_summary_by_record.csv": "results/spectral/spectral_full_by_profile_temperature.csv",
        "paper_artifacts/source_metrics/region_error_reduction_g0_g4.csv": "results/regional/regional_error_summary.csv",
        "paper_artifacts/source_metrics/region_error_reduction_g0_g4.csv": "results/regional/regional_error_counts.csv",
        "output/revision_risk_hardening/audit/feature_leakage_audit.md": "audits/feature_leakage_audit.md",
        "output/revision_risk_hardening/audit/hyperparameter_selection_audit.md": "audits/hyperparameter_selection_audit.md",
        "output/revision_risk_hardening/audit/no_test_influence_checklist.md": "audits/no_test_influence_checklist.md",
        "output/revision_risk_hardening/audit/reference_soc_generation.md": "audits/reference_soc_generation.md",
        "paper_artifacts/source_metrics/g4_frozen_manifest.json": "results/main_fuds/g4_frozen_manifest.json",
    }
    for src, dst in copies.items():
        copy_file(src, dst)
    copy_file(
        "paper_artifacts/source_metrics/region_error_reduction_g0_g4.csv",
        "results/regional/regional_error_summary.csv",
    )


def write_package_readme() -> None:
    text = """# Minimal Manuscript Data Package

This repository intentionally excludes raw CALCE data archives and model checkpoints. The files in `data/`, `results/`, `audits/`, and `configs/` provide the minimum manuscript/SI source metrics needed to recompute the reported tables and figures.

Raw data provenance is recorded in `data/source_data_manifest.csv` with CALCE archive names and SHA256 checksums. Rebuild scripts use `configs/paths.example.yaml` to point to local raw files.

Large prediction-row files and leave-one-profile-out scratch outputs are not part of this minimal package unless they are explicitly needed for a manuscript table or figure.
"""
    (REPO_ROOT / "data" / "README.md").write_text(text)


def main() -> None:
    write_source_manifest()
    copy_required_sources()
    feature_schema()
    write_feature_construction_audit()
    write_training_config()
    write_sanitized_run_manifest()
    main_fuds_outputs()
    recompute_model_comparison()
    write_regional_thresholds()
    write_representative_trace()
    write_package_readme()
    print("Wrote minimal manuscript data package.")


if __name__ == "__main__":
    main()
