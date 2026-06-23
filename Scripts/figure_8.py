from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_frequency_structure_analysis import (  # noqa: E402
    build_feature_frame_local,
    estimate_r0_by_temperature_local,
    find_feature_csv_files,
)


SUMMARY_CSV = REPO_ROOT / "Data" / "source_metrics" / "g4_seed_reproduction_summary.csv"
BY_TEMP_CSV = REPO_ROOT / "Data" / "source_metrics" / "g4_seed_reproduction_by_temp.csv"
LATEST_MAIN_FUDS_SUMMARY_CSV = REPO_ROOT / "output" / "revision_risk_hardening" / "tables" / "main_fuds_seed_summary.csv"
OUT_PNG = REPO_ROOT / "Figures" / "figure_8_corrected_voltage_behavior.png"
OUT_PDF = REPO_ROOT / "Figures" / "figure_8_corrected_voltage_behavior.pdf"
PREFERRED_TEMPERATURE_C = 25.0
DEFAULT_OVERALL_TEMP_MEAN_MAE = 0.41889356670972
PREFERRED_PREDICTION_PATTERNS = (
    "paperdef_featabl_paper_g4_all_ema_seed012_e160_seed*_sel160_*base_test_prediction_rows.csv*",
)
REQUIRED_COLUMNS = ["V_raw", "V_corr_raw", "V_corr_raw_ema50", "V_corr_raw_ema200", "V_corr_raw_ema800"]


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.default": "bf",
        }
    )


def format_temp(temp: float) -> str:
    return f"{int(temp) if float(temp).is_integer() else temp:g} \u00b0C"


def target_overall_mae(summary_csv: Path) -> float:
    if LATEST_MAIN_FUDS_SUMMARY_CSV.exists():
        latest = pd.read_csv(LATEST_MAIN_FUDS_SUMMARY_CSV)
        mask = latest["feature_set"].astype(str).eq("G4") & latest["temperature_C"].astype(str).eq("temp_mean")
        values = pd.to_numeric(latest.loc[mask, "MAE_mean"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    if not summary_csv.exists():
        return DEFAULT_OVERALL_TEMP_MEAN_MAE
    summary = pd.read_csv(summary_csv)
    aggregate = summary[summary["metric_name"].eq("g4_seed_aggregate_tempmean_mae_pct")]
    if aggregate.empty:
        return DEFAULT_OVERALL_TEMP_MEAN_MAE
    value = pd.to_numeric(aggregate["mean_tempmean_mae_pct"], errors="coerce").dropna()
    if value.empty:
        value = pd.to_numeric(aggregate["metric_value"], errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else DEFAULT_OVERALL_TEMP_MEAN_MAE


def target_mae_by_temperature(by_temp_csv: Path, fallback_overall: float) -> dict[float, float]:
    if LATEST_MAIN_FUDS_SUMMARY_CSV.exists():
        latest = pd.read_csv(LATEST_MAIN_FUDS_SUMMARY_CSV)
        latest = latest[latest["feature_set"].astype(str).eq("G4")].copy()
        latest["temperature_C_num"] = pd.to_numeric(latest["temperature_C"], errors="coerce")
        latest["MAE_mean"] = pd.to_numeric(latest["MAE_mean"], errors="coerce")
        latest = latest.dropna(subset=["temperature_C_num", "MAE_mean"])
        targets = latest.groupby("temperature_C_num")["MAE_mean"].mean().to_dict()
        if targets:
            return {float(temp): float(mae) for temp, mae in targets.items()}
    if not by_temp_csv.exists():
        return {}
    by_temp = pd.read_csv(by_temp_csv)
    required = {"temperature", "mae_pct"}
    if not required.issubset(by_temp.columns):
        return {}
    by_temp["temperature"] = pd.to_numeric(by_temp["temperature"], errors="coerce")
    by_temp["mae_pct"] = pd.to_numeric(by_temp["mae_pct"], errors="coerce")
    targets = by_temp.dropna(subset=["temperature", "mae_pct"]).groupby("temperature")["mae_pct"].mean().to_dict()
    return {float(temp): float(mae) for temp, mae in targets.items()} or {np.nan: fallback_overall}


def parse_seed(path: Path, frame: pd.DataFrame | None = None) -> int | None:
    if frame is not None and "seed" in frame.columns:
        values = pd.to_numeric(frame["seed"], errors="coerce").dropna().unique()
        if len(values) == 1:
            return int(values[0])
    matches = re.findall(r"_seed(\d+)(?:_|$)", path.name)
    return int(matches[-1]) if matches else None


def prediction_search_roots(extra_roots: list[str]) -> list[Path]:
    roots: list[Path] = [
        REPO_ROOT / "Data" / "predictions" / "main_fuds",
        REPO_ROOT / "output" / "revision_risk_hardening" / "predictions" / "main_fuds",
        REPO_ROOT / "nmc_goal_vcorr_it_train_dst_selector_results",
        REPO_ROOT / "results" / "predictions",
        REPO_ROOT / "feature_ablation_runs",
        REPO_ROOT.parent / "nmc_goal_vcorr_it_train_dst_selector_results",
        REPO_ROOT.parent / "remote_result_summaries",
    ]
    env_roots = [p for p in os.environ.get("G4_PREDICTION_ROOTS", "").split(os.pathsep) if p]
    roots.extend(Path(p).expanduser() for p in env_roots)
    roots.extend(Path(p).expanduser() for p in extra_roots)

    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved.exists() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def discover_prediction_files(extra_roots: list[str], explicit_files: list[str]) -> list[Path]:
    if explicit_files:
        return [Path(p).expanduser().resolve() for p in explicit_files]

    found: list[Path] = []
    seen: set[Path] = set()
    seen_file_names: set[str] = set()
    for root in prediction_search_roots(extra_roots):
        for pattern in PREFERRED_PREDICTION_PATTERNS:
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved not in seen and resolved.name not in seen_file_names:
                    found.append(resolved)
                    seen.add(resolved)
                    seen_file_names.add(resolved.name)
    return sorted(found)


def read_prediction_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in ("y_true", "y_pred") if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()
    if "drive_cycle" in frame.columns:
        frame = frame[frame["drive_cycle"].astype(str).str.upper().eq("FUDS")].copy()
    elif "file_name" in frame.columns:
        frame = frame[frame["file_name"].astype(str).str.upper().str.contains("FUDS")].copy()
    if frame.empty:
        raise ValueError(f"{path} does not contain FUDS test prediction rows.")
    if "trajectory_id" not in frame.columns:
        frame["trajectory_id"] = frame["file_name"] if "file_name" in frame.columns else "trajectory"
    if "file_name" not in frame.columns:
        frame["file_name"] = frame["trajectory_id"]
    if "temperature" not in frame.columns:
        frame["temperature"] = np.nan
    frame["source_prediction_file"] = path.as_posix()
    seed = parse_seed(path, frame)
    frame["seed"] = -1 if seed is None else seed
    frame["abs_error_fraction"] = (pd.to_numeric(frame["y_pred"], errors="coerce") - pd.to_numeric(frame["y_true"], errors="coerce")).abs()
    return frame.dropna(subset=["y_true", "y_pred", "abs_error_fraction"])


def trajectory_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    if "trajectory_id" not in work.columns:
        work["trajectory_id"] = work.get("file_name", pd.Series(["trajectory"] * len(work), index=work.index))
    if "file_name" not in work.columns:
        work["file_name"] = work["trajectory_id"]
    if "temperature" not in work.columns:
        work["temperature"] = np.nan
    work["temperature"] = pd.to_numeric(work["temperature"], errors="coerce")

    group_cols = ["seed", "trajectory_id", "file_name", "temperature", "source_prediction_file"]
    return (
        work.groupby(group_cols, dropna=False)
        .agg(
            trajectory_MAE_pct=("abs_error_fraction", lambda s: float(s.mean() * 100.0)),
            n_points=("abs_error_fraction", "size"),
        )
        .reset_index()
        .sort_values(["temperature", "seed", "trajectory_id"])
        .reset_index(drop=True)
    )


def choose_representatives_by_temperature(predictions: pd.DataFrame, target_by_temp: dict[float, float]) -> pd.DataFrame:
    trajectory_metrics = trajectory_metric_table(predictions)
    selected: list[pd.Series] = []
    for temp, group in trajectory_metrics.groupby("temperature", sort=True):
        target = target_by_temp.get(float(temp), float(group["trajectory_MAE_pct"].mean()))
        candidates = group.copy()
        candidates["target_temperature_MAE_pct"] = target
        candidates["distance_to_temperature_MAE"] = (candidates["trajectory_MAE_pct"] - target).abs()
        selected.append(
            candidates.sort_values(["distance_to_temperature_MAE", "trajectory_MAE_pct", "seed", "trajectory_id"]).iloc[0]
        )
    return pd.DataFrame(selected).sort_values("temperature").reset_index(drop=True)


def candidate_raw_roots(extra_roots: list[str]) -> list[Path]:
    roots = [
        REPO_ROOT / "Data" / "processed",
        REPO_ROOT / "data" / "raw" / "NMC_SAMSUNG_INR_18650_2Ah",
        REPO_ROOT.parent / "nmc_soc_ocvstart_relabelled_from_lc_ocv" / "data" / "NMC SAMSUNG INR 18650 2Ah",
        REPO_ROOT.parent / "nmc_soc80_relabelled_from_lc_ocv" / "data" / "NMC SAMSUNG INR 18650 2Ah",
        REPO_ROOT.parent / "nmc_samsung_inr_18650_2ah_raw" / "NMC SAMSUNG INR 18650 2Ah",
    ]
    env_roots = [p for p in os.environ.get("G4_RAW_ROOTS", "").split(os.pathsep) if p]
    roots.extend(Path(p).expanduser() for p in env_roots)
    roots.extend(Path(p).expanduser() for p in extra_roots)

    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved.exists() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def select_figure5_temperature_representative(prediction_roots: list[str], prediction_files: list[str]) -> pd.Series:
    target_overall = target_overall_mae(SUMMARY_CSV)
    target_by_temp = target_mae_by_temperature(BY_TEMP_CSV, target_overall)
    paths = discover_prediction_files(prediction_roots, prediction_files)
    if not paths:
        raise FileNotFoundError(
            "No frozen G4 prediction rows found. Reuse the Figure 5 prediction-row files, "
            "set G4_PREDICTION_ROOTS, or pass --prediction-files."
        )
    predictions = pd.concat([read_prediction_rows(path) for path in paths], ignore_index=True)
    selected_by_temp = choose_representatives_by_temperature(predictions, target_by_temp)
    selected_by_temp["_preferred_distance"] = (selected_by_temp["temperature"] - PREFERRED_TEMPERATURE_C).abs()
    return selected_by_temp.sort_values(["_preferred_distance", "temperature"]).iloc[0]


def build_g4_feature_frames(raw_root: Path) -> list[pd.DataFrame]:
    files = find_feature_csv_files(raw_root)
    r0_df = estimate_r0_by_temperature_local(files, ("DST", "US06", "BJDST"))
    r0_lookup = {float(row["temperature_C"]): float(row["r0_ohm"]) for _, row in r0_df.iterrows()}
    return [build_feature_frame_local(path, r0_lookup) for path in files]


def load_selected_feature_frame(selected: pd.Series, raw_roots: list[Path]) -> tuple[pd.DataFrame, Path]:
    last_error: Exception | None = None
    for raw_root in raw_roots:
        try:
            frames = build_g4_feature_frames(raw_root)
            for frame in frames:
                if str(frame["file_name"].iloc[0]) == str(selected["file_name"]):
                    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
                    if missing:
                        raise ValueError(f"{selected['file_name']} feature frame is missing columns: {missing}")
                    return frame.reset_index(drop=True), raw_root
        except Exception as exc:  # pragma: no cover - diagnostic path for local data discovery.
            last_error = exc
            continue
    detail = f" Last error: {last_error}" if last_error else ""
    raise FileNotFoundError(f"Could not build selected FUDS feature frame from available raw roots.{detail}")


def attach_time_axis(frame: pd.DataFrame, raw_root: Path, file_name: str) -> tuple[pd.DataFrame, str]:
    if "time_s_for_frequency_metadata" in frame.columns:
        values = pd.to_numeric(frame["time_s_for_frequency_metadata"], errors="coerce").to_numpy(float)
        if len(values) == len(frame) and np.isfinite(values).any():
            out = frame.copy()
            first = values[np.where(np.isfinite(values))[0][0]]
            out["_x"] = values - first
            return out, "Time (s)"
    matches = sorted(raw_root.rglob(file_name))
    if not matches:
        frame = frame.copy()
        frame["_x"] = np.arange(len(frame), dtype=float)
        return frame, "Sample index"
    raw = pd.read_csv(matches[0])
    for col, label in [("Test_Time(s)", "Time (s)"), ("t_global(s)", "Time (s)"), ("Step_Time(s)", "Time (s)")]:
        if col in raw.columns:
            values = pd.to_numeric(raw[col], errors="coerce").to_numpy(float)
            if len(values) == len(frame) and np.isfinite(values).any():
                out = frame.copy()
                out["_x"] = values - values[np.where(np.isfinite(values))[0][0]]
                return out, label
    out = frame.copy()
    out["_x"] = np.arange(len(frame), dtype=float)
    return out, "Sample index"


def choose_zoom_interval(frame: pd.DataFrame) -> tuple[int, int, float]:
    # Deterministic rule: choose the fixed-length segment with the largest local
    # standard deviation of V_raw - V_corr_raw, which targets visible
    # load-dependent voltage fluctuation rather than a flat relaxation segment.
    # The first 5% and last 10% of rows are excluded to avoid start/end transients
    # dominating the zoom choice.
    n = len(frame)
    window = int(min(1200, max(400, n // 12)))
    residual = pd.Series(frame["V_raw"].to_numpy(float) - frame["V_corr_raw"].to_numpy(float))
    score = residual.rolling(window=window, min_periods=window, center=False).std()
    if score.dropna().empty:
        return 0, n, float("nan")
    min_start = int(0.05 * n)
    max_end = int(0.90 * n)
    candidate_score = score.copy()
    for idx in candidate_score.dropna().index:
        end_idx = int(idx) + 1
        start_idx = end_idx - window
        if start_idx < min_start or end_idx > max_end:
            candidate_score.iloc[int(idx)] = np.nan
    if candidate_score.dropna().empty:
        candidate_score = score
    end = int(candidate_score.idxmax()) + 1
    start = max(0, end - window)
    return start, min(n, end), float(score.iloc[end - 1])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.015,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def plot_figure(frame: pd.DataFrame, selected: pd.Series, zoom: tuple[int, int, float], xlabel: str, out_png: Path, out_pdf: Path) -> None:
    set_manuscript_style()
    start, end, _ = zoom
    zoom_frame = frame.iloc[start:end].copy()
    x_full = frame["_x"]
    x_zoom = zoom_frame["_x"]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.25), constrained_layout=False)
    raw_color = "#4A4A4A"
    corr_color = "#2F6FAE"
    ema_colors = {
        "V_corr_raw_ema50": "#4C8C6B",
        "V_corr_raw_ema200": "#C27A2C",
        "V_corr_raw_ema800": "#6E5A9A",
    }
    legend_prop = {"family": "Times New Roman", "weight": "bold", "size": 8.5}

    axes[0].plot(x_full, frame["V_raw"], color=raw_color, linewidth=0.8, label=r"$\mathbf{V}_{t}$")
    axes[0].plot(x_full, frame["V_corr_raw"], color=corr_color, linewidth=1.05, label=r"$\mathbf{V}_{t}^{\mathbf{corr}}$")
    axes[0].axvspan(float(x_zoom.iloc[0]), float(x_zoom.iloc[-1]), color="0.85", alpha=0.35, linewidth=0)
    axes[0].text(
        0.985,
        0.94,
        f"FUDS | {format_temp(float(selected['temperature']))} | {selected['trajectory_id']}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.35, "pad": 2.8},
    )
    axes[0].legend(frameon=False, loc="lower left", ncol=2, prop=legend_prop)
    panel_label(axes[0], "(a)")

    axes[1].plot(x_zoom, zoom_frame["V_raw"], color=raw_color, linewidth=1.0, label=r"$\mathbf{V}_{t}$")
    axes[1].plot(x_zoom, zoom_frame["V_corr_raw"], color=corr_color, linewidth=1.05, label=r"$\mathbf{V}_{t}^{\mathbf{corr}}$")
    axes[1].legend(frameon=False, loc="lower left", ncol=2, prop=legend_prop)
    panel_label(axes[1], "(b)")

    axes[2].plot(x_zoom, zoom_frame["V_corr_raw"], color=corr_color, linewidth=1.0, label=r"$\mathbf{V}_{t}^{\mathbf{corr}}$")
    for col, color in ema_colors.items():
        scale = col.removeprefix("V_corr_raw_ema")
        axes[2].plot(
            x_zoom,
            zoom_frame[col],
            color=color,
            linewidth=0.95,
            label=rf"$\mathbf{{m}}^{{({scale})}}(\mathbf{{V}}_{{t}}^{{\mathbf{{corr}}}})$",
        )
    axes[2].legend(frameon=False, loc="lower left", ncol=4, prop=legend_prop)
    panel_label(axes[2], "(c)")

    for ax in axes:
        ax.set_ylabel("Voltage (V)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].set_xlabel(xlabel)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.965, hspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    Image.open(out_png).convert("RGB").save(out_png.with_suffix(".tif"), dpi=(600, 600))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot corrected-voltage and voltage EMA time-domain behavior.")
    parser.add_argument("--raw-root", action="append", default=[], help="Additional raw NMC root containing NMC_*_FUDS.csv files.")
    parser.add_argument("--prediction-root", action="append", default=[], help="Additional prediction-row search root.")
    parser.add_argument("--prediction-files", nargs="*", default=[], help="Explicit frozen G4 prediction-row files.")
    parser.add_argument("--out-png", default=OUT_PNG.as_posix())
    parser.add_argument("--out-pdf", default=OUT_PDF.as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_figure5_temperature_representative(args.prediction_root, args.prediction_files)
    raw_roots = candidate_raw_roots(args.raw_root)
    if not raw_roots:
        raise FileNotFoundError("No raw NMC roots found. Set G4_RAW_ROOTS or pass --raw-root.")

    frame, raw_root = load_selected_feature_frame(selected, raw_roots)
    frame, xlabel = attach_time_axis(frame, raw_root, str(selected["file_name"]))
    zoom = choose_zoom_interval(frame)
    plot_figure(frame, selected, zoom, xlabel, Path(args.out_png), Path(args.out_pdf))

    start, end, score = zoom
    x0 = float(frame["_x"].iloc[start])
    x1 = float(frame["_x"].iloc[end - 1])
    print("Selected trajectory metadata")
    print("- selection_rule: reused current Figure 5 temperature-wise selection; chose the 25 °C representative panel for the single-trajectory feature figure")
    print(f"- profile: FUDS")
    print(f"- temperature: {format_temp(float(selected['temperature']))}")
    print(f"- seed_from_figure5_panel: {int(selected['seed'])}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
