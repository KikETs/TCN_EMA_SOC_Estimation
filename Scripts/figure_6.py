from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = REPO_ROOT / "Data" / "source_metrics" / "g4_seed_reproduction_summary.csv"
BY_TEMP_CSV = REPO_ROOT / "Data" / "source_metrics" / "g4_seed_reproduction_by_temp.csv"
LATEST_MAIN_FUDS_SUMMARY_CSV = REPO_ROOT / "output" / "revision_risk_hardening" / "tables" / "main_fuds_seed_summary.csv"
OUT_PNG = REPO_ROOT / "Figures" / "figure_6_fuds_soc_prediction.png"
OUT_PDF = REPO_ROOT / "Figures" / "figure_6_fuds_soc_prediction.pdf"
DEFAULT_OVERALL_TEMP_MEAN_MAE = 0.41889356670972

PREFERRED_PATTERNS = (
    "paperdef_featabl_paper_g4_all_ema_seed012_e160_seed*_sel160_*base_test_prediction_rows.csv*",
)
ZOOM_WINDOW_POINTS = 1000


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


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


def format_temp(temp: float) -> str:
    return f"{int(temp) if float(temp).is_integer() else temp:g} \u00b0C"


def search_roots(extra_roots: list[str]) -> list[Path]:
    roots: list[Path] = [
        REPO_ROOT / "Data" / "predictions" / "main_fuds",
        REPO_ROOT / "output" / "revision_risk_hardening" / "predictions" / "main_fuds",
        REPO_ROOT / "nmc_goal_vcorr_it_train_dst_selector_results",
        REPO_ROOT / "results" / "predictions",
        REPO_ROOT / "feature_ablation_runs",
    ]
    roots.extend(
        [
            REPO_ROOT.parent / "nmc_goal_vcorr_it_train_dst_selector_results",
            REPO_ROOT.parent / "remote_result_summaries",
        ]
    )
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
    for root in search_roots(extra_roots):
        for pattern in PREFERRED_PATTERNS:
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved not in seen and resolved.name not in seen_file_names:
                    found.append(resolved)
                    seen.add(resolved)
                    seen_file_names.add(resolved.name)
    return sorted(found)


def read_prediction_rows(path: Path) -> pd.DataFrame:
    required = ["y_true", "y_pred"]
    frame = pd.read_csv(path)
    missing = [col for col in required if col not in frame.columns]
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


def choose_representative(predictions: pd.DataFrame, target_mae_pct: float) -> tuple[pd.Series, pd.DataFrame]:
    work = predictions.copy()
    if "trajectory_id" not in work.columns:
        work["trajectory_id"] = work.get("file_name", pd.Series(["trajectory"] * len(work), index=work.index))
    if "file_name" not in work.columns:
        work["file_name"] = work["trajectory_id"]
    if "temperature" not in work.columns:
        work["temperature"] = np.nan

    group_cols = ["seed", "trajectory_id", "file_name", "temperature", "source_prediction_file"]
    trajectory_metrics = (
        work.groupby(group_cols, dropna=False)
        .agg(
            trajectory_MAE_pct=("abs_error_fraction", lambda s: float(s.mean() * 100.0)),
            n_points=("abs_error_fraction", "size"),
        )
        .reset_index()
    )
    trajectory_metrics["distance_to_overall_tempmean_MAE"] = (
        trajectory_metrics["trajectory_MAE_pct"] - float(target_mae_pct)
    ).abs()
    trajectory_metrics = trajectory_metrics.sort_values(
        ["distance_to_overall_tempmean_MAE", "trajectory_MAE_pct", "seed", "trajectory_id"]
    ).reset_index(drop=True)
    return trajectory_metrics.iloc[0], trajectory_metrics


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


def x_axis_for(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    for col, label in [
        ("time_s", "Time (s)"),
        ("time", "Time"),
        ("Test_Time(s)", "Time (s)"),
        ("Step_Time(s)", "Time (s)"),
        ("end_time_s", "Time (s)"),
    ]:
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.notna().any():
                return values - float(values.dropna().iloc[0]), label
    if "end_index" in frame.columns:
        return pd.to_numeric(frame["end_index"], errors="coerce"), "Window endpoint index"
    if "row_id" in frame.columns:
        return pd.to_numeric(frame["row_id"], errors="coerce"), "Sample index"
    return pd.Series(np.arange(len(frame)), index=frame.index), "Sample index"


def zoom_slice_around_max_error(abs_error_pct: pd.Series, window_points: int = ZOOM_WINDOW_POINTS) -> slice:
    values = pd.to_numeric(abs_error_pct, errors="coerce").reset_index(drop=True)
    if values.empty or values.isna().all():
        return slice(0, 0)
    max_pos = int(values.idxmax())
    n_rows = int(len(values))
    width = min(int(window_points), n_rows)
    start = max(0, max_pos - width // 2)
    end = min(n_rows, start + width)
    start = max(0, end - width)
    return slice(start, end)


def add_max_error_inset(
    ax: plt.Axes,
    x: pd.Series,
    y_true_pct: pd.Series,
    y_pred_pct: pd.Series,
    abs_error_pct: pd.Series,
) -> tuple[float, int]:
    zoom_slice = zoom_slice_around_max_error(abs_error_pct)
    if zoom_slice.stop <= zoom_slice.start:
        return float("nan"), 0

    x_zoom = pd.to_numeric(x.reset_index(drop=True).iloc[zoom_slice], errors="coerce")
    y_true_zoom = pd.to_numeric(y_true_pct.reset_index(drop=True).iloc[zoom_slice], errors="coerce")
    y_pred_zoom = pd.to_numeric(y_pred_pct.reset_index(drop=True).iloc[zoom_slice], errors="coerce")
    abs_zoom = pd.to_numeric(abs_error_pct.reset_index(drop=True).iloc[zoom_slice], errors="coerce")
    valid = x_zoom.notna() & y_true_zoom.notna() & y_pred_zoom.notna()
    if not bool(valid.any()):
        return float("nan"), 0

    x_zoom = x_zoom[valid]
    y_true_zoom = y_true_zoom[valid]
    y_pred_zoom = y_pred_zoom[valid]
    abs_zoom = abs_zoom[valid]
    y_min = float(min(y_true_zoom.min(), y_pred_zoom.min()))
    y_max = float(max(y_true_zoom.max(), y_pred_zoom.max()))
    y_pad = max((y_max - y_min) * 0.12, 0.12)
    x_min = float(x_zoom.min())
    x_max = float(x_zoom.max())
    max_error = float(abs_zoom.max())

    rect = Rectangle(
        (x_min, y_min - y_pad),
        max(x_max - x_min, 1e-9),
        (y_max - y_min) + 2.0 * y_pad,
        fill=False,
        edgecolor="0.25",
        linewidth=0.55,
        linestyle="-",
        zorder=5,
    )
    ax.add_patch(rect)

    inset = ax.inset_axes([0.085, 0.065, 0.40, 0.42])
    inset.plot(x_zoom, y_true_zoom, color="black", linewidth=0.75)
    inset.plot(x_zoom, y_pred_zoom, color="#2F6FAE", linewidth=0.72)
    inset.set_xlim(x_min, x_max)
    inset.set_ylim(y_min - y_pad, y_max + y_pad)
    inset.tick_params(axis="both", labelsize=6, length=2.0, pad=0.4)
    for label in inset.get_xticklabels() + inset.get_yticklabels():
        label.set_fontweight("bold")
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("0.25")
    inset.text(
        0.04,
        0.05,
        f"max err. {max_error:.2f}",
        transform=inset.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    return max_error, int(len(x_zoom))


def plot_representative(frame: pd.DataFrame, selected: pd.Series, target_mae_pct: float, out_png: Path, out_pdf: Path) -> None:
    set_manuscript_style()
    x, xlabel = x_axis_for(frame)
    plot_frame = frame.assign(_x=x).dropna(subset=["_x"]).sort_values("_x")
    y_true_pct = pd.to_numeric(plot_frame["y_true"], errors="coerce") * 100.0
    y_pred_pct = pd.to_numeric(plot_frame["y_pred"], errors="coerce") * 100.0
    abs_error_pct = (y_pred_pct - y_true_pct).abs()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    axes[0].plot(plot_frame["_x"], y_true_pct, color="black", linewidth=1.15, label="Ground truth")
    axes[0].plot(plot_frame["_x"], y_pred_pct, color="#2F6FAE", linewidth=1.05, label="G4 prediction")
    axes[1].plot(plot_frame["_x"], abs_error_pct, color="#8F3B32", linewidth=0.95)

    annotation = (
        f"seed {int(selected['seed'])} | {selected['trajectory_id']} | "
        f"{float(selected['temperature']):g} \u00b0C | MAE={float(selected['trajectory_MAE_pct']):.3f} %SOC"
    )
    axes[0].text(
        0.99,
        0.96,
        annotation,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.4, "pad": 3.5},
    )
    axes[0].legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        prop={"family": "Times New Roman", "weight": "bold", "size": 9},
    )
    axes[0].set_ylabel("SOC (%SOC)")
    axes[1].set_ylabel("Abs. error\n(%SOC)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylim(bottom=0.0)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    Image.open(out_png).convert("RGB").save(out_png.with_suffix(".tif"), dpi=(600, 600))
    plt.close(fig)


def selected_frame(predictions: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    mask = (
        predictions["seed"].eq(selected["seed"])
        & predictions["trajectory_id"].astype(str).eq(str(selected["trajectory_id"]))
        & predictions["file_name"].astype(str).eq(str(selected["file_name"]))
        & predictions["source_prediction_file"].eq(str(selected["source_prediction_file"]))
    )
    return predictions[mask].copy()


def plot_temperature_representatives(
    predictions: pd.DataFrame,
    selected_by_temp: pd.DataFrame,
    target_overall_mae_pct: float,
    out_png: Path,
    out_pdf: Path,
) -> None:
    set_manuscript_style()
    ncols = len(selected_by_temp)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(10.2, 5.1),
        sharey="row",
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08, "wspace": 0.18},
    )
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    bottom_max = 0.0
    inset_rows = []
    for col, selected in selected_by_temp.iterrows():
        frame = selected_frame(predictions, selected)
        x, xlabel = x_axis_for(frame)
        plot_frame = frame.assign(_x=x).dropna(subset=["_x"]).sort_values("_x")
        y_true_pct = pd.to_numeric(plot_frame["y_true"], errors="coerce") * 100.0
        y_pred_pct = pd.to_numeric(plot_frame["y_pred"], errors="coerce") * 100.0
        abs_error_pct = (y_pred_pct - y_true_pct).abs()
        bottom_max = max(bottom_max, float(abs_error_pct.max()))

        ax_soc = axes[0, col]
        ax_err = axes[1, col]
        ax_soc.plot(plot_frame["_x"], y_true_pct, color="black", linewidth=1.0, label="Ground truth")
        ax_soc.plot(plot_frame["_x"], y_pred_pct, color="#2F6FAE", linewidth=0.95, label="G4 prediction")
        ax_err.plot(plot_frame["_x"], abs_error_pct, color="#8F3B32", linewidth=0.8)
        max_error, zoom_points = add_max_error_inset(
            ax_soc,
            plot_frame["_x"],
            y_true_pct,
            y_pred_pct,
            abs_error_pct,
        )
        inset_rows.append(
            {
                "temperature_C": float(selected["temperature"]),
                "max_error_pct": max_error,
                "zoom_points": zoom_points,
            }
        )
        ax_soc.text(
            0.98,
            0.95,
            f"{format_temp(float(selected['temperature']))}\n"
            f"seed {int(selected['seed'])}, MAE={float(selected['trajectory_MAE_pct']):.3f} %SOC",
            transform=ax_soc.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.35, "pad": 2.8},
        )
        if col == 0:
            ax_soc.set_ylabel("SOC (%SOC)")
            ax_err.set_ylabel("Abs. error\n(%SOC)")
        ax_err.set_xlabel("")
        ax_soc.tick_params(labelbottom=False)
        for ax in (ax_soc, ax_err):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.52, 0.055),
        prop={"family": "Times New Roman", "weight": "bold", "size": 9},
    )
    if bottom_max > 0:
        err_ylim = np.ceil(bottom_max * 10.0) / 10.0
        for ax in axes[1, :]:
            ax.set_ylim(0.0, err_ylim)
    fig.text(
        0.535,
        0.115,
        xlabel,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        fontfamily="Times New Roman",
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.875)
    for col in range(ncols):
        pos = axes[0, col].get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.006,
            f"({chr(ord('a') + col)})",
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    Image.open(out_png).convert("RGB").save(out_png.with_suffix(".tif"), dpi=(600, 600))
    plt.close(fig)
    for row in inset_rows:
        print(
            f"Zoom inset {format_temp(row['temperature_C'])}: "
            f"max_error_pct={row['max_error_pct']:.6f}, zoom_points={row['zoom_points']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot representative frozen G4 FUDS SOC trajectory.")
    parser.add_argument("--prediction-root", action="append", default=[], help="Additional directory to search for prediction rows.")
    parser.add_argument("--prediction-files", nargs="*", default=[], help="Explicit frozen G4 prediction-row files.")
    parser.add_argument("--out-png", default=OUT_PNG.as_posix())
    parser.add_argument("--out-pdf", default=OUT_PDF.as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_mae_pct = target_overall_mae(SUMMARY_CSV)
    target_by_temp = target_mae_by_temperature(BY_TEMP_CSV, target_mae_pct)
    paths = discover_prediction_files(args.prediction_root, args.prediction_files)
    if not paths:
        raise FileNotFoundError(
            "No frozen G4 FUDS prediction-row files found. "
            "Place them under results directories, set G4_PREDICTION_ROOTS, or pass --prediction-files."
        )

    frames = [read_prediction_rows(path) for path in paths]
    predictions = pd.concat(frames, ignore_index=True)
    selected_by_temp = choose_representatives_by_temperature(predictions, target_by_temp)
    plot_temperature_representatives(predictions, selected_by_temp, target_mae_pct, Path(args.out_png), Path(args.out_pdf))

    print("Selected temperature-wise representative FUDS trajectories")
    for _, selected in selected_by_temp.iterrows():
        print(
            f"- {format_temp(float(selected['temperature']))}: seed {int(selected['seed'])}, "
            f"{selected['trajectory_id']}, trajectory_MAE_pct={float(selected['trajectory_MAE_pct']):.6f}, "
            f"target_temperature_MAE_pct={float(selected['target_temperature_MAE_pct']):.6f}, "
            f"distance={float(selected['distance_to_temperature_MAE']):.6f}, n_points={int(selected['n_points'])}"
        )
        print(f"  source_prediction_file: {Path(str(selected['source_prediction_file'])).name}")
    print()
    print("All trajectory candidates")
    trajectory_metrics = trajectory_metric_table(predictions)
    trajectory_metrics["target_temperature_MAE_pct"] = trajectory_metrics["temperature"].map(target_by_temp)
    trajectory_metrics["distance_to_temperature_MAE"] = (
        trajectory_metrics["trajectory_MAE_pct"] - trajectory_metrics["target_temperature_MAE_pct"]
    ).abs()
    print(
        trajectory_metrics[
            ["seed", "trajectory_id", "temperature", "trajectory_MAE_pct", "target_temperature_MAE_pct", "distance_to_temperature_MAE", "n_points"]
        ]
        .sort_values(["temperature", "distance_to_temperature_MAE"])
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print()
    print(f"Wrote {Path(args.out_png)}")
    print(f"Wrote {Path(args.out_pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
