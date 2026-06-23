from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


def comma(values) -> str:
    return ",".join(str(v) for v in values)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the frozen G4 training/evaluation protocol using copied model code.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--config", default="configs/g4_frozen.yaml")
    p.add_argument("--seeds", default="", help="Optional override, for example 0 or 0,1,2.")
    p.add_argument("--epochs", type=int, default=0, help="Optional override. Default uses fixed_epoch from YAML.")
    p.add_argument("--loss-kind", choices=["huber", "mse"], default="", help="Optional supervised loss override.")
    p.add_argument("--save-predictions", action="store_true")
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    cfg = yaml.safe_load((base_dir / args.config).read_text(encoding="utf-8"))
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    train = cfg["training"]
    data = cfg["data"]
    model = cfg["model"]
    paths = cfg["paths"]
    features = cfg["features"]
    prefix = cfg["experiment"]["config_id"]
    epochs = int(args.epochs or train["fixed_epoch"])
    seeds = args.seeds or comma(train["seeds"])
    loss_kind = str(args.loss_kind or train.get("loss_kind", train.get("loss", "huber"))).lower()
    if loss_kind not in {"huber", "mse"}:
        loss_kind = "huber"

    cmd = [
        sys.executable,
        "-m",
        "soc_decomp.nmc_vcorr_it_train_dst_selector_run",
        "--base-dir",
        str(base_dir),
        "--raw-root",
        paths["raw_root"],
        "--output-prefix",
        prefix,
        "--seeds",
        seeds,
        "--train-profiles",
        comma(data["train_profiles"]),
        "--valid-profiles",
        comma(data.get("valid_profiles") or ["BJDST"]),
        "--test-profiles",
        comma(data["test_profiles"]),
        "--train-temperatures",
        comma(data["train_temperatures_C"]),
        "--test-temperatures",
        comma(data["test_temperatures_C"]),
        "--epochs",
        str(epochs),
        "--selector-min-epoch",
        str(epochs),
        "--selector-max-epoch",
        str(epochs),
        "--fixed-stage1-epoch",
        str(epochs),
        "--stage1-selector",
        "last_epoch",
        "--eval-every",
        str(max(1, min(20, epochs))),
        "--stage1-eval-every",
        str(max(1, min(20, epochs))),
        "--batch-size",
        str(train["batch_size"]),
        "--num-workers",
        "4",
        "--hidden-size",
        str(model["hidden_size"]),
        "--layers",
        str(model["layers"]),
        "--kernel-size",
        str(model["kernel_size"]),
        "--recurrent",
        "tcn",
        "--head-kind",
        model["head_kind"],
        "--temp-mode",
        model["temp_mode"],
        "--dropout",
        str(model["dropout"]),
        "--tcn-block-convs",
        str(model.get("tcn_block_convs", 2)),
        "--loss-kind",
        loss_kind,
        "--model-kind",
        "anchor_residual_tcn",
        "--anchor-residual-limit",
        str(model["anchor_residual_limit"]),
        "--lambda-anchor-loss",
        str(train["lambda_anchor_loss"]),
        "--lambda-rex",
        str(train["lambda_rex"]),
        "--lambda-condinv",
        str(train["lambda_condinv"]),
        "--weight-0",
        str(train["temperature_weights"]["0C"]),
        "--weight-25",
        str(train["temperature_weights"]["25C"]),
        "--weight-45",
        str(train["temperature_weights"]["45C"]),
        "--train-sampler",
        "temperature_balanced",
        "--feature-set",
        features["feature_set"],
        "--skip-stage2",
    ]
    if args.save_predictions:
        cmd.append("--save-predictions")

    subprocess.run(cmd, check=True, cwd=base_dir)

    legacy_out = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    metrics_out = base_dir / "results" / "metrics"
    metrics_out.mkdir(parents=True, exist_ok=True)
    if legacy_out.exists():
        for path in legacy_out.glob(f"{prefix}*.csv"):
            shutil.copy2(path, metrics_out / path.name)
    print(f"Copied metric CSVs to {metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
