from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


FEATURE_SETS = {
    "G0": "paper_g0_raw",
    "G1": "paper_g1_derivatives",
    "G6": "paper_g6_full23",
    "G7": "paper_g7_no_current_ema",
    "G8": "paper_g8_no_voltage_ema",
}


def comma(values: list[str]) -> str:
    return ",".join(str(v) for v in values)


def seed_token(seeds: str) -> str:
    cleaned = "".join(part.strip() for part in seeds.split(",") if part.strip())
    return f"seed{cleaned}"


def parse_feature_sets(raw: str) -> list[tuple[str, str]]:
    wanted = [part.strip() for part in raw.split(",") if part.strip()]
    out: list[tuple[str, str]] = []
    for item in wanted:
        if item in FEATURE_SETS:
            out.append((item, FEATURE_SETS[item]))
        elif item in FEATURE_SETS.values():
            label = next(k for k, v in FEATURE_SETS.items() if v == item)
            out.append((label, item))
        else:
            raise ValueError(f"Unknown feature set {item!r}. Use one of {sorted(FEATURE_SETS)}.")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Run missing seeds for the frozen feature-set ablation protocol.")
    p.add_argument("--repo-dir", default=".")
    p.add_argument("--run-base-dir", default="feature_ablation_runs")
    p.add_argument("--raw-root", default="data/raw/NMC_SAMSUNG_INR_18650_2Ah")
    p.add_argument("--feature-sets", default="G0,G1,G6,G7,G8")
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--eval-every", type=int, default=160)
    p.add_argument("--stage1-eval-every", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-predictions-for", default="G0")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_base = Path(args.run_base_dir)
    if not run_base.is_absolute():
        run_base = (repo_dir / run_base).resolve()
    run_base.mkdir(parents=True, exist_ok=True)

    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = (repo_dir / raw_root).resolve()

    save_predictions_for = {part.strip() for part in args.save_predictions_for.split(",") if part.strip()}
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_dir.as_posix() + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    for label, feature_set in parse_feature_sets(args.feature_sets):
        prefix = f"paperdef_featabl_{feature_set}_{seed_token(args.seeds)}_e{int(args.epochs)}"
        expected = run_base / "nmc_goal_vcorr_it_train_dst_selector_results" / f"{prefix}_by_temperature.csv"
        if args.skip_existing and expected.exists():
            print(f"[skip] {label} {feature_set}: {expected}")
            continue
        cmd = [
            sys.executable,
            "-m",
            "soc_decomp.nmc_vcorr_it_train_dst_selector_run",
            "--base-dir",
            run_base.as_posix(),
            "--raw-root",
            raw_root.as_posix(),
            "--output-prefix",
            prefix,
            "--seeds",
            args.seeds,
            "--train-profiles",
            comma(["DST", "US06", "BJDST"]),
            "--valid-profiles",
            "BJDST",
            "--test-profiles",
            "FUDS",
            "--train-temperatures",
            comma(["0", "25", "45"]),
            "--test-temperatures",
            comma(["0", "25", "45"]),
            "--epochs",
            str(int(args.epochs)),
            "--selector-min-epoch",
            str(int(args.epochs)),
            "--selector-max-epoch",
            str(int(args.epochs)),
            "--fixed-stage1-epoch",
            str(int(args.epochs)),
            "--stage1-selector",
            "last_epoch",
            "--eval-every",
            str(max(1, int(args.eval_every))),
            "--stage1-eval-every",
            str(max(1, int(args.stage1_eval_every))),
            "--batch-size",
            str(int(args.batch_size)),
            "--num-workers",
            str(int(args.num_workers)),
            "--hidden-size",
            "128",
            "--layers",
            "6",
            "--kernel-size",
            "5",
            "--recurrent",
            "tcn",
            "--head-kind",
            "linear",
            "--temp-mode",
            "moe",
            "--dropout",
            "0.06",
            "--model-kind",
            "anchor_residual_tcn",
            "--anchor-residual-limit",
            "0.12",
            "--lambda-anchor-loss",
            "0.1",
            "--lambda-rex",
            "2.0",
            "--lambda-condinv",
            "0.02",
            "--weight-0",
            "0.8",
            "--weight-25",
            "2.2",
            "--weight-45",
            "1.0",
            "--train-sampler",
            "temperature_balanced",
            "--feature-set",
            feature_set,
            "--skip-stage2",
        ]
        if label in save_predictions_for or feature_set in save_predictions_for:
            cmd.append("--save-predictions")
        print("[run] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=repo_dir, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
