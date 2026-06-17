from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate G4 by recomputing metrics from saved prediction CSV files.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--predictions", nargs="*", default=[])
    args = p.parse_args()
    base_dir = Path(args.base_dir).resolve()
    pred_paths = [Path(p) for p in args.predictions]
    if not pred_paths:
        pred_paths = sorted((base_dir / "results" / "predictions").glob("*.csv*"))
    if not pred_paths:
        raise FileNotFoundError("No prediction CSV files found. Re-run training with --save-predictions or pass --predictions.")
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "recompute_metrics.py"),
        "--out",
        str(base_dir / "results" / "metrics" / "evaluation_recomputed_metrics.csv"),
        "--predictions",
        *[p.as_posix() for p in pred_paths],
    ]
    subprocess.run(cmd, check=True, cwd=base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

