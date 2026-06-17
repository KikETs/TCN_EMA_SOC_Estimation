from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metric_utils import metrics_by_optional_groups, read_prediction_csv


def main() -> int:
    p = argparse.ArgumentParser(description="Recompute MAE/RMSE/MaxAE from local prediction CSV files.")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--out", default="results/metrics/recomputed_metrics.csv")
    p.add_argument("--group-cols", default="temperature_C,temperature,seed,profile,drive_cycle")
    args = p.parse_args()

    rows = []
    for pred_path in args.predictions:
        path = Path(pred_path)
        df = read_prediction_csv(path)
        group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
        metrics = metrics_by_optional_groups(df, group_cols)
        metrics.insert(0, "prediction_file", path.as_posix())
        rows.append(metrics)
    out_df = pd.concat(rows, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

