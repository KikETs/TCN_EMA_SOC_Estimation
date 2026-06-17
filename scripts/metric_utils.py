from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array.")
    err = pred - true
    return {
        "count": int(true.size),
        "MAE_pct": float(np.mean(np.abs(err))),
        "RMSE_pct": float(np.sqrt(np.mean(err**2))),
        "MaxAE_pct": float(np.max(np.abs(err))),
        "bias_pct": float(np.mean(err)),
    }


def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    aliases = {
        "y_true": ["y_true", "target", "target_pct", "soc_true", "SOC_true", "SOC_target"],
        "y_pred": ["y_pred", "prediction", "pred_pct", "soc_pred", "SOC_pred"],
    }
    resolved: dict[str, str] = {}
    for key, candidates in aliases.items():
        for col in candidates:
            if col in df.columns:
                resolved[key] = col
                break
        if key not in resolved:
            raise ValueError(f"{path} is missing a {key} column. Tried {candidates}.")
    out = df.copy()
    out["__y_true"] = pd.to_numeric(out[resolved["y_true"]], errors="coerce")
    out["__y_pred"] = pd.to_numeric(out[resolved["y_pred"]], errors="coerce")
    out = out.dropna(subset=["__y_true", "__y_pred"]).reset_index(drop=True)
    return out


def metrics_by_optional_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    usable_groups = [c for c in group_cols if c in df.columns]
    rows: list[dict[str, object]] = []
    if usable_groups:
        iterator = df.groupby(usable_groups, dropna=False)
        for keys, group in iterator:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(usable_groups, keys)}
            row.update(compute_regression_metrics(group["__y_true"], group["__y_pred"]))
            rows.append(row)
    total = {"split": "ALL"}
    total.update(compute_regression_metrics(df["__y_true"], df["__y_pred"]))
    rows.append(total)
    return pd.DataFrame(rows)

