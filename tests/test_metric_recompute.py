from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metric_utils import compute_regression_metrics


def test_metric_recompute_mae_rmse_maxae() -> None:
    y_true = [0.0, 1.0, 2.0, 3.0]
    y_pred = [0.0, 2.0, 1.0, 5.0]
    m = compute_regression_metrics(y_true, y_pred)
    err = np.array(y_pred) - np.array(y_true)
    assert m["count"] == 4
    assert np.isclose(m["MAE_pct"], np.mean(np.abs(err)))
    assert np.isclose(m["RMSE_pct"], np.sqrt(np.mean(err**2)))
    assert np.isclose(m["MaxAE_pct"], np.max(np.abs(err)))
    assert np.isclose(m["bias_pct"], np.mean(err))

