from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_ema_causality import causal_index_ema


def test_streaming_ema_prefix_is_not_changed_by_future_samples() -> None:
    x = np.linspace(-1.0, 1.0, 128)
    y0 = causal_index_ema(x, tau=50)
    x_future_changed = x.copy()
    x_future_changed[80:] += 500.0
    y1 = causal_index_ema(x_future_changed, tau=50)
    assert np.allclose(y0[:80], y1[:80])


def test_streaming_ema_recurrence_matches_manual_step() -> None:
    x = np.array([1.0, 3.0, 5.0])
    tau = 10.0
    alpha = np.exp(-1.0 / tau)
    y = causal_index_ema(x, tau=tau)
    assert y[0] == x[0]
    assert np.isclose(y[1], alpha * y[0] + (1.0 - alpha) * x[1])
    assert np.isclose(y[2], alpha * y[1] + (1.0 - alpha) * x[2])

