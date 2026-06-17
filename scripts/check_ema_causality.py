from __future__ import annotations

import argparse
import numpy as np


def causal_index_ema(values: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x.copy()
    alpha = float(np.exp(-1.0 / max(float(tau), 1e-12)))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def prefix_invariance_check() -> bool:
    x = np.linspace(0.0, 1.0, 64)
    y0 = causal_index_ema(x, 50)
    x_changed_future = x.copy()
    x_changed_future[40:] += 1000.0
    y1 = causal_index_ema(x_changed_future, 50)
    return bool(np.allclose(y0[:40], y1[:40]))


def main() -> int:
    argparse.ArgumentParser(description="Check one-sided EMA prefix invariance.").parse_args()
    if not prefix_invariance_check():
        print("EMA_CAUSALITY_FAIL")
        return 1
    print("EMA causality check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

