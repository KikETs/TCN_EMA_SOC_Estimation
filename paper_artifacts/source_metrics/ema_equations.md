# EMA Equations

For a scalar causal input `x_t`, the index-based EMA features are:

`EMA_tau(x)_0 = x_0`

`EMA_tau(x)_t = alpha_tau * EMA_tau(x)_{t-1} + (1 - alpha_tau) * x_t`

`alpha_tau = exp(-1 / tau)`

Deviation features are `x_dev_ema_tau = x_t - EMA_tau(x)_t`.

The update is one-sided causal, uses no future samples, uses no centered rolling window, and resets at every file/profile boundary.
