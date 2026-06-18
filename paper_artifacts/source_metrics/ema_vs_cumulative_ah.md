# EMA Versus Cumulative Ah

`EMA_t(I) = alpha * EMA_{t-1}(I) + (1-alpha) * I_t`

`Cumulative_Ah_t = sum_{k=0}^{t} I_k * dt / 3600`

EMA is an exponentially decayed finite-memory history variable. Cumulative Ah is a running integral of current throughput.

EMA does not implement an explicit SOC state update and is not monotone in current throughput. However, it does provide local current-history information, so the safe wording is that current is used as instantaneous and finite-memory causal excitation, not that current history is absent.
