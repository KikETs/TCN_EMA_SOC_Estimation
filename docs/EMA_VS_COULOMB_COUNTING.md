# EMA vs Coulomb Counting

## EMA Memory

```text
EMA_t = alpha * EMA_{t-1} + (1 - alpha) * x_t
```

EMA is finite-memory, exponentially decayed, one-sided, and causal.

## Coulomb Counting

```text
Q_t = Q_{t-1} + I_t * dt
SOC_t = SOC_0 - Q_t / Q_ref
```

Coulomb counting accumulates current over time into charge and updates a SOC state.

## Difference

G4 does not use an explicit SOC state equation and does not feed cumulative Ah as an input. It does use causal current-history EMA channels, so it is not correct to say current is unused.

## Correlation Caveat

EMA current-history features can correlate with cumulative charge or SOC over fixed drive trajectories. This is why the package includes `ema_vs_cumulative_correlation.csv` and forbidden-reference diagnostics.

