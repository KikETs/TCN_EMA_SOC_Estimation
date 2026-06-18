# Forbidden Reference Interpretation

- These baselines are explicitly forbidden and are included only as leakage/reference diagnostics.
- They must not be reported as valid NoCC baselines.
- Maximum absolute EMA-vs-cumulative-Ah correlation observed: 0.9881.
- EMA may correlate with cumulative charge history, especially through current-history channels.
- The valid distinction is implementation-level: G4 uses finite-memory causal EMA recurrences, not SOC_CC/cumulative Ah inputs or an explicit SOC state update.
- Do not claim EMA is completely independent of charge history.
