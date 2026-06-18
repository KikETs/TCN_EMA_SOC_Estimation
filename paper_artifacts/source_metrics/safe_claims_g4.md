# Safe Claims For G4

- G4 is a frozen candidate selected after seed0 feature ablation and then checked with seeds/profile/epoch/model-class diagnostics.
- Causal current/abs-current EMA features are important for the 25C gain under this protocol.
- Voltage EMA features are also used by the trained model, as shown by perturbation sensitivity.
- Current is used as instantaneous and finite-memory causal excitation/history, not as an explicit integrated SOC state.
- Profile rotation shows profile/observability-dependent difficulty, especially at 25C.
- G4 supports an EMA-memory analysis paper, with limitations disclosed.
