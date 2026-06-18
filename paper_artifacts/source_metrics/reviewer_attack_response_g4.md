# Reviewer Attack Response For G4

| Attack | Response | Evidence |
|---|---|---|
| G4 was tuned to FUDS. | Disclose seed0 selection; then show frozen seeds and rotations. | g4_seed_reproduction_summary.csv; g4_profile_rotation_summary.csv |
| EMA is hidden Coulomb counting. | EMA is finite-memory causal recurrence, but it can correlate with cumulative charge; disclose correlations and forbidden baselines. | ema_vs_cumulative_correlation.csv; forbidden_reference_interpretation.md |
| Current drives the model. | Yes, as causal excitation/history. It is not integrated into an SOC state and no cumulative Ah/SOC_CC input is used. | g4_frozen_manifest.md; ema_tau_group_sweep.csv |
| 25C result is cherry-picked. | Rotations show 25C sensitivity is profile-dependent; report both good and bad rotations. | g4_profile_rotation_by_seed_temp.csv |
| TCN novelty is unclear. | Controls show MLP/LSTM/GRU are weaker; still frame novelty around EMA-memory analysis, not architecture alone. | g4_model_class_baselines.csv |
| e160 is lucky. | Diagnostic sweep shows e160 is usable but not uniquely best; keep frozen e160 and disclose checkpoint sensitivity. | g4_epoch_sweep.csv |
