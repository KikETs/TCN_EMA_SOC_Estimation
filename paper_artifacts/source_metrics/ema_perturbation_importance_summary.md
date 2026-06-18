# EMA Perturbation Importance Summary

- Largest overall degradation: P4_shuffle_voltage_ema delta MAE 19.3580.
- Largest 25C degradation: P4_shuffle_voltage_ema delta MAE 18.1777.
- Shuffling voltage EMA is extremely damaging, indicating inference-time dependence on aligned voltage-memory channels.
- Zeroing current/abs-current deviation channels alone does not hurt in this perturbation, so current EMA importance is better supported by group ablation than by deviation-zeroing.
- Do not overstate a single perturbation; report group ablation and perturbation as complementary diagnostics.
