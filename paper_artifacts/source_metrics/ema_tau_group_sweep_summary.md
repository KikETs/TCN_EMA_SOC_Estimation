# EMA Tau/Group Sweep Summary

- Best 25C group in seed0 sweep: T7 (raw + current/abs-current EMA 50/200) MAE 0.3510.
- 25C raw T0 MAE: 1.9431.
- 25C voltage-only T6 MAE: 2.1183.
- 25C current/abs-current-only T7 MAE: 0.3510.
- 25C all-EMA T8/G4 MAE: 0.4607.
- Interpretation: current/abs-current EMA drives the 25C gain much more than voltage EMA alone.
- Current tau=200 alone is useful; tau=50 alone is insufficient under this seed0 protocol.
- Voltage EMA is still important at inference in perturbation tests, so do not say voltage EMA is irrelevant.
