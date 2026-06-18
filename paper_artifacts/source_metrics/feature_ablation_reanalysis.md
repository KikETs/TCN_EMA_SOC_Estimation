# Feature Ablation Reanalysis

| Group | Feature Set | Dim | 3-seed Temp-Mean MAE |
|---|---:|---:|---:|
| G4 | paper_g4_all_ema | 17 | 0.4189 |
| G6 | paper_g6_full23 | 23 | 0.4773 |
| G8 | paper_g8_no_voltage_ema | 17 | 0.4778 |
| G7 | paper_g7_no_current_ema | 15 | 0.9341 |
| G1 | paper_g1_derivatives | 8 | 0.9468 |
| G0 | paper_g0_raw | 3 | 1.0587 |

## Interpretation

- G4 reduces 3-seed temp-mean MAE versus G0 by 0.6398 %SOC point.
- G4 reduces 3-seed temp-mean MAE versus G1 by 0.5279 %SOC point.
- Removing current/abs-current EMA from G6 increases temp-mean MAE by 0.4568 %SOC point.
- Removing voltage EMA from G6 changes temp-mean MAE by 0.0005 %SOC point.
- G4 outperforms G6 in the 3-seed ablation while using fewer features.
- Do not claim every G4 feature is individually necessary; this ablation supports the broader causal EMA group.
- Derivative/interaction features are not consistently beneficial under this seed0 protocol if G4 outperforms G6.
