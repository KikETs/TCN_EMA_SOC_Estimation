# G4 Profile Rotation Interpretation

| Rotation | Test profile | mean temp-mean MAE | worst temp MAE |
|---|---:|---:|---:|
| R1 | FUDS | 0.4189 | 0.4975 |
| R2 | DST | 0.5046 | 0.9831 |
| R3 | US06 | 0.6661 | 1.4337 |
| R4 | BJDST | 0.4718 | 0.6265 |

- G4 is not uniformly profile-robust: R3/US06 holdout has a large 25C error.
- R4/BJDST holdout is strong, and R2/DST is mostly acceptable except 25C sensitivity.
- The defensible claim is profile/observability-dependent robustness, not that G4 solves all profile extrapolation.
- The 25C slice remains sensitive across rotations, so do not claim 25C is intrinsically physically harder.
