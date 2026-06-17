# G4 Features

## Frozen 17-Feature Set

1. `V_corr_raw`
2. `I_raw`
3. `T`
4. `V_corr_raw_ema50`
5. `V_corr_raw_dev_ema50`
6. `V_corr_raw_ema200`
7. `V_corr_raw_dev_ema200`
8. `V_corr_raw_ema800`
9. `V_corr_raw_dev_ema800`
10. `I_raw_ema50`
11. `I_raw_dev_ema50`
12. `I_raw_ema200`
13. `I_raw_dev_ema200`
14. `absI_ema50`
15. `absI_dev_ema50`
16. `absI_ema200`
17. `absI_dev_ema200`

## Groups

- G0: `V_corr_raw`, `I_raw`, `T`
- voltage EMA group: `V_corr_raw_ema*`, `V_corr_raw_dev_ema*`
- current EMA group: `I_raw_ema*`, `I_raw_dev_ema*`
- absolute-current EMA group: `absI_ema*`, `absI_dev_ema*`

## Dependency Table

| feature group | source | causal | forbidden SOC/Ah/progress input |
|---|---|---:|---:|
| G0 voltage/current/temperature | measured and corrected sensor streams | yes | no |
| voltage EMA | one-sided recurrence on `V_corr_raw` | yes | no |
| current EMA | one-sided recurrence on `I_raw` | yes | no |
| absolute-current EMA | one-sided recurrence on `abs(I_raw)` | yes | no |

No future samples, SOC inputs, cumulative Ah, or trajectory progress inputs are selected.

