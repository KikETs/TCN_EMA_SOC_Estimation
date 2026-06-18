# G4 Model-Class Baselines Summary

- Best seed0 model-class control: tcn_h128_l6_moe_g4 temp-mean MAE 0.4418.
- Endpoint and window-summary MLP controls underperform, so G4 features alone are not sufficient without temporal modeling.
- LSTM/GRU h128 layer=1 also underperform the TCN controls under this fixed protocol.
- TCN without temp-MoE is competitive but weaker than the MoE G4 run, so temperature-conditioned heads help but are not the entire story.
- The paper should emphasize causal EMA memory plus a causal TCN observer, not TCN novelty alone.
