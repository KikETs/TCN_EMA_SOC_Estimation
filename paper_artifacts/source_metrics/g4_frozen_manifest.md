# G4 Frozen Manifest

- Status: frozen candidate selected after seed0 minimal feature ablation; not pre-declared blind model
- Model class: `anchor_residual_tcn`
- Encoder: causal TCN
- Feature set: `paper_g4_all_ema` (17 features)

## Input Features
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

## Protocol
- Train profiles/temps: ['DST', 'US06', 'BJDST'] / [0.0, 25.0, 45.0]
- Test profiles/temps: ['FUDS'] / [0.0, 25.0, 45.0]
- Fixed epoch: 160
- Window/stride: 50 / 3
- Test stride: 1
- Stage2/correction: none

## EMA Equation
- EMA_t = alpha * EMA_{t-1} + (1-alpha) * x_t, alpha=exp(-1/tau_index), EMA_0=x_0
- x_dev_ema_tau = x_t - EMA_tau(x)_t
- One-sided causal; no future samples; resets at file/profile boundary.

## NoCC Statement
- No SOC/SOC_CC/cumulative Ah/time/progress input.
- No explicit current-integration SOC state update.
- Current is used as instantaneous and finite-memory causal excitation/history, not as cumulative charge.

## Disclosure
- G4 was selected after inspecting the seed0 minimal feature ablation.
- Therefore G4 is a frozen paper-candidate requiring confirmation, not a pre-declared blind model.
