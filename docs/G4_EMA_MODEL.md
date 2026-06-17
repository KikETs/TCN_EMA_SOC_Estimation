# G4 EMA Model

## Architecture

G4 uses a causal TCN encoder with an anchor-residual head.

- hidden size: `128`
- layers: `6`
- kernel size: `5`
- channel normalization
- dropout: `0.06`
- temperature mode: soft MoE-style temperature head in the copied model code
- output: endpoint SOC estimate for each stateless window

The hidden state does not persist between windows.

## Feature Set

The frozen feature set is `paper_g4_all_ema` with 17 inputs:

- raw corrected voltage/current/temperature
- causal voltage EMA and deviation at 50, 200, and 800 index scales
- causal current EMA/deviation at 50 and 200 index scales
- causal absolute-current EMA/deviation at 50 and 200 index scales

## EMA Equation

For an input stream `x_t`:

```text
EMA_t = alpha * EMA_{t-1} + (1 - alpha) * x_t
alpha = exp(-1 / tau_index)
EMA_0 = x_0
x_dev_ema_tau = x_t - EMA_t
```

This is one-sided and causal.

## Training Split

Training uses `DST + US06 + BJDST`; testing uses `FUDS`.

## Fixed Epoch

The frozen candidate uses epoch `160`. The epoch sweep is diagnostic only and should not be described as a new selection on the final test set.

## Profile Rotation Limitation

The package includes one profile-rotation diagnostic to reduce the appearance of FUDS-only tuning, but it is still limited and not a full blind generalization benchmark.

