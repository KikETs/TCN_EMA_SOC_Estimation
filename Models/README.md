# Models

Included model classes:

- `cema_tcn`: proposed one-sublayer CEMA-TCN
- `tcn`: same TCN implementation for baseline use
- `lstm`: 1-layer LSTM baseline
- `gru`: 1-layer GRU baseline
- `transformer`: 2-layer Transformer encoder, `d_model=128`, `nhead=8`, GELU
- `mlp`: endpoint MLP baseline

The proposed TCN block is:

```text
causal Conv1d -> LayerNorm -> SiLU -> Dropout -> residual add
```

The discarded two-sublayer TCN variant is not included.

Feature ablations are defined in `feature_sets.py`:

- `G0`: corrected voltage, current, temperature
- `G1`: G0 + local derivatives/excitation
- `G4`: G0 + voltage/current/absolute-current EMA memory
- `G6`: G4 + derivative/excitation terms
- `G7`: G6 without current/absolute-current EMA
- `G8`: G6 without voltage EMA

Training example:

```bash
python Models/train.py --model cema_tcn --feature-set G4 --test-profile FUDS
```
