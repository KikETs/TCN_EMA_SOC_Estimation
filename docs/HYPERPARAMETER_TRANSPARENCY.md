# Hyperparameter Transparency

G4 was selected after exploratory ablation, not as a pre-declared blind model.

Frozen settings:

- feature set: `paper_g4_all_ema`
- model class: `anchor_residual_tcn`
- epoch: `160`
- optimizer: AdamW
- learning rate: `8e-4`
- weight decay: `2e-4`
- Huber beta: `0.02`
- REx penalty: `2.0`
- conditional-invariance penalty: `0.02`

The epoch sweep is a diagnostic to show whether epoch 160 is isolated or stable. It should not be used to make a new post-hoc checkpoint selection claim.

