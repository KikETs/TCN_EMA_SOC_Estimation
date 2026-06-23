# Hyperparameter Selection Audit

This audit records the queued protocol. It does not prove historical manuscript-development choices.

- FUDS is assigned only as the held-out test profile in the main FUDS protocol.
- The queued fixed checkpoint rule is `last_epoch` with epoch 160.
- Validation profile for the main FUDS protocol is BJDST, which is part of the training-side profile set.
- Feature-set definitions are code-defined in `soc_decomp/nmc_vit_feature_lstm_experiment.py`.
- EMA scales are fixed before this queued run; this script does not retune them on FUDS.
