# Feature Construction Audit

- SOC input, window-start SOC, SOC_CC input, cumulative Ah input, and absolute trajectory progress are excluded from G0/G1/G4/G6/G7/G8 feature schemas.
- Current is used as instantaneous or finite-memory causal excitation (`I_raw`, `dI`, `absI`, `I_raw_ema*`, `absI_ema*`), not as an explicit SOC state update.
- EMA channels use one-sided recurrence and reset at record boundaries. They do not use future samples.
- Normalization is fitted on the training-side profiles only for each split.
- Corrected-voltage features use the repository's shared feature-construction path; raw CALCE files are not included in this repository package.
