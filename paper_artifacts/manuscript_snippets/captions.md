# Paper Artifact Captions

- `table1_main_g4_fuds_results`: generated as CSV, Markdown, and TeX.
- `table2_feature_ablation`: generated as CSV, Markdown, and TeX.
- `table3_ema_group_sweep`: generated as CSV, Markdown, and TeX.
- `table4_profile_rotation`: generated as CSV, Markdown, and TeX.
- `table5_epoch_sweep`: generated as CSV, Markdown, and TeX.
- `table6_model_class_baselines`: generated as CSV, Markdown, and TeX.
- `appendix_forbidden_reference`: generated as CSV, Markdown, and TeX.

- `fig1_main_g4_fuds_mae_by_temp`: Frozen G4 FUDS MAE across 0C, 25C, and 45C, averaged over completed seeds.
- `fig2_feature_ablation_by_temp`: Minimal feature ablation showing the contribution of derivative and EMA groups.
- `fig3_ema_group_sweep_by_temp`: EMA group sweep used to interpret which causal memory groups are useful.
- `fig4_profile_rotation_by_temp`: Profile-rotation diagnostic for FUDS and DST holdouts.
- `fig5_epoch_sweep`: Diagnostic checkpoint sweep; the paper candidate remains the frozen epoch-160 setting.
- `fig5_representative_fuds_soc_trajectory`: Temperature-wise representative FUDS profile-holdout SOC estimation trajectories of the frozen G4 CEMA-TCN model. For each temperature, the displayed seed trajectory is selected as the FUDS test trajectory whose trajectory-level MAE is closest to the corresponding three-seed temperature-mean MAE. The top row compares predicted and ground-truth SOC, and the bottom row shows the corresponding absolute error.
- `fig6_ema_perturbation_importance`: Inference-only perturbation diagnostic for G4 EMA channels.
- `fig7_model_class_baselines`: Model-class baselines compared under the same G4 feature protocol.
- `fig7_corrected_voltage_behavior`: Time-domain behavior of corrected voltage and voltage EMA features on a representative held-out FUDS trajectory. (a) Raw terminal voltage and corrected voltage over the full trajectory. (b) Zoomed view of the same segment, showing attenuation of fast load-dependent variation in the corrected voltage. (c) Corrected voltage together with representative voltage EMA channels (`ema50`, `ema200`, and `ema800`), illustrating progressively slower finite-memory coordinates.
- `fig8_region_error_reduction`: Regional error reduction from G0 to G4 across SOC bands, recent absolute-current-history regions, voltage-response-deviation regions, and local V-I ambiguity groups. Negative ΔMAE indicates that the causal EMA representation reduces error relative to the raw corrected-voltage/current/temperature input.
- `figS8_corrected_voltage_profile_temperature_grid`: Raw and corrected voltage trajectories across profile-temperature conditions. Each panel compares raw terminal voltage and corrected voltage for one profile-temperature record. Across profiles and temperatures, corrected voltage exhibits a smoother response than the raw terminal voltage, providing a more stable voltage-response coordinate for causal SOC estimation.
- `fig_appendix_ema_correlation`: Appendix diagnostic showing EMA correlation caveats with forbidden references.
