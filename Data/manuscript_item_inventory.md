# Manuscript Figure/Table Inventory

This inventory was built from the provided DOCX files:

- Main manuscript: `manuscript_energy_cited_eqnum_35_no_holdout.docx`
- Supplementary information: `Supplementary_Information_CEMA_TCN_draft.docx`

## Main Manuscript

| Item | Output script | Output stem |
|---|---|---|
| Table 1. Dataset and terminal-measurement summary | `Scripts/table_1.py` | `Tables/table_1_dataset_summary` |
| Figure 1. Current, voltage, and reference SOC trajectories | `Scripts/figure_1.py` | `Figures/figure_1_current_voltage_soc_by_profile` |
| Table 2. Voltage and current autocorrelation | `Scripts/table_2.py` | `Tables/table_2_voltage_current_autocorrelation` |
| Table 3. Causal lag correlations | `Scripts/table_3.py` | `Tables/table_3_causal_lag_correlations` |
| Figure 2. Conditional SOC spread within local voltage-current bins | `Scripts/figure_2.py` | `Figures/figure_2_vi_bin_soc_spread` |
| Table 4. SOC IQR within occupied voltage-current bins | `Scripts/table_4.py` | `Tables/table_4_vi_bin_soc_iqr` |
| Figure 3. Frequency-domain structure of raw terminal measurements and reference SOC | `Scripts/figure_3.py` | `Figures/figure_3_raw_signal_frequency_structure` |
| Figure 4. Workflow for CEMA-TCN SOC estimation | `Scripts/figure_4.py` | `Figures/figure_4_cema_tcn_workflow` |
| Table 5. Causal EMA measurement representation | `Scripts/table_5.py` | `Tables/table_5_causal_ema_measurement_representation` |
| Figure 5. CEMA-TCN endpoint estimator | `Scripts/figure_5.py` | `Figures/figure_5_cema_tcn_endpoint_estimator` |
| Table 6. FUDS test performance of CEMA-TCN estimator | `Scripts/table_6.py` | `Tables/table_6_main_fuds_performance` |
| Figure 6. SOC estimation performance on FUDS test profile | `Scripts/figure_6.py` | `Figures/figure_6_fuds_soc_prediction` |
| Table 7. Model comparison under the FUDS test protocol | `Scripts/table_7.py` | `Tables/table_7_model_comparison` |
| Table 8. Feature-set ablation under FUDS test protocol | `Scripts/table_8.py` | `Tables/table_8_feature_set_ablation` |
| Figure 7. Feature ablation across FUDS test temperature slices | `Scripts/figure_7.py` | `Figures/figure_7_feature_ablation` |
| Figure 8. Time-domain behavior of corrected voltage and voltage EMA features | `Scripts/figure_8.py` | `Figures/figure_8_corrected_voltage_behavior` |
| Table 9. Spectral energy distribution | `Scripts/table_9.py` | `Tables/table_9_spectral_energy_distribution` |
| Figure 9. Spectral behavior of corrected voltage and EMA memory channels | `Scripts/figure_9.py` | `Figures/figure_9_feature_frequency_behavior` |
| Table 10. Error reduction by SOC/current-history/voltage-response regions | `Scripts/table_10.py` | `Tables/table_10_region_error_reduction` |
| Figure 10. Regional error reduction from EMA memory | `Scripts/figure_10.py` | `Figures/figure_10_region_error_reduction` |

## Supplementary Information

| Item | Output script | Output stem |
|---|---|---|
| Table S1. Record-level sample counts and window counts | `Scripts/table_S1.py` | `Tables/table_S1_record_window_counts` |
| Table S2. Voltage-current support coverage audit | `Scripts/table_S2.py` | `Tables/table_S2_vi_support_coverage` |
| Figure S1. Additional profile-temperature voltage-current support maps | `Scripts/figure_S1.py` | `Figures/figure_S1_vi_support_maps` |
| Table S3. Local ambiguity after recent-history stratification | `Scripts/table_S3.py` | `Tables/table_S3_history_conditioned_ambiguity` |
| Table S4. Channel-level input schema | `Scripts/table_S4.py` | `Tables/table_S4_feature_schema` |
| Figure S2. Representative CEMA channels | `Scripts/figure_S2.py` | `Figures/figure_S2_representative_cema_channels` |
| Table S5. Causal feature-construction and preprocessing audit | `Scripts/table_S5.py` | `Tables/table_S5_feature_construction_audit` |
| Table S6. Fixed training configuration | `Scripts/table_S6.py` | `Tables/table_S6_training_config` |
| Table S7. Baseline model settings | `Scripts/table_S7.py` | `Tables/table_S7_baseline_model_settings` |
| Table S8. Seed-level feature-ablation results | `Scripts/table_S8.py` | `Tables/table_S8_feature_ablation_by_seed` |
| Table S9. Regional grouping criteria | `Scripts/table_S9.py` | `Tables/table_S9_regional_grouping_criteria` |
| Table S10. Spectral energy distribution of representative measurement and EMA channels | `Scripts/table_S10.py` | `Tables/table_S10_spectral_energy_by_record` |
| Table S11. Regional MAE reduction from G0 to G4 | `Scripts/table_S11.py` | `Tables/table_S11_region_error_reduction` |
| Figure S3. Profile-temperature spectral summaries | `Scripts/figure_S3.py` | `Figures/figure_S3_profile_temperature_spectral_summary` |

## Known Provenance Note

Table 6 is generated from `Data/manuscript_tables/table_6_main_fuds_performance_locked.csv`, which stores the values visible in the supplied DOCX. The local result tree did not contain a single upstream CSV matching both the MAE values and the RMSE/MaxAE values in that DOCX table.
