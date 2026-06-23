# Section 2 Measurement Structure Report

## 1. Data Discovery Summary

- files found: `12`
- total samples: `127690`
- profiles detected: `BJDST, DST, FUDS, US06`
- temperatures detected: `0 °C, 25 °C, 45 °C`
- columns used: voltage, current, temperature, SOC label, profile, and time/index when available

## 2. Key Numerical Findings

- Voltage range across discovered terminal records: `2.403` to `4.152` V.
- Current range across discovered terminal records: `-4.002` to `2.142` A.
- Mean voltage autocorrelation at lag 200 samples: `0.754`; mean current autocorrelation at lag 200 samples: `-0.021`.
- Across temperature-specific local V-I bins, the largest observed SOC IQR after min-count filtering was `0.249` SOC fraction; the largest p90 bin IQR across temperatures was `0.074`.
- Main split V-I distribution overlap coefficient, averaged over temperature, was `0.497` for train profiles versus FUDS.
- Median SOC IQR changed from `0.015` in raw V-I bins to `0.004` when also stratifying by causal current and voltage-response history tertiles.

## 3. Manuscript-Ready Cautious Bullet Points

- In this dataset, terminal voltage-current operating regions overlap across dynamic drive profiles, but the density of those regions is profile dependent.
- Under profile-holdout analysis, part of the raw measurement space contains local V-I bins with non-negligible SOC spread.
- Instantaneous voltage/current/temperature endpoints provide useful information but do not always uniquely condition the SOC inverse mapping in dynamic regions.
- Voltage and current show different lag-persistence behavior, suggesting that recent measurement context can carry complementary information.
- Causal measurement-history descriptors stratify part of the raw V-I ambiguity, motivating finite-memory voltage/current context.
- These diagnostics are dataset-level measurement-structure evidence and are not model-performance claims.
- The analysis does not claim that raw measurements are useless, that current is unused, or that NoCC proves Coulomb counting unnecessary.

## 4. Figure/Table Checklist

Suggested main-text candidates:

- `tables/table_s2_dataset_terminal_summary.csv`
- `tables/table_s2_time_lag_statistics.csv`
- `tables/table_s2_raw_vi_bin_soc_spread.csv`
- `figures/fig_s2_operating_space_vi_grid_density.png`
- `figures/fig_s2_time_lag_acf_voltage_current.png`
- `figures/fig_s2_raw_vi_soc_iqr_heatmap_by_temp.png`

Suggested SI candidates:

- `tables/table_s2_causal_lag_correlations.csv`
- `tables/table_s2_raw_vi_bin_soc_spread_sensitivity.csv`
- `tables/table_s2_raw_vi_bin_details.csv`
- `tables/table_s2_profile_shift_vi_overlap.csv`
- `tables/table_s2_history_conditioned_soc_spread.csv`
- `figures/fig_s2_history_conditioned_soc_spread_reduction.png`
- `figures/fig_s2_profile_shift_vi_overlap.png`

Generated files:

- `tables/table_s2_dataset_terminal_summary.csv`
- `tables/table_s2_time_lag_statistics.csv`
- `tables/table_s2_causal_lag_correlations.csv`
- `tables/table_s2_raw_vi_bin_soc_spread.csv`
- `tables/table_s2_raw_vi_bin_soc_spread_sensitivity.csv`
- `tables/table_s2_raw_vi_bin_details.csv`
- `tables/table_s2_profile_shift_vi_overlap.csv`
- `tables/table_s2_history_conditioned_soc_spread.csv`
- `tables/table_s2_history_conditioned_soc_spread_details.csv`
- `figures/fig_s2_operating_space_vi_grid.png`
- `figures/fig_s2_operating_space_vi_grid.pdf`
- `figures/Figure_S1_operating_space_vi_grid_density.png`
- `figures/Figure_S1_operating_space_vi_grid_density.pdf`
- `figures/Figure_S1_operating_space_vi_grid_soccolor.png`
- `figures/Figure_S1_operating_space_vi_grid_soccolor.pdf`
- `figures/fig_s2_operating_space_vi_grid_soccolor.png`
- `figures/fig_s2_operating_space_vi_grid_soccolor.pdf`
- `figures/fig_s2_operating_space_vi_grid_density.png`
- `figures/fig_s2_operating_space_vi_grid_density.pdf`
- `figures/fig_s2_time_lag_acf_voltage_current.png`
- `figures/fig_s2_time_lag_acf_voltage_current.pdf`
- `figures/fig_s2_time_lag_acf_summary.png`
- `figures/fig_s2_time_lag_acf_summary.pdf`
- `figures/fig_s2_raw_vi_soc_iqr_heatmap_by_temp.png`
- `figures/fig_s2_raw_vi_soc_iqr_heatmap_by_temp.pdf`
- `figures/fig_s2_history_conditioned_soc_spread_reduction.png`
- `figures/fig_s2_history_conditioned_soc_spread_reduction.pdf`
- `figures/fig_s2_profile_shift_vi_overlap.png`
- `figures/fig_s2_profile_shift_vi_overlap.pdf`
- `tables/table_s2_validation_checks.csv`
