# Frequency Structure Analysis Report

## 1. Data and Feature Discovery

- Data root: `/home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC/nmc_soc_ocvstart_relabelled_from_lc_ocv/data`
- Raw records included: 12 / 12
- Corrected-voltage source: V_corr_raw = causal_time_ema(Voltage(V) - Current(A) * R0_temperature, tau=120 s), with R0 estimated from BJDST/DST/US06 events only.

## 2. PSD Method and Band Definitions

- Welch PSD was used when scipy was available; otherwise the script falls back to a deterministic windowed periodogram.
- PSDs are computed within record boundaries and normalized by total nonzero-frequency power.
- Primary bands are sample based: low f < 1/200 cycles/sample, mid 1/200 <= f < 1/50 cycles/sample, high f >= 1/50 cycles/sample.

## 3. Section 2.4 Raw-Signal Findings

| signal_name   |   mean_low_frequency_energy_percent | range_low_frequency_energy_percent   |   mean_mid_frequency_energy_percent | range_mid_frequency_energy_percent   |   mean_high_frequency_energy_percent | range_high_frequency_energy_percent   |   mean_median_frequency_cycles_per_sample | range_median_frequency_cycles_per_sample   |
|:--------------|------------------------------------:|:-------------------------------------|------------------------------------:|:-------------------------------------|-------------------------------------:|:--------------------------------------|------------------------------------------:|:-------------------------------------------|
| Current       |                               14.88 | 5.39-28.34                           |                               30.81 | 15.50-42.92                          |                                54.31 | 29.42-79.08                           |                                  0.024495 | 0.009766-0.041992                          |
| Voltage       |                               20.78 | 8.97-37.39                           |                               30.97 | 16.16-41.69                          |                                48.26 | 21.54-74.05                           |                                  0.019206 | 0.005859-0.035156                          |
| Reference SOC |                               98.85 | 97.77-99.61                          |                                1.04 | 0.28-2.11                            |                                 0.11 | 0.03-0.17                             |                                  0.000977 | 0.000977-0.000977                          |

Current contains a larger high-frequency contribution than reference SOC. Reference SOC is dominated by low-frequency content because it is obtained by current integration. Voltage contains slow discharge-related variation together with faster load-induced transient response.

## 4. Section 4.x Corrected-Voltage and EMA Feature Findings

| feature_group         | representative_feature_column   |   mean_low_frequency_energy_percent |   mean_mid_frequency_energy_percent |   mean_high_frequency_energy_percent |   mean_median_frequency_cycles_per_sample |   high_frequency_energy_reduction_vs_reference_percent |
|:----------------------|:--------------------------------|------------------------------------:|------------------------------------:|-------------------------------------:|------------------------------------------:|-------------------------------------------------------:|
| Raw voltage           | V_raw                           |                               20.78 |                               30.97 |                                48.26 |                                  0.019206 |                                                   0    |
| Corrected voltage     | V_corr_raw                      |                               97.98 |                                1.94 |                                 0.08 |                                  0.000977 |                                                  99.84 |
| Short voltage EMA     | V_corr_raw_ema50                |                               99.56 |                                0.44 |                                 0    |                                  0.000977 |                                                  98.15 |
| Long voltage EMA      | V_corr_raw_ema800               |                               99.99 |                                0.01 |                                 0    |                                  0.000977 |                                                  99.94 |
| Raw current           | I_raw                           |                               14.88 |                               30.81 |                                54.31 |                                  0.024495 |                                                   0    |
| Short current EMA     | I_raw_ema50                     |                               66.37 |                               28.87 |                                 4.76 |                                  0.003662 |                                                  92.21 |
| Long current EMA      | I_raw_ema200                    |                               82.55 |                               15.46 |                                 1.98 |                                  0.002197 |                                                  96.68 |
| Short abs-current EMA | absI_ema50                      |                               72.49 |                               22.6  |                                 4.91 |                                  0.00293  |                                                  92.7  |
| Long abs-current EMA  | absI_ema200                     |                               86.97 |                               11.04 |                                 1.99 |                                  0.001953 |                                                  97.11 |

Corrected voltage attenuates part of the fast load-dependent variation in the terminal-voltage input. EMA channels shift voltage/current measurements toward lower-frequency finite-memory context, with longer EMA spans retaining slower components.

## 5. Suggested Main-Text Figures and Tables

- Section 2.4 Figure 3: `figures/Figure_3_raw_signal_frequency_structure.png`
- Section 2.4 Table 5: `tables/Table_5_raw_signal_frequency_summary_compact.csv`
- Section 4.x Figure 6: `figures/Figure_6_feature_frequency_behavior.png`
- Section 4.x Table 7: `tables/Table_7_feature_frequency_summary_compact.csv`

## 6. Suggested SI Figures and Tables

- Table S6: `tables/Table_S6_raw_signal_frequency_summary_by_record.csv`
- Table S7: `tables/Table_S7_feature_frequency_summary_by_record.csv`
- Figure S6: `figures/Figure_S6_raw_signal_frequency_by_profile_temperature.png` and `figures/Figure_S6_raw_signal_cumulative_energy.png`
- Figure S7: `figures/Figure_S7_feature_psd_by_group.png` and `figures/Figure_S7_feature_frequency_by_profile_temperature.png`

## 7. Interpretation Boundaries

- Do not describe high-frequency components as noise.
- Do not claim that EMA channels are electrochemical state variables.
- Do not claim that corrected voltage removes all current effects.
- Keep interpretation limited to measurement-structure and feature signal-processing behavior.

## Caption Drafts

**Figure 3.** Frequency-domain structure of raw voltage, current, and reference SOC trajectories. Spectra were computed within each profile-temperature record and summarized by signal type. Current contains stronger high-frequency excitation components, whereas the Coulomb-counted reference SOC trajectory is dominated by low-frequency variation. Voltage contains both slow discharge-related variation and faster load-induced transient response.

**Table 5.** Compact spectral energy summary of raw terminal measurements and reference SOC. Energy fractions were computed from normalized Welch spectra within each profile-temperature record and averaged across the twelve records.

**Figure 6.** Spectral behavior of corrected voltage and EMA memory features. Corrected voltage attenuates part of the fast load-dependent variation in the voltage input, and EMA channels shift voltage/current measurements toward lower-frequency finite-memory context.

**Table S6.** Raw-signal spectral energy summary by profile and temperature. Low-, mid-, and high-frequency energy fractions were computed from raw voltage, current, and reference SOC trajectories within each record boundary.

**Table S7.** Feature-level spectral summary of corrected voltage and EMA memory channels. Values quantify the frequency-band energy distribution and high-frequency energy reduction relative to the corresponding raw measurement stream.
