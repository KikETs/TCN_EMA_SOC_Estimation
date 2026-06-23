# Section 4 Feature Frequency Values For Manuscript

- Corrected voltage column source: V_corr_raw = causal_time_ema(Voltage(V) - Current(A) * R0_temperature, tau=120 s), with R0 estimated from BJDST/DST/US06 events only..
- EMA feature columns used: V_corr_raw_ema50, V_corr_raw_ema200, V_corr_raw_ema800, I_raw_ema50, I_raw_ema200, absI_ema50, absI_ema200, V_corr_raw_dev_ema50, V_corr_raw_dev_ema200, V_corr_raw_dev_ema800, I_raw_dev_ema50, I_raw_dev_ema200, absI_dev_ema50, absI_dev_ema200.
- Representative EMA spans selected: voltage short=50 and long=800 samples; current short=50 and long=200 samples; abs-current short=50 and long=200 samples.

## Compact Numerical Findings

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

## Section 4.x Paragraph Draft

Feature-level Welch spectra verify the intended signal-processing behavior of the constructed measurement features. Corrected voltage attenuates part of the fast load-dependent variation in the terminal-voltage input, while voltage and current EMA channels shift the representation toward lower-frequency finite-memory measurement context. Longer EMA spans retain slower components than shorter spans.

## Caption Drafts

**Figure 6.** Spectral behavior of corrected voltage and EMA memory features. Corrected voltage attenuates part of the fast load-dependent variation in the voltage input, and EMA channels shift voltage/current measurements toward lower-frequency finite-memory context.

**Table 7.** Compact frequency-domain summary of corrected-voltage and EMA feature channels. Values summarize normalized Welch spectra across profile-temperature records.

**Table S7.** Feature-level spectral summary of corrected voltage and EMA memory channels. Values quantify the frequency-band energy distribution and high-frequency energy reduction relative to the corresponding raw measurement stream.
