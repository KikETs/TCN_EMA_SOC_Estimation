# Section 2.4 Frequency Values For Manuscript

- Band definitions: low frequency is f < 1/200 cycles/sample, mid frequency is 1/200 <= f < 1/50 cycles/sample, and high frequency is f >= 1/50 cycles/sample.
- Records analyzed: 12 profile-temperature records.
- Voltage column used: Voltage(V).
- Current column used: Current(A).
- SOC column used: SOC_CC.

## Compact Numerical Findings

| signal_name   |   mean_low_frequency_energy_percent | range_low_frequency_energy_percent   |   mean_mid_frequency_energy_percent | range_mid_frequency_energy_percent   |   mean_high_frequency_energy_percent | range_high_frequency_energy_percent   |   mean_median_frequency_cycles_per_sample | range_median_frequency_cycles_per_sample   |
|:--------------|------------------------------------:|:-------------------------------------|------------------------------------:|:-------------------------------------|-------------------------------------:|:--------------------------------------|------------------------------------------:|:-------------------------------------------|
| Current       |                               14.88 | 5.39-28.34                           |                               30.81 | 15.50-42.92                          |                                54.31 | 29.42-79.08                           |                                  0.024495 | 0.009766-0.041992                          |
| Voltage       |                               20.78 | 8.97-37.39                           |                               30.97 | 16.16-41.69                          |                                48.26 | 21.54-74.05                           |                                  0.019206 | 0.005859-0.035156                          |
| Reference SOC |                               98.85 | 97.77-99.61                          |                                1.04 | 0.28-2.11                            |                                 0.11 | 0.03-0.17                             |                                  0.000977 | 0.000977-0.000977                          |

## Section 2.4 Paragraph Draft

Welch spectra computed within each profile-temperature record show that the measured current contains a larger high-frequency contribution than the reference SOC trajectory. The reference SOC trajectory is dominated by low-frequency content because it is obtained by current integration. The terminal-voltage trajectory retains slow discharge-related variation together with faster load-induced transient response, motivating the use of both slow voltage-response context and recent current-history information.

## Caption Drafts

**Figure 3.** Frequency-domain structure of raw voltage, current, and reference SOC trajectories. Spectra were computed within each profile-temperature record and summarized by signal type. Current contains stronger high-frequency excitation components, whereas the Coulomb-counted reference SOC trajectory is dominated by low-frequency variation. Voltage contains both slow discharge-related variation and faster load-induced transient response.

**Table 5.** Compact spectral energy summary of raw terminal measurements and reference SOC. Energy fractions were computed from normalized Welch spectra within each profile-temperature record and averaged across the twelve records.

**Table S6.** Raw-signal spectral energy summary by profile and temperature. Low-, mid-, and high-frequency energy fractions were computed from raw voltage, current, and reference SOC trajectories within each record boundary.
