# Filled Manuscript Tables 7-9

## Table 7. Feature-set ablation under the FUDS profile-holdout protocol.

| Feature set | Input role | Input dim. | 0 °C MAE | 25 °C MAE | 45 °C MAE | Temp-mean MAE | Worst-temp MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | Corrected voltage + current + temperature | 3 | 1.0592 | 1.8423 | 0.2766 | 1.0594 | 1.8423 |
| G1 | G0 + local derivatives/excitation | 8 | 0.7927 | 1.7248 | 0.2192 | 0.9122 | 1.7248 |
| G4 | G0 + voltage/current/absolute-current EMA memory | 17 | 0.4190 | 0.4648 | 0.3603 | 0.4147 | 0.4648 |
| G6 | G4 + derivative/excitation terms | 23 | 0.4542 | 0.6989 | 0.3710 | 0.5081 | 0.6989 |
| G7 | G6 without current/absolute-current EMA | 15 | 0.4632 | 2.0438 | 0.3359 | 0.9476 | 2.0438 |
| G8 | G6 without voltage EMA | 17 | 0.6104 | 0.6100 | 0.1756 | 0.4653 | 0.6104 |

## Table 8. Spectral energy distribution of representative measurement and EMA channels.

| feature_group | representative_channel | low_frequency_energy_percent | mid_frequency_energy_percent | high_frequency_energy_percent | median_frequency | high_frequency_reduction_vs_raw_reference_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Raw voltage | V_raw | 20.7800 | 30.9700 | 48.2600 | 0.0192 | 0.0000 |
| Corrected voltage | V_corr_raw | 97.9800 | 1.9400 | 0.0800 | 0.0010 | 99.8400 |
| Short voltage EMA | V_corr_raw_ema50 | 99.5600 | 0.4400 | 0.0000 | 0.0010 | 98.1500 |
| Long voltage EMA | V_corr_raw_ema800 | 99.9900 | 0.0100 | 0.0000 | 0.0010 | 99.9400 |
| Raw current | I_raw | 14.8800 | 30.8100 | 54.3100 | 0.0245 | 0.0000 |
| Short current EMA | I_raw_ema50 | 66.3700 | 28.8700 | 4.7600 | 0.0037 | 92.2100 |
| Absolute-current EMA | absI_ema50 | 72.4900 | 22.6000 | 4.9100 | 0.0029 | 92.7000 |

## Table 9. Error reduction by SOC, current-history, and voltage-response regions.

| region_definition | group | G0_MAE | G4_MAE | delta_MAE_G4_minus_G0 | relative_change | n_windows | seed_count_G0 | seed_count_G4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SOC band | Low SOC | 1.2398 | 0.5563 | -0.6835 | -55.1316 | 11765 | 3 | 3 |
| SOC band | Mid SOC | 0.9877 | 0.4081 | -0.5796 | -58.6837 | 12334 | 3 | 3 |
| SOC band | High SOC | 0.8568 | 0.2168 | -0.6401 | -74.7006 | 8179 | 3 | 3 |
| Recent absolute-current history | Low | 1.0788 | 0.4324 | -0.6463 | -59.9142 | 16066 | 3 | 3 |
| Recent absolute-current history | High | 1.0143 | 0.3950 | -0.6194 | -61.0620 | 16212 | 3 | 3 |
| Voltage-response deviation | Low | 1.1648 | 0.4595 | -0.7053 | -60.5523 | 16066 | 3 | 3 |
| Voltage-response deviation | High | 0.9290 | 0.3681 | -0.5609 | -60.3745 | 16212 | 3 | 3 |
| Local V-I ambiguity | Non-ambiguous bins | 1.0459 | 0.4108 | -0.6351 | -60.7247 | 31878 | 3 | 3 |
| Local V-I ambiguity | Ambiguous bins | 1.0870 | 0.6394 | -0.4475 | -41.1744 | 400 | 3 | 3 |
