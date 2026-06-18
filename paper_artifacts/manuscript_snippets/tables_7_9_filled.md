# Filled Manuscript Tables 7-9

## Table 7. Feature-set ablation under the FUDS profile-holdout protocol.

| Feature set | Input role | Input dim. | 0 °C MAE | 25 °C MAE | 45 °C MAE | Temp-mean MAE | Worst-temp MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | Corrected voltage + current + temperature | 3 | 1.0459 | 1.8893 | 0.2409 | 1.0587 | 1.8893 |
| G1 | G0 + local derivatives/excitation | 8 | 0.7955 | 1.8137 | 0.2311 | 0.9468 | 1.8137 |
| G4 | G0 + voltage/current/absolute-current EMA memory | 17 | 0.4528 | 0.4564 | 0.3474 | 0.4189 | 0.4564 |
| G6 | G4 + derivative/excitation terms | 23 | 0.4399 | 0.6100 | 0.3820 | 0.4773 | 0.6100 |
| G7 | G6 without current/absolute-current EMA | 15 | 0.4631 | 2.0031 | 0.3360 | 0.9341 | 2.0031 |
| G8 | G6 without voltage EMA | 17 | 0.6239 | 0.6257 | 0.1840 | 0.4778 | 0.6257 |

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
| SOC band | Low SOC | 1.2268 | 0.5506 | -0.6762 | -55.1178 | 11765 | 3 | 3 |
| SOC band | Mid SOC | 0.9808 | 0.4274 | -0.5534 | -56.4232 | 12334 | 3 | 3 |
| SOC band | High SOC | 0.8830 | 0.2062 | -0.6769 | -76.6534 | 8179 | 3 | 3 |
| Recent absolute-current history | Low | 1.0617 | 0.4543 | -0.6074 | -57.2118 | 16066 | 3 | 3 |
| Recent absolute-current history | High | 1.0298 | 0.3786 | -0.6513 | -63.2405 | 16212 | 3 | 3 |
| Voltage-response deviation | Low | 1.1505 | 0.4712 | -0.6793 | -59.0417 | 16066 | 3 | 3 |
| Voltage-response deviation | High | 0.9419 | 0.3618 | -0.5801 | -61.5884 | 16212 | 3 | 3 |
| Local V-I ambiguity | Non-ambiguous bins | 1.0451 | 0.4129 | -0.6321 | -60.4871 | 31878 | 3 | 3 |
| Local V-I ambiguity | Ambiguous bins | 1.0967 | 0.6808 | -0.4159 | -37.9199 | 400 | 3 | 3 |
