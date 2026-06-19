# Filled Manuscript Table 6

## Table 6. Model-class baselines under the FUDS profile-holdout protocol.

| Model class | Description | Approx. parameters | 0 °C MAE | 25 °C MAE | 45 °C MAE | Temperature-mean MAE | Worst-temperature MAE | Temperature-mean RMSE | Temperature-mean MaxAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tcn_h128_l6_moe_g4 | anchor-residual TCN h128 l6 with temp-MoE, G4 | 492806 | 0.4478 | 0.4979 | 0.3797 | 0.4418 | 0.4979 | 0.6263 | 2.5400 |
| tcn_h128_l6_no_moe | anchor-residual TCN h128 l6 without temp-MoE | 492288 | 0.4362 | 0.5375 | 0.3644 | 0.4460 | 0.5375 | 0.6378 | 2.3804 |
| lstm_h128_l1 | LSTM h128 layer=1 | 131584 | 0.5829 | 0.8056 | 0.4657 | 0.6181 | 0.8056 | 0.8108 | 3.9017 |
| window_summary_mlp | window summary MLP: endpoint/mean/std/delta | 41985 | 0.6970 | 0.9340 | 0.6378 | 0.7563 | 0.9340 | 0.9612 | 3.8651 |
| gru_h128_l1 | GRU h128 layer=1 | 98688 | 0.7758 | 1.0855 | 0.5521 | 0.8045 | 1.0855 | 1.0285 | 3.3400 |
| endpoint_mlp | endpoint MLP, no temporal order | 35457 | 0.7301 | 1.4307 | 0.7036 | 0.9548 | 1.4307 | 1.3000 | 6.0912 |

The temperature-mean MAE, RMSE, and MaxAE values are unweighted means of the corresponding 0 °C, 25 °C, and 45 °C FUDS metrics.
