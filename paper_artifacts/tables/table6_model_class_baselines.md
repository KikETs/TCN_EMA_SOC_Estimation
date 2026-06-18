| model_id | description | approx_trainable_params | 0C_MAE_pct | 25C_MAE_pct | 45C_MAE_pct | tempmean_MAE_pct | worst_temp_MAE_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tcn_h128_l6_moe_g4 | anchor-residual TCN h128 l6 with temp-MoE, G4 | 492806 | 0.4478 | 0.4979 | 0.3797 | 0.4418 | 0.4979 |
| tcn_h128_l6_no_moe | anchor-residual TCN h128 l6 without temp-MoE | 492288 | 0.4362 | 0.5375 | 0.3644 | 0.4460 | 0.5375 |
| lstm_h128_l1 | LSTM h128 layer=1 | 131584 | 0.5829 | 0.8056 | 0.4657 | 0.6181 | 0.8056 |
| window_summary_mlp | window summary MLP: endpoint/mean/std/delta | 41985 | 0.6970 | 0.9340 | 0.6378 | 0.7563 | 0.9340 |
| gru_h128_l1 | GRU h128 layer=1 | 98688 | 0.7758 | 1.0855 | 0.5521 | 0.8045 | 1.0855 |
| endpoint_mlp | endpoint MLP, no temporal order | 35457 | 0.7301 | 1.4307 | 0.7036 | 0.9548 | 1.4307 |
