# BJDST Training-Coverage Rationale Summary

- BJDST broadens the non-FUDS training-side measurement coverage: adding BJDST increased the mean occupied V-I bin count from `1286.3` to `1350.3` and changed the FUDS outside-bin fraction from `0.80%` to `0.22%` under the same temperature-wise binning.
- The main split preserves FUDS as the held-out profile: the diagnostic compares `DST+US06` and `DST+US06+BJDST` training-side measurement distributions against FUDS without using model predictions, residuals, checkpoints, or FUDS-derived training decisions.
- This reduces dependence on a deliberately narrower two-profile training set by adding a non-FUDS drive pattern with additional measurement-history coverage; the causal `absI_mean_past200` overlap changed from `0.547` to `0.561`, while the average V-I histogram overlap changed from `0.528` to `0.497`, so the diagnostic should be interpreted as coverage broadening rather than a guarantee of improved generalization.
