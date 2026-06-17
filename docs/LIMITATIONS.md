# Limitations

- G4 is a frozen candidate selected after exploratory seed-0 ablation.
- This package does not claim global SOTA performance.
- This package does not claim temperature extrapolation.
- Current-history EMA is used, so current is not unused.
- EMA features can correlate with cumulative charge on fixed drive trajectories.
- The profile-rotation diagnostic is helpful but not a complete blind benchmark.
- The 45C legacy threshold is not passed consistently under the stricter threshold noted in source metrics.
- Raw data, checkpoints, and full prediction dumps are not included.

