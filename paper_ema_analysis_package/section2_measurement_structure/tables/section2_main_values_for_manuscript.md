# Section 2 Main Values for Manuscript

Inputs: Section 2 generated tables only. No model predictions, residuals, ablations, perturbations, baselines, checkpoints, forbidden-reference diagnostics, or model-output files were used.

- Total samples: `127,690` across `12` temperature-profile terminal records.
- Global voltage range: `2.403–4.152 V`.
- Global current range: `-4.002–2.142 A`.
- Mean lag-200 autocorrelation: voltage `0.754`, current `-0.021`.
- Raw V-I median SOC IQR across temperatures: `1.31–1.72 %SOC`.
- Raw V-I p90 SOC IQR across temperatures: `4.20–7.39 %SOC`.
- Raw V-I max SOC IQR across temperatures: `20.46–24.88 %SOC`.
- Main split V-I overlap coefficient average: `0.497`.
- Main split outside-train occupied-bin fraction average: `0.22%`.

History-conditioned median SOC IQR diagnostics, averaged over temperatures:

- A raw V-I: `1.52 %SOC`.
- B + recent |I|: `0.61 %SOC`.
- C + voltage deviation: `1.75 %SOC`.
- D + both: `0.87 %SOC`.
