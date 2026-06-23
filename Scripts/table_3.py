from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def mean_min_max(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.mean():.3f} ({values.min():.3f}–{values.max():.3f})"


def main() -> None:
    src = pd.read_csv(require(DATA / "section2_tables" / "table_s2_causal_lag_correlations.csv"))
    rows = []
    for lag, g in src[src["lag_samples"].isin([1, 10, 20, 50, 200])].groupby("lag_samples"):
        rows.append(
            {
                "Lag τ (samples)": int(lag),
                "corr(I_{t-τ}, V_t)": mean_min_max(g["corr_I_t_minus_lag_with_V_t"]),
                "corr(V_{t-τ}, SOC_t)": mean_min_max(g["corr_V_t_minus_lag_with_SOC_t"]),
                "corr(I_{t-τ}, SOC_t)": mean_min_max(g["corr_I_t_minus_lag_with_SOC_t"]),
            }
        )
    save_table(pd.DataFrame(rows).sort_values("Lag τ (samples)"), "table_3_causal_lag_correlations", digits=4)


if __name__ == "__main__":
    main()
