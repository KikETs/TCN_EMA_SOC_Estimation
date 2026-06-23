from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def mean_value(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().mean())


def min_max(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.min():.3f}–{values.max():.3f}"


def main() -> None:
    src = pd.read_csv(require(DATA / "section2_tables" / "table_s2_time_lag_statistics.csv"))
    rows = []
    for lag in [1, 50, 200, 800]:
        rows.append(
            {
                "Lag τ": lag,
                "Voltage ACF mean": mean_value(src[f"rho_V_lag{lag}"]),
                "Voltage ACF range": min_max(src[f"rho_V_lag{lag}"]),
                "Current ACF mean": mean_value(src[f"rho_I_lag{lag}"]),
                "Current ACF range": min_max(src[f"rho_I_lag{lag}"]),
            }
        )
    save_table(pd.DataFrame(rows), "table_2_voltage_current_autocorrelation", digits=4)


if __name__ == "__main__":
    main()
