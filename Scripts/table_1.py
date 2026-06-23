from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


PROFILE_ORDER = {"BJDST": 0, "DST": 1, "US06": 2, "FUDS": 3}


def main() -> None:
    src = pd.read_csv(require(DATA / "section2_tables" / "table_s2_dataset_terminal_summary.csv"))
    src["_profile_order"] = src["profile"].map(PROFILE_ORDER)
    src = src.sort_values(["temperature_C", "_profile_order"]).copy()
    df = pd.DataFrame(
        {
            "Temperature (℃)": src["temperature_C"].astype(int),
            "Profile": src["profile"],
            "Samples": src["n_samples"].astype(int),
            "Duration (s)": src["duration_s"].round(0).astype(int),
            "SOC interval (%)": [
                f"{lo * 100.0:.2f}–{hi * 100.0:.1f}" for lo, hi in zip(src["SOC_min_fraction"], src["SOC_max_fraction"], strict=True)
            ],
            "Voltage range (V)": [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(src["voltage_min_V"], src["voltage_max_V"], strict=True)],
            "Current range (A)": [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(src["current_min_A"], src["current_max_A"], strict=True)],
        }
    )
    save_table(df, "table_1_dataset_summary", digits=2)


if __name__ == "__main__":
    main()
