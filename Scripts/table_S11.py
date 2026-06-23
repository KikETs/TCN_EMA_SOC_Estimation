from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "table_10_region_error_reduction.csv"))
    df = pd.DataFrame(
        {
            "Region definition": src["region_definition"],
            "Group": src["group"],
            "G0 MAE (%SOC)": src["G0_MAE"],
            "G4 MAE (%SOC)": src["G4_MAE"],
            "Reduction (%SOC)": -pd.to_numeric(src["delta_MAE_G4_minus_G0"], errors="coerce"),
            "Relative reduction (%)": -pd.to_numeric(src["relative_change"], errors="coerce"),
            "n windows": src["n_windows"],
        }
    )
    save_table(df, "table_S11_region_error_reduction", digits=4)


if __name__ == "__main__":
    main()
