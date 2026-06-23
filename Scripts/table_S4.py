from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "source_tables" / "feature_schema.csv"))
    roles = {
        "G0": "Corrected-voltage, current, and temperature baseline",
        "G1": "G0 plus local derivative and excitation descriptors",
        "G4": "Proposed causal EMA representation",
        "G6": "G4 plus local derivative, excitation, and interaction descriptors",
        "G7": "Voltage-memory-only ablation with local descriptors",
        "G8": "Current-memory-only ablation with local descriptors",
    }
    rows = []
    for feature_set in ["G0", "G1", "G4", "G6", "G7", "G8"]:
        g = src[src["feature_set"].eq(feature_set)].sort_values("channel_index")
        rows.append(
            {
                "Feature set": feature_set,
                "Dim.": int(g["input_dim"].iloc[0]),
                "Full input channels": ", ".join(g["channel"].astype(str)),
                "Role": roles[feature_set],
            }
        )
    df = pd.DataFrame(rows)
    save_table(df, "table_S4_feature_schema", digits=4)


if __name__ == "__main__":
    main()
