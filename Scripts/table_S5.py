from __future__ import annotations

import pandas as pd

from common import save_table


def main() -> None:
    rows = [
        (
            "Target/state exclusion",
            "SOC labels, SOC_CC, cumulative Ah, window-start SOC, and trajectory progress are excluded from all input schemas.",
            "No direct SOC proxy or explicit SOC-state input.",
        ),
        (
            "Current input",
            "Current is used as instantaneous current, local excitation, and causal EMA/deviation features only.",
            "No Coulomb-counted SOC input.",
        ),
        ("EMA causality", "EMA features use one-sided forward recurrences within each record.", "No future samples are used."),
        ("Record reset", "EMA states are reset at each profile-temperature record boundary.", "No carryover across independent records."),
        ("Normalization", "Standardization is fitted on training records only.", "No test-distribution leakage."),
    ]
    save_table(pd.DataFrame(rows, columns=["Check item", "Implementation", "Evaluation relevance"]), "table_S5_feature_construction_audit", digits=4)


if __name__ == "__main__":
    main()
