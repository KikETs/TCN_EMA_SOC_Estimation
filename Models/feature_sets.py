from __future__ import annotations


FEATURE_SETS: dict[str, list[str]] = {
    "G0": ["V_corr_raw", "I_raw", "T"],
    "G1": ["V_corr_raw", "I_raw", "T", "dI", "absI", "dV_corr", "abs_dV_corr", "d2V_corr"],
    "G4": [
        "V_corr_raw",
        "I_raw",
        "T",
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    ],
    "G6": [
        "V_corr_raw",
        "I_raw",
        "T",
        "dI",
        "absI",
        "dV_corr",
        "abs_dV_corr",
        "d2V_corr",
        "Vcorr_x_absI",
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    ],
}

FEATURE_SETS["G7"] = [f for f in FEATURE_SETS["G6"] if not f.startswith("I_raw_ema") and not f.startswith("I_raw_dev") and not f.startswith("absI_ema") and not f.startswith("absI_dev")]
FEATURE_SETS["G8"] = [f for f in FEATURE_SETS["G6"] if not f.startswith("V_corr_raw_ema") and not f.startswith("V_corr_raw_dev")]


def add_derived_features(frame):
    frame = frame.copy()
    frame["dI"] = frame["I_raw"].diff().fillna(0.0)
    frame["absI"] = frame["I_raw"].abs()
    frame["dV_corr"] = frame["V_corr_raw"].diff().fillna(0.0)
    frame["abs_dV_corr"] = frame["dV_corr"].abs()
    frame["d2V_corr"] = frame["dV_corr"].diff().fillna(0.0)
    frame["Vcorr_x_absI"] = frame["V_corr_raw"] * frame["absI"]
    return frame
