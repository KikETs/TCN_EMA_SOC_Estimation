from __future__ import annotations

from pathlib import Path

import yaml


def test_g4_feature_set_has_no_forbidden_soc_or_cumulative_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "g4_feature_sets.yaml").read_text(encoding="utf-8"))
    features = data["feature_sets"]["G4_all_ema"]["features"]
    forbidden = ("soc", "soc_cc", "ah", "cumulative", "progress", "time", "target", "label")
    bad = [feature for feature in features if any(token in feature.lower() for token in forbidden)]
    assert bad == []
    assert "I_raw" in features
    assert "absI_ema50" in features
    assert len(features) == 17

