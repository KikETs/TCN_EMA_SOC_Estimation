from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


FORBIDDEN_PATTERNS = [
    r"\bSOC\b",
    r"SOC_CC",
    r"\bAh\b",
    r"cumulative",
    r"progress",
    r"target",
    r"label",
]


def audit_feature_names(features: list[str]) -> list[str]:
    bad: list[str] = []
    for feature in features:
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, feature, flags=re.IGNORECASE):
                bad.append(feature)
                break
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description="Audit feature-set YAML for strict NoCC forbidden inputs.")
    p.add_argument("--feature-yaml", default="configs/g4_feature_sets.yaml")
    p.add_argument("--feature-set", default="G4_all_ema")
    args = p.parse_args()
    path = Path(args.feature_yaml)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    features = data["feature_sets"][args.feature_set]["features"]
    bad = audit_feature_names(features)
    if bad:
        print("CUMULATIVE_OR_SOC_FEATURE_LEAK: " + ", ".join(bad))
        return 1
    print(f"NoCC feature audit passed for {args.feature_set}: {len(features)} features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

