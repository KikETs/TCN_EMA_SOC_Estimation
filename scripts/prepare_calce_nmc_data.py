from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Create a local raw-file manifest after the user places CALCE/NMC data.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default="data/raw/NMC_SAMSUNG_INR_18650_2Ah")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_root = base_dir / args.raw_root
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_root.rglob("*.csv")) if raw_root.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_root}. Place raw data there first; raw data are not bundled."
        )

    rows = []
    for path in files:
        rows.append({"relative_path": path.relative_to(base_dir).as_posix(), "size_bytes": path.stat().st_size})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "raw_file_manifest.csv", index=False)
    (out_dir / "preprocessing_report.md").write_text(
        "# Preprocessing Report\n\n"
        "This script only records the local raw-file manifest. Feature construction is handled by "
        "`scripts/build_g4_features.py`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'raw_file_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

