from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_PROFILES = ("DST", "US06", "BJDST", "FUDS")
EXPECTED_TEMPS = ("0", "25", "45")


def find_expected_files(raw_root: Path) -> dict[str, list[Path]]:
    files = sorted(raw_root.rglob("*.csv")) if raw_root.exists() else []
    found: dict[str, list[Path]] = {}
    for temp in EXPECTED_TEMPS:
        for profile in EXPECTED_PROFILES:
            key = f"{temp}C_{profile}"
            matches = [p for p in files if temp in p.as_posix() and profile.lower() in p.name.lower()]
            found[key] = matches
    return found


def write_report(base_dir: Path, raw_root: Path, found: dict[str, list[Path]]) -> Path:
    report_dir = base_dir / "data_manifest"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "data_presence_report.md"
    lines = [
        "# Data Presence Report",
        "",
        f"- raw_root: `{raw_root.as_posix()}`",
        f"- csv_file_count: `{len(list(raw_root.rglob('*.csv'))) if raw_root.exists() else 0}`",
        "",
        "| expected item | status | matched file count |",
        "|---|---:|---:|",
    ]
    for key, matches in found.items():
        status = "FOUND" if matches else "MISSING"
        lines.append(f"| `{key}` | {status} | {len(matches)} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Check whether local CALCE/NMC raw files have been placed.")
    p.add_argument("--base-dir", default=".", help="Release folder root.")
    p.add_argument("--raw-root", default="data/raw/NMC_SAMSUNG_INR_18650_2Ah")
    args = p.parse_args()
    base_dir = Path(args.base_dir).resolve()
    raw_root = (base_dir / args.raw_root).resolve() if not Path(args.raw_root).is_absolute() else Path(args.raw_root)
    found = find_expected_files(raw_root)
    report = write_report(base_dir, raw_root, found)
    missing = [k for k, v in found.items() if not v]
    print(f"Wrote {report}")
    if missing:
        print("Missing expected raw items: " + ", ".join(missing))
        return 1
    print("All expected raw items are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

