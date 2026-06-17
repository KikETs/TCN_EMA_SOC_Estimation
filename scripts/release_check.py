from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5")
LOCAL_PATH_TOKENS = (
    "/home/" + "lab",
    "/home/" + "user",
    "C:" + "\\Users",
    "\ubc14\ud0d5\ud654\uba74",
    "100.121." + "61.51",
)
SOURCE_METRIC_REQUIRED = (
    "g4_seed_reproduction_summary.csv",
    "g4_seed_reproduction_by_temp.csv",
    "feature_ablation_reanalysis.csv",
    "ema_tau_group_sweep.csv",
    "ema_perturbation_importance.csv",
    "g4_profile_rotation_summary.csv",
    "g4_profile_rotation_by_seed_temp.csv",
    "g4_epoch_sweep.csv",
    "ema_vs_cumulative_correlation.csv",
    "forbidden_reference_baselines.csv",
    "g4_model_class_baselines.csv",
)


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in {".png", ".pdf", ".pyc"}:
        return False
    try:
        path.read_text(encoding="utf-8")
        return True
    except UnicodeDecodeError:
        return False


def check_data_dirs(base_dir: Path) -> tuple[bool, str]:
    allowed = {".gitkeep", "README.md"}
    problems = []
    for rel in ("data/raw", "data/processed"):
        root = base_dir / rel
        for path in root.rglob("*"):
            if path.is_file() and path.name not in allowed:
                problems.append(path.relative_to(base_dir).as_posix())
    return (not problems, "OK" if not problems else ", ".join(problems))


def check_abs_paths(base_dir: Path) -> tuple[bool, str]:
    hits = []
    for path in base_dir.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name == "RELEASE_CHECK_REPORT.md":
            continue
        if not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8")
        for token in LOCAL_PATH_TOKENS:
            if token in text:
                hits.append(f"{path.relative_to(base_dir).as_posix()}::{token}")
    return (not hits, "OK" if not hits else "; ".join(hits[:20]))


def check_no_large_or_forbidden_files(base_dir: Path) -> tuple[bool, str]:
    problems = []
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(CHECKPOINT_SUFFIXES) or name.endswith(ARCHIVE_SUFFIXES):
            problems.append(path.relative_to(base_dir).as_posix())
        elif path.stat().st_size > 10 * 1024 * 1024:
            problems.append(f"{path.relative_to(base_dir).as_posix()} >10MB")
    return (not problems, "OK" if not problems else ", ".join(problems))


def check_source_metrics(base_dir: Path) -> tuple[bool, str]:
    root = base_dir / "paper_artifacts" / "source_metrics"
    missing = [name for name in SOURCE_METRIC_REQUIRED if not (root / name).exists()]
    return (not missing, "OK" if not missing else ", ".join(missing))


def run_command(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return proc.returncode == 0, proc.stdout[-4000:]


def main() -> int:
    p = argparse.ArgumentParser(description="Run release packaging checks.")
    p.add_argument("--base-dir", default=".")
    p.add_argument(
        "--require-paper-artifacts",
        action="store_true",
        help="Fail if paper_artifacts/source_metrics is absent. Default treats it as a local optional artifact folder.",
    )
    args = p.parse_args()
    base_dir = Path(args.base_dir).resolve()
    checks: list[tuple[str, bool, str]] = []

    checks.append((".gitignore exists", (base_dir / ".gitignore").exists(), "OK" if (base_dir / ".gitignore").exists() else "missing"))
    checks.append(("raw/processed data excluded", *check_data_dirs(base_dir)))
    checks.append(("no checkpoints/archives/large files", *check_no_large_or_forbidden_files(base_dir)))
    checks.append(("no absolute local paths", *check_abs_paths(base_dir)))
    source_ok, source_detail = check_source_metrics(base_dir)
    if source_ok:
        checks.append(("source metric files exist", True, "OK"))
        ok, detail = run_command([sys.executable, "scripts/build_paper_artifacts.py", "--base-dir", "."], base_dir)
        checks.append(("build_paper_artifacts.py", ok, "OK" if ok else detail))
    elif args.require_paper_artifacts:
        checks.append(("source metric files exist", False, source_detail))
        checks.append(("build_paper_artifacts.py", False, "skipped because required source metrics are missing"))
    else:
        checks.append(("source metric files exist", True, "SKIP: paper_artifacts is optional for the current GitHub upload"))
        checks.append(("build_paper_artifacts.py", True, "SKIP: local source metrics are absent"))

    if shutil.which("pytest"):
        ok, detail = run_command([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "tests"], base_dir)
        checks.append(("pytest", ok, "OK" if ok else detail))
    else:
        checks.append(("pytest", False, "pytest not available"))

    lines = ["# Release Check Report", "", "| check | status | detail |", "|---|---:|---|"]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        safe_detail = str(detail).replace("\n", "<br>")
        lines.append(f"| {name} | {status} | {safe_detail} |")
    out = base_dir / "RELEASE_CHECK_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print("Failed checks: " + ", ".join(failed))
        return 1
    print("All release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
