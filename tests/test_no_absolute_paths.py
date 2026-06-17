from __future__ import annotations

from pathlib import Path


def test_release_text_files_do_not_contain_local_absolute_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    scan_dirs = ["README.md", "docs", "configs", "scripts", "paper_artifacts/source_metrics"]
    forbidden = (
        "/home/" + "lab",
        "/home/" + "user",
        "C:" + "\\Users",
        "\ubc14\ud0d5\ud654\uba74",
        "100.121." + "61.51",
    )
    hits: list[str] = []
    for item in scan_dirs:
        path = root / item
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for file_path in files:
            if file_path.suffix.lower() in {".png", ".pdf", ".pyc"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for token in forbidden:
                if token in text:
                    hits.append(f"{file_path.relative_to(root).as_posix()}::{token}")
    assert hits == []
