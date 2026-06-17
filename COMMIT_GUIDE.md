# Commit Guide

Recommended commands:

```bash
git init
git status
python scripts/release_check.py
pytest
git add README.md docs configs scripts soc_decomp tests data results .gitignore requirements.txt pyproject.toml CITATION.cff LICENSE_OR_TODO.md RELEASE_NOTES.md FINAL_GITHUB_RELEASE_SUMMARY.md COMMIT_GUIDE.md source_copy_manifest.csv SOURCE_OF_TRUTH_MANIFEST.csv RELEASE_CHECK_REPORT.md
git status
git commit -m "Release G4 EMA SOC estimation reproducibility package"
```

Do not push until the license and citation metadata are finalized.
