# Paper Artifacts

Regenerate tables and figures with:

```bash
python scripts/build_paper_artifacts.py
```

Source CSVs are read from `paper_artifacts/source_metrics/` when that local folder is present.

Generated outputs:

- tables: `paper_artifacts/tables/*.csv`, `*.md`, `*.tex`
- figures: `paper_artifacts/figures/*.png`, `*.pdf`
- captions: `paper_artifacts/manuscript_snippets/captions.md`

These artifacts use only small summary metrics and do not require raw data. The `paper_artifacts/` folder is intentionally ignored in the GitHub upload unless explicitly force-added later.
