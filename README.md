# tedana_group_stats

Aggregate [tedana](https://github.com/ME-ICA/tedana) multi-echo fMRI outputs across a
BIDS-derivatives tree into a single, self-contained interactive QC report — so you can
spot group trends and outlier runs without opening every report by hand.

It scans a directory for tedana output folders, pulls a handful of per-run summary
metrics (component counts, accepted fraction, kappa/rho, retained variance, elbow
thresholds), flags statistical outliers with a robust modified z-score, and writes one
HTML file with a styled summary table and three interactive Plotly charts.

It reads only tedana's `*_desc-tedana_registry.json` and the TSV/JSON it points to — it
does **not** import tedana or touch any NIfTI, so it is fast and lightweight.

If a derivatives tree was BIDS-renamed after tedana ran (e.g. files gained a `task-`
entity that the registry's recorded filenames lack), the scanner still finds the metrics
by matching on the `_desc-…` suffix, accepting a match only when it is unambiguous. This
means runs load correctly even when the registry's internal paths drift from disk.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Usage

```bash
tedana-group-stats /path/to/derivatives -o report.html
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `root` | — | BIDS-derivatives directory to scan recursively |
| `-o, --output` | `tedana_group_report.html` | Output HTML path |
| `--offline` | off | Inline `plotly.js` for a fully standalone file (no network needed to view) |
| `--threshold` | `3.5` | Modified z-score cutoff for outlier flagging |

By default the report references `plotly.js` from a CDN to keep the file small; use
`--offline` to embed it for sharing or viewing without a network.

## What counts as an outlier

For each of `accepted_fraction`, `n_components`, `median_kappa`,
`variance_explained_accepted`, and `var_exp_rejected_on_accepted`, runs are scored with
the Iglewicz–Hoaglin modified z-score (median/MAD based, robust to small samples) and
flagged when `|score| > threshold`. A run failing on any one metric is flagged. Runs
whose outputs could not be loaded are reported separately as **failed**, not as
statistical outliers.

## Python API

```python
from tedana_group_stats import discover_runs, summarize_run, build_group_table, write_report

runs = discover_runs("/path/to/derivatives")
table = build_group_table([summarize_run(r) for r in runs])
write_report(table, "report.html")
```

## Development

```bash
pre-commit install
ruff check . && ruff format --check . && pyright && pytest -q
```

## License

[LGPL-2.1](LICENSE), matching tedana.
