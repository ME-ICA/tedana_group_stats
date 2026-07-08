from __future__ import annotations

from pathlib import Path

import pandas as pd

from tedana_group_stats.aggregate import build_group_table
from tedana_group_stats.discovery import discover_runs
from tedana_group_stats.metrics import summarize_run
from tedana_group_stats.report import write_report


def _table(derivatives: Path) -> pd.DataFrame:
    return build_group_table([summarize_run(r) for r in discover_runs(derivatives)])


def test_writes_html_with_table_and_chart(derivatives: Path, tmp_path: Path) -> None:
    out = write_report(_table(derivatives), tmp_path / "report.html")
    html = out.read_text()
    assert "tedana group stats" in html
    assert "plotly" in html.lower()
    assert "<table" in html
    assert "rejected_variance_fraction" in html  # new table column
    assert "Low variance" in html  # rejection-reasons chart tag
    assert "Variance explained per run" in html  # new variance chart


def test_large_run_count_switches_to_distributions(derivatives: Path, tmp_path: Path) -> None:
    """Past the per-run-bar threshold, bars are replaced by scalable distributions."""
    table = _table(derivatives)
    big = pd.concat([table] * 10, ignore_index=True)  # 8 runs -> 80, over the 40 threshold
    html = write_report(big, tmp_path / "big.html").read_text()
    assert "Component count distribution" in html
    assert "Rejected variance fraction distribution" in html
    assert "Component counts per run" not in html  # per-run bars dropped at scale


def test_offline_inlines_plotly(derivatives: Path, tmp_path: Path) -> None:
    table = _table(derivatives)
    online = write_report(table, tmp_path / "online.html").read_text()
    offline = write_report(table, tmp_path / "offline.html", offline=True).read_text()
    assert "cdn.plot.ly" in online
    assert len(offline) > len(online)  # inlined plotly.js dwarfs a CDN script tag
