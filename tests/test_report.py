from __future__ import annotations

from pathlib import Path

import pandas as pd

from tedana_group_stats.aggregate import build_group_table
from tedana_group_stats.discovery import discover_runs
from tedana_group_stats.metrics import summarize_run
from tedana_group_stats.report import (
    _bar_components,
    _box_accepted_fraction,
    _category,
    _fmt,
    _marker_sizes,
    _table_html,
    write_report,
)


def _table(derivatives: Path) -> pd.DataFrame:
    return build_group_table([summarize_run(r) for r in discover_runs(derivatives)])


def test_writes_html_with_table_and_chart(derivatives: Path, tmp_path: Path) -> None:
    out = write_report(_table(derivatives), tmp_path / "report.html")
    html = out.read_text()
    assert "tedana group stats" in html
    assert "plotly" in html.lower()
    assert "<table" in html


def test_offline_inlines_plotly(derivatives: Path, tmp_path: Path) -> None:
    table = _table(derivatives)
    online = write_report(table, tmp_path / "online.html").read_text()
    offline = write_report(table, tmp_path / "offline.html", offline=True).read_text()
    assert "cdn.plot.ly" in online
    assert len(offline) > len(online)  # inlined plotly.js dwarfs a CDN script tag


def test_empty_table_writes_placeholder(tmp_path: Path) -> None:
    out = write_report(pd.DataFrame(), tmp_path / "empty.html")
    html = out.read_text()
    assert "No tedana runs found." in html
    assert "<b>0</b> runs" in html
    assert "plotly" not in html.lower()  # no charts generated for an empty table


def test_report_accepts_str_path(derivatives: Path, tmp_path: Path) -> None:
    out = write_report(_table(derivatives), str(tmp_path / "s.html"))
    assert out.is_file()


def test_fmt_formats_none_nan_float_and_escapes() -> None:
    assert _fmt(None) == "n/a"
    assert _fmt(float("nan")) == "n/a"
    assert _fmt(1234.5) == "1.23e+03"
    assert _fmt("<script>") == "&lt;script&gt;"  # HTML-escaped


def test_marker_sizes_edge_cases() -> None:
    all_equal = _marker_sizes(pd.DataFrame({"n_components": [3, 3, 3]}))
    assert (all_equal == 16.0).all()  # no spread -> midpoint of [6, 26]
    all_nan = _marker_sizes(pd.DataFrame({"n_components": [None, None]}))
    assert (all_nan == 6.0).all()  # nothing valid -> lo


def test_category_three_branches() -> None:
    assert _category(pd.Series({"status": "failed", "is_outlier": True})) == "failed"
    assert _category(pd.Series({"status": "ok", "is_outlier": True})) == "outlier"
    assert _category(pd.Series({"status": "ok"})) == "ok"  # missing is_outlier -> ok


def test_table_html_row_classes_and_escaping() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["<x>", "b", "c"],
            "status": ["failed", "ok", "ok"],
            "is_outlier": [False, True, False],
        }
    )
    html = _table_html(df)
    assert html.count('class="failed"') == 1
    assert html.count('class="outlier"') == 1
    assert "&lt;x&gt;" in html  # run_id escaped, no raw <x>
    assert "<x>" not in html


def test_table_html_skips_absent_columns() -> None:
    df = pd.DataFrame({"run_id": ["a"], "status": ["ok"]})
    html = _table_html(df)
    assert "median_kappa" not in html  # optional column absent -> not rendered


def test_box_accepted_fraction_without_is_outlier_column() -> None:
    df = pd.DataFrame(
        {"status": ["ok", "ok"], "accepted_fraction": [0.5, 0.6], "run_id": ["a", "b"]}
    )
    fig = _box_accepted_fraction(df)  # no is_outlier column -> defaults to False, no error
    assert len(fig.to_dict()["data"]) == 1  # a single box trace, no outlier overlay


def test_bar_components_skips_absent_count_columns() -> None:
    df = pd.DataFrame(
        {"run_id": ["a", "b"], "accepted_fraction": [0.5, 0.6], "n_accepted": [1, 2]}
    )
    fig = _bar_components(df)  # only n_accepted present; others skipped silently
    assert [trace["name"] for trace in fig.to_dict()["data"]] == ["accepted"]


def test_report_renders_all_failed_runs(tmp_path: Path) -> None:
    # Build an all-failed table through the real pipeline (empty TSVs fail to parse).
    for sub in ("01", "02"):
        func = tmp_path / f"sub-{sub}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{sub}_desc-tedana_metrics.tsv").write_text("")
    table = build_group_table([summarize_run(r) for r in discover_runs(tmp_path)])
    assert (table["status"] != "ok").all()
    out = write_report(table, tmp_path / "failed.html")
    assert "<b>2</b> failed to load" in out.read_text()
