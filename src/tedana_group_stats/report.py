"""Render the group table to a single self-contained interactive HTML report.

One HTML file: a header, a summary table (failed rows red, outlier rows amber), and
three interactive Plotly charts. By default Plotly.js is pulled from a CDN to keep the
file small; ``offline=True`` inlines it for a fully standalone file. The table is built
by hand rather than via ``DataFrame.style`` so the package needs no jinja2.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from tedana_group_stats import __version__

# Columns shown in the summary table, in order. Missing ones are skipped.
_TABLE_COLUMNS = [
    "run_id",
    "subject",
    "session",
    "task",
    "run",
    "status",
    "n_components",
    "n_accepted",
    "n_rejected",
    "accepted_fraction",
    "variance_explained_accepted",
    "variance_explained_rejected",
    "median_kappa",
    "median_rho",
    "kappa_elbow",
    "rho_elbow",
    "var_exp_rejected_on_accepted",
    "tedana_version",
    "is_outlier",
]
_CATEGORY_COLORS = {"ok": "#1f77b4", "outlier": "#d62728", "failed": "#7f7f7f"}


def _category(row: pd.Series) -> str:
    if row["status"] != "ok":
        return "failed"
    return "outlier" if bool(row.get("is_outlier", False)) else "ok"


def _marker_sizes(df: pd.DataFrame, lo: float = 6.0, hi: float = 26.0) -> pd.Series:
    counts = cast("pd.Series", pd.to_numeric(df["n_components"], errors="coerce"))
    valid = counts.dropna()
    if valid.empty:
        return pd.Series(lo, index=df.index)
    lo_val, hi_val = float(valid.min()), float(valid.max())
    if hi_val == lo_val:
        return pd.Series((lo + hi) / 2.0, index=df.index)
    scaled = lo + (counts - lo_val) / (hi_val - lo_val) * (hi - lo)
    return scaled.fillna(lo)


def _scatter_kappa_rho(df: pd.DataFrame) -> go.Figure:
    plotted = df.dropna(subset=["median_kappa", "median_rho"])
    fig = go.Figure()
    for category, color in _CATEGORY_COLORS.items():
        sub = plotted[plotted["_category"] == category]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["median_kappa"],
                y=sub["median_rho"],
                mode="markers",
                name=category,
                text=sub["run_id"],
                marker={
                    "color": color,
                    "size": sub["_marker_size"],
                    "line": {"width": 0.5, "color": "white"},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>median kappa=%{x:.1f}<br>median rho=%{y:.1f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Median kappa vs rho per run (marker size scales with component count)",
        xaxis_title="median kappa",
        yaxis_title="median rho",
        template="plotly_white",
    )
    return fig


def _bar_components(df: pd.DataFrame) -> go.Figure:
    ordered = df.sort_values("accepted_fraction", na_position="first")
    fig = go.Figure()
    for column, color in (
        ("n_accepted", "#2ca02c"),
        ("n_rejected", "#d62728"),
        ("n_ignored", "#7f7f7f"),
        ("n_other", "#9467bd"),
    ):
        if column in ordered.columns:
            fig.add_trace(
                go.Bar(
                    x=ordered["run_id"],
                    y=ordered[column],
                    name=column.removeprefix("n_"),
                    marker_color=color,
                )
            )
    fig.update_layout(
        barmode="stack",
        title="Component counts per run (sorted by accepted fraction)",
        xaxis_title="run",
        yaxis_title="components",
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=45)
    return fig


def _box_accepted_fraction(df: pd.DataFrame) -> go.Figure:
    ok = df[df["status"] == "ok"].copy()
    if "is_outlier" not in ok.columns:
        ok["is_outlier"] = False
    flagged = ok["is_outlier"].fillna(False)
    fig = go.Figure()
    normal = ok[~flagged]
    if not normal["accepted_fraction"].dropna().empty:
        fig.add_trace(
            go.Box(
                y=normal["accepted_fraction"],
                name="runs",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                marker_color=_CATEGORY_COLORS["ok"],
                text=normal["run_id"],
                hovertemplate="<b>%{text}</b><br>accepted fraction=%{y:.2f}<extra></extra>",
            )
        )
    outliers = ok[flagged]
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=["runs"] * len(outliers),
                y=outliers["accepted_fraction"],
                mode="markers",
                name="outlier",
                text=outliers["run_id"],
                marker={"color": _CATEGORY_COLORS["outlier"], "size": 11, "symbol": "diamond"},
                hovertemplate="<b>%{text}</b><br>accepted fraction=%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Accepted fraction across runs",
        yaxis_title="accepted fraction",
        template="plotly_white",
    )
    return fig


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.3g}"
    return html.escape(str(value))


def _table_html(df: pd.DataFrame) -> str:
    columns = [c for c in _TABLE_COLUMNS if c in df.columns]
    header = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    rows: list[str] = []
    for _, row in df.iterrows():
        if row["status"] != "ok":
            row_class = ' class="failed"'
        elif bool(row.get("is_outlier", False)):
            row_class = ' class="outlier"'
        else:
            row_class = ""
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in columns)
        rows.append(f"<tr{row_class}>{cells}</tr>")
    return (
        f'<table class="summary"><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _fig_to_div(fig: go.Figure, include_js: bool | str) -> str:
    # plotly's stub narrows include_plotlyjs to bool, but "cdn" is valid at runtime.
    return pio.to_html(
        fig,
        include_plotlyjs=include_js,  # pyright: ignore[reportArgumentType]
        full_html=False,
    )


def write_report(df: pd.DataFrame, out_path: str | Path, offline: bool = False) -> Path:
    """Write the group report to ``out_path`` and return it.

    Parameters
    ----------
    df : pandas.DataFrame
        The flagged group table from :func:`tedana_group_stats.build_group_table`.
    out_path : str or pathlib.Path
        Destination HTML file.
    offline : bool, default False
        If True, inline Plotly.js for a fully standalone file (no network to view).
    """
    out_path = Path(out_path)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if df.empty:
        out_path.write_text(
            _TEMPLATE.format(
                version=__version__,
                generated=generated,
                n_runs=0,
                n_outliers=0,
                n_failed=0,
                table="<p>No tedana runs found.</p>",
                charts="",
            ),
            encoding="utf-8",
        )
        return out_path

    work = df.copy()
    work["_category"] = work.apply(_category, axis=1)
    work["_marker_size"] = _marker_sizes(work)

    figures = [_scatter_kappa_rho(work), _bar_components(work), _box_accepted_fraction(work)]
    first_js: bool | str = True if offline else "cdn"
    divs = [_fig_to_div(fig, first_js if i == 0 else False) for i, fig in enumerate(figures)]

    is_outlier = df.get("is_outlier")
    n_outliers = int(is_outlier.fillna(False).sum()) if is_outlier is not None else 0

    out_path.write_text(
        _TEMPLATE.format(
            version=__version__,
            generated=generated,
            n_runs=len(df),
            n_outliers=n_outliers,
            n_failed=int((df["status"] != "ok").sum()),
            table=_table_html(df),
            charts="\n".join(divs),
        ),
        encoding="utf-8",
    )
    return out_path


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tedana group stats</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.2rem; }}
  .meta b {{ color: #222; }}
  .legend span {{ display: inline-block; padding: 2px 8px; margin-right: 8px; border-radius: 3px; }}
  .table-wrap {{ overflow-x: auto; max-height: 70vh; margin-bottom: 2rem; }}
  table.summary {{ border-collapse: collapse; font-size: 0.8rem; }}
  table.summary th, table.summary td {{ border: 1px solid #ddd; padding: 3px 6px;
    text-align: right; white-space: nowrap; }}
  table.summary th {{ background: #f4f4f4; position: sticky; top: 0; }}
  tr.failed {{ background: #f8d7da; }}
  tr.outlier {{ background: #fff3cd; }}
</style>
</head>
<body>
<h1>tedana group stats</h1>
<div class="meta">
  v{version} &middot; generated {generated}<br>
  <b>{n_runs}</b> runs &middot; <b>{n_outliers}</b> flagged as outliers &middot;
  <b>{n_failed}</b> failed to load
</div>
<div class="meta legend">
  <span style="background:#fff3cd">outlier</span>
  <span style="background:#f8d7da">failed</span>
</div>
<div class="table-wrap">
{table}
</div>
{charts}
</body>
</html>
"""
