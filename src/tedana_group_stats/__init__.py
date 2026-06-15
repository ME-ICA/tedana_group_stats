"""tedana_group_stats — group-level QC across tedana multi-echo fMRI outputs."""

from __future__ import annotations

__version__ = "0.1.0"

from tedana_group_stats.aggregate import build_group_table, flag_outliers
from tedana_group_stats.discovery import TedanaRun, discover_runs
from tedana_group_stats.metrics import RunSummary, summarize_run
from tedana_group_stats.report import write_report

__all__ = [
    "RunSummary",
    "TedanaRun",
    "build_group_table",
    "discover_runs",
    "flag_outliers",
    "summarize_run",
    "write_report",
]
