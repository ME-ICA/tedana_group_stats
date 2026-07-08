from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tedana_group_stats.aggregate import build_group_table, modified_z_score
from tedana_group_stats.discovery import discover_runs
from tedana_group_stats.metrics import summarize_run


def test_modified_z_score_flags_single_outlier() -> None:
    scores = modified_z_score(pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 100.0]))
    assert abs(scores.iloc[-1]) > 3.5
    assert (scores.iloc[:-1].abs() < 3.5).all()


def test_mad_zero_falls_back_to_mean_ad() -> None:
    scores = modified_z_score(pd.Series([5.0, 5.0, 5.0, 5.0, 9.0]))
    assert abs(scores.iloc[-1]) > 3.5


def test_no_spread_yields_no_outliers() -> None:
    assert (modified_z_score(pd.Series([5.0, 5.0, 5.0, 5.0])) == 0).all()


def test_small_sample_yields_no_outliers() -> None:
    assert (modified_z_score(pd.Series([1.0, 100.0])) == 0).all()


def test_nan_is_excluded_not_crashing() -> None:
    scores = modified_z_score(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]))
    assert scores.iloc[-1] == 0


def test_group_table_flags_injected_outlier(derivatives: Path) -> None:
    table = build_group_table([summarize_run(r) for r in discover_runs(derivatives)])
    outlier_row = table[table["subject"] == "03"].iloc[0]
    assert bool(outlier_row["is_outlier"])
    failed_row = table[table["status"] != "ok"].iloc[0]
    assert not bool(failed_row["is_outlier"])  # failed != statistical outlier


def test_new_metrics_get_outlier_columns(derivatives: Path) -> None:
    table = build_group_table([summarize_run(r) for r in discover_runs(derivatives)])
    for metric in ("rejected_variance_fraction", "kappa_rho_ratio"):
        assert f"is_outlier_{metric}" in table.columns
