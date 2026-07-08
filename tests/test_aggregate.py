from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tedana_group_stats.aggregate import (
    _dedupe_ids,
    build_group_table,
    flag_outliers,
    modified_z_score,
)
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


def test_modified_z_score_preserves_index() -> None:
    values = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    assert list(modified_z_score(values).index) == ["a", "b", "c"]


def test_threshold_boundary_is_strict() -> None:
    # A score exactly at the threshold is not flagged (comparison is strict >).
    scores = modified_z_score(pd.Series([0.0, 0.0, 0.0, 10.0]))
    at = float(abs(scores.iloc[-1]))
    df = pd.DataFrame({"status": ["ok"] * 4, "m": [0.0, 0.0, 0.0, 10.0]})
    flagged = flag_outliers(df, metrics=("m",), threshold=at)
    assert not flagged["is_outlier_m"].iloc[-1]


def test_dedupe_ids_suffixes_repeats() -> None:
    assert _dedupe_ids(["a", "a", "a", "b"]) == ["a", "a#1", "a#2", "b"]
    assert _dedupe_ids([]) == []


def test_build_group_table_empty_returns_empty() -> None:
    out = build_group_table([])
    assert out.empty
    assert "is_outlier" not in out.columns  # early return, no flag columns added


def test_flag_outliers_creates_column_for_absent_metric() -> None:
    df = pd.DataFrame({"status": ["ok", "ok", "ok"], "n_components": [1, 2, 3]})
    out = flag_outliers(df, metrics=("median_kappa",))
    assert "is_outlier_median_kappa" in out.columns
    assert not out["is_outlier_median_kappa"].any()  # metric absent -> all False


def test_flag_outliers_does_not_mutate_input() -> None:
    df = pd.DataFrame({"status": ["ok", "ok", "ok"], "median_kappa": [10.0, 11.0, 12.0]})
    flag_outliers(df)
    assert "is_outlier" not in df.columns  # worked on a copy


def test_tiny_threshold_flags_all_ok_rows() -> None:
    df = pd.DataFrame({"status": ["ok"] * 5, "median_kappa": [10.0, 11.0, 12.0, 13.0, 14.0]})
    out = flag_outliers(df, metrics=("median_kappa",), threshold=0.0)
    assert out.loc[out["median_kappa"] != 12.0, "is_outlier"].all()  # all non-median flagged


def test_group_table_flags_injected_outlier(derivatives: Path) -> None:
    table = build_group_table([summarize_run(r) for r in discover_runs(derivatives)])
    outlier_row = table[table["subject"] == "03"].iloc[0]
    assert bool(outlier_row["is_outlier"])
    failed_row = table[table["status"] != "ok"].iloc[0]
    assert not bool(failed_row["is_outlier"])  # failed != statistical outlier
