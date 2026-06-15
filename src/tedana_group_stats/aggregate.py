"""Build the group table from run summaries and flag statistical outlier runs.

Outliers use the Iglewicz-Hoaglin modified z-score (median/MAD based), which is robust
for the small samples typical of a single study. Failed runs are excluded from the
statistics and are never counted as statistical outliers.
"""

from __future__ import annotations

import pandas as pd

from tedana_group_stats.metrics import RunSummary

# Metrics whose group distribution is informative for spotting bad runs.
OUTLIER_METRICS: tuple[str, ...] = (
    "accepted_fraction",
    "n_components",
    "median_kappa",
    "variance_explained_accepted",
    "var_exp_rejected_on_accepted",
)
DEFAULT_THRESHOLD = 3.5
_MIN_N = 3
_MAD_SCALE = 0.6745  # 0.75 quantile of N(0, 1); Iglewicz & Hoaglin
_MEANAD_SCALE = 1.2533  # sqrt(pi / 2)


def modified_z_score(values: pd.Series) -> pd.Series:
    """Robust modified z-score, NaN-aware.

    Uses the median absolute deviation (MAD); falls back to the mean-absolute-deviation
    form when MAD is zero. Returns all zeros (no outliers) when fewer than three finite
    values exist or the data has no spread.
    """
    scores = pd.Series(0.0, index=values.index)
    finite = values.dropna()
    if len(finite) < _MIN_N:
        return scores
    median = float(finite.median())
    deviations = (finite - median).abs()
    mad = float(deviations.median())
    if mad > 0:
        scores.loc[finite.index] = _MAD_SCALE * (finite - median) / mad
        return scores
    mean_ad = float(deviations.mean())
    if mean_ad == 0:
        return scores
    scores.loc[finite.index] = (finite - median) / (_MEANAD_SCALE * mean_ad)
    return scores


def flag_outliers(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = OUTLIER_METRICS,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Add ``is_outlier_<metric>`` boolean columns and an OR'd ``is_outlier`` column.

    Only rows with ``status == "ok"`` participate in the statistics and can be flagged.
    """
    df = df.copy()
    is_ok = df["status"] == "ok"
    any_flag = pd.Series(False, index=df.index)
    for metric in metrics:
        column = f"is_outlier_{metric}"
        df[column] = False
        if metric not in df.columns:
            continue
        scores = modified_z_score(df.loc[is_ok, metric])
        flagged = scores.abs() > threshold
        df.loc[flagged.index, column] = flagged
        any_flag = any_flag | df[column]
    df["is_outlier"] = any_flag & is_ok
    return df


def _dedupe_ids(ids: list[str]) -> list[str]:
    """Disambiguate any repeated run_id so the table index/labels stay unique."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for value in ids:
        if value in seen:
            seen[value] += 1
            result.append(f"{value}#{seen[value]}")
        else:
            seen[value] = 0
            result.append(value)
    return result


def build_group_table(
    summaries: list[RunSummary],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Assemble run summaries into a flagged group table (one row per run)."""
    df = pd.DataFrame(list(summaries))
    if df.empty:
        return df
    df["run_id"] = _dedupe_ids(df["run_id"].tolist())
    return flag_outliers(df, threshold=threshold)
