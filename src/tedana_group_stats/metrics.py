"""Load a single tedana run and reduce it to one summary row.

``summarize_run`` is deliberately total: it never raises. A run whose outputs cannot
be read yields a summary with ``status="failed"`` and ``None`` metrics but valid
identifiers, so one broken run never aborts a group scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd

from tedana_group_stats.discovery import TedanaRun

# Registry keys are tedana's stable public contract; see
# tedana/resources/config/outputs.json. Centralized so a future rename is one edit.
ICA_METRICS_TSV_KEY = "ICA metrics tsv"
ICA_CROSSCOMP_KEY = "ICA cross component metrics json"

# Cross-component JSON keys are decision-tree dependent, so every one is optional.
_KAPPA_ELBOW_KEYS = ("kappa_elbow_kundu", "kappa_elbow")
_RHO_ELBOW_KEYS = ("rho_elbow_kundu", "rho_elbow_liberal", "rho_elbow")
_VAR_EXP_REJECTED_KEY = "total_var_exp_rejected_components_on_accepted"

_VAREX_COL = "variance explained"


class RunSummary(TypedDict):
    """One row of the group table: identifiers + reduced metrics for a single run."""

    run_id: str
    subject: str | None
    session: str | None
    task: str | None
    run: str | None
    path: str
    status: str  # "ok" or "failed"
    error: str | None
    tedana_version: str | None
    n_components: int | None
    n_accepted: int | None
    n_rejected: int | None
    n_ignored: int | None
    n_other: int | None
    accepted_fraction: float | None
    variance_explained_accepted: float | None
    variance_explained_rejected: float | None
    median_kappa: float | None
    max_kappa: float | None
    median_rho: float | None
    max_rho: float | None
    kappa_elbow: float | None
    rho_elbow: float | None
    var_exp_rejected_on_accepted: float | None


def _blank_summary(run: TedanaRun, status: str, error: str | None = None) -> RunSummary:
    return RunSummary(
        run_id=run.run_id,
        subject=run.subject,
        session=run.session,
        task=run.task,
        run=run.run,
        path=str(run.directory),
        status=status,
        error=error,
        tedana_version=None,
        n_components=None,
        n_accepted=None,
        n_rejected=None,
        n_ignored=None,
        n_other=None,
        accepted_fraction=None,
        variance_explained_accepted=None,
        variance_explained_rejected=None,
        median_kappa=None,
        max_kappa=None,
        median_rho=None,
        max_rho=None,
        kappa_elbow=None,
        rho_elbow=None,
        var_exp_rejected_on_accepted=None,
    )


def _resolve(directory: Path, name: str) -> Path | None:
    """Locate a registry-referenced output, tolerating BIDS prefix drift.

    tedana records bare-prefix names (``sub-04_desc-..``); a downstream BIDS rename
    can insert entities into the on-disk files (``sub-04_task-rest_desc-..``). When the
    recorded name is missing, fall back to the descriptor tail, accepting the match only
    when exactly one file qualifies (so we never guess between ambiguous candidates).
    """
    direct = directory / name
    if direct.is_file():
        return direct
    _, sep, tail = name.partition("_desc-")
    if sep:
        matches = sorted(directory.glob(f"*_desc-{tail}"))
        if len(matches) == 1:
            return matches[0]
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _numeric(values: Any) -> pd.Series:
    """Coerce to a numeric Series; the cast isolates pandas' noisy to_numeric typing."""
    return cast("pd.Series", pd.to_numeric(values, errors="coerce"))


def _stat(table: pd.DataFrame, col: str, op: str) -> float | None:
    if col not in table.columns:
        return None
    series = _numeric(table[col]).dropna()
    if series.empty:
        return None
    return float(getattr(series, op)())


def _sum_varex(table: pd.DataFrame, mask: pd.Series) -> float | None:
    if _VAREX_COL not in table.columns:
        return None
    return float(cast("float", _numeric(table.loc[mask, _VAREX_COL]).sum()))


def _fill_component_stats(summary: RunSummary, table: pd.DataFrame) -> None:
    n = len(table)
    summary["n_components"] = n
    classification = (
        table["classification"].astype(str)
        if "classification" in table.columns
        else pd.Series([], dtype=str)
    )
    n_accepted = int((classification == "accepted").sum())
    n_rejected = int((classification == "rejected").sum())
    n_ignored = int((classification == "ignored").sum())
    summary["n_accepted"] = n_accepted
    summary["n_rejected"] = n_rejected
    summary["n_ignored"] = n_ignored
    summary["n_other"] = n - n_accepted - n_rejected - n_ignored
    summary["accepted_fraction"] = (n_accepted / n) if n else None
    summary["variance_explained_accepted"] = _sum_varex(table, classification == "accepted")
    summary["variance_explained_rejected"] = _sum_varex(table, classification == "rejected")
    summary["median_kappa"] = _stat(table, "kappa", "median")
    summary["max_kappa"] = _stat(table, "kappa", "max")
    summary["median_rho"] = _stat(table, "rho", "median")
    summary["max_rho"] = _stat(table, "rho", "max")


def _fill_crosscomponent_stats(
    summary: RunSummary, registry: dict[str, Any], directory: Path
) -> None:
    name = registry.get(ICA_CROSSCOMP_KEY)
    if not name:
        return
    path = _resolve(directory, name)
    if path is None:
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    summary["kappa_elbow"] = _to_float(_first_present(data, _KAPPA_ELBOW_KEYS))
    summary["rho_elbow"] = _to_float(_first_present(data, _RHO_ELBOW_KEYS))
    summary["var_exp_rejected_on_accepted"] = _to_float(data.get(_VAR_EXP_REJECTED_KEY))


def _read_tedana_version(directory: Path) -> str | None:
    """Find the nearest ``dataset_description.json`` walking up and read tedana's version."""
    for parent in (directory, *directory.parents):
        description = parent / "dataset_description.json"
        if not description.is_file():
            continue
        try:
            data = json.loads(description.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        for generator in data.get("GeneratedBy", []):
            if isinstance(generator, dict) and "tedana" in str(generator.get("Name", "")).lower():
                version = generator.get("Version")
                return str(version) if version is not None else None
        return None
    return None


def summarize_run(run: TedanaRun) -> RunSummary:
    """Reduce a single tedana run to a summary row, degrading gracefully on failure."""
    try:
        registry = json.loads(run.registry.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _blank_summary(run, "failed", f"unreadable registry: {exc}")

    tsv_name = registry.get(ICA_METRICS_TSV_KEY)
    if not tsv_name:
        return _blank_summary(run, "failed", f"registry missing key {ICA_METRICS_TSV_KEY!r}")

    tsv_path = _resolve(run.directory, tsv_name)
    if tsv_path is None:
        return _blank_summary(run, "failed", f"metrics tsv not found: {tsv_name}")

    try:
        table = pd.read_csv(tsv_path, sep="\t")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return _blank_summary(run, "failed", f"unreadable metrics tsv: {exc}")

    summary = _blank_summary(run, "ok")
    summary["tedana_version"] = _read_tedana_version(run.directory)
    _fill_component_stats(summary, table)
    _fill_crosscomponent_stats(summary, registry, run.directory)
    return summary
