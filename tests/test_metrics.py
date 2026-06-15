from __future__ import annotations

from pathlib import Path

from tedana_group_stats.discovery import TedanaRun, discover_runs
from tedana_group_stats.metrics import summarize_run


def _run_for(derivatives: Path, subject: str, task: str | None = None) -> TedanaRun:
    for run in discover_runs(derivatives):
        if run.subject == subject and (task is None or run.task == task):
            return run
    raise AssertionError(f"no run for sub-{subject} task-{task}")


def test_summary_counts_and_stats(derivatives: Path) -> None:
    summary = summarize_run(_run_for(derivatives, "01", "rest"))
    assert summary["status"] == "ok"
    assert summary["n_components"] == 5
    assert summary["n_accepted"] == 2
    assert summary["n_rejected"] == 3
    assert summary["accepted_fraction"] == 2 / 5
    assert summary["variance_explained_accepted"] == 35.0
    assert summary["median_kappa"] == 25.0
    assert summary["tedana_version"] == "24.0.1"
    assert summary["kappa_elbow"] == 30.0


def test_broken_run_degrades_without_raising(derivatives: Path) -> None:
    summary = summarize_run(_run_for(derivatives, "04"))
    assert summary["status"] == "failed"
    assert summary["error"]
    assert summary["n_components"] is None
    assert summary["subject"] == "04"  # identifiers still populated


def test_missing_crosscomponent_json(derivatives: Path) -> None:
    summary = summarize_run(_run_for(derivatives, "05"))
    assert summary["status"] == "ok"
    assert summary["n_components"] == 3
    assert summary["kappa_elbow"] is None
    assert summary["rho_elbow"] is None
