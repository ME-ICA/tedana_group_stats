from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def test_registryless_run_summarizes_component_stats(derivatives: Path) -> None:
    """A run with only a metrics tsv (no registry) still yields component metrics."""
    run = _run_for(derivatives, "06")
    assert run.registry is None
    summary = summarize_run(run)
    assert summary["status"] == "ok"
    assert summary["n_components"] == 3
    assert summary["n_accepted"] == 2
    assert summary["median_kappa"] == 58.0
    assert summary["kappa_elbow"] is None  # crosscomponent json is absent in this fixture
    assert summary["rho_elbow"] is None


def test_bids_renamed_files_resolve_despite_registry_prefix_drift(tmp_path: Path) -> None:
    """Registry records a bare tedana prefix; on-disk cross-component JSON gained a BIDS entity.

    Mirrors ds000258, where files were renamed to add ``task-rest`` after tedana ran, so the
    cross-component name recorded in the registry no longer matches disk. The metrics tsv is
    read directly off disk now, so it needs no resolution; the ``_resolve`` tail-match still
    matters for the registry-recorded cross-component JSON.
    """
    func = tmp_path / "sub-06" / "func"
    func.mkdir(parents=True)
    registry_name = "sub-06_task-rest_desc-tedana_registry.json"
    # Registry references the bare prefix (no task entity).
    (func / registry_name).write_text(
        json.dumps(
            {
                "ICA metrics tsv": "sub-06_desc-tedana_metrics.tsv",
                "ICA cross component metrics json": "sub-06_desc-ICACrossComponent_metrics.json",
            }
        )
    )
    # Files on disk carry the extra task-rest entity.
    pd.DataFrame(
        {
            "kappa": [80.0, 25.0],
            "rho": [10.0, 30.0],
            "variance explained": [20.0, 8.0],
            "classification": ["accepted", "rejected"],
        }
    ).to_csv(func / "sub-06_task-rest_desc-tedana_metrics.tsv", sep="\t", index=False)
    (func / "sub-06_task-rest_desc-ICACrossComponent_metrics.json").write_text(
        json.dumps({"kappa_elbow_kundu": 30.0, "rho_elbow_kundu": 18.0})
    )

    summary = summarize_run(_run_for(tmp_path, "06"))
    assert summary["status"] == "ok"
    assert summary["n_components"] == 2
    assert summary["n_accepted"] == 1
    assert summary["kappa_elbow"] == 30.0
    assert summary["rho_elbow"] == 18.0


def test_empty_prefix_registry_names_resolve_despite_drift(tmp_path: Path) -> None:
    """Registry records empty-prefix ``desc-..`` names; on-disk files gained a BIDS entity.

    tedana can emit prefix-less outputs (``desc-ICACrossComponent_metrics.json``); after a
    rename those drift to ``sub-07_task-rest_desc-..`` on disk. The cross-component resolver
    must match on the ``desc-`` tail even without a leading underscore.
    """
    func = tmp_path / "sub-07" / "func"
    func.mkdir(parents=True)
    (func / "sub-07_task-rest_desc-tedana_registry.json").write_text(
        json.dumps(
            {
                "ICA metrics tsv": "desc-tedana_metrics.tsv",
                "ICA cross component metrics json": "desc-ICACrossComponent_metrics.json",
            }
        )
    )
    pd.DataFrame(
        {
            "kappa": [80.0, 25.0],
            "rho": [10.0, 30.0],
            "variance explained": [20.0, 8.0],
            "classification": ["accepted", "rejected"],
        }
    ).to_csv(func / "sub-07_task-rest_desc-tedana_metrics.tsv", sep="\t", index=False)
    (func / "sub-07_task-rest_desc-ICACrossComponent_metrics.json").write_text(
        json.dumps({"kappa_elbow_kundu": 30.0, "rho_elbow_kundu": 18.0})
    )

    summary = summarize_run(_run_for(tmp_path, "07"))
    assert summary["status"] == "ok"
    assert summary["n_components"] == 2
    assert summary["n_accepted"] == 1
    assert summary["kappa_elbow"] == 30.0
    assert summary["rho_elbow"] == 18.0
