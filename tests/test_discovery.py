from __future__ import annotations

from pathlib import Path

from tedana_group_stats.discovery import discover_runs


def test_finds_all_registries(derivatives: Path) -> None:
    assert len(discover_runs(derivatives)) == 8


def test_parses_bids_entities(derivatives: Path) -> None:
    by_id = {run.run_id: run for run in discover_runs(derivatives)}
    assert "sub-01_ses-01_task-rest" in by_id
    assert "sub-01_ses-01_task-task2" in by_id  # two runs share one session/func dir
    run = by_id["sub-01_ses-01_task-rest"]
    assert (run.subject, run.session, run.task) == ("01", "01", "rest")


def test_empty_prefix_uses_directory_path(derivatives: Path) -> None:
    sub05 = [run for run in discover_runs(derivatives) if run.subject == "05"]
    assert len(sub05) == 1
    assert sub05[0].session == "01"
    assert sub05[0].task is None


def test_discovery_is_deterministic(derivatives: Path) -> None:
    first = [run.run_id for run in discover_runs(derivatives)]
    second = [run.run_id for run in discover_runs(derivatives)]
    assert first == second
