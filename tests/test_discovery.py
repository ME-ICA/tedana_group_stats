from __future__ import annotations

from pathlib import Path

import pytest

from tedana_group_stats.discovery import TedanaRun, _parse_entities, discover_runs


def _touch_metrics(directory: Path, prefix: str) -> Path:
    """Create an (empty) metrics TSV so discovery finds a run; content is irrelevant here."""
    directory.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""
    path = directory / f"{pfx}desc-tedana_metrics.tsv"
    path.write_text("")
    return path


def test_finds_all_runs(derivatives: Path) -> None:
    assert len(discover_runs(derivatives)) == 9


def test_finds_registryless_run(derivatives: Path) -> None:
    sub06 = [run for run in discover_runs(derivatives) if run.subject == "06"]
    assert len(sub06) == 1
    assert sub06[0].registry is None  # discovered by its metrics tsv alone
    assert sub06[0].metrics_tsv.name.endswith("desc-tedana_metrics.tsv")


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


def test_run_id_falls_back_to_directory_name(tmp_path: Path) -> None:
    run = TedanaRun(
        metrics_tsv=tmp_path / "myrun" / "desc-tedana_metrics.tsv",
        registry=None,
        subject=None,
        session=None,
        task=None,
        acquisition=None,
        run=None,
    )
    assert run.run_id == "myrun"  # no BIDS entities -> parent directory name


def test_parses_acq_and_run_entities_in_order(tmp_path: Path) -> None:
    _touch_metrics(tmp_path / "func", "sub-01_ses-02_task-rest_acq-hi_run-3")
    (run,) = discover_runs(tmp_path)
    assert (run.subject, run.session, run.task, run.acquisition, run.run) == (
        "01",
        "02",
        "rest",
        "hi",
        "3",
    )
    assert run.run_id == "sub-01_ses-02_task-rest_acq-hi_run-3"  # fixed entity order


def test_entities_parsed_from_directory_path(tmp_path: Path) -> None:
    entities = _parse_entities(tmp_path / "sub-07" / "ses-01" / "func" / "desc-tedana_metrics.tsv")
    assert entities["subject"] == "07"
    assert entities["session"] == "01"
    assert entities["task"] is None


def test_dedupes_symlinked_duplicate(tmp_path: Path) -> None:
    real = _touch_metrics(tmp_path / "real" / "func", "sub-01_task-rest")
    link_dir = tmp_path / "mirror" / "func"
    link_dir.mkdir(parents=True)
    link = link_dir / real.name
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    # Both the real file and its symlink resolve to one file -> a single run.
    assert len(discover_runs(tmp_path)) == 1


def test_registry_that_is_a_directory_is_ignored(tmp_path: Path) -> None:
    func = tmp_path / "func"
    _touch_metrics(func, "sub-01_task-rest")
    (func / "sub-01_task-rest_desc-tedana_registry.json").mkdir()  # a dir, not a file
    (run,) = discover_runs(tmp_path)
    assert run.registry is None  # .is_file() guard rejects the directory


def test_multiple_runs_in_one_dir_pair_own_registry(tmp_path: Path) -> None:
    func = tmp_path / "func"
    _touch_metrics(func, "sub-01_task-rest")
    _touch_metrics(func, "sub-01_task-nback")
    (func / "sub-01_task-rest_desc-tedana_registry.json").write_text("{}")
    # nback run has no registry on disk.
    by_task = {run.task: run for run in discover_runs(tmp_path)}
    assert by_task["rest"].registry is not None
    assert by_task["nback"].registry is None


def test_accepts_str_root(derivatives: Path) -> None:
    assert len(discover_runs(str(derivatives))) == 9


def test_nonexistent_root_returns_empty(tmp_path: Path) -> None:
    assert discover_runs(tmp_path / "does-not-exist") == []
