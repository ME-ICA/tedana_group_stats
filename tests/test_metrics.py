from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tedana_group_stats.discovery import TedanaRun, discover_runs
from tedana_group_stats.metrics import (
    _blank_summary,
    _fill_crosscomponent_stats,
    _find_crosscomponent,
    _first_present,
    _load_registry,
    _read_tedana_version,
    _resolve,
    _stat,
    _sum_varex,
    _to_float,
    summarize_run,
)


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


def _run(metrics_tsv: Path, registry: Path | None = None) -> TedanaRun:
    return TedanaRun(
        metrics_tsv=metrics_tsv,
        registry=registry,
        subject=None,
        session=None,
        task=None,
        acquisition=None,
        run=None,
    )


# --- _resolve ---------------------------------------------------------------


def test_resolve_direct_hit(tmp_path: Path) -> None:
    (tmp_path / "sub-01_desc-tedana_metrics.tsv").write_text("")
    assert _resolve(tmp_path, "sub-01_desc-tedana_metrics.tsv") is not None


def test_resolve_single_drift_match(tmp_path: Path) -> None:
    (tmp_path / "sub-01_task-rest_desc-tedana_metrics.tsv").write_text("")
    resolved = _resolve(tmp_path, "sub-01_desc-tedana_metrics.tsv")
    assert resolved is not None and resolved.name.startswith("sub-01_task-rest")


def test_resolve_ambiguous_returns_none(tmp_path: Path) -> None:
    (tmp_path / "sub-01_task-a_desc-tedana_metrics.tsv").write_text("")
    (tmp_path / "sub-01_task-b_desc-tedana_metrics.tsv").write_text("")
    assert _resolve(tmp_path, "sub-01_desc-tedana_metrics.tsv") is None


def test_resolve_name_without_desc_returns_none(tmp_path: Path) -> None:
    assert _resolve(tmp_path, "some_other_file.tsv") is None


# --- small pure helpers -----------------------------------------------------


def test_to_float_coercions() -> None:
    assert _to_float("1.5") == 1.5
    assert _to_float(None) is None
    assert _to_float("abc") is None
    assert _to_float([]) is None


def test_first_present_skips_none_but_keeps_zero() -> None:
    assert _first_present({"a": None, "b": 0}, ("a", "b")) == 0
    assert _first_present({"a": 1, "b": 2}, ("a", "b")) == 1  # first key wins
    assert _first_present({}, ("a",)) is None


def test_stat_missing_and_all_nan() -> None:
    assert _stat(pd.DataFrame({"x": [1.0]}), "missing", "median") is None
    assert _stat(pd.DataFrame({"kappa": ["a", "b"]}), "kappa", "median") is None


def test_sum_varex_empty_mask_vs_missing_column() -> None:
    table = pd.DataFrame({"variance explained": [1.0, 2.0]})
    empty_mask = pd.Series([False, False])
    assert _sum_varex(table, empty_mask) == 0.0  # selection empty -> 0.0
    assert _sum_varex(pd.DataFrame({"x": [1.0]}), pd.Series([True])) is None  # column absent


# --- cross-component resolution & parsing -----------------------------------


def test_find_crosscomponent_registry_missing_key(tmp_path: Path) -> None:
    tsv = tmp_path / "sub-01_desc-tedana_metrics.tsv"
    assert _find_crosscomponent({}, tsv) is None  # registry present but no crosscomp key


def test_find_crosscomponent_derived_from_prefix(tmp_path: Path) -> None:
    tsv = tmp_path / "sub-01_desc-tedana_metrics.tsv"
    tsv.write_text("")
    (tmp_path / "sub-01_desc-ICACrossComponent_metrics.json").write_text("{}")
    assert _find_crosscomponent(None, tsv) is not None  # no registry -> derive sibling


def test_find_crosscomponent_derived_absent(tmp_path: Path) -> None:
    tsv = tmp_path / "sub-01_desc-tedana_metrics.tsv"
    assert _find_crosscomponent(None, tsv) is None


def test_crosscomponent_malformed_json_leaves_fields_none(tmp_path: Path) -> None:
    tsv = tmp_path / "sub-01_desc-tedana_metrics.tsv"
    tsv.write_text("")
    (tmp_path / "sub-01_desc-ICACrossComponent_metrics.json").write_text("{ not json")
    summary = _blank_summary(_run(tsv), "ok")
    _fill_crosscomponent_stats(summary, None, tsv)
    assert summary["kappa_elbow"] is None
    assert summary["rho_elbow"] is None


def test_elbow_key_precedence(tmp_path: Path) -> None:
    tsv = tmp_path / "sub-01_desc-tedana_metrics.tsv"
    tsv.write_text("")
    (tmp_path / "sub-01_desc-ICACrossComponent_metrics.json").write_text(
        json.dumps({"rho_elbow_liberal": 9.0, "kappa_elbow": 7.0})
    )
    summary = _blank_summary(_run(tsv), "ok")
    _fill_crosscomponent_stats(summary, None, tsv)
    assert summary["kappa_elbow"] == 7.0  # falls back from kundu to plain
    assert summary["rho_elbow"] == 9.0  # liberal chosen when kundu absent


# --- tedana version walk-up -------------------------------------------------


def test_version_absent_returns_none(tmp_path: Path) -> None:
    assert _read_tedana_version(tmp_path) is None


def test_version_corrupt_nearest_stops_walk(tmp_path: Path) -> None:
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"GeneratedBy": [{"Name": "tedana", "Version": "9.9"}]})
    )
    child = tmp_path / "sub-01" / "func"
    child.mkdir(parents=True)
    (tmp_path / "sub-01" / "dataset_description.json").write_text("{ broken")
    assert _read_tedana_version(child) is None  # nearest corrupt file stops the walk


def test_version_non_tedana_generator(tmp_path: Path) -> None:
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"GeneratedBy": [{"Name": "fmriprep", "Version": "23.0"}]})
    )
    assert _read_tedana_version(tmp_path) is None


def test_version_missing_version_field(tmp_path: Path) -> None:
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"GeneratedBy": [{"Name": "tedana"}]})
    )
    assert _read_tedana_version(tmp_path) is None


# --- registry loading -------------------------------------------------------


def test_load_registry_non_dict_returns_none(tmp_path: Path) -> None:
    reg = tmp_path / "sub-01_desc-tedana_registry.json"
    reg.write_text("[]")
    assert _load_registry(_run(tmp_path / "m.tsv", registry=reg)) is None


def test_load_registry_none_when_absent(tmp_path: Path) -> None:
    assert _load_registry(_run(tmp_path / "m.tsv", registry=None)) is None


def test_load_registry_malformed_json_returns_none(tmp_path: Path) -> None:
    reg = tmp_path / "sub-01_desc-tedana_registry.json"
    reg.write_text("{ not json")
    assert _load_registry(_run(tmp_path / "m.tsv", registry=reg)) is None


# --- summarize_run integration edge cases -----------------------------------


def test_header_only_tsv_is_ok_with_zero_components(tmp_path: Path) -> None:
    func = tmp_path / "sub-09" / "func"
    func.mkdir(parents=True)
    (func / "sub-09_desc-tedana_metrics.tsv").write_text("kappa\trho\tclassification\n")
    summary = summarize_run(_run_for(tmp_path, "09"))
    assert summary["status"] == "ok"
    assert summary["n_components"] == 0
    assert summary["accepted_fraction"] is None  # no div-by-zero


def test_unexpected_labels_counted_as_other(tmp_path: Path) -> None:
    func = tmp_path / "sub-10" / "func"
    func.mkdir(parents=True)
    pd.DataFrame(
        {
            "variance explained": [1.0, 2.0, 3.0],
            "classification": ["accepted", "ignored", "Accepted"],
        }
    ).to_csv(func / "sub-10_desc-tedana_metrics.tsv", sep="\t", index=False)
    summary = summarize_run(_run_for(tmp_path, "10"))
    assert summary["n_accepted"] == 1
    assert summary["n_ignored"] == 1
    assert summary["n_other"] == 1  # "Accepted" (capital) is not "accepted"


def test_tsv_without_classification_column_does_not_raise(tmp_path: Path) -> None:
    func = tmp_path / "sub-11" / "func"
    func.mkdir(parents=True)
    pd.DataFrame(
        {"kappa": [80.0, 25.0], "rho": [10.0, 30.0], "variance explained": [20.0, 8.0]}
    ).to_csv(func / "sub-11_desc-tedana_metrics.tsv", sep="\t", index=False)
    summary = summarize_run(_run_for(tmp_path, "11"))
    assert summary["status"] == "ok"  # never-raise contract holds
    assert summary["n_components"] == 2
    assert summary["n_accepted"] == 0
    assert summary["variance_explained_accepted"] == 0.0


def test_var_exp_rejected_on_accepted_is_read(derivatives: Path) -> None:
    summary = summarize_run(_run_for(derivatives, "01", "rest"))
    assert summary["var_exp_rejected_on_accepted"] == 5.0
