"""Synthetic tedana derivatives tree for tests — registry + TSV/JSON only, no NIfTI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_run(
    func_dir: Path,
    prefix: str,
    classifications: list[str],
    kappas: list[float],
    rhos: list[float],
    var_exps: list[float],
    *,
    crosscomp: bool = True,
    break_tsv: bool = False,
) -> Path:
    func_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""
    metrics_name = f"{pfx}desc-tedana_metrics.tsv"
    cross_name = f"{pfx}desc-ICACrossComponent_metrics.json"
    registry_name = f"{pfx}desc-tedana_registry.json"

    registry: dict[str, str] = {
        "ICA metrics tsv": metrics_name,
        "registry json": registry_name,
    }
    if crosscomp:
        registry["ICA cross component metrics json"] = cross_name
        (func_dir / cross_name).write_text(
            json.dumps(
                {
                    "kappa_elbow_kundu": 30.0,
                    "rho_elbow_kundu": 18.0,
                    "total_var_exp_rejected_components_on_accepted": 5.0,
                }
            )
        )

    if not break_tsv:
        table = pd.DataFrame(
            {
                "Component": [f"ICA_{i:03d}" for i in range(len(classifications))],
                "kappa": kappas,
                "rho": rhos,
                "variance explained": var_exps,
                "classification": classifications,
                "classification_tags": ["" for _ in classifications],
            }
        )
        table.to_csv(func_dir / metrics_name, sep="\t", index=False)

    registry_path = func_dir / registry_name
    registry_path.write_text(json.dumps(registry))
    return registry_path


@pytest.fixture(scope="session")
def derivatives(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build an 8-run derivatives tree covering the edge cases and return its root."""
    root = tmp_path_factory.mktemp("derivatives")
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "test",
                "BIDSVersion": "1.9.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "tedana", "Version": "24.0.1"}],
            }
        )
    )

    # Four healthy runs: 2 subjects x 2 sessions.
    for sub in ("01", "02"):
        for ses in ("01", "02"):
            _write_run(
                root / f"sub-{sub}" / f"ses-{ses}" / "func",
                f"sub-{sub}_ses-{ses}_task-rest",
                classifications=["accepted", "accepted", "rejected", "rejected", "rejected"],
                kappas=[80.0, 60.0, 25.0, 20.0, 15.0],
                rhos=[10.0, 12.0, 30.0, 28.0, 26.0],
                var_exps=[20.0, 15.0, 8.0, 7.0, 6.0],
            )

    # A second task in one session -> two runs share a func dir (multi-run per session).
    _write_run(
        root / "sub-01" / "ses-01" / "func",
        "sub-01_ses-01_task-task2",
        classifications=["accepted", "accepted", "accepted", "rejected"],
        kappas=[70.0, 65.0, 55.0, 22.0],
        rhos=[11.0, 13.0, 14.0, 29.0],
        var_exps=[18.0, 16.0, 14.0, 9.0],
    )

    # An outlier run: many components, almost nothing accepted, low kappa.
    _write_run(
        root / "sub-03" / "ses-01" / "func",
        "sub-03_ses-01_task-rest",
        classifications=["accepted"] + ["rejected"] * 19,
        kappas=[12.0] + [10.0] * 19,
        rhos=[40.0] + [38.0] * 19,
        var_exps=[2.0] + [4.0] * 19,
    )

    # A broken run: registry present, metrics tsv missing.
    _write_run(
        root / "sub-04" / "ses-01" / "func",
        "sub-04_ses-01_task-rest",
        classifications=["accepted"],
        kappas=[50.0],
        rhos=[10.0],
        var_exps=[10.0],
        break_tsv=True,
    )

    # An empty-prefix run: entities come from the directory path; no crosscomponent json.
    _write_run(
        root / "sub-05" / "ses-01" / "func",
        "",
        classifications=["accepted", "rejected", "rejected"],
        kappas=[55.0, 20.0, 18.0],
        rhos=[12.0, 30.0, 28.0],
        var_exps=[14.0, 8.0, 7.0],
        crosscomp=False,
    )

    return root
