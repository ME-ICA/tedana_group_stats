"""Discover tedana output directories in a BIDS-derivatives tree.

A completed tedana run always writes a ``{prefix}_desc-tedana_registry.json`` file
(bare ``desc-tedana_registry.json`` when the prefix is empty). That registry is the
reliable, unique marker we glob for. BIDS entities are parsed from the full path so
discovery works whether they live in the filename prefix or the directory hierarchy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REGISTRY_GLOB = "*desc-tedana_registry.json"

# Entity name -> regex capturing its value, and the BIDS key used to render run_id.
_ENTITIES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("subject", "sub", re.compile(r"sub-([a-zA-Z0-9]+)")),
    ("session", "ses", re.compile(r"ses-([a-zA-Z0-9]+)")),
    ("task", "task", re.compile(r"task-([a-zA-Z0-9]+)")),
    ("acquisition", "acq", re.compile(r"acq-([a-zA-Z0-9]+)")),
    ("run", "run", re.compile(r"run-([a-zA-Z0-9]+)")),
)


@dataclass(frozen=True)
class TedanaRun:
    """A discovered tedana output, identified by its registry file."""

    registry: Path
    subject: str | None
    session: str | None
    task: str | None
    acquisition: str | None
    run: str | None

    @property
    def directory(self) -> Path:
        """Directory containing the tedana outputs."""
        return self.registry.parent

    @property
    def run_id(self) -> str:
        """Human-readable, stable identifier built from the BIDS entities.

        Falls back to the parent directory name when no entities are present.
        """
        parts = [
            f"{key}-{getattr(self, attr)}"
            for attr, key, _ in _ENTITIES
            if getattr(self, attr) is not None
        ]
        return "_".join(parts) if parts else self.directory.name


def _parse_entities(path: Path) -> dict[str, str | None]:
    text = str(path)
    return {attr: (m.group(1) if (m := pat.search(text)) else None) for attr, _, pat in _ENTITIES}


def discover_runs(root: str | Path) -> list[TedanaRun]:
    """Find every tedana run under ``root``.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory searched recursively for tedana registry files.

    Returns
    -------
    list of TedanaRun
        One entry per registry found, sorted by path and de-duplicated by resolved
        path for deterministic output.
    """
    root = Path(root)
    seen: set[Path] = set()
    runs: list[TedanaRun] = []
    for registry in sorted(root.rglob(REGISTRY_GLOB)):
        resolved = registry.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        runs.append(TedanaRun(registry=registry, **_parse_entities(registry)))
    return runs
