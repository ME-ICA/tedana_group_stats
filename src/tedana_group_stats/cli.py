"""Command-line entry point for tedana_group_stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tedana_group_stats import __version__
from tedana_group_stats.aggregate import DEFAULT_THRESHOLD, build_group_table
from tedana_group_stats.discovery import discover_runs
from tedana_group_stats.metrics import summarize_run
from tedana_group_stats.report import write_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tedana-group-stats",
        description=(
            "Aggregate tedana outputs across a BIDS-derivatives tree into a group QC report."
        ),
    )
    parser.add_argument("root", type=Path, help="BIDS-derivatives directory to scan recursively")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tedana_group_report.html"),
        help="output HTML path (default: %(default)s)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="inline plotly.js for a fully standalone file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="modified z-score cutoff for outlier flagging (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Discover tedana runs under ``root``, build the group table, and write the report."""
    args = _build_parser().parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    runs = discover_runs(root)
    if not runs:
        print(f"error: no tedana outputs found under {root}", file=sys.stderr)
        return 1
    summaries = [summarize_run(run) for run in runs]
    table = build_group_table(summaries, threshold=args.threshold)
    out = write_report(table, args.output, offline=args.offline)
    n_failed = sum(summary["status"] != "ok" for summary in summaries)
    print(f"Wrote {out} — {len(runs)} runs ({n_failed} failed to load).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
