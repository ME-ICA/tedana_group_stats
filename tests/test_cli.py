from __future__ import annotations

from pathlib import Path

from tedana_group_stats.cli import main


def test_cli_writes_report(derivatives: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    assert main([str(derivatives), "-o", str(out)]) == 0
    assert out.is_file()
    assert "<table" in out.read_text()


def test_cli_no_outputs_returns_1(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 1


def test_cli_bad_root_returns_2(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing")]) == 2
