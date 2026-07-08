from __future__ import annotations

from pathlib import Path

import pytest

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


def test_cli_root_that_is_a_file_returns_2(tmp_path: Path) -> None:
    a_file = tmp_path / "afile.txt"
    a_file.write_text("x")
    assert main([str(a_file)]) == 2


def test_cli_success_message_and_default_output(
    derivatives: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main([str(derivatives)]) == 0
    assert (tmp_path / "tedana_group_report.html").is_file()  # default output path
    out = capsys.readouterr().out
    assert "9 runs (1 failed to load)." in out  # sub-04 has a broken tsv


def test_cli_offline_inlines_plotly(derivatives: Path, tmp_path: Path) -> None:
    online = tmp_path / "online.html"
    offline = tmp_path / "offline.html"
    assert main([str(derivatives), "-o", str(online)]) == 0
    assert main([str(derivatives), "-o", str(offline), "--offline"]) == 0
    # Inlined plotly.js makes the offline file far larger than the CDN-linked one.
    assert offline.stat().st_size > online.stat().st_size * 10


def test_cli_threshold_is_passed_through(derivatives: Path, tmp_path: Path) -> None:
    # A huge threshold flags nothing; the report should report 0 outliers.
    out = tmp_path / "r.html"
    assert main([str(derivatives), "-o", str(out), "--threshold", "1e9"]) == 0
    assert "<b>0</b> flagged as outliers" in out.read_text()


def test_cli_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tedana-group-stats" in capsys.readouterr().out
