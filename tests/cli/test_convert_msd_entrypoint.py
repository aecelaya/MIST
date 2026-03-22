"""Tests for mist.cli.convert_msd_entrypoint."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from mist.cli import convert_msd_entrypoint as entry


# ---------------------------------------------------------------------------
# _parse_convert_msd_args
# ---------------------------------------------------------------------------

class TestParseConvertMsdArgs:
    """Tests for convert_msd_entrypoint._parse_convert_msd_args."""

    def test_required_args_are_parsed(self, tmp_path):
        """--source and --output are captured correctly."""
        ns = entry._parse_convert_msd_args([
            "--source", str(tmp_path / "msd"),
            "--output", str(tmp_path / "out"),
        ])
        assert ns.source == str(tmp_path / "msd")
        assert ns.output == str(tmp_path / "out")

    def test_num_workers_defaults_to_one(self, tmp_path):
        """--num-workers-conversion is optional and defaults to 1."""
        ns = entry._parse_convert_msd_args([
            "--source", str(tmp_path / "msd"),
            "--output", str(tmp_path / "out"),
        ])
        assert ns.num_workers_conversion == 1

    def test_num_workers_is_parsed(self, tmp_path):
        """--num-workers-conversion is captured as an integer."""
        ns = entry._parse_convert_msd_args([
            "--source", str(tmp_path / "msd"),
            "--output", str(tmp_path / "out"),
            "--num-workers-conversion", "4",
        ])
        assert ns.num_workers_conversion == 4

    def test_missing_source_exits(self, tmp_path):
        """Omitting --source causes SystemExit."""
        with pytest.raises(SystemExit):
            entry._parse_convert_msd_args(["--output", str(tmp_path / "out")])

    def test_missing_output_exits(self, tmp_path):
        """Omitting --output causes SystemExit."""
        with pytest.raises(SystemExit):
            entry._parse_convert_msd_args(["--source", str(tmp_path / "msd")])


# ---------------------------------------------------------------------------
# run_convert_msd
# ---------------------------------------------------------------------------

class TestRunConvertMsd:
    """Tests for convert_msd_entrypoint.run_convert_msd."""

    def test_creates_output_dir_and_calls_converter(self, monkeypatch, tmp_path):
        """run_convert_msd creates the output directory and calls convert_msd."""
        captured = {}

        monkeypatch.setattr(
            entry,
            "convert_msd",
            lambda source, output, max_workers=None: captured.update(
                source=source, output=output, max_workers=max_workers
            ),
            raising=True,
        )

        ns = argparse.Namespace(
            source=str(tmp_path / "msd"),
            output=str(tmp_path / "out"),
            num_workers_conversion=None,
        )
        entry.run_convert_msd(ns)

        assert (tmp_path / "out").is_dir()
        assert captured["source"] == (tmp_path / "msd").resolve()
        assert captured["output"] == (tmp_path / "out").resolve()
        assert captured["max_workers"] is None

    def test_num_workers_forwarded(self, monkeypatch, tmp_path):
        """num_workers is passed through to convert_msd."""
        captured = {}

        monkeypatch.setattr(
            entry,
            "convert_msd",
            lambda source, output, max_workers=None: captured.update(
                max_workers=max_workers
            ),
            raising=True,
        )

        ns = argparse.Namespace(
            source=str(tmp_path / "msd"),
            output=str(tmp_path / "out"),
            num_workers_conversion=8,
        )
        entry.run_convert_msd(ns)
        assert captured["max_workers"] == 8


# ---------------------------------------------------------------------------
# convert_msd_entry (integration)
# ---------------------------------------------------------------------------

class TestConvertMsdEntry:
    """Tests for convert_msd_entrypoint.convert_msd_entry."""

    def test_parses_then_runs(self, monkeypatch):
        """convert_msd_entry calls _parse_convert_msd_args then run_convert_msd."""
        ns = SimpleNamespace(source="/src", output="/out", num_workers_conversion=None)
        called = {"parsed": False, "ran": False}

        monkeypatch.setattr(
            entry,
            "_parse_convert_msd_args",
            lambda argv=None: (
                called.__setitem__("parsed", True) or ns
            ),
            raising=True,
        )
        monkeypatch.setattr(
            entry,
            "run_convert_msd",
            lambda n: called.__setitem__("ran", True),
            raising=True,
        )

        entry.convert_msd_entry(["--source", "/src", "--output", "/out"])

        assert called["parsed"] is True
        assert called["ran"] is True
