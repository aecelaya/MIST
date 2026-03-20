"""Tests for mist.cli.convert_csv_entrypoint."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from mist.cli import convert_csv_entrypoint as entry


# ---------------------------------------------------------------------------
# _parse_convert_csv_args
# ---------------------------------------------------------------------------

class TestParseConvertCsvArgs:
    """Tests for convert_csv_entrypoint._parse_convert_csv_args."""

    def test_required_args_are_parsed(self, tmp_path):
        """--train-csv and --output are captured correctly."""
        ns = entry._parse_convert_csv_args([
            "--train-csv", str(tmp_path / "train.csv"),
            "--output", str(tmp_path / "out"),
        ])
        assert ns.train_csv == str(tmp_path / "train.csv")
        assert ns.output == str(tmp_path / "out")

    def test_test_csv_defaults_to_none(self, tmp_path):
        """--test-csv is optional and defaults to None."""
        ns = entry._parse_convert_csv_args([
            "--train-csv", str(tmp_path / "train.csv"),
            "--output", str(tmp_path / "out"),
        ])
        assert ns.test_csv is None

    def test_test_csv_is_parsed(self, tmp_path):
        """--test-csv is captured when provided."""
        ns = entry._parse_convert_csv_args([
            "--train-csv", str(tmp_path / "train.csv"),
            "--output", str(tmp_path / "out"),
            "--test-csv", str(tmp_path / "test.csv"),
        ])
        assert ns.test_csv == str(tmp_path / "test.csv")

    def test_num_workers_defaults_to_one(self, tmp_path):
        """--num-workers is optional and defaults to None."""
        ns = entry._parse_convert_csv_args([
            "--train-csv", str(tmp_path / "train.csv"),
            "--output", str(tmp_path / "out"),
        ])
        assert ns.num_workers == 1

    def test_num_workers_is_parsed(self, tmp_path):
        """--num-workers is captured as an integer."""
        ns = entry._parse_convert_csv_args([
            "--train-csv", str(tmp_path / "train.csv"),
            "--output", str(tmp_path / "out"),
            "--num-workers", "4",
        ])
        assert ns.num_workers == 4

    def test_missing_train_csv_exits(self, tmp_path):
        """Omitting --train-csv causes SystemExit."""
        with pytest.raises(SystemExit):
            entry._parse_convert_csv_args(["--output", str(tmp_path / "out")])

    def test_missing_output_exits(self, tmp_path):
        """Omitting --output causes SystemExit."""
        with pytest.raises(SystemExit):
            entry._parse_convert_csv_args(
                ["--train-csv", str(tmp_path / "train.csv")]
            )


# ---------------------------------------------------------------------------
# run_convert_csv
# ---------------------------------------------------------------------------

class TestRunConvertCsv:
    """Tests for convert_csv_entrypoint.run_convert_csv."""

    def test_creates_output_dir_and_calls_converter(self, monkeypatch, tmp_path):
        """run_convert_csv creates the output directory and calls convert_csv."""
        captured = {}

        monkeypatch.setattr(
            entry,
            "convert_csv",
            lambda train_csv, output, test_csv=None, max_workers=None: (
                captured.update(
                    train_csv=train_csv, output=output,
                    test_csv=test_csv, max_workers=max_workers,
                )
            ),
            raising=True,
        )

        ns = argparse.Namespace(
            train_csv=str(tmp_path / "train.csv"),
            output=str(tmp_path / "out"),
            test_csv=None,
            num_workers=None,
        )
        entry.run_convert_csv(ns)

        assert (tmp_path / "out").is_dir()
        assert captured["train_csv"] == (tmp_path / "train.csv").resolve()
        assert captured["output"] == (tmp_path / "out").resolve()
        assert captured["test_csv"] is None
        assert captured["max_workers"] is None

    def test_test_csv_resolved_when_provided(self, monkeypatch, tmp_path):
        """test_csv is resolved to a Path when provided."""
        captured = {}

        monkeypatch.setattr(
            entry,
            "convert_csv",
            lambda train_csv, output, test_csv=None, max_workers=None: (
                captured.update(test_csv=test_csv)
            ),
            raising=True,
        )

        ns = argparse.Namespace(
            train_csv=str(tmp_path / "train.csv"),
            output=str(tmp_path / "out"),
            test_csv=str(tmp_path / "test.csv"),
            num_workers=None,
        )
        entry.run_convert_csv(ns)
        assert captured["test_csv"] == (tmp_path / "test.csv").resolve()

    def test_num_workers_forwarded(self, monkeypatch, tmp_path):
        """num_workers is passed through to convert_csv."""
        captured = {}

        monkeypatch.setattr(
            entry,
            "convert_csv",
            lambda train_csv, output, test_csv=None, max_workers=None: (
                captured.update(max_workers=max_workers)
            ),
            raising=True,
        )

        ns = argparse.Namespace(
            train_csv=str(tmp_path / "train.csv"),
            output=str(tmp_path / "out"),
            test_csv=None,
            num_workers=8,
        )
        entry.run_convert_csv(ns)
        assert captured["max_workers"] == 8


# ---------------------------------------------------------------------------
# convert_csv_entry (integration)
# ---------------------------------------------------------------------------

class TestConvertCsvEntry:
    """Tests for convert_csv_entrypoint.convert_csv_entry."""

    def test_parses_then_runs(self, monkeypatch):
        """convert_csv_entry calls _parse_convert_csv_args then run_convert_csv."""
        ns = SimpleNamespace(
            train_csv="/train.csv", output="/out",
            test_csv=None, num_workers=None,
        )
        called = {"parsed": False, "ran": False}

        monkeypatch.setattr(
            entry,
            "_parse_convert_csv_args",
            lambda argv=None: (
                called.__setitem__("parsed", True) or ns
            ),
            raising=True,
        )
        monkeypatch.setattr(
            entry,
            "run_convert_csv",
            lambda n: called.__setitem__("ran", True),
            raising=True,
        )

        entry.convert_csv_entry(
            ["--train-csv", "/train.csv", "--output", "/out"]
        )

        assert called["parsed"] is True
        assert called["ran"] is True
