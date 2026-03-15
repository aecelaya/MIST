"""Tests for mist.cli.evaluation_entrypoint."""
import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from mist.cli import evaluation_entrypoint as entry


# ---------------------------------------------------------------------------
# _parse_eval_args
# ---------------------------------------------------------------------------

class TestParseEvalArgs:
    """Tests for evaluation_entrypoint._parse_eval_args."""

    def test_required_args_are_parsed(self, tmp_path):
        """All required flags are captured in the returned Namespace."""
        cfg = tmp_path / "config.json"
        csv = tmp_path / "paths.csv"
        out = tmp_path / "out" / "metrics.csv"

        ns = entry._parse_eval_args([
            "--config", str(cfg),
            "--paths-csv", str(csv),
            "--output-csv", str(out),
        ])

        assert ns.config == str(cfg)
        assert ns.paths_csv == str(csv)
        assert ns.output_csv == str(out)

    def test_num_workers_defaults_to_none(self, tmp_path):
        """--num-workers is optional and defaults to None."""
        ns = entry._parse_eval_args([
            "--config", str(tmp_path / "c.json"),
            "--paths-csv", str(tmp_path / "p.csv"),
            "--output-csv", str(tmp_path / "o.csv"),
        ])
        assert ns.num_workers is None

    def test_num_workers_is_parsed(self, tmp_path):
        """--num-workers is captured as an integer."""
        ns = entry._parse_eval_args([
            "--config", str(tmp_path / "c.json"),
            "--paths-csv", str(tmp_path / "p.csv"),
            "--output-csv", str(tmp_path / "o.csv"),
            "--num-workers", "4",
        ])
        assert ns.num_workers == 4

    def test_missing_required_arg_exits(self, tmp_path):
        """Omitting a required flag causes SystemExit."""
        with pytest.raises(SystemExit):
            entry._parse_eval_args([
                "--paths-csv", str(tmp_path / "p.csv"),
                "--output-csv", str(tmp_path / "o.csv"),
                # --config is missing
            ])


# ---------------------------------------------------------------------------
# _ensure_output_dir
# ---------------------------------------------------------------------------

class TestEnsureOutputDir:
    """Tests for evaluation_entrypoint._ensure_output_dir."""

    def test_creates_parent_directory(self, tmp_path):
        """Creates nested parent directories for the output CSV."""
        out_csv = tmp_path / "nested" / "dir" / "metrics.csv"
        entry._ensure_output_dir(out_csv)
        assert out_csv.parent.is_dir()

    def test_existing_directory_does_not_raise(self, tmp_path):
        """Calling on an already-existing parent directory is a no-op."""
        out_csv = tmp_path / "metrics.csv"
        entry._ensure_output_dir(out_csv)  # tmp_path already exists
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    """Tests for evaluation_entrypoint.run_evaluation."""

    def _make_ns(self, tmp_path, num_workers=None):
        """Return a minimal Namespace for run_evaluation."""
        return argparse.Namespace(
            config=str(tmp_path / "cfg.json"),
            paths_csv=str(tmp_path / "paths.csv"),
            output_csv=str(tmp_path / "out" / "metrics.csv"),
            num_workers=num_workers,
        )

    def test_evaluator_constructed_with_correct_args(
        self, monkeypatch, tmp_path
    ):
        """Evaluator receives filepaths_dataframe, evaluation_config, and
        output_csv_path derived from the Namespace."""
        fake_config = {
            "evaluation": {"tumor": {"labels": [1], "metrics": {"dice": {}}}}
        }
        monkeypatch.setattr(
            entry.io, "read_json_file", lambda _: fake_config, raising=True
        )
        monkeypatch.setattr(
            entry.pd, "read_csv",
            lambda _: pd.DataFrame([{"id": "p1"}]),
            raising=True,
        )

        captured = {}

        class _EvalStub:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, max_workers=None):
                captured["max_workers"] = max_workers

        monkeypatch.setattr(entry, "Evaluator", _EvalStub, raising=True)

        entry.run_evaluation(self._make_ns(tmp_path))

        assert isinstance(captured["filepaths_dataframe"], pd.DataFrame)
        assert captured["evaluation_config"] == fake_config["evaluation"]
        assert "metrics.csv" in captured["output_csv_path"]
        assert captured["max_workers"] is None

    def test_num_workers_forwarded_to_run(self, monkeypatch, tmp_path):
        """The num_workers value from the Namespace is passed to run()."""
        monkeypatch.setattr(
            entry.io, "read_json_file",
            lambda _: {"tumor": {"labels": [1], "metrics": {"dice": {}}}},
            raising=True,
        )
        monkeypatch.setattr(
            entry.pd, "read_csv",
            lambda _: pd.DataFrame([{"id": "p1"}]),
            raising=True,
        )

        captured = {}

        class _EvalStub:
            def __init__(self, **kwargs):
                pass

            def run(self, max_workers=None):
                captured["max_workers"] = max_workers

        monkeypatch.setattr(entry, "Evaluator", _EvalStub, raising=True)

        entry.run_evaluation(self._make_ns(tmp_path, num_workers=8))

        assert captured["max_workers"] == 8

    def test_config_without_evaluation_key_used_directly(
        self, monkeypatch, tmp_path
    ):
        """If the JSON has no 'evaluation' key the full dict is forwarded."""
        flat_config = {
            "tumor": {"labels": [1], "metrics": {"dice": {}}}
        }
        monkeypatch.setattr(
            entry.io, "read_json_file", lambda _: flat_config, raising=True
        )
        monkeypatch.setattr(
            entry.pd, "read_csv",
            lambda _: pd.DataFrame([{"id": "p1"}]),
            raising=True,
        )

        captured = {}

        class _EvalStub:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, max_workers=None):
                pass

        monkeypatch.setattr(entry, "Evaluator", _EvalStub, raising=True)

        entry.run_evaluation(self._make_ns(tmp_path))

        assert captured["evaluation_config"] == flat_config

    def test_output_directory_is_created(self, monkeypatch, tmp_path):
        """run_evaluation creates the output directory if it is absent."""
        monkeypatch.setattr(
            entry.io, "read_json_file",
            lambda _: {"tumor": {"labels": [1], "metrics": {"dice": {}}}},
            raising=True,
        )
        monkeypatch.setattr(
            entry.pd, "read_csv",
            lambda _: pd.DataFrame([{"id": "p1"}]),
            raising=True,
        )

        class _EvalStub:
            def __init__(self, **kwargs):
                pass

            def run(self, max_workers=None):
                pass

        monkeypatch.setattr(entry, "Evaluator", _EvalStub, raising=True)

        ns = self._make_ns(tmp_path)
        entry.run_evaluation(ns)

        assert (tmp_path / "out").is_dir()


# ---------------------------------------------------------------------------
# evaluation_entry (integration)
# ---------------------------------------------------------------------------

class TestEvaluationEntry:
    """Tests for evaluation_entrypoint.evaluation_entry."""

    def test_parses_then_runs(self, monkeypatch):
        """evaluation_entry calls _parse_eval_args then run_evaluation."""
        ns = SimpleNamespace(
            config="/cfg.json",
            paths_csv="/paths.csv",
            output_csv="/out/metrics.csv",
            num_workers=None,
        )
        called = {"parsed": False, "ran": False}

        monkeypatch.setattr(
            entry, "_parse_eval_args",
            lambda argv=None: (
                called.__setitem__("parsed", True) or ns
            ),
            raising=True,
        )
        monkeypatch.setattr(
            entry, "run_evaluation",
            lambda n: called.__setitem__("ran", True),
            raising=True,
        )

        entry.evaluation_entry(["--config", "/c.json",
                                 "--paths-csv", "/p.csv",
                                 "--output-csv", "/o.csv"])

        assert called["parsed"] is True
        assert called["ran"] is True
