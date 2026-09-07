"""Tests for the mist_finalize CLI entrypoint."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

# MIST imports.
import mist.cli.finalize_entrypoint as entry
from mist.utils import console as console_mod


class _NoGrad:
    """Minimal no_grad context manager."""

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context."""
        return False


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_required_files(results_dir: Path, include_test: bool = False) -> None:
    """Write the files run_finalize requires to exist."""
    _write_json(
        results_dir / "config.json",
        {
            "training": {"nfolds": 2},
            "evaluation": {
                "final_classes": {"background": [0], "foreground": [1]},
                "metrics": ["dice"],
            },
        },
    )
    (results_dir / "train_paths.csv").write_text("id,mask,ct\n0,m0,i0\n1,m1,i1\n")
    if include_test:
        (results_dir / "test_paths.csv").write_text("id,ct\n9,it\n")


# =============================================================================
# Tests for _parse_finalize_args.
# =============================================================================


def test_parse_finalize_args_fallback_results(tmp_path, monkeypatch):
    """It sets default ./results when --results is not provided."""
    monkeypatch.chdir(tmp_path)
    ns = entry._parse_finalize_args(argv=[])
    assert Path(ns.results) == (tmp_path / "results").resolve()
    assert ns.num_workers_evaluate == 1
    assert ns.device == "cuda"


def test_parse_finalize_args_explicit(tmp_path):
    """It keeps explicit --results/--num-workers-evaluate/--device values."""
    res = tmp_path / "out"
    ns = entry._parse_finalize_args(
        [
            "--results",
            str(res),
            "--num-workers-evaluate",
            "4",
            "--device",
            "cpu",
        ]
    )
    assert ns.results == str(res)
    assert ns.num_workers_evaluate == 4
    assert ns.device == "cpu"


def test_parse_finalize_args_num_workers_must_be_positive(tmp_path):
    """--num-workers-evaluate must be a positive integer."""
    with pytest.raises(SystemExit):
        entry._parse_finalize_args(["--results", str(tmp_path), "--num-workers-evaluate", "0"])


# =============================================================================
# Tests for _check_folds_complete.
# =============================================================================


def test_check_folds_complete_no_warning_when_all_present(tmp_path, monkeypatch):
    """No warning is printed when every configured fold has a saved model."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "fold_0.pt").write_bytes(b"\x00")
    (models_dir / "fold_1.pt").write_bytes(b"\x00")

    logs = []
    monkeypatch.setattr(entry, "print_warning", lambda msg: logs.append(msg), raising=True)

    entry._check_folds_complete(tmp_path, {"training": {"nfolds": 2}})

    assert logs == []


def test_check_folds_complete_warns_on_missing_folds(tmp_path, monkeypatch):
    """Missing fold checkpoints are named in a warning."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "fold_0.pt").write_bytes(b"\x00")
    # fold_1.pt intentionally missing.

    logs = []
    monkeypatch.setattr(entry, "print_warning", lambda msg: logs.append(msg), raising=True)

    entry._check_folds_complete(tmp_path, {"training": {"nfolds": 2}})

    assert len(logs) == 1
    assert "1" in logs[0]


# =============================================================================
# Tests for run_finalize.
# =============================================================================


def test_run_finalize_raises_on_missing_results_dir(tmp_path):
    """It raises FileNotFoundError when the results directory doesn't exist."""
    ns = argparse.Namespace(results=str(tmp_path / "nope"), num_workers_evaluate=1, device="cpu")
    with pytest.raises(FileNotFoundError, match="Results directory"):
        entry.run_finalize(ns)


def test_run_finalize_raises_on_missing_config(tmp_path):
    """It raises FileNotFoundError when config.json is missing."""
    ns = argparse.Namespace(results=str(tmp_path), num_workers_evaluate=1, device="cpu")
    with pytest.raises(FileNotFoundError, match="Configuration file"):
        entry.run_finalize(ns)


def test_run_finalize_raises_on_missing_train_paths_csv(tmp_path):
    """It raises FileNotFoundError when train_paths.csv is missing."""
    _write_json(tmp_path / "config.json", {"training": {"nfolds": 1}, "evaluation": {}})
    ns = argparse.Namespace(results=str(tmp_path), num_workers_evaluate=1, device="cpu")
    with pytest.raises(FileNotFoundError, match="Training paths file"):
        entry.run_finalize(ns)


def test_run_finalize_empty_eval_prints_error_and_skips_evaluator(tmp_path, monkeypatch):
    """No valid prediction-mask pairs: skip the Evaluator, print an error."""
    _write_required_files(tmp_path, include_test=False)

    monkeypatch.setattr(
        entry.evaluation_utils,
        "build_evaluation_dataframe",
        lambda **kwargs: (pd.DataFrame(), "[warn] something"),
        raising=True,
    )
    monkeypatch.setattr(
        entry,
        "Evaluator",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Evaluator should not be constructed")
        ),
        raising=True,
    )

    logs = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg, **k: logs.append(str(msg)))

    ns = argparse.Namespace(results=str(tmp_path), num_workers_evaluate=1, device="cpu")
    entry.run_finalize(ns)

    assert any("warn" in m.lower() for m in logs)
    assert any("No valid prediction-mask pairs" in m for m in logs)
    assert not (tmp_path / "evaluation_paths.csv").exists()


def test_run_finalize_writes_results_and_runs_test_inference(tmp_path, monkeypatch):
    """Full happy path: evaluation_paths.csv + results.csv + test inference."""
    _write_required_files(tmp_path, include_test=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fold_0.pt").write_bytes(b"\x00")
    (tmp_path / "models" / "fold_1.pt").write_bytes(b"\x00")

    eval_df = pd.DataFrame({"prediction": ["p.nii.gz"], "mask": ["m.nii.gz"], "id": [0]})
    monkeypatch.setattr(
        entry.evaluation_utils,
        "build_evaluation_dataframe",
        lambda **kwargs: (eval_df, ""),
        raising=True,
    )

    runs = {"called": False}

    def _mk_eval(filepaths_dataframe, evaluation_config, output_csv_path):
        class _E:
            def run(self, max_workers=None):
                runs["called"] = True
                Path(output_csv_path).write_text("metric,value\n")

        return _E()

    monkeypatch.setattr(entry, "Evaluator", _mk_eval, raising=True)
    monkeypatch.setattr(entry.torch, "no_grad", _NoGrad, raising=True)

    infer_calls: list[tuple[str, str]] = []

    def _infer_from_dataframe(
        paths_dataframe,
        output_directory,
        mist_configuration,
        models_directory,
        postprocessing_strategy_filepath,
        device,
    ):
        infer_calls.append((output_directory, models_directory))
        p = Path(output_directory)
        assert p.name == "test"
        assert p.parent.name == "predictions"
        assert Path(models_directory).name == "models"

    monkeypatch.setattr(entry, "infer_from_dataframe", _infer_from_dataframe, raising=True)

    ns = argparse.Namespace(results=str(tmp_path), num_workers_evaluate=1, device="cpu")
    entry.run_finalize(ns)

    assert runs["called"] is True
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "evaluation_paths.csv").is_file()
    assert len(infer_calls) == 1


def test_run_finalize_skips_test_inference_when_no_test_paths_csv(tmp_path, monkeypatch):
    """No test_paths.csv: skip test-set inference entirely."""
    _write_required_files(tmp_path, include_test=False)

    eval_df = pd.DataFrame({"prediction": ["p.nii.gz"], "mask": ["m.nii.gz"], "id": [0]})
    monkeypatch.setattr(
        entry.evaluation_utils,
        "build_evaluation_dataframe",
        lambda **kwargs: (eval_df, ""),
        raising=True,
    )

    def _mk_eval(filepaths_dataframe, evaluation_config, output_csv_path):
        class _E:
            def run(self, max_workers=None):
                Path(output_csv_path).write_text("metric,value\n")

        return _E()

    monkeypatch.setattr(entry, "Evaluator", _mk_eval, raising=True)

    monkeypatch.setattr(
        entry,
        "infer_from_dataframe",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("infer_from_dataframe should not be called")
        ),
        raising=True,
    )

    ns = argparse.Namespace(results=str(tmp_path), num_workers_evaluate=1, device="cpu")
    entry.run_finalize(ns)

    assert (tmp_path / "results.csv").is_file()
    assert not (tmp_path / "predictions" / "test").exists()


# =============================================================================
# Tests for finalize_entry.
# =============================================================================


def test_finalize_entry_calls_run_finalize(monkeypatch, tmp_path):
    """finalize_entry parses args and delegates to run_finalize."""
    captured = {}
    monkeypatch.setattr(
        entry, "run_finalize", lambda ns: captured.setdefault("ns", ns), raising=True
    )

    entry.finalize_entry(["--results", str(tmp_path)])

    assert captured["ns"].results == str(tmp_path)
