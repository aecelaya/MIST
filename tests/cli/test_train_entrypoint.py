"""Tests for MIST training entrypoint CLI."""
from typing import List, Tuple
import argparse
import json
import os
from pathlib import Path
import pandas as pd
import pytest

# MIST imports.
import mist.cli.train_entrypoint as entry
from mist.utils import console as console_mod


# =============================================================================
# Helpers and minimal patching for ArgParser.
# =============================================================================

# pylint: disable=protected-access
class _DummyTrainer:
    """A tiny stub trainer that records if fit() was called."""

    def __init__(self, ns: argparse.Namespace) -> None:
        """Initialize and capture the namespace."""
        self.ns = ns
        self.fit_called = False

    def fit(self) -> None:
        """Mark that fit() was called."""
        self.fit_called = True


def _patch_minimal_cli(monkeypatch) -> None:
    """Patch argmod.* to provide a minimal, deterministic CLI."""

    def _mk_parser(**kwargs):
        return argparse.ArgumentParser(**kwargs)

    def _add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--results", type=str, default=None)
        parser.add_argument("--numpy", type=str, default=None)
        parser.add_argument("--gpus", nargs="+", type=int, default=[-1])
        parser.add_argument("--num-workers-evaluate", type=int, default=1)
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--resume", action="store_true")

    monkeypatch.setattr(entry.argmod, "ArgParser", _mk_parser, raising=True)
    monkeypatch.setattr(
        entry.argmod, "add_train_args", _add_train_args, raising=True
    )


class _NoGrad:
    """Minimal no_grad context manager."""

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context."""
        return False


# =============================================================================
# Tests for _parse_train_args.
# =============================================================================


def test_parse_train_args_fallbacks(tmp_path, monkeypatch):
    """It sets default ./results and ./numpy when not provided."""
    _patch_minimal_cli(monkeypatch)
    monkeypatch.chdir(tmp_path)

    ns = entry._parse_train_args(argv=[])
    assert Path(ns.results) == (tmp_path / "results").resolve()
    assert Path(ns.numpy) == (tmp_path / "numpy").resolve()


def test_parse_train_args_explicit(tmp_path, monkeypatch):
    """It keeps explicit --results and --numpy values."""
    _patch_minimal_cli(monkeypatch)
    res = tmp_path / "out"
    npy = tmp_path / "np"
    ns = entry._parse_train_args(
        argv=["--results", str(res), "--numpy", str(npy)]
    )
    assert ns.results == str(res)
    assert ns.numpy == str(npy)


# =============================================================================
# Tests for _ensure_required_artifacts.
# =============================================================================


def _write_required_files(base: Path, include_test: bool = False) -> None:
    """Write required results files; optionally include test_paths.csv."""
    (base / "config.json").write_text(json.dumps({"k": "v"}))
    (base / "train_paths.csv").write_text("id,mask,ct\n0,m0,i0\n")
    (base / "fg_bboxes.csv").write_text("id,x_start,x_end\n0,1,2\n")
    if include_test:
        (base / "test_paths.csv").write_text("id,mask,ct\n9,mt,it\n")


def _ensure_numpy_dirs(numpy_dir: Path) -> None:
    """Create numpy/images and numpy/labels subdirs."""
    (numpy_dir / "images").mkdir(parents=True, exist_ok=True)
    (numpy_dir / "labels").mkdir(parents=True, exist_ok=True)


def test_ensure_required_artifacts_happy_and_test_flag(tmp_path):
    """It returns (results_dir, has_test_paths) when structure is valid."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=True)
    _ensure_numpy_dirs(numpy_dir)

    ns = argparse.Namespace(results=str(results_dir), numpy=str(numpy_dir))
    got_results, has_test = entry._ensure_required_artifacts(
        ns
    )
    assert got_results == results_dir.resolve()
    assert has_test is True


def test_ensure_required_artifacts_missing_results_dir(tmp_path):
    """It raises when results directory does not exist."""
    numpy_dir = tmp_path / "numpy"
    numpy_dir.mkdir()
    ns = argparse.Namespace(
        results=str(tmp_path / "missing"), numpy=str(numpy_dir)
    )
    with pytest.raises(FileNotFoundError):
        entry._ensure_required_artifacts(ns)


def test_ensure_required_artifacts_missing_results_files(tmp_path):
    """It raises when required files in results are missing."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    # Write only one file; others missing.
    (results_dir / "config.json").write_text("{}")
    _ensure_numpy_dirs(numpy_dir)

    ns = argparse.Namespace(results=str(results_dir), numpy=str(numpy_dir))
    with pytest.raises(FileNotFoundError) as e:
        entry._ensure_required_artifacts(ns)
    assert "train_paths.csv" in str(e.value) and "fg_bboxes.csv" in str(
        e.value
    )


def test_ensure_required_artifacts_missing_numpy_dirs(tmp_path):
    """It raises when numpy subfolders images/labels are missing."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    # Intentionally do not create images/labels under numpy

    ns = argparse.Namespace(results=str(results_dir), numpy=str(numpy_dir))
    with pytest.raises(FileNotFoundError) as e:
        entry._ensure_required_artifacts(ns)
    assert "images" in str(e.value) and "labels" in str(e.value)


def test_ensure_required_artifacts_missing_numpy_dir(tmp_path):
    """It raises when the NumPy directory itself does not exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_required_files(
        results_dir, include_test=False
    )  # helper from this test module

    ns = argparse.Namespace(
        results=str(results_dir), numpy=str(tmp_path / "numpy_missing")
    )
    with pytest.raises(FileNotFoundError) as e:
        entry._ensure_required_artifacts(ns)
    assert "NumPy directory does not exist" in str(e.value)


def test_ensure_required_artifacts_numpy_path_is_file(tmp_path):
    """It raises when the NumPy path exists but is a file, not a directory."""
    results_dir = tmp_path / "results"
    numpy_path = tmp_path / "numpy_file"
    results_dir.mkdir()
    numpy_path.write_text("not a dir")
    _write_required_files(results_dir, include_test=False)

    ns = argparse.Namespace(results=str(results_dir), numpy=str(numpy_path))
    with pytest.raises(FileNotFoundError) as e:
        entry._ensure_required_artifacts(ns)
    assert "NumPy directory does not exist" in str(e.value)


# =============================================================================
# Tests for _create_train_dirs.
# =============================================================================


@pytest.mark.parametrize("has_test_paths", [False, True])
def test_create_train_dirs_makes_structure(tmp_path, has_test_paths):
    """It creates logs, models, and prediction directories."""
    entry._create_train_dirs(tmp_path, has_test_paths)
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "predictions" / "train" / "raw").is_dir()
    if has_test_paths:
        assert (tmp_path / "predictions" / "test").is_dir()
    else:
        assert not (tmp_path / "predictions" / "test").exists()


# =============================================================================
# Tests for _set_visible_devices.
# =============================================================================


def test_set_visible_devices_no_gpus_raises(monkeypatch):
    """It raises a RuntimeError when torch reports 0 GPUs."""
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 0, raising=True
    )
    with pytest.raises(RuntimeError):
        entry._set_visible_devices(
            argparse.Namespace(gpus=[-1])
        )


@pytest.mark.parametrize("cli_gpus", [None, [], [-1]])
def test_set_visible_devices_all_gpus(monkeypatch, cli_gpus):
    """It sets CUDA_VISIBLE_DEVICES to all indices for None/[]/[-1]."""
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 3, raising=True
    )
    ns = argparse.Namespace(gpus=cli_gpus)
    entry._set_visible_devices(ns)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2"


def test_set_visible_devices_specific_indices(monkeypatch):
    """It sets CUDA_VISIBLE_DEVICES to the requested indices."""
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 4, raising=True
    )
    ns = argparse.Namespace(gpus=[1, 3])
    entry._set_visible_devices(ns)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1,3"


def test_set_visible_devices_invalid_indices(monkeypatch):
    """It raises ValueError when indices are out of range."""
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 2, raising=True
    )
    ns = argparse.Namespace(gpus=[3])
    with pytest.raises(ValueError):
        entry._set_visible_devices(ns)


# =============================================================================
# Tests for train_entry — integration behavior.
# =============================================================================


def test_train_entry_resume_and_overwrite_are_mutually_exclusive(
    tmp_path, monkeypatch
):
    """It raises ValueError when both --resume and --overwrite are passed."""
    _patch_minimal_cli(monkeypatch)

    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    _ensure_numpy_dirs(numpy_dir)

    argv = [
        "--results", str(results_dir),
        "--numpy", str(numpy_dir),
        "--resume",
        "--overwrite",
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        entry.train_entry(argv)


def test_train_entry_blocks_existing_results_csv_without_overwrite(
    tmp_path, monkeypatch
):
    """It raises FileExistsError when results.csv exists without --overwrite."""
    _patch_minimal_cli(monkeypatch)

    # Build valid results and numpy trees.
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    _ensure_numpy_dirs(numpy_dir)
    (results_dir / "results.csv").write_text("already here")

    # Ensure no GPU-driven crash by mocking device_count.
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 1, raising=True
    )

    # If trainer is constructed, fail—we should block earlier.
    monkeypatch.setattr(
        entry,
        "Patch3DTrainer",
        lambda _ns: (_ for _ in ()).throw(
            AssertionError("Trainer should not be created")
        ),
        raising=True,
    )

    argv = ["--results", str(results_dir), "--numpy", str(numpy_dir)]
    with pytest.raises(FileExistsError):
        entry.train_entry(argv)


def test_train_entry_happy_path_no_test_empty_eval(tmp_path, monkeypatch):
    """It trains, folds infer, and prints warnings with empty eval pairs."""
    _patch_minimal_cli(monkeypatch)

    # Folder structure & required files (no test_paths.csv).
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    _ensure_numpy_dirs(numpy_dir)

    # Config returned by io.read_json_file.
    config = {
        "training": {"folds": [0, 1]},
        "evaluation": {
            "final_classes": {"background": [0], "foreground": [1]},
            "metrics": ["dice"],
            "params": {"surf_dice_tol": 1.0},
        },
    }
    monkeypatch.setattr(
        entry.io, "read_json_file", lambda p: config, raising=True
    )

    # GPU availability
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 1, raising=True
    )
    monkeypatch.setattr(
        entry.torch, "no_grad", lambda: _NoGrad(), raising=True
    )

    # Trainer stub.
    created = {}
    monkeypatch.setattr(
        entry,
        "Patch3DTrainer",
        lambda ns: created.setdefault("trainer", _DummyTrainer(ns)),
        raising=True,
    )

    # Record folds for test_on_fold calls.
    folds_called: List[int] = []
    monkeypatch.setattr(
        entry,
        "test_on_fold",
        lambda ns, f: folds_called.append(f),
        raising=True,
    )

    # Evaluation dataframe is empty, with warnings.
    monkeypatch.setattr(
        entry.evaluation_utils,
        "build_evaluation_dataframe",
        lambda **kwargs: (pd.DataFrame(), "[warn] something"),
        raising=True,
    )

    # Capture console prints.
    logs = []
    monkeypatch.setattr(
        console_mod.console, "print",
        lambda msg, **k: logs.append(str(msg)),
    )

    # Evaluator should not be called since dataframe is empty.
    monkeypatch.setattr(
        entry,
        "Evaluator",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Evaluator should not be constructed")
        ),
        raising=True,
    )

    argv = [
        "--results", str(results_dir),
        "--numpy", str(numpy_dir),
        "--overwrite",
    ]
    entry.train_entry(argv)

    # Trainer.fit called.
    assert created["trainer"].fit_called is True
    # Folds inferred.
    assert folds_called == [0, 1]
    # Warnings surfaced, and empty-eval message printed.
    assert any("warn" in m.lower() for m in logs)
    assert any("No valid prediction-mask pairs" in m for m in logs)
    # Predictions/train/raw created.
    assert (results_dir / "predictions" / "train" / "raw").is_dir()
    # No evaluation_paths.csv written.
    assert not (results_dir / "evaluation_paths.csv").exists()


def test_train_entry_happy_path_with_eval_and_test_infer(
    tmp_path, monkeypatch
):
    """Run eval and test inference when pairs exist and test set present."""
    _patch_minimal_cli(monkeypatch)

    # Files and dirs including test paths.
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=True)
    _ensure_numpy_dirs(numpy_dir)

    # Config for folds/eval params.
    config = {
        "training": {
            "folds": [2],
            "hardware": {"num_cpu_workers": 4},
        },
        "evaluation": {
            "background": {"labels": [0], "metrics": {"dice": {}}},
            "foreground": {"labels": [1], "metrics": {"dice": {}}},
        },
    }
    (results_dir / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        entry.io, "read_json_file", lambda p: config, raising=True
    )

    # GPU availability.
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 2, raising=True
    )
    monkeypatch.setattr(
        entry.torch, "no_grad", lambda: _NoGrad(), raising=True
    )

    # Trainer stub.
    trainer = _DummyTrainer(argparse.Namespace())
    monkeypatch.setattr(
        entry, "Patch3DTrainer", lambda ns: trainer, raising=True
    )

    # Fold test tracker.
    folds_called: List[int] = []
    monkeypatch.setattr(
        entry,
        "test_on_fold",
        lambda ns, f: folds_called.append(f),
        raising=True,
    )

    # Non-empty evaluation dataframe.
    eval_df = pd.DataFrame(
        {"prediction": ["p.nii.gz"], "mask": ["m.nii.gz"], "id": [0]}
    )
    monkeypatch.setattr(
        entry.evaluation_utils,
        "build_evaluation_dataframe",
        lambda **kwargs: (eval_df, ""),
        raising=True,
    )

    # Evaluator stub that writes results.csv.
    runs = {"called": False}

    def _mk_eval(filepaths_dataframe, evaluation_config, output_csv_path):
        class _E:
            def run(self_nonlocal, max_workers=None):
                runs["called"] = True
                Path(output_csv_path).write_text("metric,value\n")

        return _E()

    monkeypatch.setattr(entry, "Evaluator", _mk_eval, raising=True)

    # test inference: patch read_csv + infer runner.
    test_df = pd.DataFrame({"id": [9]})
    monkeypatch.setattr(pd, "read_csv", lambda p: test_df, raising=True)

    infer_calls: List[Tuple[str, str]] = []

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
        # output_directory is .../predictions/test (the folder itself).
        assert p.name == "test"
        assert p.parent.name == "predictions"
        assert Path(models_directory).name == "models"

    monkeypatch.setattr(
        entry, "infer_from_dataframe", _infer_from_dataframe, raising=True
    )

    argv = [
        "--results", str(results_dir),
        "--numpy", str(numpy_dir),
        "--overwrite",
    ]
    entry.train_entry(argv)

    # Trainer.fit called.
    assert trainer.fit_called is True
    # Fold tested
    assert folds_called == [2]
    # Evaluation ran and produced results.csv.
    assert runs["called"] is True
    assert (results_dir / "results.csv").is_file()
    # evaluation_paths.csv written.
    assert (results_dir / "evaluation_paths.csv").is_file()
    # Test inference invoked with expected directories.
    assert len(infer_calls) == 1
