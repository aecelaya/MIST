"""Tests for MIST training entrypoint CLI."""

import argparse
import json
from pathlib import Path

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
        parser.add_argument("--folds", nargs="+", type=int, default=None)
        parser.add_argument("--num-workers-evaluate", type=int, default=1)
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--resume", action="store_true")

    monkeypatch.setattr(entry.argmod, "ArgParser", _mk_parser, raising=True)
    monkeypatch.setattr(entry.argmod, "add_train_args", _add_train_args, raising=True)


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
    ns = entry._parse_train_args(argv=["--results", str(res), "--numpy", str(npy)])
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
    got_results, has_test = entry._ensure_required_artifacts(ns)
    assert got_results == results_dir.resolve()
    assert has_test is True


def test_ensure_required_artifacts_missing_results_dir(tmp_path):
    """It raises when results directory does not exist."""
    numpy_dir = tmp_path / "numpy"
    numpy_dir.mkdir()
    ns = argparse.Namespace(results=str(tmp_path / "missing"), numpy=str(numpy_dir))
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
    assert "train_paths.csv" in str(e.value) and "fg_bboxes.csv" in str(e.value)


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
    _write_required_files(results_dir, include_test=False)  # helper from this test module

    ns = argparse.Namespace(results=str(results_dir), numpy=str(tmp_path / "numpy_missing"))
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
# Tests for train_entry — integration behavior.
# =============================================================================


def test_train_entry_resume_and_overwrite_are_mutually_exclusive(tmp_path, monkeypatch):
    """It raises ValueError when both --resume and --overwrite are passed."""
    _patch_minimal_cli(monkeypatch)

    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    _ensure_numpy_dirs(numpy_dir)

    argv = [
        "--results",
        str(results_dir),
        "--numpy",
        str(numpy_dir),
        "--resume",
        "--overwrite",
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        entry.train_entry(argv)


def test_train_entry_blocks_existing_results_csv_without_overwrite(tmp_path, monkeypatch):
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

    # If trainer is constructed, fail—we should block earlier.
    monkeypatch.setattr(
        entry,
        "Patch3DTrainer",
        lambda _ns: (_ for _ in ()).throw(AssertionError("Trainer should not be created")),
        raising=True,
    )

    argv = ["--results", str(results_dir), "--numpy", str(numpy_dir)]
    with pytest.raises(FileExistsError):
        entry.train_entry(argv)


def test_train_entry_calls_finalize_when_covers_all_folds(tmp_path, monkeypatch):
    """When this invocation trains every configured fold, it finalizes directly."""
    _patch_minimal_cli(monkeypatch)

    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    _ensure_numpy_dirs(numpy_dir)

    # No --folds passed, so own_folds falls back to config's "folds", which
    # equals range(nfolds): this invocation covers the whole run.
    config = {"training": {"folds": [0, 1], "nfolds": 2}}
    monkeypatch.setattr(entry.io, "read_json_file", lambda p: config, raising=True)

    created = {}
    monkeypatch.setattr(
        entry,
        "Patch3DTrainer",
        lambda ns: created.setdefault("trainer", _DummyTrainer(ns)),
        raising=True,
    )

    folds_called: list[int] = []
    monkeypatch.setattr(
        entry,
        "test_on_fold",
        lambda ns, f: folds_called.append(f),
        raising=True,
    )

    finalize_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(
        entry,
        "run_finalize",
        lambda ns: finalize_calls.append(ns),
        raising=True,
    )

    argv = ["--results", str(results_dir), "--numpy", str(numpy_dir), "--overwrite"]
    entry.train_entry(argv)

    assert created["trainer"].fit_called is True
    assert folds_called == [0, 1]
    assert len(finalize_calls) == 1
    assert finalize_calls[0].results == str(results_dir)
    assert finalize_calls[0].device == "cuda"


def test_train_entry_skips_finalize_when_folds_are_a_subset(tmp_path, monkeypatch):
    """One-fold-per-node pattern: skip finalize, point the user at mist_finalize."""
    _patch_minimal_cli(monkeypatch)

    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=True)
    _ensure_numpy_dirs(numpy_dir)

    # Total run has 2 folds; this invocation only trains fold 0.
    config = {"training": {"folds": [0, 1], "nfolds": 2}}
    monkeypatch.setattr(entry.io, "read_json_file", lambda p: config, raising=True)
    monkeypatch.setattr(entry, "Patch3DTrainer", lambda ns: _DummyTrainer(ns), raising=True)

    folds_called: list[int] = []
    monkeypatch.setattr(entry, "test_on_fold", lambda ns, f: folds_called.append(f), raising=True)

    finalize_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(entry, "run_finalize", lambda ns: finalize_calls.append(ns), raising=True)

    logs = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg, **k: logs.append(str(msg)))

    argv = [
        "--results",
        str(results_dir),
        "--numpy",
        str(numpy_dir),
        "--folds",
        "0",
        "--overwrite",
    ]
    entry.train_entry(argv)

    # Only the requested fold was tested; finalize was never called.
    assert folds_called == [0]
    assert finalize_calls == []
    assert any("mist_finalize" in m for m in logs)


def test_train_entry_uses_cli_folds_not_config_folds_for_test_on_fold(tmp_path, monkeypatch):
    """test_on_fold uses this invocation's own --folds, not a re-read of the
    shared config.json's "folds" key.

    Regression test: config.json's "folds" key is only safe to read *before*
    training starts. BaseTrainer persists a --folds override into that same,
    possibly shared, file, so another per-node invocation racing on the same
    --results directory could have overwritten it by the time this process
    finishes training. Simulate that by having config's "folds" disagree with
    what this invocation was actually asked to train.
    """
    _patch_minimal_cli(monkeypatch)

    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    _write_required_files(results_dir, include_test=False)
    _ensure_numpy_dirs(numpy_dir)

    # Deliberately disagrees with --folds 0 below, simulating another node's
    # concurrent write to the shared config.json.
    config = {"training": {"folds": [5, 6], "nfolds": 2}}
    monkeypatch.setattr(entry.io, "read_json_file", lambda p: config, raising=True)
    monkeypatch.setattr(entry, "Patch3DTrainer", lambda ns: _DummyTrainer(ns), raising=True)

    folds_called: list[int] = []
    monkeypatch.setattr(entry, "test_on_fold", lambda ns, f: folds_called.append(f), raising=True)
    monkeypatch.setattr(entry, "run_finalize", lambda ns: None, raising=True)

    argv = [
        "--results",
        str(results_dir),
        "--numpy",
        str(numpy_dir),
        "--folds",
        "0",
        "--overwrite",
    ]
    entry.train_entry(argv)

    assert folds_called == [0]
