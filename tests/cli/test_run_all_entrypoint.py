"""Tests for mist_run_all CLI entrypoint."""
import argparse
from pathlib import Path
import pytest

# MIST imports.
import mist.cli.run_all_entrypoint as entry


# pylint: disable=protected-access
# =============================================================================
# Minimal CLI patching
# =============================================================================


def _patch_minimal_cli(monkeypatch) -> None:
    """Patch argmod.* to a minimal CLI for deterministic parsing."""
    def _mk_parser(**kwargs):
        return argparse.ArgumentParser(**kwargs)

    def _add_analyzer_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data", type=str)
        parser.add_argument("--results", type=str)
        parser.add_argument("--nfolds", type=int)
        parser.add_argument("--num-workers-analyze", type=int, default=1)
        parser.add_argument("--overwrite", action="store_true")

    def _add_preprocess_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--results", type=str)
        parser.add_argument("--numpy", type=str)
        parser.add_argument("--no-preprocess", action="store_true")
        parser.add_argument("--compute-dtms", action="store_true")
        parser.add_argument("--overwrite", action="store_true")

    def _add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--results", type=str)
        parser.add_argument("--numpy", type=str)
        parser.add_argument("--gpus", nargs="+", type=int)
        parser.add_argument("--model", type=str)
        parser.add_argument("--pocket", action="store_true")
        parser.add_argument("--patch-size", nargs=3, type=int)
        parser.add_argument("--loss", type=str)
        parser.add_argument("--use-dtms", action="store_true")
        parser.add_argument("--composite-loss-weighting", type=str)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--batch-size-per-gpu", type=int)
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--lr-scheduler", type=str)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--l2-penalty", type=float)
        parser.add_argument("--folds", nargs="+", type=int)
        parser.add_argument("--overwrite", action="store_true")

    monkeypatch.setattr(entry.argmod, "ArgParser", _mk_parser, raising=True)
    monkeypatch.setattr(
        entry.argmod, "add_analyzer_args", _add_analyzer_args, raising=True
    )
    monkeypatch.setattr(
        entry.argmod, "add_preprocess_args", _add_preprocess_args, raising=True
    )
    monkeypatch.setattr(
        entry.argmod, "add_train_args", _add_train_args, raising=True
    )


# =============================================================================
# Tests for _ns_to_argv.
# =============================================================================

def test_ns_to_argv_includes_scalars_bools_lists_and_converts_dashes():
    """Converts scalars, True flags, lists/tuples and underscores -> dashes."""
    ns = argparse.Namespace(
        data="ds.json",
        nfolds=5,
        overwrite=True,             # Boolean True included.
        gpus=[0, 1],                # List flattened.
        patch_size=(32, 32, 32),    # Tuple flattened.
        pocket=False,               # Boolean False omitted.
        model="mednext",
    )
    keys = [
        "data", "nfolds", "overwrite", "gpus", "patch_size", "pocket", "model"
    ]
    out = entry._ns_to_argv(ns, keys)
    assert out == [
        "--data", "ds.json",
        "--nfolds", "5",
        "--overwrite",
        "--gpus", "0", "1",
        "--patch-size", "32", "32", "32",
        "--model", "mednext",
    ]


def test_ns_to_argv_skips_missing_none_false_and_empty_list():
    """Skip missing attributes, None values, False booleans, and empty lists."""
    ns = argparse.Namespace(
        data=None,              # Skipped.
        overwrite=False,        # Skipped.
        folds=[],               # Skipped.
        results="out",          # Included.
    )
    keys = ["data", "overwrite", "folds", "results", "not_present"]
    out = entry._ns_to_argv(ns, keys)
    assert out == ["--results", "out"]


# =============================================================================
# Tests for _parse_run_all_args.
# =============================================================================

def test_parse_run_all_args_requires_data(monkeypatch):
    """It raises SystemExit when --data is missing."""
    _patch_minimal_cli(monkeypatch)
    with pytest.raises(SystemExit):
        entry._parse_run_all_args(argv=[])


def test_parse_run_all_args_defaults_results(tmp_path, monkeypatch):
    """It supplies a default results path when --results is not given."""
    _patch_minimal_cli(monkeypatch)
    monkeypatch.chdir(tmp_path)
    ns = entry._parse_run_all_args(argv=["--data", "d.json"])
    assert Path(ns.results) == (tmp_path / "results").resolve()
    assert Path(ns.numpy) == (tmp_path / "numpy").resolve()


def test_parse_run_all_args_explicit_values(monkeypatch, tmp_path):
    """It retains explicit values for all provided arguments."""
    _patch_minimal_cli(monkeypatch)
    res = tmp_path / "r"
    npy = tmp_path / "n"
    ns = entry._parse_run_all_args(
        argv=[
            "--data", "d.json",
            "--results", str(res),
            "--numpy", str(npy),
            "--nfolds", "3",
            "--overwrite",
            "--gpus", "0", "1",
            "--pocket",
            "--use-dtms",
            "--epochs", "10",
            "--batch-size-per-gpu", "2",
        ]
    )
    assert ns.data == "d.json"
    assert ns.results == str(res)
    assert ns.numpy == str(npy)
    assert ns.nfolds == 3
    assert ns.overwrite is True
    assert ns.gpus == [0, 1]
    assert ns.pocket is True
    assert ns.use_dtms is True
    assert ns.epochs == 10
    assert ns.batch_size_per_gpu == 2


# =============================================================================
# Tests for run_all_entry.
# =============================================================================

def test_run_all_entry_forwards_subsets_correctly(monkeypatch, tmp_path):
    """It forwards correct subsets to analyze, preprocess, and train."""
    _patch_minimal_cli(monkeypatch)

    # Capture argv passed to stage entrypoints.
    calls = {"analyze": None, "preprocess": None, "train": None}

    def _an(argv):
        calls["analyze"] = list(argv)

    def _pre(argv):
        calls["preprocess"] = list(argv)

    def _tr(argv):
        calls["train"] = list(argv)

    monkeypatch.setattr(entry, "analyze_entry", _an, raising=True)
    monkeypatch.setattr(entry, "preprocess_entry", _pre, raising=True)
    monkeypatch.setattr(entry, "train_entry", _tr, raising=True)

    argv = [
        # Analyzer.
        "--data", "d.json",
        "--results", str(tmp_path / "out"),
        "--nfolds", "4",
        "--overwrite",
        # Preprocess.
        "--numpy", str(tmp_path / "np"),
        "--no-preprocess",
        "--compute-dtms",
        # Train.
        "--gpus", "0", "1",
        "--model", "mednext",
        "--patch-size", "48", "64", "32",
        "--loss", "dice",
        "--use-dtms",       # True -> include.
        "--epochs", "20",
        "--batch-size-per-gpu", "2",
        "--learning-rate", "0.001",
        "--lr-scheduler", "cos",
        "--optimizer", "adam",
        "--l2-penalty", "0.0005",
        "--folds", "0", "2",
        "--overwrite",     # Appears multiple times; conflict_handler=resolve.
    ]

    # Build expected subsets from the parsed Namespace.
    ns = entry._parse_run_all_args(argv=argv)
    analyzer_keys = ["data", "results", "nfolds", "num_workers_analyze", "overwrite"]
    preprocess_keys = [
        "results", "numpy", "num_workers_preprocess",
        "no_preprocess", "compute_dtms", "overwrite"
    ]
    train_keys = [
        "results", "numpy",
        "gpus",
        "model", "pocket", "patch_size",
        "loss", "use_dtms", "composite_loss_weighting",
        "epochs", "batch_size_per_gpu", "learning_rate",
        "lr_scheduler", "optimizer", "l2_penalty",
        "folds", "overwrite",
    ]
    expected_an = entry._ns_to_argv(ns, analyzer_keys)
    expected_pre = entry._ns_to_argv(ns, preprocess_keys)
    expected_tr = entry._ns_to_argv(ns, train_keys)

    # Now actually run orchestrator.
    entry.run_all_entry(argv=argv)

    # Verify exact argv passed into each stage.
    assert calls["analyze"] == expected_an
    assert calls["preprocess"] == expected_pre
    assert calls["train"] == expected_tr


def test_run_all_entry_handles_false_flags_and_empty_lists(monkeypatch):
    """Omit False boolean flags and empty list-valued args from stage argv."""
    _patch_minimal_cli(monkeypatch)

    calls = {"analyze": None, "preprocess": None, "train": None}
    monkeypatch.setattr(
        entry,
        "analyze_entry",
        lambda a: calls.__setitem__("analyze", a),
        raising=True,
    )
    monkeypatch.setattr(
        entry,
        "preprocess_entry",
        lambda a: calls.__setitem__("preprocess", a),
        raising=True,
    )
    monkeypatch.setattr(
        entry,
        "train_entry",
        lambda a: calls.__setitem__("train", a),
        raising=True,
    )

    argv = [
        "--data", "d.json",
        "--results", "r",
        "--numpy", "n",
        # Explicitly false flags and empty list.
        # (with our minimal CLI, absence of flag == False; sim. by not passing).
        "--folds",  # Empty list not representable on CLI, verify via ns below.
    ]

    # Parse and construct expected.
    ns = entry._parse_run_all_args(
        argv=["--data", "d.json", "--results", "r", "--numpy", "n"]
    )
    ns.pocket = False
    ns.use_dtms = False
    ns.folds = []  # Empty -> should not appear.

    analyze_keys = ["data", "results", "nfolds", "num_workers_analyze", "overwrite"]
    preprocess_keys = [
        "results", "numpy", "num_workers_preprocess",
        "no_preprocess", "compute_dtms", "overwrite"
    ]
    train_keys = [
        "results", "numpy",
        "gpus",
        "model", "pocket", "patch_size",
        "loss", "use_dtms", "composite_loss_weighting",
        "epochs", "batch_size_per_gpu", "learning_rate",
        "lr_scheduler", "optimizer", "l2_penalty",
        "folds", "overwrite",
    ]

    expected_an = entry._ns_to_argv(ns, analyze_keys)
    expected_pre = entry._ns_to_argv(ns, preprocess_keys)
    expected_tr = entry._ns_to_argv(ns, train_keys)

    # Run orchestrator with minimal args (false/empty are omitted naturally).
    entry.run_all_entry(
        argv=["--data", "d.json", "--results", "r", "--numpy", "n"]
    )

    assert calls["analyze"] == expected_an
    assert calls["preprocess"] == expected_pre
    assert calls["train"] == expected_tr


def test_run_all_entry_forwards_num_workers_analyze(monkeypatch):
    """--num-workers-analyze is forwarded to analyze_entry."""
    _patch_minimal_cli(monkeypatch)

    calls = {"analyze": None}
    monkeypatch.setattr(
        entry, "analyze_entry", lambda a: calls.__setitem__("analyze", list(a)),
        raising=True,
    )
    monkeypatch.setattr(entry, "preprocess_entry", lambda _: None, raising=True)
    monkeypatch.setattr(entry, "train_entry", lambda _: None, raising=True)

    entry.run_all_entry(argv=["--data", "d.json", "--num-workers-analyze", "4"])

    assert "--num-workers-analyze" in calls["analyze"]
    assert calls["analyze"][calls["analyze"].index("--num-workers-analyze") + 1] == "4"
