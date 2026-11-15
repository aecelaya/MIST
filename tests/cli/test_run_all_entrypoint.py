"""Tests for mist_run_all CLI entrypoint."""

import argparse

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
        parser.add_argument("--data", type=str, required=True)
        parser.add_argument("--results", type=str, required=True)
        parser.add_argument("--nfolds", type=int)
        parser.add_argument("--num-workers-analyze", type=int, default=1)
        parser.add_argument("--overwrite", action="store_true")

    def _add_preprocess_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--results", type=str, required=True)
        parser.add_argument("--numpy", type=str, required=True)
        parser.add_argument("--no-preprocess", action="store_true")
        parser.add_argument("--compute-dtms", action="store_true")
        parser.add_argument("--overwrite", action="store_true")

    def _add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--results", type=str, required=True)
        parser.add_argument("--numpy", type=str, required=True)
        parser.add_argument("--model", type=str)
        parser.add_argument("--patch-size", nargs=3, type=int)
        parser.add_argument("--loss", type=str)
        parser.add_argument("--composite-loss-weighting", type=str)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--warmup-epochs", type=int)
        parser.add_argument("--batch-size-per-gpu", type=int)
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--lr-scheduler", type=str)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--l2-penalty", type=float)
        parser.add_argument("--folds", nargs="+", type=int)
        parser.add_argument("--overwrite", action="store_true")

    monkeypatch.setattr(entry.argmod, "ArgParser", _mk_parser, raising=True)
    monkeypatch.setattr(entry.argmod, "add_analyzer_args", _add_analyzer_args, raising=True)
    monkeypatch.setattr(entry.argmod, "add_preprocess_args", _add_preprocess_args, raising=True)
    monkeypatch.setattr(entry.argmod, "add_train_args", _add_train_args, raising=True)


# =============================================================================
# Tests for _ns_to_argv.
# =============================================================================


def test_ns_to_argv_includes_scalars_bools_lists_and_converts_dashes():
    """Converts scalars, True flags, lists/tuples and underscores -> dashes."""
    ns = argparse.Namespace(
        data="ds.json",
        nfolds=5,
        overwrite=True,  # Boolean True included.
        patch_size=(32, 32, 32),  # Tuple flattened.
        model="mednext",
    )
    keys = ["data", "nfolds", "overwrite", "patch_size", "model"]
    out = entry._ns_to_argv(ns, keys)
    assert out == [
        "--data",
        "ds.json",
        "--nfolds",
        "5",
        "--overwrite",
        "--patch-size",
        "32",
        "32",
        "32",
        "--model",
        "mednext",
    ]


def test_ns_to_argv_skips_missing_none_false_and_empty_list():
    """Skip missing attributes, None values, False booleans, and empty lists."""
    ns = argparse.Namespace(
        data=None,  # Skipped.
        overwrite=False,  # Skipped.
        folds=[],  # Skipped.
        results="out",  # Included.
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


def test_parse_run_all_args_requires_paths(monkeypatch):
    """It errors when --results or --numpy is missing."""
    _patch_minimal_cli(monkeypatch)
    with pytest.raises(SystemExit):
        entry._parse_run_all_args(argv=["--data", "d.json", "--numpy", "n"])
    with pytest.raises(SystemExit):
        entry._parse_run_all_args(argv=["--data", "d.json", "--results", "r"])


def test_parse_run_all_args_real_parser_requires_paths(capsys):
    """With MIST's real parser, missing paths are named in the error."""
    with pytest.raises(SystemExit):
        entry._parse_run_all_args(argv=["--data", "d.json"])
    err = capsys.readouterr().err
    assert "--results" in err and "--numpy" in err


def test_parse_run_all_args_explicit_values(monkeypatch, tmp_path):
    """It retains explicit values for all provided arguments."""
    _patch_minimal_cli(monkeypatch)
    res = tmp_path / "r"
    npy = tmp_path / "n"
    ns = entry._parse_run_all_args(
        argv=[
            "--data",
            "d.json",
            "--results",
            str(res),
            "--numpy",
            str(npy),
            "--nfolds",
            "3",
            "--overwrite",
            "--epochs",
            "10",
            "--batch-size-per-gpu",
            "2",
        ]
    )
    assert ns.data == "d.json"
    assert ns.results == str(res)
    assert ns.numpy == str(npy)
    assert ns.nfolds == 3
    assert ns.overwrite is True
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
        "--data",
        "d.json",
        "--results",
        str(tmp_path / "out"),
        "--nfolds",
        "4",
        "--overwrite",
        # Preprocess.
        "--numpy",
        str(tmp_path / "np"),
        "--no-preprocess",
        "--compute-dtms",
        # Train.
        "--model",
        "mednext",
        "--patch-size",
        "48",
        "64",
        "32",
        "--loss",
        "dice",
        "--epochs",
        "20",
        "--batch-size-per-gpu",
        "2",
        "--learning-rate",
        "0.001",
        "--lr-scheduler",
        "cos",
        "--optimizer",
        "adam",
        "--l2-penalty",
        "0.0005",
        "--folds",
        "0",
        "2",
        "--overwrite",  # Appears multiple times; conflict_handler=resolve.
    ]

    # Build expected subsets from the parsed Namespace.
    ns = entry._parse_run_all_args(argv=argv)
    analyzer_keys = [
        "data",
        "results",
        "nfolds",
        "num_workers_analyze",
        "verify",
        "overwrite",
    ]
    preprocess_keys = [
        "results",
        "numpy",
        "num_workers_preprocess",
        "no_preprocess",
        "compute_dtms",
        "overwrite",
    ]
    train_keys = [
        "results",
        "numpy",
        "model",
        "patch_size",
        "loss",
        "composite_loss_weighting",
        "epochs",
        "warmup_epochs",
        "batch_size_per_gpu",
        "learning_rate",
        "lr_scheduler",
        "optimizer",
        "l2_penalty",
        "folds",
        "overwrite",
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

    # Parse and construct expected.
    ns = entry._parse_run_all_args(argv=["--data", "d.json", "--results", "r", "--numpy", "n"])
    ns.folds = []  # Empty -> should not appear.

    analyze_keys = ["data", "results", "nfolds", "num_workers_analyze", "overwrite"]
    preprocess_keys = [
        "results",
        "numpy",
        "num_workers_preprocess",
        "no_preprocess",
        "compute_dtms",
        "overwrite",
    ]
    train_keys = [
        "results",
        "numpy",
        "model",
        "patch_size",
        "loss",
        "composite_loss_weighting",
        "epochs",
        "warmup_epochs",
        "batch_size_per_gpu",
        "learning_rate",
        "lr_scheduler",
        "optimizer",
        "l2_penalty",
        "folds",
        "overwrite",
    ]

    expected_an = entry._ns_to_argv(ns, analyze_keys)
    expected_pre = entry._ns_to_argv(ns, preprocess_keys)
    expected_tr = entry._ns_to_argv(ns, train_keys)

    # Run orchestrator with minimal args (false/empty are omitted naturally).
    entry.run_all_entry(argv=["--data", "d.json", "--results", "r", "--numpy", "n"])

    assert calls["analyze"] == expected_an
    assert calls["preprocess"] == expected_pre
    assert calls["train"] == expected_tr


def test_run_all_entry_passes_warmup_epochs_to_train(monkeypatch):
    """--warmup-epochs is forwarded to the train stage."""
    _patch_minimal_cli(monkeypatch)

    captured_train_argv = []
    monkeypatch.setattr(entry, "analyze_entry", lambda _: None, raising=True)
    monkeypatch.setattr(entry, "preprocess_entry", lambda _: None, raising=True)
    monkeypatch.setattr(
        entry,
        "train_entry",
        lambda a: captured_train_argv.extend(a),
        raising=True,
    )

    entry.run_all_entry(
        argv=["--data", "d.json", "--results", "r", "--numpy", "n", "--warmup-epochs", "0"]
    )

    assert "--warmup-epochs" in captured_train_argv
    idx = captured_train_argv.index("--warmup-epochs")
    assert captured_train_argv[idx + 1] == "0"


def test_run_all_entry_forwards_num_workers_analyze(monkeypatch):
    """--num-workers-analyze is forwarded to analyze_entry."""
    _patch_minimal_cli(monkeypatch)

    calls = {"analyze": None}
    monkeypatch.setattr(
        entry,
        "analyze_entry",
        lambda a: calls.__setitem__("analyze", list(a)),
        raising=True,
    )
    monkeypatch.setattr(entry, "preprocess_entry", lambda _: None, raising=True)
    monkeypatch.setattr(entry, "train_entry", lambda _: None, raising=True)

    entry.run_all_entry(
        argv=["--data", "d.json", "--results", "r", "--numpy", "n", "--num-workers-analyze", "4"]
    )

    assert "--num-workers-analyze" in calls["analyze"]
    assert calls["analyze"][calls["analyze"].index("--num-workers-analyze") + 1] == "4"


def test_run_all_entry_forwards_every_train_flag_except_resume(monkeypatch):
    """Every mist_train flag except --resume reaches train_entry."""
    from mist.cli import args as real_argmod  # pylint: disable=import-outside-toplevel

    parser = real_argmod.ArgParser()
    real_argmod.add_train_args(parser)
    train_dests = {a.dest for a in parser._actions if a.dest != "help"}

    captured = []
    monkeypatch.setattr(entry, "analyze_entry", lambda _: None, raising=True)
    monkeypatch.setattr(entry, "preprocess_entry", lambda _: None, raising=True)
    monkeypatch.setattr(entry, "train_entry", captured.extend, raising=True)

    entry.run_all_entry(
        argv=[
            "--data",
            "d.json",
            "--results",
            "r",
            "--numpy",
            "n",
            "--val-percent",
            "0.1",
            "--validate-every",
            "5",
            "--pretrained-weights",
            "w.pt",
            "--pretrained-config",
            "c.json",
            "--input-channel-strategy",
            "first",
        ]
    )

    def _value(flag):
        return captured[captured.index(flag) + 1]

    assert _value("--val-percent") == "0.1"
    assert _value("--validate-every") == "5"
    assert _value("--pretrained-weights") == "w.pt"
    assert _value("--pretrained-config") == "c.json"
    assert _value("--input-channel-strategy") == "first"

    # Guard against new train flags that run_all silently drops.
    import inspect  # pylint: disable=import-outside-toplevel

    source = inspect.getsource(entry.run_all_entry)
    missing = {d for d in train_dests - {"resume"} if f'"{d}"' not in source}
    assert not missing, f"mist_run_all does not forward: {sorted(missing)}"


def test_run_all_entry_rejects_resume():
    """--resume errors and points to mist_train --resume."""
    with pytest.raises(SystemExit):
        entry._parse_run_all_args(
            argv=["--data", "d.json", "--results", "r", "--numpy", "n", "--resume"]
        )
