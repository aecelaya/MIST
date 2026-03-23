"""Tests for mist.scripts.postprocess_entrypoint."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import pytest

# MIST imports.
from mist.cli import postprocess_entrypoint as entry


def test_parse_postprocess_args_ok(tmp_path):
    """_parse_postprocess_args parses valid arguments."""
    base = tmp_path / "preds"
    out = tmp_path / "out"
    strat = tmp_path / "post.json"

    ns = entry._parse_postprocess_args([
        "--base-predictions", str(base),
        "--output", str(out),
        "--postprocess-strategy", str(strat),
    ])

    assert ns.base_predictions == str(base)
    assert ns.output == str(out)
    assert ns.postprocess_strategy == str(strat)
    assert ns.paths_csv is None
    assert ns.eval_config is None
    assert ns.num_workers_evaluate == 1


def test_parse_postprocess_args_num_workers_evaluate(tmp_path):
    """--num-workers-evaluate is parsed correctly."""
    ns = entry._parse_postprocess_args([
        "--base-predictions", str(tmp_path / "preds"),
        "--output", str(tmp_path / "out"),
        "--postprocess-strategy", str(tmp_path / "post.json"),
        "--num-workers-evaluate", "4",
    ])
    assert ns.num_workers_evaluate == 4


def test_parse_postprocess_args_with_eval_args(tmp_path):
    """_parse_postprocess_args parses optional evaluation arguments."""
    base = tmp_path / "preds"
    out = tmp_path / "out"
    strat = tmp_path / "post.json"
    paths_csv = tmp_path / "paths.csv"
    eval_cfg = tmp_path / "config.json"

    ns = entry._parse_postprocess_args([
        "--base-predictions", str(base),
        "--output", str(out),
        "--postprocess-strategy", str(strat),
        "--paths-csv", str(paths_csv),
        "--eval-config", str(eval_cfg),
    ])

    assert ns.paths_csv == str(paths_csv)
    assert ns.eval_config == str(eval_cfg)


def test_parse_postprocess_args_missing_required_raises(tmp_path):
    """_parse_postprocess_args raises when required args are missing."""
    with pytest.raises(SystemExit):
        entry._parse_postprocess_args([
            "--output", str(tmp_path / "out"),
            "--postprocess-strategy", str(tmp_path / "post.json"),
        ])


def _touch_json(p: Path, payload=None):
    """Create a JSON file at path p with optional payload."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload if payload is not None else {}, f)


def _write_csv(p: Path, rows: list, fieldnames: list):
    """Write a CSV file with given rows."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# _prepare_io
# ---------------------------------------------------------------------------

def test_prepare_io_creates_predictions_subdir(tmp_path):
    """_prepare_io creates output/predictions/ and returns correct paths."""
    base = tmp_path / "preds"
    base.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "nested" / "out"
    strat = tmp_path / "post.json"
    _touch_json(strat, {"rules": []})

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
    )

    b, o, p, s = entry._prepare_io(ns)
    assert b == base.resolve()
    assert o == out.resolve()
    assert p == out.resolve() / "predictions"
    assert s == strat.resolve()
    assert p.exists()


def test_prepare_io_missing_base_raises(tmp_path):
    """_prepare_io raises when base predictions directory is missing."""
    out = tmp_path / "out"
    strat = tmp_path / "post.json"
    _touch_json(strat)

    ns = argparse.Namespace(
        base_predictions=str(tmp_path / "nope"),
        output=str(out),
        postprocess_strategy=str(strat),
    )
    with pytest.raises(
        FileNotFoundError, match="Base predictions directory not found"
    ):
        entry._prepare_io(ns)


def test_prepare_io_missing_strategy_raises(tmp_path):
    """_prepare_io raises when postprocess strategy json is missing."""
    base = tmp_path / "preds"; base.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(tmp_path / "missing.json"),
    )
    with pytest.raises(
        FileNotFoundError, match="Postprocess strategy file not found"
    ):
        entry._prepare_io(ns)


# ---------------------------------------------------------------------------
# _validate_eval_args
# ---------------------------------------------------------------------------

def test_validate_eval_args_passes_when_neither_provided():
    """No error when both paths_csv and eval_config are None."""
    ns = argparse.Namespace(paths_csv=None, eval_config=None)
    entry._validate_eval_args(ns)  # Should not raise.


def test_validate_eval_args_passes_when_both_provided():
    """No error when both paths_csv and eval_config are set."""
    ns = argparse.Namespace(paths_csv="paths.csv", eval_config="cfg.json")
    entry._validate_eval_args(ns)  # Should not raise.


def test_validate_eval_args_raises_when_only_paths_csv():
    """Raises ValueError when only --paths-csv is provided."""
    ns = argparse.Namespace(paths_csv="paths.csv", eval_config=None)
    with pytest.raises(ValueError, match="--paths-csv and --eval-config"):
        entry._validate_eval_args(ns)


def test_validate_eval_args_raises_when_only_eval_config():
    """Raises ValueError when only --eval-config is provided."""
    ns = argparse.Namespace(paths_csv=None, eval_config="cfg.json")
    with pytest.raises(ValueError, match="--paths-csv and --eval-config"):
        entry._validate_eval_args(ns)


# ---------------------------------------------------------------------------
# _build_eval_filepaths_df
# ---------------------------------------------------------------------------

def test_build_eval_filepaths_df_correct_columns(tmp_path):
    """Builds a DataFrame with id, mask, and derived prediction columns."""
    paths_csv = tmp_path / "paths.csv"
    predictions_dir = tmp_path / "out" / "predictions"
    _write_csv(
        paths_csv,
        [
            {"id": "patient001", "mask": "/data/patient001.nii.gz"},
            {"id": "patient002", "mask": "/data/patient002.nii.gz"},
        ],
        fieldnames=["id", "mask"],
    )

    df = entry._build_eval_filepaths_df(paths_csv, predictions_dir)

    assert list(df.columns) == ["id", "mask", "prediction"]
    assert df.loc[0, "prediction"] == str(
        predictions_dir / "patient001.nii.gz"
    )
    assert df.loc[1, "prediction"] == str(
        predictions_dir / "patient002.nii.gz"
    )


def test_build_eval_filepaths_df_raises_on_missing_columns(tmp_path):
    """Raises ValueError when CSV is missing 'id' or 'mask' columns."""
    paths_csv = tmp_path / "paths.csv"
    _write_csv(
        paths_csv,
        [{"patient": "p001", "file": "/data/p001.nii.gz"}],
        fieldnames=["patient", "file"],
    )
    with pytest.raises(ValueError, match="missing required column"):
        entry._build_eval_filepaths_df(paths_csv, tmp_path / "predictions")


# ---------------------------------------------------------------------------
# _run_evaluation_after_postprocess
# ---------------------------------------------------------------------------

def test_run_evaluation_after_postprocess_missing_paths_csv_raises(tmp_path):
    """Raises FileNotFoundError when paths_csv does not exist."""
    ns = argparse.Namespace(
        paths_csv=str(tmp_path / "nope.csv"),
        eval_config=str(tmp_path / "cfg.json"),
    )
    with pytest.raises(FileNotFoundError, match="Paths CSV not found"):
        entry._run_evaluation_after_postprocess(
            ns, tmp_path / "out", tmp_path / "out" / "predictions"
        )


def test_run_evaluation_after_postprocess_missing_eval_config_raises(tmp_path):
    """Raises FileNotFoundError when eval_config does not exist."""
    paths_csv = tmp_path / "paths.csv"
    _write_csv(paths_csv, [{"id": "p001", "mask": "/m.nii.gz"}], ["id", "mask"])
    ns = argparse.Namespace(
        paths_csv=str(paths_csv),
        eval_config=str(tmp_path / "missing.json"),
    )
    with pytest.raises(FileNotFoundError, match="Evaluation config not found"):
        entry._run_evaluation_after_postprocess(
            ns, tmp_path / "out", tmp_path / "out" / "predictions"
        )


def test_run_evaluation_after_postprocess_invokes_evaluator(tmp_path, monkeypatch):
    """Constructs Evaluator and calls .run with the correct arguments."""
    output_dir = tmp_path / "out"
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True)

    paths_csv = tmp_path / "paths.csv"
    _write_csv(
        paths_csv,
        [{"id": "pat1", "mask": "/masks/pat1.nii.gz"}],
        fieldnames=["id", "mask"],
    )

    eval_cfg = tmp_path / "config.json"
    _touch_json(eval_cfg, {"Tumor": {"labels": [1], "metrics": {"dice": {}}}})

    ns = argparse.Namespace(
        paths_csv=str(paths_csv),
        eval_config=str(eval_cfg),
    )

    captured = {}

    class _EvalStub:
        def __init__(self, *, filepaths_dataframe, evaluation_config,
                     output_csv_path):
            captured["df"] = filepaths_dataframe
            captured["config"] = evaluation_config
            captured["csv"] = output_csv_path

        def run(self, max_workers=1):
            captured["ran"] = True
            captured["max_workers"] = max_workers

    monkeypatch.setattr(entry, "Evaluator", _EvalStub, raising=True)

    entry._run_evaluation_after_postprocess(
        ns, output_dir, predictions_dir, num_workers=3
    )

    assert captured["ran"] is True
    assert captured["max_workers"] == 3
    assert "prediction" in captured["df"].columns
    assert captured["config"] == {
        "Tumor": {"labels": [1], "metrics": {"dice": {}}}
    }
    assert captured["csv"] == output_dir / "postprocess_results.csv"


# ---------------------------------------------------------------------------
# run_postprocess
# ---------------------------------------------------------------------------

def test_run_postprocess_creates_expected_output_structure(tmp_path, monkeypatch):
    """run_postprocess writes predictions/ and strategy.json to output_dir."""
    base = tmp_path / "preds"; base.mkdir()
    out = tmp_path / "out"
    strat = tmp_path / "post.json"; _touch_json(strat, [])

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
        num_workers_postprocess=1,
        num_workers_evaluate=1,
        paths_csv=None,
        eval_config=None,
    )

    class _PPStub:
        def __init__(self, *, strategy_path): pass
        def run(self, *, base_dir, output_dir, num_workers): pass

    monkeypatch.setattr(entry, "Postprocessor", _PPStub, raising=True)

    entry.run_postprocess(ns)

    assert (out / "predictions").is_dir()
    assert (out / "strategy.json").exists()


def test_run_postprocess_passes_predictions_dir_to_postprocessor(
    tmp_path, monkeypatch
):
    """Postprocessor.run receives output/predictions/ as its output_dir."""
    base = tmp_path / "preds"; base.mkdir()
    out = tmp_path / "out"
    strat = tmp_path / "post.json"; _touch_json(strat, [])

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
        num_workers_postprocess=2,
        num_workers_evaluate=1,
        paths_csv=None,
        eval_config=None,
    )

    captured = {}

    class _PPStub:
        def __init__(self, *, strategy_path):
            captured["ctor"] = strategy_path
        def run(self, *, base_dir, output_dir, num_workers):
            captured["run"] = (base_dir, output_dir, num_workers)

    monkeypatch.setattr(entry, "Postprocessor", _PPStub, raising=True)

    entry.run_postprocess(ns)

    assert captured["run"] == (
        base.resolve(), out.resolve() / "predictions", 2
    )


def test_run_postprocess_with_eval_runs_evaluation(tmp_path, monkeypatch):
    """run_postprocess calls _run_evaluation_after_postprocess when args given."""
    base = tmp_path / "preds"; base.mkdir()
    out = tmp_path / "out"
    strat = tmp_path / "post.json"; _touch_json(strat, [])
    paths_csv = tmp_path / "paths.csv"
    eval_cfg = tmp_path / "cfg.json"
    _write_csv(paths_csv, [{"id": "p1", "mask": "/m.nii.gz"}], ["id", "mask"])
    _touch_json(eval_cfg, {})

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
        num_workers_postprocess=1,
        num_workers_evaluate=2,
        paths_csv=str(paths_csv),
        eval_config=str(eval_cfg),
    )

    class _PPStub:
        def __init__(self, *, strategy_path): pass
        def run(self, *, base_dir, output_dir, num_workers): pass

    eval_called = {}

    def _fake_eval(ns_arg, output_dir_arg, predictions_dir_arg, num_workers=1):
        eval_called["output_dir"] = output_dir_arg
        eval_called["predictions_dir"] = predictions_dir_arg
        eval_called["num_workers"] = num_workers

    monkeypatch.setattr(entry, "Postprocessor", _PPStub, raising=True)
    monkeypatch.setattr(
        entry, "_run_evaluation_after_postprocess", _fake_eval, raising=True
    )

    entry.run_postprocess(ns)

    assert eval_called["output_dir"] == out.resolve()
    assert eval_called["predictions_dir"] == out.resolve() / "predictions"
    assert eval_called["num_workers"] == 2


def test_run_postprocess_invalid_eval_args_raises(tmp_path):
    """run_postprocess raises if only one of paths-csv/eval-config given."""
    base = tmp_path / "preds"; base.mkdir()
    out = tmp_path / "out"
    strat = tmp_path / "post.json"; _touch_json(strat, {})

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
        num_workers_postprocess=1,
        num_workers_evaluate=1,
        paths_csv="paths.csv",
        eval_config=None,
    )

    with pytest.raises(ValueError, match="--paths-csv and --eval-config"):
        entry.run_postprocess(ns)


def test_postprocess_entry_integration(monkeypatch):
    """postprocess_entry wires parsing -> running."""
    observed = {"parsed": False, "ran": False}

    def _parse(argv=None):
        observed["parsed"] = True
        return SimpleNamespace(
            base_predictions="/preds",
            output="/out",
            postprocess_strategy="/post.json",
        )

    def _run(ns):
        assert ns.base_predictions == "/preds"
        observed["ran"] = True

    monkeypatch.setattr(entry, "_parse_postprocess_args", _parse, raising=True)
    monkeypatch.setattr(entry, "run_postprocess", _run, raising=True)

    entry.postprocess_entry([
        "--base-predictions", "/preds",
        "--output", "/out",
        "--postprocess-strategy", "/post.json",
    ])

    assert observed["parsed"] is True
    assert observed["ran"] is True
