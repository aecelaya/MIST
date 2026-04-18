"""Tests for mist.scripts.inference_entrypoint."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import pandas as pd
import pytest

# MIST imports.
from mist.cli import inference_entrypoint as entry


def test_parse_inference_args_ok(tmp_path, monkeypatch):
    """Test _parse_inference_args with valid arguments."""
    models = tmp_path / "models"
    cfg = tmp_path / "config.json"
    paths = tmp_path / "paths.csv"
    out = tmp_path / "out"

    ns = entry._parse_inference_args([
        "--models-dir", str(models),
        "--config", str(cfg),
        "--paths-csv", str(paths),
        "--output", str(out),
        "--device", "cpu",
    ])

    assert ns.models_dir == str(models)
    assert ns.config == str(cfg)
    assert ns.paths_csv == str(paths)
    assert ns.output == str(out)
    assert ns.device == "cpu"
    assert ns.postprocess_strategy is None


def test_parse_inference_args_missing_required_raises(tmp_path):
    """Test _parse_inference_args raises on missing required arguments."""
    # Missing --models-dir.
    with pytest.raises(SystemExit):
        entry._parse_inference_args([
            "--config", str(tmp_path / "config.json"),
            "--paths-csv", str(tmp_path / "paths.csv"),
            "--output", str(tmp_path / "out"),
        ])


@pytest.mark.parametrize(
    "is_avail, dev_in, expected_type",
    [
        (False, "cpu", "cpu"),
        (False, "cuda", "cpu"),   # fall back when unavailable
        (True, "cuda", "cuda"),
    ],
)
def test_resolve_device_cpu_cuda(monkeypatch, is_avail, dev_in, expected_type):
    """Test _resolve_device for CPU and CUDA."""
    monkeypatch.setattr(
        entry.torch.cuda, "is_available", lambda: is_avail, raising=True
    )
    dev = entry._resolve_device(dev_in)
    assert isinstance(dev, entry.torch.device)
    assert dev.type == expected_type


def test_resolve_device_numeric_available(monkeypatch):
    """Test _resolve_device with numeric CUDA index when available."""
    monkeypatch.setattr(
        entry.torch.cuda, "is_available", lambda: True, raising=True
    )
    monkeypatch.setattr(
        entry.torch.cuda, "device_count", lambda: 2, raising=True
    )
    dev = entry._resolve_device("1")
    assert dev.type == "cuda" and dev.index == 1


def test_resolve_device_numeric_unavailable_warns_and_cpu(monkeypatch):
    """Test _resolve_device with numeric CUDA index when unavailable."""
    monkeypatch.setattr(
        entry.torch.cuda, "is_available", lambda: False, raising=True
    )
    with pytest.warns(UserWarning, match="falling back to CPU"):
        dev = entry._resolve_device("0")
    assert dev.type == "cpu"


def test_resolve_device_invalid_string_raises():
    """Test _resolve_device raises on invalid device string."""
    with pytest.raises(ValueError, match="Invalid device specification"):
        entry._resolve_device("cuda:0")  # Invalid per our CLI (expects "0").


def _touch_json(p: Path, payload=None):
    """Create a JSON file with the given payload."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload if payload is not None else {}, f)


def _touch_csv(p: Path, rows=None):
    """Create a CSV file with the given rows."""
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows if rows is not None else [{"id": "p1", "image": "/tmp/p1.nii.gz"}]
    )
    df.to_csv(p, index=False)


def test_prepare_io_success_with_optional_postprocess(tmp_path, monkeypatch):
    """Test _prepare_io with all required and optional arguments."""
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "results" / "config.json"
    paths = tmp_path / "paths.csv"
    out = tmp_path / "out"
    pps = tmp_path / "post.json"

    _touch_json(cfg, {})
    _touch_csv(paths, [{"id": "p1", "image": "/tmp/p1.nii.gz"}])
    _touch_json(pps, {"post": True})

    ns = argparse.Namespace(
        models_dir=str(models),
        config=str(cfg),
        paths_csv=str(paths),
        output=str(out),
        postprocess_strategy=str(pps),
    )

    md, cp, pc, od = entry._prepare_io(ns)
    assert md == models.resolve()
    assert cp == cfg.resolve()
    assert pc == paths.resolve()
    assert od == out.resolve()
    assert od.exists()


@pytest.mark.parametrize(
    "missing_field",
    ["models_dir", "config", "paths_csv", "postprocess_strategy"]
)
def test_prepare_io_missing_raises(tmp_path, missing_field):
    """Test _prepare_io raises on missing required fields."""
    # Create paths but remove one based on missing_field.
    models = tmp_path / "models"
    cfg = tmp_path / "config.json"
    paths = tmp_path / "paths.csv"
    out = tmp_path / "out"

    # Create only some of them.
    models.mkdir(parents=True, exist_ok=True)
    _touch_json(cfg, {})
    _touch_csv(paths)

    pps = tmp_path / "pps.json"  # Do NOT create.

    ns = argparse.Namespace(
        models_dir=str(models),
        config=str(cfg),
        paths_csv=str(paths),
        output=str(out),
        postprocess_strategy=(
            str(pps) if missing_field == "postprocess_strategy" else None
        ),
    )

    # Remove whichever path we want missing.
    if missing_field == "models_dir":
        ns.models_dir = str(tmp_path / "no_models")
    elif missing_field == "config":
        ns.config = str(tmp_path / "no_cfg.json")
    elif missing_field == "paths_csv":
        ns.paths_csv = str(tmp_path / "no_paths.csv")
    with pytest.raises(FileNotFoundError):
        entry._prepare_io(ns)


def test_run_inference_calls_infer_with_expected_args(tmp_path, monkeypatch):
    """Test run_inference initializes inference_runners.infer_from_dataframe."""
    # Files and dirs.
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "config.json"
    _touch_json(cfg, {"foo": "bar"})
    paths = tmp_path / "paths.csv"
    _touch_csv(paths)
    out = tmp_path / "out"

    # Stub validation + runners.
    monkeypatch.setattr(
        entry.inference_utils,
        "validate_paths_dataframe",
        lambda df: None,
        raising=True,
    )

    captured = {}

    def _infer_from_df(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        entry.inference_runners,
        "infer_from_dataframe",
        _infer_from_df,
        raising=True,
    )

    # Ensure device resolution yields a stable device (CPU).
    monkeypatch.setattr(
        entry.torch.cuda, "is_available", lambda: False, raising=True
    )

    ns = argparse.Namespace(
        models_dir=str(models),
        config=str(cfg),
        paths_csv=str(paths),
        output=str(out),
        device="cuda",  # Will fall back to CPU.
        postprocess_strategy=None,
    )

    entry.run_inference(ns)

    # Output dir created.
    assert out.exists()

    # Verify we passed expected arguments to infer_from_dataframe.
    assert isinstance(captured["paths_dataframe"], pd.DataFrame)
    assert captured["output_directory"] == str(out.resolve())
    assert captured["mist_configuration"] == {"foo": "bar"}
    assert captured["models_directory"] == str(models.resolve())
    assert captured["postprocessing_strategy_filepath"] is None
    assert isinstance(captured["device"], entry.torch.device)
    assert captured["device"].type in ("cpu", "cuda")  # Depending on env mock.


def test_run_inference_with_postprocess_strategy(tmp_path, monkeypatch):
    """Test run_inference with postprocess strategy."""
    # Files and dirs.
    models = tmp_path / "models"; models.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "config.json"; _touch_json(cfg, {})
    paths = tmp_path / "paths.csv"; _touch_csv(paths)
    out = tmp_path / "out"
    pps = tmp_path / "pps.json"; _touch_json(pps, {"post": True})

    # Stubs.
    monkeypatch.setattr(
        entry.inference_utils,
        "validate_paths_dataframe",
        lambda df: None,
        raising=True
    )

    got = {}
    monkeypatch.setattr(
        entry.inference_runners, "infer_from_dataframe",
        lambda **kwargs: got.update(kwargs), raising=True
    )
    monkeypatch.setattr(
        entry.torch.cuda, "is_available", lambda: False, raising=True
    )

    ns = argparse.Namespace(
        models_dir=str(models),
        config=str(cfg),
        paths_csv=str(paths),
        output=str(out),
        device="cpu",
        postprocess_strategy=str(pps),
    )

    entry.run_inference(ns)

    assert got["postprocessing_strategy_filepath"] == str(pps.resolve())


def test_inference_entry_integration(monkeypatch):
    """Test inference_entry integrates parsing and running."""
    observed = {"parsed": False, "ran": False}

    def _parse(argv=None):
        observed["parsed"] = True
        return SimpleNamespace(
            models_dir="/m",
            config="/c.json",
            paths_csv="/p.csv",
            output="/o",
            device="cpu",
            postprocess_strategy=None,
        )

    def _run(ns):
        assert ns.models_dir == "/m"
        observed["ran"] = True

    monkeypatch.setattr(entry, "_parse_inference_args", _parse, raising=True)
    monkeypatch.setattr(entry, "run_inference", _run, raising=True)

    entry.inference_entry([
        "--models-dir", "/m",
        "--config", "/c.json",
        "--paths-csv", "/p.csv",
        "--output", "/o",
    ])
    assert observed["parsed"] is True
    assert observed["ran"] is True
