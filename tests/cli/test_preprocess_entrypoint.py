"""Tests for mist.scripts.preprocess_entrypoint."""
from pathlib import Path
import argparse
import pytest

# MIST imports.
from mist.cli import preprocess_entrypoint as entry


def _touch(path: Path, text: str = "") -> None:
    """Create a file at `path` with optional text content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _mk_results_with_required_artifacts(base: Path) -> Path:
    """Create a results/ dir with the 3 files required by preprocess."""
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)
    _touch(results / "config.json", "{}")
    _touch(results / "train_paths.csv", "id\n0\n")
    _touch(
        results / "fg_bboxes.csv",
        "id,x_start,x_end,y_start,y_end,z_start,z_end\n"
    )
    return results


def test_parse_preprocess_args_requires_results(tmp_path, monkeypatch):
    """_parse_preprocess_args should error if --results is missing."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        entry._parse_preprocess_args([])


def test_parse_preprocess_args_defaults_numpy(tmp_path, monkeypatch):
    """If --numpy not provided, default to ./numpy under CWD."""
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "res"
    ns = entry._parse_preprocess_args(["--results", str(results)])
    assert ns.results == str(results)
    # New default is ./numpy (CWD), not <results>/numpy
    assert Path(ns.numpy) == (tmp_path / "numpy").resolve()


def test_parse_preprocess_args_keeps_explicit_numpy(tmp_path, monkeypatch):
    """If --numpy provided, preserve it."""
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "res"
    numpy_dir = tmp_path / "custom_numpy"
    ns = entry._parse_preprocess_args(
        ["--results", str(results), "--numpy", str(numpy_dir)]
    )
    assert Path(ns.numpy) == numpy_dir.resolve()


def test_prepare_preprocess_dirs_creates_dir(tmp_path):
    """_prepare_preprocess_dirs should create the numpy directory."""
    numpy_dir = tmp_path / "np_out"
    ns = argparse.Namespace(numpy=str(numpy_dir), overwrite=False)
    entry._prepare_preprocess_dirs(ns)
    assert numpy_dir.exists() and numpy_dir.is_dir()


def test_prepare_preprocess_dirs_blocks_without_overwrite(tmp_path):
    """Raises if numpy dir exists and is non-empty without --overwrite."""
    numpy_dir = tmp_path / "np_out"
    numpy_dir.mkdir(parents=True, exist_ok=True)
    _touch(numpy_dir / "foo.bin", "x")
    ns = argparse.Namespace(numpy=str(numpy_dir), overwrite=False)

    with pytest.raises(FileExistsError):
        entry._prepare_preprocess_dirs(ns)


def test_prepare_preprocess_dirs_allows_overwrite_on_nonempty(tmp_path):
    """Succeeds when numpy dir exists & non-empty but --overwrite is set."""
    numpy_dir = tmp_path / "np_out"
    numpy_dir.mkdir(parents=True, exist_ok=True)
    _touch(numpy_dir / "bar.bin", "x")
    ns = argparse.Namespace(numpy=str(numpy_dir), overwrite=True)

    entry._prepare_preprocess_dirs(ns)  # should not raise
    assert numpy_dir.exists()


def test_ensure_analyze_artifacts_success(tmp_path):
    """_ensure_analyze_artifacts succeeds when required files are present."""
    results = _mk_results_with_required_artifacts(tmp_path)
    ns = argparse.Namespace(results=str(results))
    out = entry._ensure_analyze_artifacts(ns)
    assert out == results


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (
            lambda r: r.rename(r.parent / "gone"),
            "Results directory does not exist"
        ),
        (
            lambda r: (r / "config.json").unlink(),
            "Missing required file(s)"
        ),
        (
            lambda r: (r / "train_paths.csv").unlink(),
            "Missing required file(s)"
        ),
        (
            lambda r: (r / "fg_bboxes.csv").unlink(),
            "Missing required file(s)"
        ),
    ],
)
def test_ensure_analyze_artifacts_raises(tmp_path, mutator, expected):
    """_ensure_analyze_artifacts raises when files are missing."""
    results = _mk_results_with_required_artifacts(tmp_path)
    mutator(results)
    ns = argparse.Namespace(results=str(results))
    with pytest.raises(FileNotFoundError) as exc:
        entry._ensure_analyze_artifacts(ns)
    assert expected in str(exc.value)


def test_preprocess_entry_integration_parsing_and_run(tmp_path, monkeypatch):
    """Test parses args, check artifacts, create dirs, and runs."""
    monkeypatch.chdir(tmp_path)
    results = _mk_results_with_required_artifacts(tmp_path)
    argv = [
        "--results", str(results),
        "--no-preprocess",
        "--compute-dtms",
    ]

    observed = {
        "called": False, "ns": None, "prep_dir": False
    }

    def _fake_preprocess_dataset(ns):
        observed["called"] = True
        observed["ns"] = ns

    def _fake_set_warning_levels():
        observed["set_warn"] = True

    real_prepare = entry._prepare_preprocess_dirs

    def _wrapped_prepare(ns):
        observed["prep_dir"] = True
        return real_prepare(ns)

    monkeypatch.setattr(
        entry.preprocess,
        "preprocess_dataset",
        _fake_preprocess_dataset,
        raising=True,
    )
    monkeypatch.setattr(
        entry, "_prepare_preprocess_dirs", _wrapped_prepare, raising=True
    )

    entry.preprocess_entry(argv)

    assert observed["prep_dir"] is True
    assert observed["called"] is True

    ns = observed["ns"]
    assert ns is not None
    assert Path(ns.results) == results
    # Default numpy is ./numpy in CWD (tmp_path)
    assert Path(ns.numpy) == (tmp_path / "numpy").resolve()
    assert ns.no_preprocess is True
    assert ns.compute_dtms is True
    assert Path(ns.numpy).exists()


def test_preprocess_entry_blocks_when_numpy_nonempty_without_overwrite(
    tmp_path, monkeypatch
):
    """Raises if default ./numpy contains files and --overwrite not set."""
    monkeypatch.chdir(tmp_path)
    results = _mk_results_with_required_artifacts(tmp_path)

    # Fill the default ./numpy (since we won't pass --numpy).
    default_numpy = tmp_path / "numpy"
    default_numpy.mkdir(parents=True, exist_ok=True)
    _touch(default_numpy / "existing.npy", "x")

    # Keep preprocess from actually running if our guard failed.
    monkeypatch.setattr(
        entry.preprocess, "preprocess_dataset", lambda *_: (_), raising=True
    )

    argv = ["--results", str(results)]
    with pytest.raises(FileExistsError):
        entry.preprocess_entry(argv)


def test_preprocess_entry_raises_when_analyze_artifacts_missing(
    tmp_path, monkeypatch
):
    """Raises FileNotFoundError if required analyze artifacts are missing."""
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)  # Missing the required files.

    monkeypatch.setattr(
        entry.preprocess, "preprocess_dataset", lambda *_: (_), raising=True
    )

    argv = ["--results", str(results)]
    with pytest.raises(FileNotFoundError):
        entry.preprocess_entry(argv)
