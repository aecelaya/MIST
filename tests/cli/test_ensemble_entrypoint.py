"""Tests for mist_ensemble CLI entrypoint."""

import numpy as np
import pytest
import SimpleITK as sitk

from mist.cli.ensemble_entrypoint import (
    _get_patient_ids,
    _parse_ensemble_args,
    _validate_prediction_dirs,
    ensemble_entry,
    run_ensemble,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_label_map(path, arr: np.ndarray) -> None:
    """Write a numpy array as a uint8 NIfTI file."""
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    sitk.WriteImage(img, str(path))


def _make_pred_dir(tmp_path, name: str, patient_ids: list[str], value: int = 1) -> str:
    """Create a prediction directory with one NIfTI per patient."""
    d = tmp_path / name
    d.mkdir()
    for pid in patient_ids:
        arr = np.full((4, 4, 4), value, dtype=np.uint8)
        _write_label_map(d / f"{pid}.nii.gz", arr)
    return str(d)


# ---------------------------------------------------------------------------
# _parse_ensemble_args tests
# ---------------------------------------------------------------------------


def test_parse_ensemble_args_defaults(tmp_path):
    """Default ensemble backend should be 'staple'."""
    d1 = _make_pred_dir(tmp_path, "p1", ["a"])
    d2 = _make_pred_dir(tmp_path, "p2", ["a"])
    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert ns.ensemble_backend == "staple"


def test_parse_ensemble_args_majority_vote(tmp_path):
    """--ensemble-backend majority_vote should be parsed correctly."""
    d1 = _make_pred_dir(tmp_path, "p1", ["a"])
    d2 = _make_pred_dir(tmp_path, "p2", ["a"])
    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            str(tmp_path / "out"),
            "--ensemble-backend",
            "majority_vote",
        ]
    )
    assert ns.ensemble_backend == "majority_vote"


def test_parse_ensemble_args_num_workers_default(tmp_path):
    """Default number of ensemble workers should be 1."""
    d1 = _make_pred_dir(tmp_path, "p1", ["a"])
    d2 = _make_pred_dir(tmp_path, "p2", ["a"])
    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert ns.num_workers_ensemble == 1


def test_parse_ensemble_args_num_workers_custom(tmp_path):
    """--num-workers-ensemble should be parsed correctly."""
    d1 = _make_pred_dir(tmp_path, "p1", ["a"])
    d2 = _make_pred_dir(tmp_path, "p2", ["a"])
    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            str(tmp_path / "out"),
            "--num-workers-ensemble",
            "4",
        ]
    )
    assert ns.num_workers_ensemble == 4


def test_parse_ensemble_args_num_workers_non_positive_raises(tmp_path):
    """--num-workers-ensemble must be a positive integer."""
    d1 = _make_pred_dir(tmp_path, "p1", ["a"])
    d2 = _make_pred_dir(tmp_path, "p2", ["a"])
    with pytest.raises(SystemExit):
        _parse_ensemble_args(
            [
                "--predictions",
                d1,
                d2,
                "--output",
                str(tmp_path / "out"),
                "--num-workers-ensemble",
                "0",
            ]
        )


# ---------------------------------------------------------------------------
# _validate_prediction_dirs tests
# ---------------------------------------------------------------------------


def test_validate_prediction_dirs_valid(tmp_path):
    """Valid existing directories should be returned as resolved Paths."""
    d1 = tmp_path / "dir1"
    d1.mkdir()
    d2 = tmp_path / "dir2"
    d2.mkdir()
    result = _validate_prediction_dirs([str(d1), str(d2)])
    assert len(result) == 2
    assert all(p.is_dir() for p in result)


def test_validate_prediction_dirs_missing_raises(tmp_path):
    """A missing directory should raise FileNotFoundError."""
    d1 = tmp_path / "exists"
    d1.mkdir()
    with pytest.raises(FileNotFoundError, match="not found"):
        _validate_prediction_dirs([str(d1), str(tmp_path / "missing")])


def test_validate_prediction_dirs_fewer_than_two_raises(tmp_path):
    """Fewer than two directories should raise ValueError."""
    d1 = tmp_path / "only"
    d1.mkdir()
    with pytest.raises(ValueError, match="at least two"):
        _validate_prediction_dirs([str(d1)])


# ---------------------------------------------------------------------------
# _get_patient_ids tests
# ---------------------------------------------------------------------------


def test_get_patient_ids_matching(tmp_path):
    """Matching patient IDs across directories should be returned sorted."""
    d1 = _make_pred_dir(tmp_path, "d1", ["pat_b", "pat_a"])
    d2 = _make_pred_dir(tmp_path, "d2", ["pat_a", "pat_b"])
    dirs = _validate_prediction_dirs([d1, d2])
    ids = _get_patient_ids(dirs)
    assert ids == ["pat_a", "pat_b"]


def test_get_patient_ids_mismatch_raises(tmp_path):
    """Mismatched patient IDs should raise ValueError."""
    d1 = _make_pred_dir(tmp_path, "d1", ["pat_a", "pat_b"])
    d2 = _make_pred_dir(tmp_path, "d2", ["pat_a", "pat_c"])
    dirs = _validate_prediction_dirs([d1, d2])
    with pytest.raises(ValueError, match="do not match"):
        _get_patient_ids(dirs)


# ---------------------------------------------------------------------------
# run_ensemble / ensemble_entry happy path tests
# ---------------------------------------------------------------------------


def test_run_ensemble_staple_produces_output(tmp_path):
    """run_ensemble with staple backend should write one file per patient."""
    pids = ["p1", "p2"]
    d1 = _make_pred_dir(tmp_path, "pred1", pids, value=1)
    d2 = _make_pred_dir(tmp_path, "pred2", pids, value=1)
    out = str(tmp_path / "out")

    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            out,
            "--ensemble-backend",
            "staple",
        ]
    )
    run_ensemble(ns)

    for pid in pids:
        assert (tmp_path / "out" / f"{pid}.nii.gz").exists()


def test_run_ensemble_majority_vote_produces_output(tmp_path):
    """run_ensemble with majority_vote backend should write one file per patient."""
    pids = ["p1"]
    d1 = _make_pred_dir(tmp_path, "pred1", pids, value=1)
    d2 = _make_pred_dir(tmp_path, "pred2", pids, value=1)
    d3 = _make_pred_dir(tmp_path, "pred3", pids, value=0)
    out = str(tmp_path / "out_mv")

    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            d3,
            "--output",
            out,
            "--ensemble-backend",
            "majority_vote",
        ]
    )
    run_ensemble(ns)

    assert (tmp_path / "out_mv" / "p1.nii.gz").exists()


def test_run_ensemble_output_values_correct(tmp_path):
    """Majority vote of 2 foreground vs 1 background should yield foreground."""
    pid = "patient"
    foreground = np.ones((4, 4, 4), dtype=np.uint8)
    background = np.zeros((4, 4, 4), dtype=np.uint8)

    for name, arr in [("a", foreground), ("b", foreground), ("c", background)]:
        d = tmp_path / name
        d.mkdir()
        _write_label_map(d / f"{pid}.nii.gz", arr)

    out = str(tmp_path / "out")
    ns = _parse_ensemble_args(
        [
            "--predictions",
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            str(tmp_path / "c"),
            "--output",
            out,
            "--ensemble-backend",
            "majority_vote",
        ]
    )
    run_ensemble(ns)

    result = sitk.GetArrayFromImage(
        sitk.ReadImage(str(tmp_path / "out" / f"{pid}.nii.gz"))
    )
    assert np.array_equal(result, foreground)


def test_run_ensemble_multiple_workers_produces_output(tmp_path):
    """run_ensemble with num_workers_ensemble > 1 should still write all files."""
    pids = ["p1", "p2", "p3", "p4"]
    d1 = _make_pred_dir(tmp_path, "pred1", pids, value=1)
    d2 = _make_pred_dir(tmp_path, "pred2", pids, value=1)
    out = str(tmp_path / "out")

    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            out,
            "--ensemble-backend",
            "majority_vote",
            "--num-workers-ensemble",
            "2",
        ]
    )
    run_ensemble(ns)

    for pid in pids:
        assert (tmp_path / "out" / f"{pid}.nii.gz").exists()


def test_ensemble_entry_runs_without_error(tmp_path):
    """ensemble_entry should complete without raising."""
    pids = ["p1"]
    d1 = _make_pred_dir(tmp_path, "e1", pids, value=1)
    d2 = _make_pred_dir(tmp_path, "e2", pids, value=1)
    out = str(tmp_path / "entry_out")

    ensemble_entry(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            out,
        ]
    )

    assert (tmp_path / "entry_out" / "p1.nii.gz").exists()


# ---------------------------------------------------------------------------
# run_ensemble error handling tests
# ---------------------------------------------------------------------------


def test_run_ensemble_missing_dir_raises(tmp_path):
    """run_ensemble should raise FileNotFoundError for missing directories."""
    d1 = _make_pred_dir(tmp_path, "good", ["p1"])
    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(FileNotFoundError):
        run_ensemble(ns)


def test_run_ensemble_mismatched_ids_raises(tmp_path):
    """run_ensemble should raise ValueError for mismatched patient IDs."""
    d1 = _make_pred_dir(tmp_path, "d1", ["p1", "p2"])
    d2 = _make_pred_dir(tmp_path, "d2", ["p1", "p3"])
    ns = _parse_ensemble_args(
        [
            "--predictions",
            d1,
            d2,
            "--output",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="do not match"):
        run_ensemble(ns)


def test_run_ensemble_per_patient_error_does_not_crash(tmp_path):
    """A corrupt file for one patient should not crash the entire run."""
    pids = ["good", "bad"]
    _make_pred_dir(tmp_path, "d1", pids, value=1)
    _make_pred_dir(tmp_path, "d2", pids, value=1)

    # Corrupt one file in d2.
    (tmp_path / "d2" / "bad.nii.gz").write_bytes(b"not a nifti file")

    out = str(tmp_path / "out")
    ns = _parse_ensemble_args(
        [
            "--predictions",
            str(tmp_path / "d1"),
            str(tmp_path / "d2"),
            "--output",
            out,
        ]
    )
    run_ensemble(ns)  # Should not raise.

    # The good patient should still be written.
    assert (tmp_path / "out" / "good.nii.gz").exists()
