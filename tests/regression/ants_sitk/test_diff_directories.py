"""Tests for harness.diff_directories: the real-data-workflow differ.

Stage 2 onward validates migrated stages against a real, perturbed dataset
run out-of-band (e.g. on an HPC box), landing as two directory trees on
disk rather than something this harness regenerates itself (contrast
test_stage0_selfdiff.py, which drives the pipeline directly). These tests
build small directory trees by hand -- postprocessed-mask-style NIfTI files
plus a results.csv, matching what mist_postprocess/mist_evaluate actually
produce -- and prove the differ reports zero differences when the two
directories genuinely match, and catches the specific corruption cases the
migration is worried about (axis/shape mismatch, reorientation, label
corruption) when they don't.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from mist.utils import sitk_io
from tests.regression.ants_sitk import harness


def _write_mask(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    image = sitk_io.image_from_array(array.astype(np.uint8), spacing=spacing)
    sitk_io.write_image(image, str(path))


def _make_run_dir(root: Path, name: str, patient_ids: list[str], seed: int) -> Path:
    """Build a directory shaped like a Stage 2 real-data run: postprocessed/
    masks + results.csv, matching what mist_postprocess/mist_evaluate write.

    `name` (not `seed`) determines the directory, so two runs built with the
    same seed -- meant to be content-identical -- don't collide on disk.
    """
    run_dir = root / name
    masks_dir = run_dir / "postprocessed"
    masks_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    lines = ["id,tumor_dice"]
    for patient_id in patient_ids:
        arr = (rng.integers(0, 3, size=(8, 8, 8))).astype(np.uint8)
        _write_mask(masks_dir / f"{patient_id}.nii.gz", arr)
        lines.append(f"{patient_id},0.9")
    (run_dir / "results.csv").write_text("\n".join(lines) + "\n")
    return run_dir


@pytest.fixture
def matched_run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Two directories built from the *same* seed -- genuinely identical."""
    patient_ids = ["p1", "p2", "p3"]
    golden = _make_run_dir(tmp_path, "golden", patient_ids, seed=0)
    candidate = _make_run_dir(tmp_path, "candidate", patient_ids, seed=0)
    return golden, candidate


def test_collect_directory_artifacts_finds_expected_files(matched_run_dirs):
    """Walks the tree and keys artifacts by path relative to root."""
    golden, _ = matched_run_dirs
    artifacts = harness.collect_directory_artifacts(golden)
    assert "results.csv" in artifacts
    assert "postprocessed/p1.nii.gz" in artifacts
    assert "postprocessed/p2.nii.gz" in artifacts
    assert "postprocessed/p3.nii.gz" in artifacts


def test_collect_directory_artifacts_ignores_unknown_extensions(tmp_path):
    """Files with unrecognized extensions are not collected."""
    (tmp_path / "notes.txt").write_text("scratch")
    (tmp_path / "mask.nii.gz").touch()
    artifacts = harness.collect_directory_artifacts(tmp_path)
    assert "mask.nii.gz" in artifacts
    assert "notes.txt" not in artifacts


def test_identical_directories_diff_as_identical(matched_run_dirs):
    """The release-gate case: two matching runs report zero differences."""
    golden, candidate = matched_run_dirs
    report = harness.diff_directories(golden, candidate, atol=0.0, rtol=0.0)
    assert report.identical, str(report)


def test_missing_patient_file_is_detected(tmp_path):
    """A patient present in golden but absent from candidate is reported."""
    golden = _make_run_dir(tmp_path, "golden", ["p1", "p2"], seed=1)
    candidate = _make_run_dir(tmp_path, "candidate", ["p1"], seed=1)

    report = harness.diff_directories(golden, candidate)
    assert not report.identical
    assert any("p2" in d and "missing in candidate" in d for d in report.differences), str(report)


def test_shape_mismatch_is_detected(matched_run_dirs, tmp_path):
    """A resized mask is reported as a possible axis transpose, not silently
    passed through voxel comparison (which would crash or misalign anyway).
    """
    golden, candidate = matched_run_dirs
    _write_mask(candidate / "postprocessed" / "p1.nii.gz", np.zeros((4, 4, 4), dtype=np.uint8))

    report = harness.diff_directories(golden, candidate)
    assert not report.identical
    assert any(
        "postprocessed/p1.nii.gz" in d and "possible axis transpose" in d
        for d in report.differences
    ), str(report)


def test_label_corruption_is_detected(matched_run_dirs):
    """A single flipped label voxel is caught by the exact integer compare."""
    golden, candidate = matched_run_dirs
    path = candidate / "postprocessed" / "p1.nii.gz"
    image = sitk_io.read_image(str(path))
    array = sitk_io.array_from_image(image)
    array = array.copy()
    array[0, 0, 0] = (int(array[0, 0, 0]) + 1) % 3
    _write_mask(path, array)

    report = harness.diff_directories(golden, candidate)
    assert not report.identical
    assert any(
        "postprocessed/p1.nii.gz" in d and "exact integer compare" in d for d in report.differences
    ), str(report)


def test_reorientation_is_detected(matched_run_dirs):
    """A silently reoriented mask (same shape, different direction) is caught
    even though voxel comparison alone wouldn't necessarily catch it (this is
    the class of bug Stage 1 found and fixed in sitk_io's orientation
    handling -- this test is the corresponding real-data-shaped regression
    check for that same failure mode).
    """
    golden, candidate = matched_run_dirs
    path = candidate / "postprocessed" / "p1.nii.gz"
    image = sitk.ReadImage(str(path))
    image.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    sitk.WriteImage(image, str(path))

    report = harness.diff_directories(golden, candidate)
    assert not report.identical
    assert any(
        "postprocessed/p1.nii.gz" in d and "reorientation" in d for d in report.differences
    ), str(report)


def test_results_csv_metric_difference_is_detected(matched_run_dirs):
    """A results.csv metric value outside tolerance is reported."""
    golden, candidate = matched_run_dirs
    lines = (candidate / "results.csv").read_text().splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0] + ",0.5"  # was 0.9
    (candidate / "results.csv").write_text("\n".join(lines) + "\n")

    report = harness.diff_directories(golden, candidate, atol=1e-5, rtol=1e-5)
    assert not report.identical
    assert any("results.csv" in d for d in report.differences), str(report)


def test_results_csv_within_tolerance_is_identical(matched_run_dirs):
    """A tiny floating-point difference within tolerance is not flagged."""
    golden, candidate = matched_run_dirs
    lines = (candidate / "results.csv").read_text().splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0] + ",0.900001"  # was 0.9
    (candidate / "results.csv").write_text("\n".join(lines) + "\n")

    report = harness.diff_directories(golden, candidate, atol=1e-4, rtol=1e-4)
    assert report.identical, str(report)
