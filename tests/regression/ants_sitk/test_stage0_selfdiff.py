"""Stage 0 release gate for the ANTs -> SimpleITK migration.

Per the migration plan, Stage 0 ships only once "the diffing script correctly
reports zero differences when diffed against itself as a sanity check." These
tests are that gate. They also prove the differ is not trivially passing by
corrupting an artifact and asserting the diff is detected -- specifically the
axis-transpose case the whole migration is worried about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.regression.ants_sitk import fixtures, harness


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Generate the fixtures once and run the pipeline twice, independently.

    Returns the two artifact maps plus the path replacements needed to compare
    them across their different working directories.
    """
    dataset_root = tmp_path_factory.mktemp("dataset")
    dataset = fixtures.generate_dataset(dataset_root)

    run_a = tmp_path_factory.mktemp("run_a")
    run_b = tmp_path_factory.mktemp("run_b")

    outputs_a = harness.run_pipeline(dataset.dataset_json, run_a / "results", run_a / "numpy")
    outputs_b = harness.run_pipeline(dataset.dataset_json, run_b / "results", run_b / "numpy")

    return {
        "golden": harness.collect_artifacts(outputs_a),
        "candidate": harness.collect_artifacts(outputs_b),
        "replacements": [
            (str(run_a), harness._ROOT_PLACEHOLDER),
            (str(run_b), harness._ROOT_PLACEHOLDER),
        ],
        "patient_ids": dataset.patient_ids,
    }


def test_all_expected_artifacts_present(two_runs: dict[str, object]) -> None:
    """Every fixture must yield config/csv artifacts and per-patient npy files."""
    golden = two_runs["golden"]
    assert "config.json" in golden
    assert "train_paths.csv" in golden
    assert "fg_bboxes.csv" in golden

    patient_ids = two_runs["patient_ids"]
    assert patient_ids, "expected a non-empty fixture set"
    for patient_id in patient_ids:
        assert f"images/{patient_id}.npy" in golden
        assert f"labels/{patient_id}.npy" in golden
        assert f"dtms/{patient_id}.npy" in golden


def test_selfdiff_is_exactly_zero(two_runs: dict[str, object]) -> None:
    """The release gate: two independent runs must be byte-for-byte identical.

    Exact tolerance (atol=rtol=0) proves both determinism of the current
    pipeline and correctness of the differ -- the baseline later stages diff
    against.
    """
    report = harness.diff_artifacts(
        two_runs["golden"],
        two_runs["candidate"],
        replacements=two_runs["replacements"],
        atol=0.0,
        rtol=0.0,
    )
    assert report.identical, str(report)


def test_differ_detects_axis_transpose(two_runs: dict[str, object], tmp_path: Path) -> None:
    """A transposed image array must be reported (the core silent-bug case)."""
    golden = dict(two_runs["golden"])
    candidate = dict(two_runs["candidate"])

    patient_id = two_runs["patient_ids"][0]
    key = f"images/{patient_id}.npy"

    original = np.load(candidate[key])
    transposed = np.ascontiguousarray(np.swapaxes(original, 0, -1))
    corrupted = tmp_path / "transposed.npy"
    np.save(corrupted, transposed)
    candidate[key] = corrupted

    report = harness.diff_artifacts(golden, candidate, replacements=two_runs["replacements"])
    assert not report.identical
    assert any(key in d for d in report.differences), str(report)


def test_differ_detects_label_corruption(two_runs: dict[str, object], tmp_path: Path) -> None:
    """A single flipped label voxel must be reported (exact integer compare)."""
    golden = dict(two_runs["golden"])
    candidate = dict(two_runs["candidate"])

    patient_id = two_runs["patient_ids"][0]
    key = f"labels/{patient_id}.npy"

    labels = np.load(candidate[key]).copy()
    idx = np.unravel_index(0, labels.shape)
    labels[idx] = labels[idx] + 1  # flip one voxel to a different label value.
    corrupted = tmp_path / "labels.npy"
    np.save(corrupted, labels)
    candidate[key] = corrupted

    report = harness.diff_artifacts(golden, candidate, replacements=two_runs["replacements"])
    assert not report.identical
    assert any(key in d for d in report.differences), str(report)


def test_capture_then_diff_roundtrip(tmp_path: Path) -> None:
    """End-to-end CLI path: capture a golden set, then diff regenerates zero."""
    golden_dir = tmp_path / "golden"
    harness.capture(golden_dir)

    # Golden artifacts and manifest were written.
    assert (golden_dir / "config.json").is_file()
    assert (golden_dir / "manifest.json").is_file()

    report = harness.diff_against_golden(golden_dir, atol=0.0, rtol=0.0)
    assert report.identical, str(report)
