"""Tests for mist.analyze_data.data_dump_utils."""
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ants

from mist.analyze_data import data_dump_utils as ddu
from mist.analyze_data.analyzer_constants import AnalyzeConstants as constants
from mist.utils import progress_bar as pb_mod
from tests.analyze_data.helpers import FakePB


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ants_image(arr: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """Wrap a numpy array in an ANTs image with given spacing."""
    img = ants.from_numpy(arr.astype(np.float32))
    img.set_spacing(spacing)
    return img


def _make_paths_df(
    mask_paths, channel_paths: dict[str, list]
) -> pd.DataFrame:
    """Build a minimal paths DataFrame for the DataDumper's expected layout."""
    data = {
        "id": list(range(len(mask_paths))),
        "fold": list(range(len(mask_paths))),
        "mask": mask_paths,
    }
    data.update(channel_paths)
    return pd.DataFrame(data)


def _make_dataset_info(
    labels=(0, 1, 2),
    channels=("t1",),
    final_classes=None,
) -> dict[str, Any]:
    """Return a minimal dataset_info dictionary."""
    if final_classes is None:
        final_classes = {"tumor": [lbl for lbl in labels if lbl != 0]}
    return {
        "labels": list(labels),
        "images": {ch: [ch] for ch in channels},
        "final_classes": final_classes,
    }


def _make_config(resampled=(64, 64, 32)) -> dict[str, Any]:
    """Return a minimal MIST config dict."""
    return {"preprocessing": {"median_resampled_image_size": list(resampled)}}


def _make_raw_stats(n=3, non_bg=(1, 2), effective_dims=None) -> dict[str, Any]:
    """Build a controlled raw_stats dict for pure-function tests."""
    original = np.tile([64.0, 64.0, 32.0], (n, 1))
    return {
        "spacings": np.tile([1.0, 1.0, 2.5], (n, 1)),
        "original_dims": original,
        "effective_dims": effective_dims if effective_dims is not None else original,
        "foreground_fractions": np.full(n, 0.8),
        "total_fg_voxels": [1000] * n,
        "channel_intensities": {
            "t1": list(np.linspace(-100, 400, 200))
        },
        "label_voxel_counts": {
            lbl: [100, 200, 150][:n] for lbl in non_bg
        },
        "label_presence": {lbl: [1] * n for lbl in non_bg},
        "label_shape_descriptors": {
            lbl: [
                {
                    "linearity": 0.7,
                    "planarity": 0.2,
                    "sphericity": 0.1,
                    "shape_class": "tubular",
                    "compactness": 0.05,
                    "skeleton_ratio": 0.12,
                }
            ]
            for lbl in non_bg
        },
    }


# ---------------------------------------------------------------------------
# get_dataset_size_gb
# ---------------------------------------------------------------------------

class TestGetDatasetSizeGb:
    """Tests for the get_dataset_size_gb helper."""

    def test_sums_existing_files(self, tmp_path: Path):
        """Size equals sum of mask and channel file sizes."""
        f1 = tmp_path / "a.nii.gz"
        f2 = tmp_path / "b.nii.gz"
        f1.write_bytes(b"x" * 1000)
        f2.write_bytes(b"x" * 2000)
        df = pd.DataFrame({"id": [0], "mask": [str(f1)], "t1": [str(f2)]})
        result = ddu.get_dataset_size_gb(df)
        assert result == round(3000 / 1e9, 4)

    def test_skips_id_and_fold_columns(self, tmp_path: Path):
        """'id' and 'fold' column values are never stat-ed, even if valid paths."""
        f = tmp_path / "real.nii.gz"
        f.write_bytes(b"x" * 500)
        df = pd.DataFrame({"id": [str(f)], "fold": [str(f)], "mask": [str(f)]})
        result = ddu.get_dataset_size_gb(df)
        # Only 'mask' counted, not 'id' or 'fold'.
        assert result == round(500 / 1e9, 4)

    def test_skips_nonexistent_and_nan(self, tmp_path: Path):
        """Non-existent paths and NaN values contribute nothing."""
        df = pd.DataFrame({
            "id": [0],
            "mask": [str(tmp_path / "missing.nii.gz")],
            "t1": [float("nan")],
        })
        assert ddu.get_dataset_size_gb(df) == 0.0

    def test_empty_df_returns_zero(self):
        """Empty DataFrame yields zero."""
        df = pd.DataFrame({"id": [], "mask": []})
        assert ddu.get_dataset_size_gb(df) == 0.0


# ---------------------------------------------------------------------------
# compute_shape_descriptors
# ---------------------------------------------------------------------------

class TestComputeShapeDescriptors:
    """Tests for PCA-based shape descriptor computation."""

    def test_fewer_than_4_points_returns_none(self):
        """Degenerate input with < 4 points returns None."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        assert ddu.compute_shape_descriptors(coords) is None

    def test_tubular_shape_detected(self):
        """Points along one axis → linearity dominant → 'tubular'."""
        # 1000 points along x-axis; minimal spread in y, z.
        rng = np.random.default_rng(42)
        coords = np.column_stack([
            np.linspace(0, 100, 1000),
            rng.normal(0, 0.1, 1000),
            rng.normal(0, 0.1, 1000),
        ])
        result = ddu.compute_shape_descriptors(coords)
        assert result is not None
        assert result["shape_class"] == "tubular"
        assert result["linearity"] > result["planarity"]
        assert result["linearity"] > result["sphericity"]
        total = (
            result["linearity"]
            + result["planarity"]
            + result["sphericity"]
        )
        assert abs(total - 1.0) < 1e-4

    def test_planar_shape_detected(self):
        """Points in a 2D plane → planarity dominant → shape_class = 'planar'."""
        rng = np.random.default_rng(0)
        coords = np.column_stack([
            rng.uniform(0, 100, 2000),
            rng.uniform(0, 100, 2000),
            rng.normal(0, 0.1, 2000),     # negligible spread along z
        ])
        result = ddu.compute_shape_descriptors(coords)
        assert result is not None
        assert result["shape_class"] == "planar"
        assert result["planarity"] > result["sphericity"]

    def test_blob_shape_detected(self):
        """Points in a sphere → sphericity dominant → shape_class = 'blob'."""
        rng = np.random.default_rng(7)
        coords = rng.normal(0, 10, (2000, 3))
        result = ddu.compute_shape_descriptors(coords)
        assert result is not None
        assert result["shape_class"] == "blob"
        assert result["sphericity"] > 0.2

    def test_large_input_is_subsampled(self):
        """Inputs > MAX_SHAPE_COORDS are subsampled and still valid."""
        rng = np.random.default_rng(1)
        n = constants.MAX_SHAPE_COORDS + 500
        coords = np.column_stack([
            np.linspace(0, 100, n),
            rng.normal(0, 0.1, n),
            rng.normal(0, 0.1, n),
        ])
        result = ddu.compute_shape_descriptors(coords)
        assert result is not None
        assert result["shape_class"] == "tubular"

    def test_degenerate_identical_points_returns_none(self):
        """All-identical coordinates → zero covariance → returns None."""
        coords = np.zeros((100, 3))
        assert ddu.compute_shape_descriptors(coords) is None

    def test_wrong_ndim_input_returns_none(self):
        """2-column coords produce a (2,2) covariance → returns None."""
        coords_2d = np.random.default_rng(0).normal(0, 1, (50, 2))
        assert ddu.compute_shape_descriptors(coords_2d) is None


# ---------------------------------------------------------------------------
# compute_compactness
# ---------------------------------------------------------------------------

class TestComputeCompactness:
    """Tests for the isoperimetric-quotient compactness helper."""

    def test_empty_mask_returns_none(self):
        """All-zero mask → voxel_count == 0 → returns None."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        assert ddu.compute_compactness(mask, np.array([1.0, 1.0, 1.0])) is None

    def test_near_zero_surface_area_returns_none(self):
        """Single voxel with negligibly small spacing → SA < 1e-8 → None."""
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        # Tiny spacing makes each face contribution ~ (1e-4)^2 = 1e-8.
        # With 6 faces × (1e-4)^2 = 6e-8 — still above threshold, so use
        # extremely small spacing to force SA below 1e-8.
        spacing = np.array([1e-5, 1e-5, 1e-5])
        result = ddu.compute_compactness(mask, spacing)
        assert result is None

    def test_valid_sphere_like_mask_returns_float_in_range(self):
        """A compact cube returns a float value in (0, 1]."""
        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[1:6, 1:6, 1:6] = True  # 5^3 = 125 voxel cube
        spacing = np.array([1.0, 1.0, 1.0])
        result = ddu.compute_compactness(mask, spacing)
        assert result is not None
        assert 0.0 < result <= 1.0

    def test_thin_mask_has_lower_compactness_than_cube(self):
        """A thin tube has lower compactness (more SA per volume) than a cube."""
        # Cube
        cube = np.zeros((10, 10, 10), dtype=bool)
        cube[2:8, 2:8, 2:8] = True
        spacing = np.array([1.0, 1.0, 1.0])
        iq_cube = ddu.compute_compactness(cube, spacing)

        # Thin tube: 1×1×8
        tube = np.zeros((10, 10, 10), dtype=bool)
        tube[4, 4, 1:9] = True
        iq_tube = ddu.compute_compactness(tube, spacing)

        assert iq_cube is not None and iq_tube is not None
        assert iq_tube < iq_cube


# ---------------------------------------------------------------------------
# compute_skeleton_ratio
# ---------------------------------------------------------------------------

class TestComputeSkeletonRatio:
    """Tests for the skeleton-ratio helper."""

    def test_empty_mask_returns_none(self):
        """All-zero mask → voxel_count == 0 → returns None."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        assert ddu.compute_skeleton_ratio(mask) is None

    def test_oversized_mask_returns_none(self):
        """Labels exceeding MAX_SKELETON_VOXELS → skipped → returns None."""
        # 80^3 = 512 000 > MAX_SKELETON_VOXELS (500 000)
        mask = np.ones((80, 80, 80), dtype=bool)
        assert ddu.compute_skeleton_ratio(mask) is None

    def test_thin_line_has_high_ratio(self):
        """A 1-voxel-wide line → nearly all voxels on skeleton → high ratio."""
        mask = np.zeros((3, 3, 20), dtype=bool)
        mask[1, 1, :] = True  # 20 voxels in a straight line
        result = ddu.compute_skeleton_ratio(mask)
        assert result is not None
        assert result > 0.5

    def test_solid_cube_has_low_ratio(self):
        """A filled cube has far more interior than skeleton voxels → low ratio."""
        mask = np.zeros((15, 15, 15), dtype=bool)
        mask[2:13, 2:13, 2:13] = True  # 11^3 = 1331 voxels
        result = ddu.compute_skeleton_ratio(mask)
        assert result is not None
        assert result < 0.2


# ---------------------------------------------------------------------------
# _axis_stats  (accessed via module for coverage)
# ---------------------------------------------------------------------------

class TestAxisStats:
    """Tests for the _axis_stats helper."""

    def test_all_statistics_correct(self):
        """Each statistic is computed correctly for a known array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ddu._axis_stats(arr)
        assert result["mean"] == round(float(np.mean(arr)), 4)
        assert result["std"] == round(float(np.std(arr)), 4)
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["median"] == 3.0
        assert set(result.keys()) == {
            "mean", "std", "min", "p25", "median", "p75", "max"
        }

    def test_single_element(self):
        """Single-element array: mean == median == min == max == the value."""
        result = ddu._axis_stats(np.array([7.0]))
        assert result["mean"] == 7.0
        assert result["std"] == 0.0
        assert result["min"] == result["max"] == result["median"] == 7.0


# ---------------------------------------------------------------------------
# _size_category
# ---------------------------------------------------------------------------

class TestSizeCategory:
    """Tests for the _size_category helper."""

    @pytest.mark.parametrize("vol_frac,expected", [
        pytest.param(0.0, "tiny", id="zero"),
        pytest.param(0.05, "tiny", id="tiny_mid"),
        pytest.param(0.1, "small", id="small_boundary"),
        pytest.param(0.5, "small", id="small_mid"),
        pytest.param(1.0, "medium", id="medium_boundary"),
        pytest.param(2.5, "medium", id="medium_mid"),
        pytest.param(5.0, "large", id="large_boundary"),
        pytest.param(20.0, "large", id="large_high"),
    ])
    def test_category_boundaries(self, vol_frac, expected):
        """Correct category is returned for each boundary and interior value."""
        assert ddu._size_category(vol_frac) == expected


# ---------------------------------------------------------------------------
# build_image_statistics
# ---------------------------------------------------------------------------

class TestBuildImageStatistics:
    """Tests for build_image_statistics."""

    def test_structure_has_required_keys(self):
        """Output contains spacing, dimensions, and intensity top-level keys."""
        raw = _make_raw_stats()
        config = _make_config()
        result = ddu.build_image_statistics(raw, config)
        assert set(result.keys()) == {
            "spacing", "dimensions", "intensity"
        }
        assert "per_axis" in result["spacing"]
        assert "anisotropy_ratio" in result["spacing"]
        assert "is_anisotropic" in result["spacing"]
        assert "original" in result["dimensions"]
        assert "resampled_median" in result["dimensions"]
        assert "per_channel" in result["intensity"]
        assert "foreground_fraction" in result["intensity"]

    def test_spacing_stats_per_axis(self):
        """Per-axis spacing statistics match the input spacings."""
        raw = _make_raw_stats()
        result = ddu.build_image_statistics(raw, _make_config())
        # Spacing is identical for all patients: [1.0, 1.0, 2.5].
        assert result["spacing"]["per_axis"]["axis_0"]["mean"] == 1.0
        assert result["spacing"]["per_axis"]["axis_2"]["mean"] == 2.5

    def test_isotropic_not_flagged(self):
        """Equal spacing produces is_anisotropic = False."""
        raw = _make_raw_stats()
        raw["spacings"] = np.ones((3, 3))
        result = ddu.build_image_statistics(raw, _make_config())
        assert result["spacing"]["is_anisotropic"] is False
        assert result["spacing"]["anisotropy_ratio"] == pytest.approx(
            1.0, abs=1e-3
        )

    def test_anisotropic_flagged(self):
        """Spacing with ratio > 3 is flagged as anisotropic."""
        raw = _make_raw_stats()
        raw["spacings"] = np.tile([1.0, 1.0, 5.0], (3, 1))
        result = ddu.build_image_statistics(raw, _make_config())
        assert result["spacing"]["is_anisotropic"] is True
        assert result["spacing"]["anisotropy_ratio"] == pytest.approx(
            5.0, abs=0.01
        )

    def test_resampled_median_comes_from_config(self):
        """resampled_median is taken directly from config, not recomputed."""
        result = ddu.build_image_statistics(
            _make_raw_stats(), _make_config([32, 64, 48])
        )
        assert result["dimensions"]["resampled_median"] == [32, 64, 48]

    def test_empty_channel_is_skipped(self):
        """A channel with no intensity samples is absent from per_channel."""
        raw = _make_raw_stats()
        raw["channel_intensities"] = {"t1": [1.0, 2.0, 3.0], "t2": []}
        result = ddu.build_image_statistics(raw, _make_config())
        assert "t1" in result["intensity"]["per_channel"]
        assert "t2" not in result["intensity"]["per_channel"]

    def test_intensity_percentiles_correct(self):
        """Intensity statistics match known numpy percentile outputs."""
        vals = list(range(1, 101))  # 1 to 100
        raw = _make_raw_stats()
        raw["channel_intensities"] = {"t1": vals}
        result = ddu.build_image_statistics(raw, _make_config())
        stats = result["intensity"]["per_channel"]["t1"]
        assert stats["p50"] == pytest.approx(
            np.percentile(vals, 50), abs=0.01
        )
        assert stats["p01"] == pytest.approx(
            np.percentile(vals, 1), abs=0.01
        )
        assert stats["p99"] == pytest.approx(
            np.percentile(vals, 99), abs=0.01
        )


# ---------------------------------------------------------------------------
# build_label_statistics
# ---------------------------------------------------------------------------

class TestBuildLabelStatistics:
    """Tests for build_label_statistics."""

    def test_structure_has_required_keys(self):
        """Output contains per_label, final_classes, and class_imbalance."""
        ds_info = _make_dataset_info()
        result = ddu.build_label_statistics(_make_raw_stats(), ds_info)
        assert set(result.keys()) == {
            "per_label", "final_classes", "class_imbalance"
        }

    def test_per_label_voxel_stats_correct(self):
        """Mean voxel count matches the average of per-patient counts."""
        raw = _make_raw_stats(n=2, non_bg=(1,))
        raw["label_voxel_counts"] = {1: [100, 200]}
        ds_info = _make_dataset_info(labels=(0, 1), final_classes={"fg": [1]})
        result = ddu.build_label_statistics(raw, ds_info)
        vc = result["per_label"]["1"]["voxel_count"]
        assert vc["mean"] == pytest.approx(150.0)
        assert vc["min"] == 100
        assert vc["max"] == 200

    def test_image_fraction_correct(self):
        """Vol. fraction of image = mean label voxels / mean effective image voxels."""
        raw = _make_raw_stats(n=2, non_bg=(1,))
        # effective_dims defaults to original_dims: 64×64×32 = 131072 voxels.
        raw["label_voxel_counts"] = {1: [1310, 1310]}  # ~1% of 131072
        ds_info = _make_dataset_info(labels=(0, 1), final_classes={"fg": [1]})
        result = ddu.build_label_statistics(raw, ds_info)
        img_frac = result["per_label"]["1"]["mean_volume_fraction_of_image_pct"]
        expected = 1310 / (64 * 64 * 32) * 100
        assert img_frac == pytest.approx(expected, rel=0.01)

    def test_image_fraction_uses_effective_dims_when_crop_active(self):
        """When effective_dims differs from original_dims, uses effective_dims."""
        # Simulate crop_to_foreground: bounding box is 32×32×20 per patient.
        eff = np.tile([32.0, 32.0, 20.0], (2, 1))
        raw = _make_raw_stats(n=2, non_bg=(1,), effective_dims=eff)
        raw["label_voxel_counts"] = {1: [200, 200]}
        ds_info = _make_dataset_info(labels=(0, 1), final_classes={"fg": [1]})
        result = ddu.build_label_statistics(raw, ds_info)
        img_frac = result["per_label"]["1"]["mean_volume_fraction_of_image_pct"]
        # Denominator is 32*32*20 = 20480, not 64*64*32 = 131072.
        expected = 200 / (32 * 32 * 20) * 100
        assert img_frac == pytest.approx(expected, rel=0.01)

    def test_presence_rate_correct(self):
        """Presence rate is percentage of patients where label appears."""
        raw = _make_raw_stats(n=4, non_bg=(1,))
        raw["label_voxel_counts"] = {1: [100, 0, 50, 0]}
        raw["label_presence"] = {1: [1, 0, 1, 0]}
        ds_info = _make_dataset_info(
            labels=(0, 1), final_classes={"fg": [1]}
        )
        result = ddu.build_label_statistics(raw, ds_info)
        assert result["per_label"]["1"]["presence_rate_pct"] == (
            pytest.approx(50.0)
        )

    def test_shape_with_descriptors_populated(self):
        """Shape fields are populated when descriptors are present."""
        raw = _make_raw_stats(non_bg=(1,))
        raw["label_shape_descriptors"] = {
            1: [
                {
                    "linearity": 0.7,
                    "planarity": 0.2,
                    "sphericity": 0.1,
                    "shape_class": "tubular",
                }
            ]
        }
        ds_info = _make_dataset_info(labels=(0, 1), final_classes={"fg": [1]})
        result = ddu.build_label_statistics(raw, ds_info)
        shape = result["per_label"]["1"]["shape"]
        assert shape["shape_class"] == "tubular"
        assert shape["linearity"] == pytest.approx(0.7)
        assert shape["planarity"] is not None

    def test_shape_empty_descriptors_gives_unknown(self):
        """Empty descriptor list produces shape_class 'unknown' with Nones."""
        raw = _make_raw_stats(non_bg=(1,))
        raw["label_shape_descriptors"] = {1: []}
        ds_info = _make_dataset_info(labels=(0, 1), final_classes={"fg": [1]})
        result = ddu.build_label_statistics(raw, ds_info)
        shape = result["per_label"]["1"]["shape"]
        assert shape["shape_class"] == "unknown"
        assert shape["linearity"] is None
        assert shape["planarity"] is None
        assert shape["sphericity"] is None

    def test_class_imbalance_two_labels(self):
        """Imbalance ratio equals fraction of dominant / fraction of minor."""
        # Label 1: mean 900/1000 = 90% FG.  Label 2: mean 10/1000 = 1% FG.
        raw = _make_raw_stats(n=2, non_bg=(1, 2))
        raw["total_fg_voxels"] = [1000, 1000]
        raw["label_voxel_counts"] = {1: [900, 900], 2: [10, 10]}
        raw["label_presence"] = {1: [1, 1], 2: [1, 1]}
        ds_info = _make_dataset_info(
            labels=(0, 1, 2), final_classes={"a": [1], "b": [2]}
        )
        result = ddu.build_label_statistics(raw, ds_info)
        ci = result["class_imbalance"]
        assert ci["dominant_label"] == 1
        assert ci["minority_label"] == 2
        assert ci["imbalance_ratio"] == pytest.approx(90.0, rel=0.01)

    def test_single_label_no_imbalance(self):
        """One non-background label: minority_label is None, ratio is 1.0."""
        raw = _make_raw_stats(n=2, non_bg=(1,))
        ds_info = _make_dataset_info(
            labels=(0, 1), final_classes={"fg": [1]}
        )
        result = ddu.build_label_statistics(raw, ds_info)
        ci = result["class_imbalance"]
        assert ci["minority_label"] is None
        assert ci["imbalance_ratio"] == 1.0

    def test_final_classes_combines_constituent_labels(self):
        """Final class aggregates voxel counts across all constituent labels."""
        raw = _make_raw_stats(n=2, non_bg=(1, 2))
        raw["total_fg_voxels"] = [1000, 1000]
        raw["label_voxel_counts"] = {1: [300, 300], 2: [100, 100]}
        raw["label_presence"] = {1: [1, 1], 2: [1, 1]}
        ds_info = _make_dataset_info(
            labels=(0, 1, 2), final_classes={"combined": [1, 2]}
        )
        result = ddu.build_label_statistics(raw, ds_info)
        # Combined mean = (300+100) / 1000 * 100 = 40%
        key = "mean_volume_fraction_of_foreground_pct"
        assert (
            result["final_classes"]["combined"][key]
            == pytest.approx(40.0, rel=0.01)
        )

    def test_final_classes_skips_background_label(self):
        """Background label 0 in final_classes is safely ignored."""
        raw = _make_raw_stats(n=2, non_bg=(1,))
        raw["total_fg_voxels"] = [1000, 1000]
        raw["label_voxel_counts"] = {1: [200, 200]}
        raw["label_presence"] = {1: [1, 1]}
        # final_classes includes 0 which is background
        ds_info = _make_dataset_info(
            labels=(0, 1), final_classes={"fg": [0, 1]}
        )
        # Should not raise and should count only label 1
        result = ddu.build_label_statistics(raw, ds_info)
        frac = result["final_classes"]["fg"][
            "mean_volume_fraction_of_foreground_pct"
        ]
        assert frac == pytest.approx(20.0, rel=0.01)


# ---------------------------------------------------------------------------
# generate_observations
# ---------------------------------------------------------------------------

def _base_image_stats() -> dict[str, Any]:
    """Baseline image_stats with no observations triggered."""
    return {
        "spacing": {"anisotropy_ratio": 1.5, "is_anisotropic": False},
        "intensity": {
            "foreground_fraction": {"mean": 0.9},
            "per_channel": {
                "t1": {
                    "p01": -100.0,
                    "p99": 300.0,
                    "mean": 50.0,
                    "std": 80.0,
                }
            },
        },
    }


def _base_label_stats() -> dict[str, Any]:
    """Baseline label_stats with no observations triggered."""
    return {
        "class_imbalance": {
            "imbalance_ratio": 1.5,
            "dominant_label": 1,
            "minority_label": 2,
        },
        "per_label": {
            "1": {
                "mean_volume_fraction_of_foreground_pct": 30.0,
                "mean_volume_fraction_of_image_pct": 5.0,
                "presence_rate_pct": 100.0,
                "size_category": "large",
                "shape": {
                    "shape_class": "blob",
                    "linearity": 0.1,
                    "planarity": 0.2,
                    "sphericity": 0.7,
                    "compactness": None,
                    "skeleton_ratio": None,
                },
            }
        },
    }


def _base_summary(n=100, channels=1) -> dict[str, Any]:
    """Baseline dataset_summary with no observations triggered."""
    return {
        "modality": "mr",
        "num_patients": n,
        "num_channels": channels,
        "channel_names": [f"ch{i}" for i in range(channels)],
    }


class TestGenerateObservations:
    """Tests for the rule-based observation generator."""

    # --- dataset size ---

    def test_small_dataset_triggers_obs(self):
        """n < 50 → small dataset observation."""
        obs = ddu.generate_observations(
            _base_image_stats(),
            _base_label_stats(),
            _base_summary(n=30),
        )
        assert any("Small dataset" in o for o in obs)

    def test_large_dataset_triggers_obs(self):
        """n > 500 → large dataset observation."""
        obs = ddu.generate_observations(
            _base_image_stats(),
            _base_label_stats(),
            _base_summary(n=600),
        )
        assert any("Large dataset" in o for o in obs)

    def test_medium_dataset_no_size_obs(self):
        """50 <= n <= 500 → no dataset-size observation."""
        obs = ddu.generate_observations(
            _base_image_stats(),
            _base_label_stats(),
            _base_summary(n=100),
        )
        assert not any(
            "dataset" in o.lower() and "patient" in o for o in obs
        )

    # --- multi-channel ---

    def test_multi_channel_triggers_obs(self):
        """More than 1 channel → multi-channel observation."""
        obs = ddu.generate_observations(
            _base_image_stats(),
            _base_label_stats(),
            _base_summary(channels=4),
        )
        assert any("Multi-channel" in o for o in obs)

    def test_single_channel_no_obs(self):
        """1 channel → no multi-channel observation."""
        obs = ddu.generate_observations(
            _base_image_stats(),
            _base_label_stats(),
            _base_summary(channels=1),
        )
        assert not any("Multi-channel" in o for o in obs)

    # --- anisotropy ---

    def test_anisotropic_triggers_obs(self):
        """is_anisotropic = True → anisotropy observation."""
        img = _base_image_stats()
        img["spacing"]["is_anisotropic"] = True
        img["spacing"]["anisotropy_ratio"] = 6.0
        obs = ddu.generate_observations(
            img, _base_label_stats(), _base_summary()
        )
        assert any("Anisotropic" in o for o in obs)

    def test_isotropic_no_obs(self):
        """is_anisotropic = False → no anisotropy observation."""
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), _base_summary()
        )
        assert not any("Anisotropic" in o for o in obs)

    # --- sparse images ---

    def test_low_foreground_density_triggers_obs_multi_label(self):
        """foreground_fraction < 0.2 with multiple labels → obs emitted."""
        img = _base_image_stats()
        img["intensity"]["foreground_fraction"]["mean"] = 0.1
        lbl = _base_label_stats()
        lbl["per_label"]["2"] = lbl["per_label"]["1"].copy()
        obs = ddu.generate_observations(img, lbl, _base_summary())
        assert any("foreground density" in o for o in obs)

    def test_low_foreground_density_suppressed_for_single_label(self):
        """foreground_fraction < 0.2 with one label → obs suppressed."""
        img = _base_image_stats()
        img["intensity"]["foreground_fraction"]["mean"] = 0.1
        obs = ddu.generate_observations(
            img, _base_label_stats(), _base_summary()
        )
        assert not any("foreground density" in o for o in obs)

    def test_high_foreground_density_no_obs(self):
        """foreground_fraction >= 0.2 → no foreground density observation."""
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), _base_summary()
        )
        assert not any("foreground density" in o for o in obs)

    # --- CT modality ---

    def test_ct_modality_emits_hu_obs(self):
        """CT modality → HU range observation for each channel."""
        summary = _base_summary()
        summary["modality"] = "ct"
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), summary
        )
        assert any("HU" in o and "t1" in o for o in obs)

    def test_mr_modality_no_hu_obs(self):
        """Non-CT modality → no HU observation."""
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), _base_summary()
        )
        assert not any("HU" in o for o in obs)

    # --- class imbalance ---

    def test_severe_imbalance_triggers_obs(self):
        """Ratio > 10 → severe imbalance observation."""
        lbl = _base_label_stats()
        lbl["class_imbalance"]["imbalance_ratio"] = 15.0
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("Severe" in o for o in obs)

    def test_moderate_imbalance_triggers_obs(self):
        """3 < ratio <= 10 → moderate imbalance observation."""
        lbl = _base_label_stats()
        lbl["class_imbalance"]["imbalance_ratio"] = 5.0
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("Moderate" in o for o in obs)

    def test_balanced_no_imbalance_obs(self):
        """Ratio <= 3 → no imbalance observation."""
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), _base_summary()
        )
        assert not any("imbalance" in o.lower() for o in obs)

    def test_no_minority_label_no_obs(self):
        """minority_label = None → no imbalance observation."""
        lbl = _base_label_stats()
        lbl["class_imbalance"]["minority_label"] = None
        lbl["class_imbalance"]["imbalance_ratio"] = 20.0
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert not any("imbalance" in o.lower() for o in obs)

    # --- tiny / low-presence labels ---

    def test_tiny_label_triggers_obs(self):
        """size_category == 'tiny' → tiny-label observation."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "tiny"
        lbl["per_label"]["1"][
            "mean_volume_fraction_of_foreground_pct"
        ] = 0.05
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("very small" in o for o in obs)

    def test_tiny_label_suppresses_low_presence_obs(self):
        """A tiny label with low presence emits only the 'tiny' obs."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "tiny"
        lbl["per_label"]["1"]["presence_rate_pct"] = 20.0
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("very small" in o for o in obs)
        assert not any("absent in" in o for o in obs)

    def test_low_presence_non_tiny_triggers_obs(self):
        """Non-tiny label with presence < 50% → low-presence observation."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "small"
        lbl["per_label"]["1"]["presence_rate_pct"] = 30.0
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("absent in" in o for o in obs)

    def test_high_presence_no_obs(self):
        """Label with presence >= 50% and non-tiny size → no presence obs."""
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), _base_summary()
        )
        assert not any("absent in" in o for o in obs)

    # --- image volume fraction ---

    def test_very_sparse_image_fraction_triggers_obs(self):
        """img_frac < VERY_SPARSE threshold → 'very sparse' obs."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["mean_volume_fraction_of_image_pct"] = 0.1
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("very sparse" in o and "image volume" in o for o in obs)

    def test_sparse_image_fraction_triggers_obs(self):
        """VERY_SPARSE <= img_frac < SPARSE threshold → 'sparse' (not 'very sparse') obs."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["mean_volume_fraction_of_image_pct"] = 2.0
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        matching = [o for o in obs if "image volume" in o]
        assert len(matching) == 1
        assert "sparse" in matching[0]
        assert "very sparse" not in matching[0]

    def test_large_image_fraction_no_obs(self):
        """img_frac >= SPARSE threshold → no image volume obs."""
        obs = ddu.generate_observations(
            _base_image_stats(), _base_label_stats(), _base_summary()
        )
        assert not any("image volume" in o for o in obs)

    # --- shape-based observations ---

    @pytest.mark.parametrize("shape_class,expected_keyword", [
        pytest.param("planar", "planar/sheet-like", id="planar"),
        pytest.param("blob", "compact/blob-like", id="blob"),
    ])
    def test_shape_class_triggers_correct_obs(
        self, shape_class, expected_keyword
    ):
        """Each shape class emits its corresponding descriptive keyword."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["shape"]["shape_class"] = shape_class
        lin, plan, sph = {
            "planar": (0.3, 0.6, 0.1),
            "blob": (0.1, 0.2, 0.7),
        }[shape_class]
        lbl["per_label"]["1"]["shape"].update({
            "linearity": lin,
            "planarity": plan,
            "sphericity": sph,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any(expected_keyword in o for o in obs)

    def test_high_skeleton_ratio_triggers_branching_obs(self):
        """skeleton_ratio above threshold → thin or branching structure obs."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["shape"].update({
            "shape_class": "tubular",
            "linearity": 0.6,
            "planarity": 0.3,
            "sphericity": 0.1,
            "skeleton_ratio": 0.12,
            "compactness": 0.04,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("thin or branching" in o for o in obs)

    def test_tubular_without_skeleton_ratio_emits_elongated_obs(self):
        """Tubular shape_class with no skeleton ratio → elongated observation."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["shape"].update({
            "shape_class": "tubular",
            "linearity": 0.6,
            "planarity": 0.3,
            "sphericity": 0.1,
            "skeleton_ratio": None,
            "compactness": None,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("elongated" in o for o in obs)

    # --- resampling risk ---

    def test_small_thin_label_triggers_resampling_warning(self):
        """Small thin label → resampling risk observation."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "small"
        lbl["per_label"]["1"]["shape"].update({
            "shape_class": "tubular",
            "skeleton_ratio": 0.12,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("target spacing" in o for o in obs)

    def test_tiny_planar_label_triggers_resampling_warning(self):
        """Tiny planar label → resampling risk observation."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "tiny"
        lbl["per_label"]["1"]["mean_volume_fraction_of_foreground_pct"] = 0.05
        lbl["per_label"]["1"]["shape"].update({
            "shape_class": "planar",
            "linearity": 0.3,
            "planarity": 0.6,
            "sphericity": 0.1,
            "skeleton_ratio": None,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert any("target spacing" in o for o in obs)

    def test_large_thin_label_no_resampling_warning(self):
        """Large thin label → no resampling warning (only small/tiny trigger)."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "large"
        lbl["per_label"]["1"]["shape"].update({
            "shape_class": "tubular",
            "skeleton_ratio": 0.12,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert not any("target spacing" in o for o in obs)

    def test_small_blob_label_no_resampling_warning(self):
        """Small blob label → no resampling warning (only thin structures)."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["size_category"] = "small"
        lbl["per_label"]["1"]["shape"].update({
            "shape_class": "blob",
            "skeleton_ratio": 0.01,
        })
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert not any("target spacing" in o for o in obs)

    def test_shape_unknown_is_skipped(self):
        """shape_class == 'unknown' → no shape observation emitted."""
        lbl = _base_label_stats()
        lbl["per_label"]["1"]["shape"] = {
            "shape_class": "unknown",
            "linearity": None,
            "planarity": None,
            "sphericity": None,
        }
        obs = ddu.generate_observations(
            _base_image_stats(), lbl, _base_summary()
        )
        assert not any(
            "tubular" in o or "planar" in o or "blob" in o
            for o in obs
        )


# ---------------------------------------------------------------------------
# collect_per_patient_stats  (integration with ants + progress bar mocked)
# ---------------------------------------------------------------------------

def _make_mask_image(shape=(10, 10, 10), spacing=(1.0, 1.0, 1.5)):
    """Return an ANTs mask with label 1 in a 3x3x3 cube and label 2 in
    a 2x2x2 cube."""
    arr = np.zeros(shape, dtype=np.float32)
    arr[1:4, 1:4, 1:4] = 1   # 27 voxels of label 1
    arr[6:8, 6:8, 6:8] = 2   # 8 voxels of label 2
    return _ants_image(arr, spacing)


def _make_channel_image(shape=(10, 10, 10), fill=100.0, spacing=(1.0, 1.0, 1.5)):
    """Return a constant-value ANTs image for a channel."""
    return _ants_image(np.full(shape, fill, dtype=np.float32), spacing)


class TestCollectPerPatientStats:
    """Tests for the single-pass statistics collector."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        """Patch ants and progress_bar for all tests in this class."""
        def _image_read(path):
            if "mask" in str(path):
                return _make_mask_image()
            return _make_channel_image()

        monkeypatch.setattr(ants, "image_read", _image_read, raising=True)
        monkeypatch.setattr(
            pb_mod, "get_progress_bar", lambda _: FakePB(), raising=True
        )

    def _run(self, n=2, labels=(0, 1, 2), nan_channel=False):
        """Set up a DataFrame and run collect_per_patient_stats."""
        t1_vals = [f"p{i}_t1.nii.gz" for i in range(n)]
        if nan_channel:
            t1_vals[0] = float("nan")
        df = _make_paths_df(
            [f"p{i}_mask.nii.gz" for i in range(n)],
            {"t1": t1_vals},
        )
        ds_info = _make_dataset_info(labels=labels, channels=("t1",))
        return ddu.collect_per_patient_stats(df, ds_info)

    def test_returns_all_required_keys(self):
        """Output contains all required keys including effective_dims."""
        result = self._run()
        assert set(result.keys()) == {
            "spacings",
            "original_dims",
            "effective_dims",
            "foreground_fractions",
            "total_fg_voxels",
            "channel_intensities",
            "label_voxel_counts",
            "label_presence",
            "label_shape_descriptors",
        }

    def test_spacings_shape_and_values(self):
        """Spacings array has shape (n_patients, 3) with correct values."""
        result = self._run(n=3)
        assert result["spacings"].shape == (3, 3)
        # All patients use the same mask image with spacing (1.0, 1.0, 1.5).
        np.testing.assert_allclose(result["spacings"][:, 2], 1.5)

    def test_original_dims_shape(self):
        """original_dims array has shape (n_patients, 3)."""
        result = self._run(n=2)
        assert result["original_dims"].shape == (2, 3)
        np.testing.assert_array_equal(result["original_dims"][0], [10, 10, 10])

    def test_foreground_fraction_full_foreground(self):
        """Non-zero fraction is > 0 when foreground voxels are present."""
        result = self._run()
        assert all(f > 0 for f in result["foreground_fractions"])

    def test_foreground_fraction_empty_foreground(self, monkeypatch):
        """Non-zero fraction is 0 when the mask is all background."""
        empty_mask = _ants_image(np.zeros((10, 10, 10), dtype=np.float32))
        monkeypatch.setattr(
            ants, "image_read", lambda _: empty_mask, raising=True
        )
        df = _make_paths_df(["p0_mask.nii.gz"], {"t1": ["p0_t1.nii.gz"]})
        ds_info = _make_dataset_info(labels=(0, 1), channels=("t1",))
        result = ddu.collect_per_patient_stats(df, ds_info)
        assert result["foreground_fractions"][0] == 0.0

    def test_label_voxel_counts_match_mask(self):
        """Label 1 has 27 voxels and label 2 has 8 voxels in the test mask."""
        result = self._run(n=1)
        assert result["label_voxel_counts"][1][0] == 27
        assert result["label_voxel_counts"][2][0] == 8

    def test_label_presence_correct(self):
        """Presence flag is 1 when label exists, 0 otherwise."""
        result = self._run(n=2)
        assert all(p == 1 for p in result["label_presence"][1])
        assert all(p == 1 for p in result["label_presence"][2])

    def test_label_absent_has_zero_presence(self, monkeypatch):
        """Label absent in a patient gets presence = 0 and no shape descriptors."""
        # Mask with only label 1; label 2 is absent.
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[1:4, 1:4, 1:4] = 1

        def _read(p):
            """Return mask or channel image depending on path."""
            if "mask" in str(p):
                return _ants_image(arr)
            return _make_channel_image()

        monkeypatch.setattr(ants, "image_read", _read, raising=True)
        df = _make_paths_df(["p0_mask.nii.gz"], {"t1": ["p0_t1.nii.gz"]})
        ds_info = _make_dataset_info(labels=(0, 1, 2), channels=("t1",))
        result = ddu.collect_per_patient_stats(df, ds_info)
        assert result["label_presence"][2][0] == 0
        assert result["label_shape_descriptors"][2] == []

    def test_shape_descriptors_populated_for_present_label(self):
        """Shape descriptor list is non-empty for a label that is present."""
        result = self._run(n=2)
        assert len(result["label_shape_descriptors"][1]) == 2

    def test_channel_intensities_sampled_from_foreground(self):
        """Channel intensities list is non-empty after processing."""
        result = self._run(n=2)
        assert len(result["channel_intensities"]["t1"]) > 0

    def test_effective_dims_defaults_to_original_dims(self):
        """When effective_dims is None, effective_dims key equals original_dims."""
        result = self._run(n=2)
        np.testing.assert_array_equal(
            result["effective_dims"], result["original_dims"]
        )

    def test_effective_dims_override_used_for_foreground_fraction(self):
        """Passing effective_dims uses them as the fg-fraction denominator."""
        n = 2
        # Each mask is 10×10×10 = 1000 voxels; supply a smaller effective dim.
        eff = np.tile([5.0, 5.0, 5.0], (n, 1))  # 125 voxels per patient
        df = _make_paths_df(
            [f"p{i}_mask.nii.gz" for i in range(n)],
            {"t1": [f"p{i}_t1.nii.gz" for i in range(n)]},
        )
        ds_info = _make_dataset_info(labels=(0, 1, 2), channels=("t1",))
        result = ddu.collect_per_patient_stats(df, ds_info, effective_dims=eff)
        # fg voxels = 27 + 8 = 35; fraction relative to 125, not 1000.
        assert result["effective_dims"].shape == (n, 3)
        assert all(f > 35 / 125 * 0.9 for f in result["foreground_fractions"])

    def test_missing_channel_value_skipped(self):
        """NaN channel path is silently skipped without error."""
        result = self._run(n=2, nan_channel=True)
        # Still collects intensity from patient 1 (valid path).
        assert len(result["channel_intensities"]["t1"]) > 0
