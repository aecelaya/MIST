"""Tests for mist.analyze_data.analyze_utils."""
from typing import Dict, Any, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# MIST imports.
from mist.analyze_data import analyzer_utils as au


def _make_header(
    dims: Tuple[int, ...] = (64, 64, 32),
    origin: Tuple[float, ...] = (0.0, 0.0, 0.0),
    spacing: Tuple[float, ...] = (1.0, 1.0, 2.5),
    direction: Union[
        np.ndarray, List[float], Tuple[float, ...]
    ] = np.eye(3),
) -> Dict[str, Any]:
    """Helper to construct a header dict."""
    return {
        "dimensions": list(dims),
        "origin": list(origin),
        "spacing": list(spacing),
        "direction": np.asarray(direction),
    }


def _touch(path: Path):
    """Create an empty file at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# ---------------------------------------------------------------------------
# compare_headers
# ---------------------------------------------------------------------------

class TestCompareHeaders:
    """Tests for analyzer_utils.compare_headers."""

    def test_matching_headers_return_true(self):
        """Headers that match within float tolerance return True."""
        h1 = _make_header()
        h2 = _make_header(spacing=(1.0 + 1e-12, 1.0, 2.5))
        assert au.compare_headers(h1, h2) is True

    def test_dimension_mismatch_returns_false(self):
        """Differing dimensions return False."""
        h1 = _make_header(dims=(64, 64, 32))
        h2 = _make_header(dims=(64, 64, 33))
        assert au.compare_headers(h1, h2) is False

    def test_origin_mismatch_returns_false(self):
        """Differing origins return False."""
        h1 = _make_header(origin=(0.0, 0.0, 0.0))
        h2 = _make_header(origin=(0.0, 0.0, 1.0))
        assert au.compare_headers(h1, h2) is False

    def test_spacing_mismatch_beyond_tolerance_returns_false(self):
        """Spacing differences beyond atol return False."""
        h1 = _make_header(spacing=(1.0, 1.0, 2.5))
        h2 = _make_header(spacing=(1.1, 1.0, 2.5))
        assert au.compare_headers(h1, h2) is False

    def test_direction_mismatch_returns_false(self):
        """Differing direction matrices return False."""
        bad_dir = np.eye(3)
        bad_dir[0, 0] = -1.0
        h1 = _make_header(direction=np.eye(3))
        h2 = _make_header(direction=bad_dir)
        assert au.compare_headers(h1, h2) is False


# ---------------------------------------------------------------------------
# is_image_3d
# ---------------------------------------------------------------------------

class TestIsImage3D:
    """Tests for analyzer_utils.is_image_3d."""

    def test_3d_header_returns_true(self):
        """A header with 3 dimensions returns True."""
        assert au.is_image_3d(_make_header(dims=(64, 64, 64))) is True

    def test_2d_header_returns_false(self):
        """A header with 2 dimensions returns False."""
        assert au.is_image_3d(_make_header(dims=(128, 128))) is False


# ---------------------------------------------------------------------------
# get_resampled_image_dimensions
# ---------------------------------------------------------------------------

class TestGetResampledImageDimensions:
    """Tests for analyzer_utils.get_resampled_image_dimensions."""

    @pytest.mark.parametrize(
        "dims, spacing, target, expected",
        [
            pytest.param(
                (100, 80, 20),
                (1.0, 1.0, 2.0),
                (2.0, 2.0, 2.0),
                (50, 40, 20),
                id="halved_xy_spacing",
            ),
            pytest.param(
                (96, 64, 48),
                (1.2, 1.5, 2.0),
                (1.0, 1.0, 2.0),
                (115, 96, 48),
                id="finer_xy_target",
            ),
            pytest.param(
                (64, 64, 64),
                (0.8, 0.8, 0.8),
                (1.6, 1.6, 1.6),
                (32, 32, 32),
                id="isotropic_halved",
            ),
        ],
    )
    def test_dimensions_computed_correctly(self, dims, spacing, target, expected):
        """Resampled dimensions are computed via round(dim * spc / target)."""
        assert au.get_resampled_image_dimensions(dims, spacing, target) == expected


# ---------------------------------------------------------------------------
# get_float32_example_memory_size
# ---------------------------------------------------------------------------

class TestGetFloat32ExampleMemorySize:
    """Tests for analyzer_utils.get_float32_example_memory_size."""

    @pytest.mark.parametrize(
        "dims, channels, labels",
        [
            pytest.param((32, 32, 32), 1, 2, id="small_1ch"),
            pytest.param((64, 48, 16), 3, 1, id="medium_3ch"),
            pytest.param((10, 20, 30), 4, 4, id="small_4ch_4lbl"),
        ],
    )
    def test_memory_size_correct(self, dims, channels, labels):
        """Memory equals 4 * prod(dims) * (channels + labels)."""
        expected = (
            4 * int(np.prod(np.array(dims))) * (channels + labels)
        )
        assert (
            au.get_float32_example_memory_size(dims, channels, labels)
            == expected
        )


# ---------------------------------------------------------------------------
# get_files_df
# ---------------------------------------------------------------------------

def _make_dataset_info(base: Path) -> Dict[str, Any]:
    """Create a dataset_info structure for mocking utils.io.read_json_file."""
    return {
        "train-data": str(base / "train"),
        "test-data": str(base / "test"),
        "images": {
            "image_1": ["image_1"],
            "image_2": ["image_2"],
            "image_3": ["image_3"],
        },
        "mask": ["mask"],
    }


class TestGetFilesDF:
    """Tests for analyzer_utils.get_files_df."""

    def test_patient_ids_are_sorted(self, monkeypatch, tmp_path: Path):
        """Patient IDs are returned in sorted (deterministic) order."""
        base = tmp_path
        ds_info = _make_dataset_info(base)

        for name in ("patient_c", "patient_a", "patient_b"):
            (base / "train" / name).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr("mist.utils.io.read_json_file", lambda _: ds_info)
        df = au.get_files_df("fake/path/dataset.json", "train")
        assert df["id"].tolist() == ["patient_a", "patient_b", "patient_c"]

    def test_missing_image_emits_warning(
        self, monkeypatch, tmp_path: Path, caplog
    ):
        """A patient missing an image file produces a warning."""
        import logging

        base = tmp_path
        ds_info = _make_dataset_info(base)
        p = base / "train" / "patient_x"
        p.mkdir(parents=True, exist_ok=True)
        # Only image_1 present; image_2 and image_3 are missing.
        _touch(p / "image_1.nii.gz")
        _touch(p / "mask.nii.gz")

        monkeypatch.setattr("mist.utils.io.read_json_file", lambda _: ds_info)
        with caplog.at_level(logging.WARNING):
            au.get_files_df("fake/path/dataset.json", "train")

        messages = [r.getMessage() for r in caplog.records]
        patient_msgs = [m for m in messages if "patient_x" in m]
        assert any("image_2" in m for m in patient_msgs)
        assert any("image_3" in m for m in patient_msgs)

    def test_absent_segmentation_file_emits_warning(
        self, monkeypatch, tmp_path: Path, caplog
    ):
        """A patient missing a mask file produces a warning in train mode."""
        import logging

        base = tmp_path / "dataset"
        ds_info = _make_dataset_info(base)
        p = base / "train" / "patient_y"
        p.mkdir(parents=True, exist_ok=True)
        _touch(p / "image_1.nii.gz")
        # No mask file.

        monkeypatch.setattr("mist.utils.io.read_json_file", lambda _: ds_info)
        with caplog.at_level(logging.WARNING):
            au.get_files_df("fake/path/dataset.json", "train")

        messages = [r.getMessage() for r in caplog.records]
        assert any("patient_y" in m and "mask" in m for m in messages)

    def test_train_mode_maps_paths(self, monkeypatch, tmp_path: Path):
        """Train mode includes mask and modality columns with absolute paths."""
        base = tmp_path
        ds_info = _make_dataset_info(base)

        # Hidden folder should be ignored by implementation.
        (base / "train" / ".DS_Store").mkdir(parents=True, exist_ok=True)

        p1 = base / "train" / "patient_1"
        p2 = base / "train" / "patient_2"
        p3 = base / "train" / "patient_3"  # Missing some files on purpose.
        for p in (p1, p2, p3):
            p.mkdir(parents=True, exist_ok=True)

        # Patient 1: all images + mask.
        _touch(p1 / "image_1.nii.gz")
        _touch(p1 / "image_2.nii.gz")
        _touch(p1 / "image_3.nii.gz")
        _touch(p1 / "mask.nii.gz")

        # Patient 2: two images + mask.
        _touch(p2 / "image_1_time0.nii.gz")
        _touch(p2 / "image_2_alt.nii.gz")
        _touch(p2 / "mask_final.nii.gz")

        # Patient 3: only image_1, no others.
        _touch(p3 / "image_1_only.nii.gz")

        monkeypatch.setattr("mist.utils.io.read_json_file", lambda _: ds_info)

        df = au.get_files_df("fake/path/dataset.json", "train")

        assert list(df.columns) == [
            "id", "mask", "image_1", "image_2", "image_3"
        ]
        assert set(df["id"].tolist()) == {
            "patient_1", "patient_2", "patient_3"
        }

        row1 = df[df["id"] == "patient_1"].iloc[0]
        assert row1["mask"].endswith("train/patient_1/mask.nii.gz")
        assert row1["image_1"].endswith(
            "train/patient_1/image_1.nii.gz"
        )
        assert row1["image_2"].endswith(
            "train/patient_1/image_2.nii.gz"
        )
        assert row1["image_3"].endswith(
            "train/patient_1/image_3.nii.gz"
        )

        row2 = df[df["id"] == "patient_2"].iloc[0]
        assert row2["image_1"].endswith(
            "train/patient_2/image_1_time0.nii.gz"
        )
        assert row2["image_2"].endswith(
            "train/patient_2/image_2_alt.nii.gz"
        )
        assert row2["mask"].endswith(
            "train/patient_2/mask_final.nii.gz"
        )
        assert pd.isna(row2["image_3"])

        row3 = df[df["id"] == "patient_3"].iloc[0]
        assert row3["image_1"].endswith("train/patient_3/image_1_only.nii.gz")
        assert pd.isna(row3["image_2"])
        assert pd.isna(row3["image_3"])
        assert pd.isna(row3["mask"])

    def test_test_mode_omits_mask_column(self, monkeypatch, tmp_path: Path):
        """Test mode omits 'mask' column and still maps modality files."""
        base = tmp_path
        ds_info = _make_dataset_info(base)

        t1 = base / "test" / "patient_A"
        t2 = base / "test" / "patient_B"
        for p in (t1, t2):
            p.mkdir(parents=True, exist_ok=True)

        _touch(t1 / "image_1.nii.gz")
        _touch(t1 / "image_2.nii.gz")
        _touch(t2 / "image_3.nii.gz")

        monkeypatch.setattr("mist.utils.io.read_json_file", lambda _: ds_info)

        df = au.get_files_df("fake/path/dataset.json", "test")

        assert list(df.columns) == [
            "id", "image_1", "image_2", "image_3"
        ]
        assert set(df["id"].tolist()) == {"patient_A", "patient_B"}

        rowA = df[df["id"] == "patient_A"].iloc[0]
        assert rowA["image_1"].endswith(
            "test/patient_A/image_1.nii.gz"
        )
        assert rowA["image_2"].endswith(
            "test/patient_A/image_2.nii.gz"
        )
        assert pd.isna(rowA["image_3"])

        rowB = df[df["id"] == "patient_B"].iloc[0]
        assert pd.isna(rowB["image_1"])
        assert pd.isna(rowB["image_2"])
        assert rowB["image_3"].endswith(
            "test/patient_B/image_3.nii.gz"
        )


# ---------------------------------------------------------------------------
# add_folds_to_df
# ---------------------------------------------------------------------------

class TestAddFoldsToDf:
    """Tests for analyzer_utils.add_folds_to_df."""

    def test_adds_fold_column_and_sorts(self):
        """Adds a 'fold' column with deterministic stratification."""
        df = pd.DataFrame(
            {
                "id": [f"p{i}" for i in range(10)],
                "mask": [f"/path/m{i}.nii.gz" for i in range(10)],
                "image_1": [f"/path/i1_{i}.nii.gz" for i in range(10)],
            }
        )

        out = au.add_folds_to_df(df.copy(), n_splits=5)

        assert "fold" in out.columns
        assert list(out.columns).index("fold") == 1
        assert set(out["fold"].unique()) == {0, 1, 2, 3, 4}
        assert out["fold"].isna().sum() == 0
        assert len(out) == len(df)
        assert out["fold"].is_monotonic_increasing
        counts = out["fold"].value_counts().to_dict()
        assert counts == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}


# ---------------------------------------------------------------------------
# _largest_multiple_of_32_leq
# ---------------------------------------------------------------------------

class TestLargestMultipleOf32Leq:
    """Tests for analyzer_utils._largest_multiple_of_32_leq."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(128.0, 128, id="exact_multiple"),
            pytest.param(159.9, 128, id="just_below_160"),
            pytest.param(160.0, 160, id="exact_160"),
            pytest.param(170.0, 160, id="between_160_and_192"),
            pytest.param(512.0, 512, id="exact_512"),
        ],
    )
    def test_snaps_down_to_multiple_of_32(self, value, expected):
        """Returns the largest multiple of 32 not exceeding value."""
        assert au._largest_multiple_of_32_leq(value) == expected

    def test_minimum_floor_applied_when_value_below_32(self):
        """Values below 32 return the minimum (32 by default)."""
        assert au._largest_multiple_of_32_leq(10.0) == 32

    def test_custom_minimum(self):
        """Custom minimum is respected when snapped value is below it."""
        assert au._largest_multiple_of_32_leq(10.0, minimum=16) == 16


# ---------------------------------------------------------------------------
# _get_voxel_budget
# ---------------------------------------------------------------------------

class TestGetVoxelBudget:
    """Tests for analyzer_utils._get_voxel_budget."""

    def test_returns_default_when_cuda_unavailable(self, monkeypatch):
        """Falls back to PATCH_BUDGET_DEFAULT_VOXELS when CUDA is absent."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        from mist.analyze_data.analyzer_constants import AnalyzeConstants as C
        assert au._get_voxel_budget() == C.PATCH_BUDGET_DEFAULT_VOXELS

    def test_returns_default_when_no_devices(self, monkeypatch):
        """Falls back when CUDA reports zero devices."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        from mist.analyze_data.analyzer_constants import AnalyzeConstants as C
        assert au._get_voxel_budget() == C.PATCH_BUDGET_DEFAULT_VOXELS

    def test_scales_linearly_with_gpu_memory(self, monkeypatch):
        """Budget scales linearly: 32 GB GPU → 2× the reference budget."""
        import torch
        from mist.analyze_data.analyzer_constants import AnalyzeConstants as C
        from unittest.mock import MagicMock

        fake_props = MagicMock()
        fake_props.total_memory = 2 * C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: fake_props
        )

        expected = 2 * C.PATCH_BUDGET_REFERENCE_VOXELS
        assert au._get_voxel_budget() == expected

    def test_uses_minimum_across_gpus(self, monkeypatch):
        """With multiple GPUs the smallest VRAM determines the budget."""
        import torch
        from mist.analyze_data.analyzer_constants import AnalyzeConstants as C
        from unittest.mock import MagicMock

        small = MagicMock()
        small.total_memory = C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES // 2  # 8 GB
        large = MagicMock()
        large.total_memory = C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES * 2   # 32 GB

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda i: small if i == 0 else large,
        )

        expected = C.PATCH_BUDGET_REFERENCE_VOXELS // 2
        assert au._get_voxel_budget() == expected

    def test_budget_scales_inversely_with_batch_size(self, monkeypatch):
        """Doubling batch size halves the per-patch voxel budget."""
        import torch
        from mist.analyze_data.analyzer_constants import AnalyzeConstants as C
        from unittest.mock import MagicMock

        props = MagicMock()
        props.total_memory = C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)

        budget_bs2 = au._get_voxel_budget(batch_size_per_gpu=2)
        budget_bs4 = au._get_voxel_budget(batch_size_per_gpu=4)
        assert budget_bs4 == budget_bs2 // 2


# ---------------------------------------------------------------------------
# get_best_patch_size
# ---------------------------------------------------------------------------

class TestGetBestPatchSize:
    """Tests for analyzer_utils.get_best_patch_size."""

    # Fixture: pin the voxel budget to 128^3 so tests are GPU-independent.
    @pytest.fixture(autouse=True)
    def pin_budget(self, monkeypatch):
        monkeypatch.setattr(
            au, "_get_voxel_budget", lambda batch_size_per_gpu=2: 128 ** 3
        )

    # --- 3D isotropic mode ---

    def test_isotropic_no_clamping(self):
        """Isotropic large image: budget distributes equally, snaps to mult-32."""
        # target_mm = (128^3 * 1.0^3)^(1/3) = 128mm; raw = 128/1.0 = 128
        # 128 < 512 → no clamping; snap(128) = 128
        result = au.get_best_patch_size([512, 512, 512], [1.0, 1.0, 1.0])
        assert result == [128, 128, 128]

    def test_isotropic_clamped_one_axis_redistributes_budget(self):
        """Small z forces clamping; freed budget pumps up x and y."""
        # Isotropic spacing → 3D mode.
        # Iter 1: target_mm = 128mm, raw_z = 128 >= median_z=40
        #   → fix z = snap(40) = 32  (stores snapped value for accurate budget)
        # Iter 2: remaining = 128^3/32 = 65536; target_mm_xy = sqrt(65536) = 256
        #   → snap(256) = 256; both x,y = 256
        result = au.get_best_patch_size([512, 512, 40], [1.0, 1.0, 1.0])
        assert result == [256, 256, 32]

    def test_isotropic_all_axes_clamped_to_median(self):
        """Tiny image: all axes clamped to median, snapped down."""
        # budget >> image → all axes raw > median → fix all at median
        result = au.get_best_patch_size([32, 32, 32], [1.0, 1.0, 1.0])
        assert result == [32, 32, 32]

    def test_anisotropic_spacing_but_below_threshold_uses_3d_mode(self):
        """Spacing ratio < 3 → 3D mode even if spacing is not uniform."""
        # spacing ratio = 2.0/1.0 = 2.0, below threshold of 3.0 → 3D mode.
        # target_mm = (128^3 * 1.0 * 1.0 * 2.0)^(1/3) ≈ 161.3mm
        # raw = [161.3, 161.3, 80.6]; all < [512,512,512] → no clamp
        # snap(161.3) = 160; snap(80.6) = 64
        result = au.get_best_patch_size([512, 512, 512], [1.0, 1.0, 2.0])
        assert result == [160, 160, 64]

    # --- quasi-2D mode ---

    def test_quasi_2d_thick_slice_z_axis(self):
        """Thick-slice CT (z=low-res): z gets small patch, xy get full res."""
        # anisotropy = 3.0/0.8 = 3.75 > 3.0 → quasi-2D; low_res_axis=2
        # lr_raw = 128^3 / 512^2 = 8; clamp(8, 5, 40) = 8
        # ip_raw = sqrt(128^3 / 8) = 512; snap(512)=512; min(512,512)=512
        result = au.get_best_patch_size([512, 512, 40], [0.8, 0.8, 3.0])
        assert result == [512, 512, 8]

    def test_quasi_2d_low_res_axis_is_x(self):
        """Low-resolution axis is detected correctly when it is axis 0."""
        # spacing = [3.0, 0.8, 0.8] → low_res_axis = 0
        # median_lr = 40 (axis 0), median_ip = max(512,512) = 512
        # same arithmetic as above but result reordered
        result = au.get_best_patch_size([40, 512, 512], [3.0, 0.8, 0.8])
        assert result == [8, 512, 512]

    def test_quasi_2d_low_res_axis_is_y(self):
        """Low-resolution axis is detected correctly when it is axis 1."""
        result = au.get_best_patch_size([512, 40, 512], [0.8, 3.0, 0.8])
        assert result == [512, 8, 512]

    def test_quasi_2d_lr_raw_below_minimum_clamped_up(self):
        """lr_raw < MIN_LOW_RES_AXIS_PATCH_SIZE is clamped to that minimum."""
        # Very large in-plane: 1024^2; lr_raw = 128^3/1024^2 = 2 < 5
        # → lr_patch = min(5, median_lr=40) = 5
        # ip_raw = sqrt(128^3/5) ≈ 648; snap(648) = 640; min(640,1024)=640
        result = au.get_best_patch_size([1024, 1024, 40], [0.8, 0.8, 3.0])
        assert result == [640, 640, 5]

    def test_quasi_2d_tiny_z_uses_all_available_slices(self):
        """When median_lr < MIN_LOW_RES_AXIS_PATCH_SIZE, patch = median_lr."""
        # median_lr = 3 < 5; min_lr = min(5, 3) = 3 → lr_patch clamped to 3
        result = au.get_best_patch_size([512, 512, 3], [0.8, 0.8, 3.0])
        assert result[2] == 3

    def test_quasi_2d_ip_patch_capped_to_median_ip(self):
        """In-plane patch is capped at median_ip when budget would exceed it."""
        # Small budget relative to image: ip_raw may snap down; confirm cap.
        result = au.get_best_patch_size([64, 64, 10], [0.8, 0.8, 3.0])
        for ax in (0, 1):
            assert result[ax] <= 64

    def test_larger_batch_size_yields_smaller_patch(self, monkeypatch):
        """Doubling batch size from 2 → 4 halves the budget and shrinks patch."""
        # Unpin the budget so batch_size_per_gpu is forwarded to _get_voxel_budget.
        monkeypatch.setattr(
            au,
            "_get_voxel_budget",
            lambda batch_size_per_gpu=2: 128 ** 3 // batch_size_per_gpu * 2,
        )
        result_bs2 = au.get_best_patch_size(
            [512, 512, 512], [1.0, 1.0, 1.0], batch_size_per_gpu=2
        )
        result_bs4 = au.get_best_patch_size(
            [512, 512, 512], [1.0, 1.0, 1.0], batch_size_per_gpu=4
        )
        # Smaller budget → smaller or equal patch per axis.
        for a, b in zip(result_bs4, result_bs2):
            assert a <= b


# ---------------------------------------------------------------------------
# build_base_config
# ---------------------------------------------------------------------------

class TestBuildBaseConfig:
    """Tests for analyzer_utils.build_base_config."""

    def test_returns_expected_top_level_structure(self):
        """build_base_config returns a dict with all required top-level keys."""
        cfg = au.build_base_config()
        for key in (
            "mist_version", "dataset_info", "spatial_config",
            "preprocessing", "model", "training", "inference",
        ):
            assert key in cfg

    def test_spatial_config_has_patch_size_and_target_spacing(self):
        """spatial_config contains patch_size and target_spacing, both None."""
        cfg = au.build_base_config()
        assert "patch_size" in cfg["spatial_config"]
        assert "target_spacing" in cfg["spatial_config"]
        assert cfg["spatial_config"]["patch_size"] is None
        assert cfg["spatial_config"]["target_spacing"] is None

    def test_patch_size_not_in_model_params(self):
        """model.params no longer contains patch_size or target_spacing."""
        cfg = au.build_base_config()
        assert "patch_size" not in cfg["model"]["params"]
        assert "target_spacing" not in cfg["model"]["params"]

    def test_target_spacing_not_in_preprocessing(self):
        """preprocessing no longer contains target_spacing."""
        cfg = au.build_base_config()
        assert "target_spacing" not in cfg["preprocessing"]

    def test_patch_size_not_in_inference_inferer_params(self):
        """inference.inferer.params no longer contains patch_size."""
        cfg = au.build_base_config()
        assert "patch_size" not in cfg["inference"]["inferer"]["params"]

    def test_modality_starts_as_none(self):
        """dataset_info.modality is None before the analyzer fills it in."""
        cfg = au.build_base_config()
        assert cfg["dataset_info"]["modality"] is None

    def test_preprocessing_not_skipped_by_default(self):
        """preprocessing.skip defaults to False."""
        cfg = au.build_base_config()
        assert not cfg["preprocessing"]["skip"]

    def test_model_architecture_is_nnunet(self):
        """model.architecture defaults to 'nnunet'."""
        cfg = au.build_base_config()
        assert cfg["model"]["architecture"] == "nnunet"


# ---------------------------------------------------------------------------
# build_evaluation_config
# ---------------------------------------------------------------------------

class TestBuildEvaluationConfig:
    """Tests for analyzer_utils.build_evaluation_config."""

    def test_returns_correct_structure(self):
        """build_evaluation_config returns correct structure for valid input."""
        dataset = {"final_classes": {"tumor": [1, 2], "edema": [3]}}
        result = au.build_evaluation_config(dataset)
        assert result == {
            "evaluation": {
                "tumor": {
                    "labels": [1, 2],
                    "metrics": {"dice": {}, "haus95": {}},
                },
                "edema": {
                    "labels": [3],
                    "metrics": {"dice": {}, "haus95": {}},
                },
            }
        }

    def test_missing_final_classes_raises_value_error(self):
        """build_evaluation_config raises ValueError when absent."""
        with pytest.raises(ValueError, match="Missing 'final_classes'"):
            au.build_evaluation_config({})
