"""Tests for mist.analyze_data.analyze_utils."""
import logging
from typing import Any
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

# MIST imports.
from mist.analyze_data import analyzer_utils as au
from mist.analyze_data import analyzer_constants as constants
from mist.models.nnunet.nnunet_utils import get_unet_params

_C = constants.AnalyzeConstants()
_REF_BUDGET = _C.PATCH_BUDGET_REFERENCE_VOXELS  # 128^3 = 2,097,152


def _make_header(
    dims: tuple[int, ...] = (64, 64, 32),
    origin: tuple[float, ...] = (0.0, 0.0, 0.0),
    spacing: tuple[float, ...] = (1.0, 1.0, 2.5),
    direction: (
        np.ndarray | list[float] | tuple[float, ...]
    ) = np.eye(3),
) -> dict[str, Any]:
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

def _make_dataset_info(base: Path) -> dict[str, Any]:
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
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert au._get_voxel_budget() == _C.PATCH_BUDGET_DEFAULT_VOXELS

    def test_returns_default_when_no_devices(self, monkeypatch):
        """Falls back when CUDA reports zero devices."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert au._get_voxel_budget() == _C.PATCH_BUDGET_DEFAULT_VOXELS

    def test_scales_linearly_with_gpu_memory(self, monkeypatch):
        """Budget scales linearly: 32 GB GPU → 2× the reference budget."""
        fake_props = MagicMock()
        fake_props.total_memory = 2 * _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: fake_props
        )

        expected = 2 * _C.PATCH_BUDGET_REFERENCE_VOXELS
        assert au._get_voxel_budget() == expected

    def test_uses_minimum_across_gpus(self, monkeypatch):
        """With multiple GPUs the smallest VRAM determines the budget."""
        small = MagicMock()
        small.total_memory = _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES // 2  # 8 GB
        large = MagicMock()
        large.total_memory = _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES * 2   # 32 GB

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda i: small if i == 0 else large,
        )

        expected = _C.PATCH_BUDGET_REFERENCE_VOXELS // 2
        assert au._get_voxel_budget() == expected

    def test_budget_scales_inversely_with_batch_size(self, monkeypatch):
        """Doubling batch size halves the per-patch voxel budget."""
        props = MagicMock()
        props.total_memory = _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)

        budget_bs2 = au._get_voxel_budget(batch_size_per_gpu=2)
        budget_bs4 = au._get_voxel_budget(batch_size_per_gpu=4)
        assert budget_bs4 == budget_bs2 // 2


# ---------------------------------------------------------------------------
# _snap_lr_to_nnunet_compatible
# ---------------------------------------------------------------------------

class TestSnapLrToNnunetCompatible:
    """Unit tests for analyzer_utils._snap_lr_to_nnunet_compatible.

    get_unet_params is mocked so that each test fully controls the cumulative
    z_divisor without depending on the real nnUNet architecture planner.
    """

    @staticmethod
    def _make_strides(z_divisor: int, low_res_axis: int) -> list:
        """Build a minimal strides list that yields the desired z_divisor.

        strides[0] is always [1,1,1] (the input block that the function skips).
        Subsequent entries each contribute a factor-of-2 stride on the low-res
        axis until the cumulative product equals z_divisor.
        """
        strides = [[1, 1, 1]]
        remaining = z_divisor
        while remaining > 1:
            factor = 2 if remaining % 2 == 0 else remaining
            s = [1, 1, 1]
            s[low_res_axis] = factor
            strides.append(s)
            remaining //= factor
        return strides

    def _patch_get_unet_params(self, monkeypatch, z_divisor, low_res_axis):
        strides = self._make_strides(z_divisor, low_res_axis)
        monkeypatch.setattr(
            au,
            "get_unet_params",
            lambda patch_size, spacing: (None, strides, None),
        )

    # Parametrized cases:
    # (lr_patch, z_divisor, low_res_axis, median_ip, median_lr, min_lr,
    #  expected, test_id)
    @pytest.mark.parametrize(
        "lr_patch,z_divisor,low_res_axis,median_ip,median_lr,min_lr,expected",
        [
            # --- Already compatible (no change needed) ---
            pytest.param(8, 2, 2, 512, 40, 5, 8, id="already_compat_z2"),
            pytest.param(10, 2, 2, 512, 40, 5, 10, id="already_compat_z2_10"),
            pytest.param(20, 4, 2, 512, 20, 5, 20, id="already_compat_z4"),
            pytest.param(16, 8, 2, 512, 40, 5, 16, id="already_compat_z8"),
            # --- z_divisor=1: function returns lr_patch unchanged ---
            pytest.param(7, 1, 2, 512, 40, 5, 7, id="zdiv1_no_snap"),
            pytest.param(13, 1, 2, 512, 20, 5, 13, id="zdiv1_no_snap_odd"),
            # --- z_divisor=2: snap up ---
            pytest.param(7, 2, 2, 512, 40, 5, 8, id="zdiv2_snap_up_7"),
            pytest.param(9, 2, 2, 512, 40, 5, 10, id="zdiv2_snap_up_9"),
            pytest.param(11, 2, 2, 512, 40, 5, 12, id="zdiv2_snap_up_11"),
            # --- z_divisor=4: snap up (prostate-bug regime) ---
            pytest.param(18, 4, 2, 512, 20, 5, 20, id="zdiv4_snap_up_prostate"),
            pytest.param(17, 4, 2, 512, 20, 5, 20, id="zdiv4_snap_up_17"),
            pytest.param(15, 4, 2, 512, 20, 5, 16, id="zdiv4_snap_up_15"),
            pytest.param(13, 4, 2, 512, 20, 5, 16, id="zdiv4_snap_up_13"),
            # --- z_divisor=4: snap up exceeds median_lr → snap down ---
            pytest.param(18, 4, 2, 512, 19, 5, 16, id="zdiv4_snap_down_18"),
            pytest.param(14, 4, 2, 512, 15, 5, 12, id="zdiv4_snap_down_14"),
            # --- z_divisor=8 ---
            pytest.param(6, 8, 2, 512, 20, 5, 8, id="zdiv8_snap_up_6"),
            pytest.param(10, 8, 2, 512, 20, 5, 16, id="zdiv8_snap_up_10"),
            pytest.param(18, 8, 2, 512, 23, 5, 16, id="zdiv8_snap_down_18"),
            # --- min_lr floor protection ---
            pytest.param(3, 4, 2, 512, 20, 3, 4, id="min_lr_floor_snap_up"),
            pytest.param(3, 4, 2, 512, 3, 3, 3, id="min_lr_floor_snap_down"),
            # --- low_res_axis variety ---
            pytest.param(7, 2, 0, 512, 40, 5, 8, id="axis0_snap_up"),
            pytest.param(7, 2, 1, 512, 40, 5, 8, id="axis1_snap_up"),
        ],
    )
    def test_snapping_logic(
        self,
        monkeypatch,
        lr_patch,
        z_divisor,
        low_res_axis,
        median_ip,
        median_lr,
        min_lr,
        expected,
    ):
        """Snapping logic is correct across the full parameter space."""
        self._patch_get_unet_params(monkeypatch, z_divisor, low_res_axis)

        # Build a target_spacing whose argmax equals low_res_axis.
        spacing = [0.8, 0.8, 0.8]
        spacing[low_res_axis] = 3.0

        result = au._snap_lr_to_nnunet_compatible(
            lr_patch=lr_patch,
            low_res_axis=low_res_axis,
            median_ip=median_ip,
            median_lr=median_lr,
            min_lr=min_lr,
            target_spacing=spacing,
        )
        assert result == expected

    def test_result_always_divisible_by_z_divisor(self, monkeypatch):
        """For any input, the returned value is divisible by z_divisor (or at
        least as large as min_lr when the image is too small to allow it)."""
        low_res_axis = 2
        spacing = [0.8, 0.8, 3.0]
        cases = [
            # (lr_patch, z_divisor, median_ip, median_lr, min_lr)
            (18, 4, 320, 20, 5),
            (7, 2, 512, 40, 5),
            (10, 8, 512, 32, 5),
            (5, 4, 512, 12, 5),
        ]
        for lr_patch, z_divisor, median_ip, median_lr, min_lr in cases:
            self._patch_get_unet_params(monkeypatch, z_divisor, low_res_axis)
            result = au._snap_lr_to_nnunet_compatible(
                lr_patch=lr_patch,
                low_res_axis=low_res_axis,
                median_ip=median_ip,
                median_lr=median_lr,
                min_lr=min_lr,
                target_spacing=spacing,
            )
            # Result is either divisible by z_divisor or equals min_lr
            # (when snapping down would go below min_lr).
            assert result % z_divisor == 0 or result == min_lr, (
                f"lr_patch={lr_patch}, z_divisor={z_divisor}: "
                f"result={result} not divisible and not min_lr"
            )

    def test_snap_never_exceeds_median_lr(self, monkeypatch):
        """The returned patch is always ≤ median_lr for valid (clamped) inputs.

        get_best_patch_size always clips lr_patch to [min_lr, median_lr] before
        calling this helper, so we only test that contract-valid range.
        """
        low_res_axis = 2
        spacing = [0.8, 0.8, 3.0]
        median_lr = 20
        for lr_patch in range(1, median_lr + 1):
            for z_divisor in [2, 4, 8]:
                self._patch_get_unet_params(monkeypatch, z_divisor, low_res_axis)
                median_lr = 20
                result = au._snap_lr_to_nnunet_compatible(
                    lr_patch=lr_patch,
                    low_res_axis=low_res_axis,
                    median_ip=512,
                    median_lr=median_lr,
                    min_lr=1,
                    target_spacing=spacing,
                )
                assert result <= median_lr, (
                    f"lr_patch={lr_patch}, z_divisor={z_divisor}: "
                    f"result={result} exceeds median_lr={median_lr}"
                )


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

    # --- nnUNet compatibility snapping ---

    def test_quasi_2d_prostate_lr_snapped_up_to_nnunet_compatible(
        self, monkeypatch
    ):
        """Prostate-like dataset: lr_raw=18 is not divisible by z_divisor=4.

        Budget is pinned so that budget/320^2 = 18 exactly, reproducing the
        real failure mode reported against a 320x320x20 prostate dataset with
        spacing [0.625, 0.625, 3.6].  The nnUNet planner assigns two z-stride-2
        stages → z_divisor=4.  Snapping 18 up to 20 (nearest mult-4 ≤ median_lr)
        fixes the decoder skip-connection mismatch.
        """
        # budget = 18 * 320^2 → lr_raw = 18 exactly.
        monkeypatch.setattr(
            au, "_get_voxel_budget", lambda batch_size_per_gpu=2: 18 * 320 ** 2
        )
        result = au.get_best_patch_size(
            [320, 320, 20], [0.625, 0.625, 3.6]
        )
        # z must be snapped from 18 → 20 (divisible by 4, ≤ median_lr=20).
        assert result[2] == 20
        # In-plane recomputed with lr_patch=20: snap(sqrt(18*320^2/20))=288.
        assert result[0] == result[1] == 288

    def test_quasi_2d_snap_down_when_snap_up_exceeds_median_lr(
        self, monkeypatch
    ):
        """When snapping up exceeds median_lr, the patch snaps down instead.

        median_lr=19, z_divisor=4.  Snapping 18 up → 20 > 19, so we snap
        down to 16 instead.
        """
        monkeypatch.setattr(
            au, "_get_voxel_budget", lambda batch_size_per_gpu=2: 18 * 320 ** 2
        )
        result = au.get_best_patch_size(
            [320, 320, 19], [0.625, 0.625, 3.6]
        )
        # snapped_up=20 > median_lr=19 → snapped_down=16.
        assert result[2] == 16

    def test_quasi_2d_result_divisible_by_nnunet_z_divisor(self, monkeypatch):
        """The low-res axis of the returned patch is always nnUNet-compatible.

        For each realistic scenario, verify that the result patch size on the
        low-res axis is exactly divisible by the cumulative nnUNet z-stride.
        """
        # budget = 18 * 320^2 triggers the prostate-like regime.
        monkeypatch.setattr(
            au, "_get_voxel_budget", lambda batch_size_per_gpu=2: 18 * 320 ** 2
        )
        scenarios = [
            # (median, spacing)
            ([320, 320, 20], [0.625, 0.625, 3.6]),   # prostate
            ([512, 512, 40], [0.8, 0.8, 3.0]),        # thick-slice CT
            ([512, 512,  8], [0.8, 0.8, 3.0]),        # very thin z
            ([40, 512, 512], [3.0, 0.8, 0.8]),       # low-res on axis 0
            ([512,  40, 512], [0.8, 3.0, 0.8]),       # low-res on axis 1
        ]
        for median, spacing in scenarios:
            result = au.get_best_patch_size(median, spacing)
            low_res_axis = spacing.index(max(spacing))
            median_ip = max(median[i] for i in range(3) if i != low_res_axis)
            lr = result[low_res_axis]

            trial = [median_ip, median_ip, median_ip]
            trial[low_res_axis] = lr
            _, strides, _ = get_unet_params(trial, spacing)
            z_divisor = 1
            for s in strides[1:]:
                if s[low_res_axis] > 1:
                    z_divisor *= s[low_res_axis]

            assert lr % z_divisor == 0, (
                f"median={median}, spacing={spacing}: "
                f"lr_patch={lr} not divisible by z_divisor={z_divisor}"
            )

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


# ---------------------------------------------------------------------------
# _largest_multiple_of_32_leq
# ---------------------------------------------------------------------------

class TestLargestMultipleOf32Leq:
    """Tests for analyzer_utils._largest_multiple_of_32_leq."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(32,  32,  id="exact_32"),
            pytest.param(64,  64,  id="exact_64"),
            pytest.param(128, 128, id="exact_128"),
            pytest.param(256, 256, id="exact_256"),
            pytest.param(512, 512, id="exact_512"),
        ],
    )
    def test_exact_multiples_unchanged(self, value, expected):
        """Values that are already multiples of 32 are returned as-is."""
        assert au._largest_multiple_of_32_leq(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(33,  32,  id="33_to_32"),
            pytest.param(63,  32,  id="63_to_32"),
            pytest.param(65,  64,  id="65_to_64"),
            pytest.param(96,  96,  id="96_exact"),
            pytest.param(100, 96,  id="100_to_96"),
            pytest.param(127, 96,  id="127_to_96"),
            pytest.param(480, 480, id="480_exact"),
            pytest.param(491, 480, id="491_to_480"),  # the bug scenario
            pytest.param(511, 480, id="511_to_480"),
        ],
    )
    def test_non_multiples_snap_down(self, value, expected):
        """Non-multiples snap down to the nearest multiple of 32."""
        assert au._largest_multiple_of_32_leq(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0,  id="zero"),
            pytest.param(1,  id="one"),
            pytest.param(20, id="twenty"),
            pytest.param(31, id="thirty_one"),
        ],
    )
    def test_below_minimum_returns_32(self, value):
        """Values below 32 are floored to the default minimum of 32."""
        assert au._largest_multiple_of_32_leq(value) == 32

    def test_custom_minimum_respected(self):
        """A custom minimum is returned when the snap result is smaller."""
        assert au._largest_multiple_of_32_leq(10, minimum=64) == 64

    def test_result_is_always_multiple_of_32(self):
        """Output is always divisible by 32 for a wide range of inputs."""
        for v in range(1, 600):
            result = au._largest_multiple_of_32_leq(v)
            assert result % 32 == 0, f"_largest_multiple_of_32_leq({v}) = {result}"

    def test_result_never_exceeds_input(self):
        """Output never exceeds the input value."""
        for v in range(32, 600):
            assert au._largest_multiple_of_32_leq(v) <= v


# ---------------------------------------------------------------------------
# _get_voxel_budget
# ---------------------------------------------------------------------------

class TestGetVoxelBudget:
    """Tests for analyzer_utils._get_voxel_budget."""

    def test_returns_reference_voxels_when_no_cuda(self, monkeypatch):
        """Falls back to PATCH_BUDGET_DEFAULT_VOXELS when CUDA is unavailable."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert au._get_voxel_budget() == _C.PATCH_BUDGET_DEFAULT_VOXELS

    def test_scales_inversely_with_batch_size(self, monkeypatch):
        """Doubling batch_size halves the per-patch voxel budget (CUDA path)."""
        class _FakeProps:
            total_memory = _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES  # 16 GB

        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
        monkeypatch.setattr("torch.cuda.get_device_properties", lambda i: _FakeProps())

        b1 = au._get_voxel_budget(batch_size_per_gpu=1)
        b2 = au._get_voxel_budget(batch_size_per_gpu=2)
        b4 = au._get_voxel_budget(batch_size_per_gpu=4)
        assert b1 == b2 * 2
        assert b2 == b4 * 2

    def test_scales_with_gpu_memory(self, monkeypatch):
        """Budget scales linearly with GPU memory relative to the 16 GB reference."""
        class _FakeProps:
            def __init__(self, mem):
                self.total_memory = mem

        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

        ref_mem = _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES  # 16 GB
        ref_vox = _C.PATCH_BUDGET_REFERENCE_VOXELS
        ref_bs = _C.PATCH_BUDGET_REFERENCE_BATCH_SIZE

        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda i: _FakeProps(ref_mem),
        )
        assert au._get_voxel_budget(ref_bs) == ref_vox

        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda i: _FakeProps(ref_mem * 2),
        )
        assert au._get_voxel_budget(ref_bs) == ref_vox * 2

    def test_uses_minimum_gpu_memory_across_devices(self, monkeypatch):
        """When multiple GPUs are present the smallest memory device is used."""
        class _FakeProps:
            def __init__(self, mem):
                self.total_memory = mem

        mems = [32 * (1024 ** 3), 16 * (1024 ** 3)]  # 32 GB and 16 GB
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda i: _FakeProps(mems[i]),
        )
        expected = int(
            min(mems)
            / _C.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES
            * _C.PATCH_BUDGET_REFERENCE_VOXELS
            * _C.PATCH_BUDGET_REFERENCE_BATCH_SIZE
            / 2
        )
        assert au._get_voxel_budget(batch_size_per_gpu=2) == expected


# ---------------------------------------------------------------------------
# get_best_patch_size — shared helpers
# ---------------------------------------------------------------------------

def _patch_budget(budget: int):
    """Context manager: pin the voxel budget to a fixed value."""
    return patch(
        "mist.analyze_data.analyzer_utils._get_voxel_budget",
        return_value=budget,
    )


def _patch_snap_identity():
    """Context manager: make _snap_lr_to_nnunet_compatible a no-op."""
    return patch(
        "mist.analyze_data.analyzer_utils._snap_lr_to_nnunet_compatible",
        side_effect=lambda lr, *_a, **_kw: lr,
    )


# ---------------------------------------------------------------------------
# get_best_patch_size — invariants
# ---------------------------------------------------------------------------

def _is_quasi2d(spacing: list) -> bool:
    """Return True if spacing triggers quasi-2D mode."""
    return max(spacing) / min(spacing) > _C.MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD


def _low_res_axis(spacing: list) -> int:
    """Return the index of the low-resolution axis."""
    return int(np.argmax(spacing))


class TestGetBestPatchSizeInvariants:
    """Universal invariants that must hold for every get_best_patch_size call."""

    # (median_resampled_size, target_spacing, budget) triples that exercise
    # both quasi-2D and isotropic branches across a range of GPU sizes.
    _CASES = [
        # Quasi-2D cases (anisotropy > 3)
        ([491, 404, 52],  [0.81, 0.81, 3.98],  _REF_BUDGET),   # reported bug
        ([256, 256, 10],  [0.5,  0.5,  3.0],   _REF_BUDGET),   # thin-slice CT (ratio=6)
        ([128, 128, 20],  [1.0,  1.0,  5.0],   _REF_BUDGET * 4),
        ([300, 300, 8],   [0.6,  0.6,  4.0],   _REF_BUDGET // 2),
        ([200, 180, 15],  [0.8,  0.8,  4.0],   _REF_BUDGET),
        # Isotropic cases (anisotropy ≤ 3)
        ([128, 128, 128], [1.0,  1.0,  1.0],   _REF_BUDGET),
        ([64,  64,  64],  [1.5,  1.5,  1.5],   _REF_BUDGET),
        ([200, 180, 160], [0.8,  0.9,  1.0],   _REF_BUDGET),
        ([512, 512, 400], [0.5,  0.5,  0.5],   _REF_BUDGET * 2),
        ([96,  80,  72],  [1.2,  1.2,  2.0],   _REF_BUDGET),   # mild anisotropy
    ]

    @pytest.mark.parametrize(
        "median, spacing, budget",
        [pytest.param(*c, id=f"case{i}") for i, c in enumerate(_CASES)],
    )
    def test_in_plane_dims_are_multiples_of_32(self, median, spacing, budget):
        """In-plane patch dimensions must be divisible by 32 in all modes.

        In quasi-2D mode the low-res axis is snapped by _snap_lr_to_nnunet_compatible
        (which uses nnUNet-specific strides, not 32), so only in-plane axes are
        required to be multiples of 32.  In isotropic mode all axes are in-plane.
        """
        with _patch_budget(budget), _patch_snap_identity():
            patch_size = au.get_best_patch_size(median, spacing)
        if _is_quasi2d(spacing):
            lr = _low_res_axis(spacing)
            in_plane = [i for i in range(3) if i != lr]
        else:
            in_plane = [0, 1, 2]
        for i in in_plane:
            assert patch_size[i] % 32 == 0, (
                f"In-plane axis {i}: patch={patch_size} has non-multiple-of-32 dim"
            )

    @pytest.mark.parametrize(
        "median, spacing, budget",
        [pytest.param(*c, id=f"case{i}") for i, c in enumerate(_CASES)],
    )
    def test_in_plane_dims_at_least_32(self, median, spacing, budget):
        """In-plane patch dimensions must be at least 32.

        The low-res axis in quasi-2D mode may legitimately be smaller than 32
        (e.g. 8 slices for thick-slice CT) since MIN_LOW_RES_AXIS_PATCH_SIZE=5.
        """
        with _patch_budget(budget), _patch_snap_identity():
            patch_size = au.get_best_patch_size(median, spacing)
        if _is_quasi2d(spacing):
            lr = _low_res_axis(spacing)
            in_plane = [i for i in range(3) if i != lr]
        else:
            in_plane = [0, 1, 2]
        for i in in_plane:
            assert patch_size[i] >= 32, (
                f"In-plane axis {i}: patch={patch_size} has dim < 32"
            )

    @pytest.mark.parametrize(
        "median, spacing, budget",
        [pytest.param(*c, id=f"case{i}") for i, c in enumerate(_CASES)],
    )
    def test_patch_fits_within_median_image(self, median, spacing, budget):
        """No patch dimension should exceed the corresponding median image size."""
        with _patch_budget(budget), _patch_snap_identity():
            patch_size = au.get_best_patch_size(median, spacing)
        for p, m in zip(patch_size, median):
            assert p <= m, f"patch dim {p} exceeds median size {m}"

    @pytest.mark.parametrize(
        "median, spacing, budget",
        [pytest.param(*c, id=f"case{i}") for i, c in enumerate(_CASES)],
    )
    def test_returns_three_dimensions(self, median, spacing, budget):
        """Patch size must always be a list of exactly 3 integers."""
        with _patch_budget(budget), _patch_snap_identity():
            patch_size = au.get_best_patch_size(median, spacing)
        assert len(patch_size) == 3
        assert all(isinstance(d, int) for d in patch_size)


# ---------------------------------------------------------------------------
# get_best_patch_size — quasi-2D mode
# ---------------------------------------------------------------------------

class TestGetBestPatchSizeQuasi2D:
    """Tests for the quasi-2D branch (anisotropy ratio > MAX threshold)."""

    # Spacing where z-axis is the low-res axis (3.98/0.81 ≈ 4.9 > 3.0).
    _ANISO_SPACING = [0.81, 0.81, 3.98]
    _ANISO_MEDIAN = [491, 404, 52]

    def test_regression_bug_ip_patch_is_multiple_of_32(self):
        """Regression: ip_patch must be a multiple of 32 even when clamped
        to median_ip.  Before the fix min(512, 491) = 491 was returned raw."""
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                self._ANISO_MEDIAN, self._ANISO_SPACING
            )
        assert patch_size[0] % 32 == 0
        assert patch_size[1] % 32 == 0
        assert patch_size[0] != 491, "491 is not a multiple of 32 — snap-order bug"

    def test_regression_bug_exact_values(self):
        """Regression: with REF_BUDGET and snap=identity the result is [384,384,8].

        median_ip = min(491, 404) = 404 → largest multiple of 32 ≤ 404 = 384.
        (Before the max→min fix ip_patch was 480, which exceeded axis-1 size 404.)
        """
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                self._ANISO_MEDIAN, self._ANISO_SPACING
            )
        assert patch_size[0] == 384
        assert patch_size[1] == 384

    def test_low_res_axis_identified_correctly(self):
        """The axis with the largest spacing gets the small (lr) patch size."""
        # z has the largest spacing → index 2 should be the small dimension.
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [200, 200, 30], [0.8, 0.8, 4.0]
            )
        # In quasi-2D mode in-plane ≥ low-res is not guaranteed by maths, but
        # the low-res patch should be smaller than the in-plane patches.
        assert patch_size[2] <= patch_size[0]
        assert patch_size[2] <= patch_size[1]

    def test_lr_patch_respects_min_low_res_size(self):
        """lr_patch never drops below MIN_LOW_RES_AXIS_PATCH_SIZE."""
        # Tiny image depth + tiny budget → lr_patch could underflow without the clip.
        with _patch_budget(32 ** 3), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [64, 64, 6], [0.5, 0.5, 4.0]
            )
        # low-res axis is index 2; must be ≥ min(MIN_LOW_RES_AXIS_PATCH_SIZE, 6)
        min_lr = min(_C.MIN_LOW_RES_AXIS_PATCH_SIZE, 6)
        assert patch_size[2] >= min_lr

    def test_lr_patch_does_not_exceed_median_lr(self):
        """lr_patch never exceeds the median depth of the image."""
        median = [300, 300, 20]
        with _patch_budget(_REF_BUDGET * 8), _patch_snap_identity():
            patch_size = au.get_best_patch_size(median, [0.8, 0.8, 5.0])
        # low-res axis is index 2
        assert patch_size[2] <= median[2]

    def test_in_plane_both_axes_equal(self):
        """Both in-plane axes always receive the same patch size."""
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                self._ANISO_MEDIAN, self._ANISO_SPACING
            )
        assert patch_size[0] == patch_size[1]

    def test_quasi2d_triggered_just_above_threshold(self):
        """Anisotropy ratio just above threshold enters quasi-2D mode."""
        # ratio = 3.1 / 1.0 = 3.1 > 3.0 → quasi-2D
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [200, 200, 40], [1.0, 1.0, 3.1]
            )
        # Low-res axis (z) should be the smallest.
        assert patch_size[2] <= patch_size[0]

    def test_x_axis_as_low_res(self):
        """Quasi-2D correctly handles x as the low-resolution axis."""
        # x spacing is largest → axis 0 is low-res
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [15, 256, 256], [5.0, 0.8, 0.8]
            )
        # axis 0 should be the small dimension
        assert patch_size[0] <= patch_size[1]
        assert patch_size[0] <= patch_size[2]

    @pytest.mark.parametrize(
        "batch_size, expected_ordering",
        [
            pytest.param(1, "larger",  id="batch1_larger_patch"),
            pytest.param(4, "smaller", id="batch4_smaller_patch"),
        ],
    )
    def test_larger_batch_size_gives_smaller_patch(self, batch_size, expected_ordering):
        """Budget scales inversely with batch_size so patch volume shrinks."""
        with _patch_snap_identity():
            patch_b2 = au.get_best_patch_size(
                [300, 300, 40], [0.8, 0.8, 4.0], batch_size_per_gpu=2
            )
            patch_bx = au.get_best_patch_size(
                [300, 300, 40], [0.8, 0.8, 4.0], batch_size_per_gpu=batch_size
            )
        vol_b2 = int(np.prod(patch_b2))
        vol_bx = int(np.prod(patch_bx))
        if expected_ordering == "larger":
            assert vol_bx >= vol_b2
        else:
            assert vol_bx <= vol_b2


# ---------------------------------------------------------------------------
# get_best_patch_size — 3D isotropic mode
# ---------------------------------------------------------------------------

class TestGetBestPatchSizeIsotropic:
    """Tests for the 3D isotropic branch (anisotropy ratio ≤ MAX threshold)."""

    def test_isotropic_spacing_yields_roughly_cubic_patch(self):
        """Isotropic spacing → all three patch dims should be equal."""
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [256, 256, 256], [1.0, 1.0, 1.0]
            )
        assert patch_size[0] == patch_size[1] == patch_size[2]

    def test_budget_constrains_patch_volume(self):
        """Patch voxel count should not exceed the budget by more than 32³."""
        budget = _REF_BUDGET
        with _patch_budget(budget), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [256, 256, 256], [1.0, 1.0, 1.0]
            )
        volume = int(np.prod(patch_size))
        # Allow one 32-voxel snap-up per axis beyond the theoretical budget.
        assert volume <= budget + 3 * 32 * (budget ** (2 / 3))

    def test_small_image_clamped_to_median_size(self):
        """When the image is smaller than the budget allows, patch ≤ image size."""
        # Tiny image: budget would suggest larger than image → clamp.
        with _patch_budget(_REF_BUDGET * 10), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [64, 64, 64], [1.0, 1.0, 1.0]
            )
        assert patch_size[0] <= 64
        assert patch_size[1] <= 64
        assert patch_size[2] <= 64

    def test_non_cubic_image_distributes_budget_by_spacing(self):
        """Anisotropic image (but below threshold) is handled in isotropic mode."""
        # ratio = 2.0/1.0 = 2.0 ≤ 3.0 → isotropic mode
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [200, 180, 100], [1.0, 1.0, 2.0]
            )
        # All invariants should still hold.
        assert all(d % 32 == 0 for d in patch_size)
        assert all(d >= 32 for d in patch_size)

    def test_isotropic_just_below_anisotropy_threshold(self):
        """Anisotropy ratio just below threshold stays in isotropic mode."""
        # ratio = 3.0 / 1.0 = 3.0, which is NOT > threshold → isotropic
        with _patch_budget(_REF_BUDGET), _patch_snap_identity():
            patch_size = au.get_best_patch_size(
                [128, 128, 128], [1.0, 1.0, 3.0]
            )
        assert all(d % 32 == 0 for d in patch_size)
        assert all(d >= 32 for d in patch_size)

    def test_large_budget_clamped_to_image_size(self):
        """A very large budget results in patch ≤ image on all axes."""
        with _patch_budget(_REF_BUDGET * 100), _patch_snap_identity():
            median = [96, 80, 72]
            patch_size = au.get_best_patch_size(median, [1.0, 1.0, 1.0])
        for p, m in zip(patch_size, median):
            assert p <= m
