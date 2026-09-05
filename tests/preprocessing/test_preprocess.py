"""Tests for mist.preprocessing.preprocess."""

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

# MIST imports.
from mist.preprocessing import preprocess as pp
from mist.utils import console as console_mod
from mist.utils import sitk_io


def _make_sitk_image(
    arr_xyz: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
) -> sitk.Image:
    """Build a real SimpleITK image from an (x, y, z)-ordered array."""
    return sitk_io.image_from_array(
        arr_xyz.astype(np.float32),
        spacing=spacing,
        origin=origin,
        direction=direction if direction is not None else np.eye(3),
    )


class _PB:
    """Very small progress-bar stub used by tests."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def track(self, iterable, **kwargs):
        """Track progress of an iterable."""
        return iterable


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a CSV to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Return a minimal base config for preprocessing."""
    return {
        "dataset_info": {
            "modality": "ct",
            "labels": [0, 1],
            "images": ["image"],
        },
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": True,
            "compute_dtms": False,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }


def test_window_and_normalize_ct_uses_config_values():
    """CT path uses configured window and z-score parameters."""
    img = np.array([-200.0, -100.0, 0.0, 50.0, 200.0], dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "ct"},
        "preprocessing": {
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 10.0,
            },
        },
    }
    out = pp.window_and_normalize(img, cfg)
    expected = np.array([-10.0, -10.0, 0.0, 5.0, 10.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32


def test_window_and_normalize_nonct_with_nonzero_mask():
    """Non-CT path with nonzero-mask normalization behavior."""
    img = np.array([0.0, 0.0, 1.0, 3.0], dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": True},
    }
    out = pp.window_and_normalize(img, cfg)

    mask = np.array([0, 0, 1, 1], dtype=np.float32)
    clip_low = np.percentile(img[mask > 0], 0.5)
    clip_high = np.percentile(img[mask > 0], 99.5)
    mean = img[mask > 0].mean()
    std = img[mask > 0].std()
    expected = np.clip(img, clip_low, clip_high) * mask
    expected = ((expected - mean) / std) * mask

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32


def test_window_and_normalize_nonct_full_image():
    """Non-CT path with full-image normalization."""
    img = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": False},
    }
    out = pp.window_and_normalize(img, cfg)
    assert out.dtype == np.float32
    assert np.isclose(out.mean(), 0.0, atol=1e-5)


def test_window_and_normalize_nonzero_mask_all_zeros_falls_back_to_full_image():
    """All-zero image with nonzero-mask flag falls back to full-image normalization."""
    img = np.zeros(5, dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": True},
    }
    out = pp.window_and_normalize(img, cfg)
    assert out.dtype == np.float32
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_window_and_normalize_zero_std_returns_zeros():
    """Constant (non-zero) image with std=0 returns all-zeros without warning."""
    img = np.full(5, 3.0, dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": False},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes an error
        out = pp.window_and_normalize(img, cfg)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_resample_image_isotropic_changes_spacing_and_size():
    """Isotropic resample changes spacing/size and preserves origin/direction."""
    arr = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    img = _make_sitk_image(arr, spacing=(2.0, 2.0, 2.0), origin=(1.0, 2.0, 3.0))

    out = pp.resample_image(img, target_spacing=(1.0, 1.0, 1.0))

    assert out.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))
    assert out.GetSize() == (8, 8, 8)
    assert out.GetOrigin() == pytest.approx((1.0, 2.0, 3.0))
    assert np.allclose(out.GetDirection(), np.eye(3).flatten())


def test_resample_image_new_size_overrides_computed_size():
    """An explicit new_size is used as-is instead of being computed."""
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))

    out = pp.resample_image(img, target_spacing=(1.0, 1.0, 1.0), new_size=(6, 6, 6))
    assert out.GetSize() == (6, 6, 6)


def test_resample_image_aniso_axis_type_error(monkeypatch):
    """Anisotropic path with non-int axis raises ValueError."""
    img = _make_sitk_image(np.zeros((4, 4, 4), dtype=np.float32))
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _s: {"is_anisotropic": True, "low_resolution_axis": "z"},
        raising=True,
    )

    with pytest.raises(ValueError, match="must be an integer"):
        pp.resample_image(img, (1.0, 1.0, 1.0))


def test_resample_image_aniso_intermediate_called():
    """Anisotropic image is resampled via the nearest-neighbor intermediate step."""
    arr = np.ones((4, 4, 2), dtype=np.float32)
    img = _make_sitk_image(arr, spacing=(1.0, 1.0, 5.0))

    out = pp.resample_image(img, target_spacing=(1.0, 1.0, 1.0))

    # The low-resolution axis (z, index 2 -- spacing ratio 5 > 3) ends up at
    # the target spacing, same as every other axis.
    assert out.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))
    assert out.GetSize() == (4, 4, 10)


def test_resample_mask_happy_path():
    """Resampling a mask upsamples it and preserves origin/direction."""
    labels = [0, 1, 2]
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[:2, :, :] = 1
    arr[2:, :2, :] = 2
    mask = _make_sitk_image(arr, spacing=(2.0, 2.0, 2.0), origin=(1.0, 2.0, 3.0))

    out = pp.resample_mask(mask, labels=labels, target_spacing=(1.0, 1.0, 1.0))

    assert out.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))
    assert out.GetSize() == (8, 8, 8)
    assert out.GetOrigin() == pytest.approx((1.0, 2.0, 3.0))

    out_arr = sitk_io.array_from_image(out)
    assert set(np.unique(out_arr)).issubset({0.0, 1.0, 2.0})


def test_resample_mask_aniso_axis_not_int_raises(monkeypatch):
    """Anisotropic resample with non-int axis raises ValueError."""
    mask = _make_sitk_image(np.zeros((4, 4, 4), dtype=np.float32))
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _s: {"is_anisotropic": True, "low_resolution_axis": "z"},
        raising=True,
    )

    with pytest.raises(ValueError, match="low resolution axis must be an integer"):
        pp.resample_mask(mask, labels=[0, 1], target_spacing=(1.0, 1.0, 1.0))


def test_resample_mask_aniso_intermediate_used_for_each_label():
    """Anisotropic mask resample runs the intermediate step for every label."""
    labels = [0, 1]
    arr = np.zeros((4, 4, 2), dtype=np.float32)
    arr[2:, :, :] = 1
    mask = _make_sitk_image(arr, spacing=(1.0, 1.0, 5.0))

    out = pp.resample_mask(mask, labels=labels, target_spacing=(1.0, 1.0, 1.0))

    assert out.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))
    assert out.GetSize() == (4, 4, 10)
    out_arr = sitk_io.array_from_image(out)
    assert set(np.unique(out_arr)).issubset({0.0, 1.0})


def test_resample_mask_oblique_direction_preserved():
    """A mask with oblique direction cosines keeps that direction after resample."""
    # A small rotation about the z axis -- not axis-aligned.
    theta = 0.2
    direction = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[1:3, 1:3, 1:3] = 1
    mask = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0), direction=direction)

    out = pp.resample_mask(mask, labels=[0, 1], target_spacing=(0.5, 0.5, 0.5))

    assert np.allclose(out.GetDirection(), direction.flatten())


def test_compute_dtm_shapes_and_types():
    """compute_dtm returns expected shape/dtype with one empty class."""
    labels = [0, 1]
    arr = np.zeros((5, 6, 7), dtype=np.float32)
    arr[2, 3, 4] = 1  # Label 1 present at a single voxel; label 0 elsewhere.
    mask = _make_sitk_image(arr)

    out = pp.compute_dtm(mask, labels=labels, normalize_dtm=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 6, 7, 2)
    assert out.dtype == np.float32


def test_compute_dtm_normalized_range_is_bounded():
    """Normalized DTM values stay within [-1, 1] for a non-empty, non-full mask."""
    labels = [0, 1]
    arr = np.zeros((6, 6, 6), dtype=np.float32)
    arr[2:4, 2:4, 2:4] = 1
    mask = _make_sitk_image(arr)

    out = pp.compute_dtm(mask, labels=labels, normalize_dtm=True)
    assert np.all(out >= -1.0 - 1e-6)
    assert np.all(out <= 1.0 + 1e-6)


def test_compute_dtm_empty_mask_diagonal_distance():
    """Empty mask with normalize_dtm=False uses diagonal distance."""
    shape = (2, 2, 2)
    mask = _make_sitk_image(np.zeros(shape, dtype=np.float32))
    labels = [0]

    out = pp.compute_dtm(mask, labels=labels, normalize_dtm=False)
    # Label 0 is the whole (non-empty) background, so this exercises the
    # "empty mask" branch only when a label is entirely absent -- use a label
    # that never appears instead.
    out_missing = pp.compute_dtm(mask, labels=[7], normalize_dtm=False)
    expected = np.sqrt(sum(s**2 for s in shape))
    assert out_missing.shape == (2, 2, 2, 1)
    assert np.allclose(out_missing[..., 0], expected)
    assert out.shape == (2, 2, 2, 1)


def test_preprocess_example_full_flow_no_skip_with_crop_and_dtm(monkeypatch):
    """Full flow: crop, resample, normalize, and compute DTM."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": True,
            "compute_dtms": True,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    img0 = _make_sitk_image(np.ones((4, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    img1 = _make_sitk_image(2 * np.ones((4, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    mask_arr = np.zeros((4, 4, 4), dtype=np.float32)
    mask_arr[1:3, 1:3, 1:3] = 1
    mask_img = _make_sitk_image(mask_arr, spacing=(1.0, 1.0, 1.0))

    seq = iter([img0, img1, mask_img])
    monkeypatch.setattr(sitk_io, "read_image", lambda _p: next(seq), raising=True)
    fg_bbox = {
        "x_start": 0,
        "x_end": 3,
        "y_start": 0,
        "y_end": 3,
        "z_start": 0,
        "z_end": 3,
    }

    calls = {"crop_calls": 0}

    def _crop(im, _bb):
        calls["crop_calls"] += 1
        return im

    monkeypatch.setattr(pp.preprocessing_utils, "crop_to_fg", _crop, raising=True)

    out = pp.preprocess_example(
        cfg,
        image_paths_list=["i0", "i1"],
        mask_path="m.nii.gz",
        fg_bbox=fg_bbox,
    )

    assert out["image"].shape == (4, 4, 4, 2)
    assert out["image"].dtype == np.float32
    assert out["mask"].shape == (4, 4, 4, 1)
    assert out["mask"].dtype == np.uint8
    assert out["dtm"].shape == (4, 4, 4, 2)
    assert out["dtm"].dtype == np.float32
    assert calls["crop_calls"] == 3  # Two images + one mask.


def test_preprocess_example_skip_true_no_resample_no_normalize(monkeypatch):
    """skip=True is a pure pass-through: no reorient, resample, or normalize."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "mri"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": True,
            "crop_to_foreground": False,
            "compute_dtms": False,
            "normalize_dtms": False,
            "normalize_with_nonzero_mask": False,
        },
    }

    img0 = _make_sitk_image(np.ones((2, 2, 2), dtype=np.float32), spacing=(1.5, 1.5, 1.5))
    mask_img = _make_sitk_image(np.zeros((2, 2, 2), dtype=np.float32))

    seq = iter([img0, mask_img])
    monkeypatch.setattr(sitk_io, "read_image", lambda _p: next(seq), raising=True)
    monkeypatch.setattr(
        sitk_io,
        "reorient_image",
        lambda *_a, **_k: pytest.fail("reorient_image must not be called when skip=True."),
        raising=True,
    )
    monkeypatch.setattr(
        pp,
        "resample_image",
        lambda *_a, **_k: pytest.fail("resample_image must not be called when skip=True."),
    )
    monkeypatch.setattr(
        pp,
        "window_and_normalize",
        lambda *_a, **_k: pytest.fail("window_and_normalize must not be called when skip=True."),
    )

    out = pp.preprocess_example(cfg, image_paths_list=["i0"], mask_path="m.nii.gz", fg_bbox=None)
    assert out["image"].shape == (2, 2, 2, 1)
    assert out["mask"].shape == (2, 2, 2, 1)
    assert out["dtm"] is None
    assert out["fg_bbox"] is None


def test_preprocess_example_crop_requires_bbox_error(monkeypatch):
    """skip=False + crop=True + get_fg_mask_bbox returns None raises ValueError."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1, 1, 1),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": True,
            "compute_dtms": False,
            "normalize_dtms": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    monkeypatch.setattr(
        sitk_io,
        "read_image",
        lambda _p: _make_sitk_image(np.zeros((2, 2, 2), dtype=np.float32)),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "get_fg_mask_bbox",
        lambda _im: None,
        raising=True,
    )

    with pytest.raises(ValueError, match="Foreground bounding box is required"):
        pp.preprocess_example(cfg, image_paths_list=["img.nii.gz"], mask_path=None, fg_bbox=None)


def test_preprocess_example_skip_true_crop_flag_ignored(monkeypatch):
    """skip=True ignores crop_to_foreground: no crop, no reorient, fg_bbox=None."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": True,
            "crop_to_foreground": True,
            "compute_dtms": False,
            "normalize_dtms": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    monkeypatch.setattr(
        sitk_io,
        "read_image",
        lambda _p: _make_sitk_image(np.zeros((2, 2, 2), dtype=np.float32)),
        raising=True,
    )
    monkeypatch.setattr(
        sitk_io,
        "reorient_image",
        lambda *_a, **_k: pytest.fail("reorient_image must not be called when skip=True."),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "get_fg_mask_bbox",
        lambda *_a, **_k: pytest.fail("get_fg_mask_bbox must not be called when skip=True."),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "crop_to_fg",
        lambda *_a, **_k: pytest.fail("crop_to_fg must not be called when skip=True."),
        raising=True,
    )

    out = pp.preprocess_example(cfg, image_paths_list=["img.nii.gz"], mask_path=None, fg_bbox=None)
    assert out["fg_bbox"] is None
    assert out["image"].shape == (2, 2, 2, 1)


def test_preprocess_example_inference_mode_sets_mask_and_dtm_none(monkeypatch):
    """Inference mode sets mask=None and dtm=None and does not compute DTM."""
    monkeypatch.setattr(
        sitk_io,
        "read_image",
        lambda _p: _make_sitk_image(np.zeros((4, 5, 6), dtype=np.float32)),
        raising=True,
    )
    monkeypatch.setattr(
        sitk_io,
        "reorient_image",
        lambda *_a, **_k: pytest.fail("reorient_image must not be called when skip=True."),
        raising=True,
    )

    def _no_compute_dtm(*_a, **_k):
        pytest.fail("compute_dtm should not be called in inference mode.")

    monkeypatch.setattr(pp, "compute_dtm", _no_compute_dtm, raising=True)

    config = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": True,
            "crop_to_foreground": False,
            "compute_dtms": True,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    out = pp.preprocess_example(
        config=config,
        image_paths_list=["/fake/image.nii.gz"],
        mask_path=None,
        fg_bbox=None,
    )

    assert out["mask"] is None
    assert out["dtm"] is None
    assert isinstance(out["image"], np.ndarray)
    assert out["image"].shape == (4, 5, 6, 1)
    assert out["image"].dtype == np.float32
    assert out["fg_bbox"] is None


def test_preprocess_dataset_end_to_end_saves_arrays_and_updates_config(
    tmp_path, monkeypatch, base_config
):
    """End-to-end preprocess_dataset writes arrays and updates config."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results / "models").mkdir(parents=True, exist_ok=True)
    _write_json(results / "config.json", base_config)

    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame(
            [
                {
                    "id": "p1",
                    "fold": 0,
                    "image": "/tmp/p1.nii.gz",
                    "mask": "/tmp/p1_mask.nii.gz",
                }
            ]
        ),
    )
    _write_csv(
        results / "fg_bboxes.csv",
        pd.DataFrame([{"id": "p1", "x0": 0, "x1": 2, "y0": 0, "y1": 2, "z0": 0, "z1": 2}]),
    )

    monkeypatch.setattr(pp.progress_bar, "get_progress_bar", lambda *_: _PB(), raising=True)
    monkeypatch.setattr(pp.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)

    def _pe(config, image_paths_list, mask_path, fg_bbox):
        img = np.ones((2, 2, 2, 1), dtype=np.float32)
        mask = np.zeros((2, 2, 2, 1), dtype=np.uint8)
        dtm = np.full((2, 2, 2, 2), 2.0, dtype=np.float32)
        return {"image": img, "mask": mask, "dtm": dtm, "fg_bbox": fg_bbox}

    monkeypatch.setattr(pp, "preprocess_example", _pe, raising=True)

    ns = argparse.Namespace(
        results=str(results),
        numpy=str(numpy_dir),
        compute_dtms=True,
        no_preprocess=True,
        num_workers_preprocess=None,
    )
    pp.preprocess_dataset(ns)

    img_npy = numpy_dir / "images" / "p1.npy"
    lab_npy = numpy_dir / "labels" / "p1.npy"
    dtm_npy = numpy_dir / "dtms" / "p1.npy"
    assert img_npy.exists()
    assert lab_npy.exists()
    assert dtm_npy.exists()
    assert np.load(img_npy).dtype == np.float32
    assert np.load(lab_npy).dtype == np.uint8
    assert np.load(dtm_npy).dtype == np.float32

    cfg = json.loads((results / "config.json").read_text(encoding="utf-8"))
    assert cfg["preprocessing"]["compute_dtms"] is True
    assert cfg["preprocessing"]["skip"] is True


def test_preprocess_dataset_missing_files_raise(tmp_path, base_config):
    """Missing config/train_paths/fg_bboxes should raise FileNotFoundError."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        pp.preprocess_dataset(
            argparse.Namespace(
                results=str(results),
                numpy=str(numpy_dir),
                compute_dtms=False,
                no_preprocess=False,
            )
        )

    _write_json(results / "config.json", base_config)
    with pytest.raises(FileNotFoundError):
        pp.preprocess_dataset(
            argparse.Namespace(
                results=str(results),
                numpy=str(numpy_dir),
                compute_dtms=False,
                no_preprocess=False,
            )
        )

    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame(
            [
                {
                    "id": "p1",
                    "fold": 0,
                    "image": "/tmp/p1.nii.gz",
                    "mask": "/tmp/p1_mask.nii.gz",
                }
            ]
        ),
    )
    with pytest.raises(FileNotFoundError):
        pp.preprocess_dataset(
            argparse.Namespace(
                results=str(results),
                numpy=str(numpy_dir),
                compute_dtms=False,
                no_preprocess=False,
            )
        )


def test_preprocess_dataset_sets_fg_bbox_none_when_crop_disabled(tmp_path, monkeypatch):
    """When cropping disabled, pass fg_bbox=None to preprocess_example."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results.mkdir(parents=True, exist_ok=True)
    numpy_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset_info": {"images": ["image"], "labels": [0, 1]},
        "spatial_config": {
            "target_spacing": [1.0, 1.0, 1.0],
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": False,
            "compute_dtms": False,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }
    (results / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame(
            [
                {
                    "id": "p1",
                    "image": "/tmp/p1_image.nii.gz",
                    "mask": "/tmp/p1_mask.nii.gz",
                }
            ]
        ),
    )
    _write_csv(
        results / "fg_bboxes.csv",
        pd.DataFrame([{"id": "p1", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}]),
    )

    monkeypatch.setattr(
        pp.progress_bar,
        "get_progress_bar",
        lambda *_a, **_k: _PB(),
        raising=True,
    )
    monkeypatch.setattr(
        pp.io,
        "read_json_file",
        lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
        raising=True,
    )
    monkeypatch.setattr(pp.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)

    observed: dict[str, Any] = {}

    def _fake_preprocess_example(**kwargs):
        observed["fg_bbox"] = kwargs.get("fg_bbox", "MISSING")
        return {
            "image": np.zeros((2, 2, 2, 1), dtype=np.float32),
            "mask": np.zeros((2, 2, 2, 1), dtype=np.uint8),
            "dtm": np.zeros((2, 2, 2, 1), dtype=np.float32),
        }

    monkeypatch.setattr(pp, "preprocess_example", _fake_preprocess_example, raising=True)

    ns = SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        no_preprocess=False,
        compute_dtms=False,
        num_workers_preprocess=None,
    )
    pp.preprocess_dataset(ns)

    assert "fg_bbox" in observed
    assert observed["fg_bbox"] is None
    assert (numpy_dir / "images" / "p1.npy").exists()
    assert (numpy_dir / "labels" / "p1.npy").exists()


# ---------------------------------------------------------------------------
# _preprocess_single_patient
# ---------------------------------------------------------------------------


def test_preprocess_single_patient_happy_path_saves_arrays(tmp_path, monkeypatch):
    """_preprocess_single_patient saves image, mask, and dtm and returns None."""
    output_dirs = {
        "images": tmp_path / "images",
        "labels": tmp_path / "labels",
        "dtms": tmp_path / "dtms",
    }
    for d in output_dirs.values():
        d.mkdir()

    def _pe(config, image_paths_list, mask_path, fg_bbox):
        return {
            "image": np.ones((2, 2, 2, 1), dtype=np.float32),
            "mask": np.zeros((2, 2, 2, 1), dtype=np.uint8),
            "dtm": np.full((2, 2, 2, 2), 3.0, dtype=np.float32),
            "fg_bbox": fg_bbox,
        }

    monkeypatch.setattr(pp, "preprocess_example", _pe, raising=True)

    config = {
        "dataset_info": {"labels": [0, 1]},
        "preprocessing": {"skip": False},
    }
    patient = {"id": "pat_01", "image": "/fake/img.nii.gz", "mask": "/fake/msk.nii.gz"}

    err = pp._preprocess_single_patient(
        config=config,
        patient=patient,
        image_columns=["image"],
        fg_bbox=None,
        output_dirs=output_dirs,
        compute_dtms=True,
    )

    assert err is None
    assert (output_dirs["images"] / "pat_01.npy").exists()
    assert (output_dirs["labels"] / "pat_01.npy").exists()
    assert (output_dirs["dtms"] / "pat_01.npy").exists()
    np.testing.assert_array_equal(
        np.load(output_dirs["images"] / "pat_01.npy"),
        np.ones((2, 2, 2, 1), dtype=np.float32),
    )


def test_preprocess_single_patient_returns_error_on_exception(tmp_path, monkeypatch):
    """_preprocess_single_patient returns an error string when processing fails."""
    output_dirs = {
        "images": tmp_path / "images",
        "labels": tmp_path / "labels",
        "dtms": tmp_path / "dtms",
    }
    for d in output_dirs.values():
        d.mkdir()

    def _pe_raises(**kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pp, "preprocess_example", _pe_raises, raising=True)

    patient = {"id": "bad_pat", "image": "/fake/img.nii.gz", "mask": "/fake/msk.nii.gz"}

    err = pp._preprocess_single_patient(
        config={},
        patient=patient,
        image_columns=["image"],
        fg_bbox=None,
        output_dirs=output_dirs,
        compute_dtms=False,
    )

    assert err is not None
    assert "bad_pat" in err
    assert "simulated failure" in err


def test_preprocess_single_patient_skips_dtm_when_disabled(tmp_path, monkeypatch):
    """_preprocess_single_patient does not write dtm file when compute_dtms=False."""
    output_dirs = {
        "images": tmp_path / "images",
        "labels": tmp_path / "labels",
        "dtms": tmp_path / "dtms",
    }
    for d in output_dirs.values():
        d.mkdir()

    def _pe(**kwargs):
        return {
            "image": np.zeros((2, 2, 2, 1), dtype=np.float32),
            "mask": np.zeros((2, 2, 2, 1), dtype=np.uint8),
            "dtm": np.zeros((2, 2, 2, 2), dtype=np.float32),
            "fg_bbox": None,
        }

    monkeypatch.setattr(pp, "preprocess_example", _pe, raising=True)

    patient = {"id": "p1", "image": "/img.nii.gz", "mask": "/msk.nii.gz"}
    pp._preprocess_single_patient(
        config={},
        patient=patient,
        image_columns=["image"],
        fg_bbox=None,
        output_dirs=output_dirs,
        compute_dtms=False,
    )

    assert (output_dirs["images"] / "p1.npy").exists()
    assert (output_dirs["labels"] / "p1.npy").exists()
    assert not (output_dirs["dtms"] / "p1.npy").exists()


def test_preprocess_dataset_prints_error_summary_on_failures(tmp_path, monkeypatch, base_config):
    """preprocess_dataset prints 'N of M patients had errors' on failure."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results.mkdir(parents=True)
    _write_json(results / "config.json", base_config)
    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame(
            [
                {"id": "p1", "image": "/a.nii.gz", "mask": "/a_msk.nii.gz"},
                {"id": "p2", "image": "/b.nii.gz", "mask": "/b_msk.nii.gz"},
            ]
        ),
    )
    _write_csv(
        results / "fg_bboxes.csv",
        pd.DataFrame(
            [
                {"id": "p1", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1},
                {"id": "p2", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1},
            ]
        ),
    )

    monkeypatch.setattr(pp.progress_bar, "get_progress_bar", lambda *_: _PB(), raising=True)
    monkeypatch.setattr(pp.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)

    def _pe_always_fail(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pp, "preprocess_example", _pe_always_fail, raising=True)

    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg, **k: printed.append(str(msg)))

    ns = argparse.Namespace(
        results=str(results),
        numpy=str(numpy_dir),
        compute_dtms=False,
        no_preprocess=False,
        num_workers_preprocess=None,
    )
    pp.preprocess_dataset(ns)

    assert any("2 of 2" in p for p in printed)
