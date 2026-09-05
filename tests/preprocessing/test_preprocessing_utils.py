"""Tests for MIST preprocessing utilities."""

import numpy as np
import pytest
import SimpleITK as sitk

# MIST imports.
from mist.preprocessing import preprocessing_utils as pu
from mist.utils import sitk_io


def _make_sitk_image_from_xyz(
    arr_xyz: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction_mat: np.ndarray | None = None,
):
    """Create a SimpleITK image from an XYZ-ordered array and set metadata.

    SimpleITK expects array order (z, y, x), so we transpose before creating.
    """
    img = sitk.GetImageFromArray(arr_xyz.T)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    direction = (
        tuple(direction_mat.flatten()) if direction_mat is not None else tuple(np.eye(3).flatten())
    )
    img.SetDirection(direction)
    return img


def test_get_fg_mask_bbox_detects_cube(monkeypatch):
    """Foreground bbox is correctly found for a bright cube in zeros."""
    # Neutralize percentile clipping to simplify behavior.
    monkeypatch.setattr(pu.pc, "FOREGROUND_BBOX_PERCENTILE_LOW", 0)
    monkeypatch.setattr(pu.pc, "FOREGROUND_BBOX_PERCENTILE_HIGH", 100)

    vol = np.zeros((16, 17, 18), dtype=np.float32)
    xs, xe = 3, 9
    ys, ye = 4, 12
    zs, ze = 5, 15
    vol[xs : xe + 1, ys : ye + 1, zs : ze + 1] = 100.0  # Bright cube.

    img = _make_sitk_image_from_xyz(vol)
    bbox = pu.get_fg_mask_bbox(img)

    assert bbox["x_start"] == xs and bbox["x_end"] == ze - (ze - xe)
    assert bbox["y_start"] == ys and bbox["y_end"] == ye
    assert bbox["z_start"] == zs and bbox["z_end"] == ze
    assert bbox["x_og_size"] == 16 and bbox["y_og_size"] == 17 and bbox["z_og_size"] == 18


def test_get_fg_mask_bbox_returns_full_when_empty(monkeypatch):
    """If no foreground exists, bbox should cover full image extent."""
    monkeypatch.setattr(pu.pc, "FOREGROUND_BBOX_PERCENTILE_LOW", 0)
    monkeypatch.setattr(pu.pc, "FOREGROUND_BBOX_PERCENTILE_HIGH", 100)

    vol = np.zeros((8, 9, 10), dtype=np.float32)
    img = _make_sitk_image_from_xyz(vol)
    bbox = pu.get_fg_mask_bbox(img)

    assert bbox == {
        "x_start": 0,
        "x_end": 7,
        "y_start": 0,
        "y_end": 8,
        "z_start": 0,
        "z_end": 9,
        "x_og_size": 8,
        "y_og_size": 9,
        "z_og_size": 10,
    }


def test_get_fg_mask_bbox_oblique_direction_does_not_affect_bbox(monkeypatch):
    """Bbox is computed from voxel indices, independent of direction cosines."""
    monkeypatch.setattr(pu.pc, "FOREGROUND_BBOX_PERCENTILE_LOW", 0)
    monkeypatch.setattr(pu.pc, "FOREGROUND_BBOX_PERCENTILE_HIGH", 100)

    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2:5, 3:6, 1:4] = 100.0

    theta = 0.3
    direction = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    img = _make_sitk_image_from_xyz(vol, direction_mat=direction)
    bbox = pu.get_fg_mask_bbox(img)

    assert (bbox["x_start"], bbox["x_end"]) == (2, 4)
    assert (bbox["y_start"], bbox["y_end"]) == (3, 5)
    assert (bbox["z_start"], bbox["z_end"]) == (1, 3)


def test_aniso_intermediate_resample_changes_only_low_axis():
    """NN resampling along the low-res axis adjusts only that axis."""
    arr = np.zeros((6, 7, 3), dtype=np.float32)
    img = _make_sitk_image_from_xyz(
        arr,
        spacing=(1.0, 1.0, 5.0),
        origin=(0.0, 0.0, 0.0),
        direction_mat=np.eye(3),
    )

    low_axis = 2
    target_spacing = (1.0, 1.0, 2.5)
    new_size = (6, 7, 6)  # Double along low-res axis only.

    out = pu.aniso_intermediate_resample(img, new_size, target_spacing, low_axis)

    assert out.GetSize() == new_size
    assert out.GetSpacing()[low_axis] == pytest.approx(target_spacing[low_axis])
    assert out.GetSpacing()[0] == 1.0 and out.GetSpacing()[1] == 1.0


def test_check_anisotropic_true_and_false():
    """Detect anisotropy if spacing ratio > 3 and return correct axis."""
    # True case: ratio > 3.
    img1 = _make_sitk_image_from_xyz(np.zeros((4, 4, 2), np.float32), spacing=(1.0, 1.0, 4.5))
    res1 = pu.check_anisotropic(img1)
    assert res1["is_anisotropic"] is True
    assert res1["low_resolution_axis"] == 2

    # False case: ratio == 3 -> not anisotropic by strict '>' check.
    img2 = _make_sitk_image_from_xyz(np.zeros((4, 4, 2), np.float32), spacing=(1.0, 1.0, 3.0))
    res2 = pu.check_anisotropic(img2)
    assert res2["is_anisotropic"] is False
    assert res2["low_resolution_axis"] is None


def test_make_onehot_creates_binary_masks_with_metadata():
    """One-hot conversion creates binary SITK images with preserved metadata."""
    rng = np.random.default_rng(0)
    shape = (6, 5, 4)
    labels = np.array([0, 1, 2], dtype=np.int32)
    vol = rng.integers(low=0, high=3, size=shape).astype(np.int32)

    spacing = (1.1, 2.2, 3.3)
    origin = (10.0, -2.0, 7.0)
    direction = np.eye(3)
    mask_sitk = _make_sitk_image_from_xyz(
        vol.astype(np.float32), spacing=spacing, origin=origin, direction_mat=direction
    )

    out_masks = pu.make_onehot(mask_sitk, labels_list=[0, 1, 2])
    assert len(out_masks) == 3

    # Each is binary with matching metadata and correct voxel counts.
    for idx, lbl in enumerate(labels):
        m = out_masks[idx]
        assert m.GetSpacing() == spacing
        assert m.GetOrigin() == origin
        assert np.allclose(m.GetDirection(), direction.flatten())

        m_arr = sitk_io.array_from_image(m)  # Back to XYZ order.
        assert set(np.unique(m_arr)).issubset({0.0, 1.0})
        assert m_arr.sum() == np.count_nonzero(vol == lbl)


def test_sitk_get_min_max_and_sum():
    """Statistics filters return correct min, max, and sum."""
    arr = np.array(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        dtype=np.float32,
    )  # Shape (2, 2, 2).
    img = _make_sitk_image_from_xyz(arr)

    mn, mx = pu.sitk_get_min_max(img)
    s = pu.sitk_get_sum(img)
    assert mn == 0.0 and mx == 7.0
    assert s == float(arr.sum())


def test_crop_to_fg_returns_expected_shape():
    """Cropping with bbox yields a correctly shaped SimpleITK image."""
    vol = np.zeros((12, 10, 8), dtype=np.float32)
    # Define a block to crop to.
    xs, xe = 2, 7
    ys, ye = 3, 9
    zs, ze = 1, 6
    vol[xs : xe + 1, ys : ye + 1, zs : ze + 1] = 5.0

    img = _make_sitk_image_from_xyz(vol)

    bbox = {
        "x_start": xs,
        "x_end": xe,
        "y_start": ys,
        "y_end": ye,
        "z_start": zs,
        "z_end": ze,
    }
    cropped = pu.crop_to_fg(img, bbox)
    cropped_arr = sitk_io.array_from_image(cropped)

    exp_shape = (xe - xs + 1, ye - ys + 1, ze - zs + 1)
    assert cropped_arr.shape == exp_shape
    assert np.allclose(cropped_arr, vol[xs : xe + 1, ys : ye + 1, zs : ze + 1])


def test_crop_to_fg_preserves_spacing_origin_direction():
    """Cropping keeps the source image's spacing and direction, shifts origin."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    spacing = (1.5, 2.0, 0.5)
    direction = np.eye(3)
    img = _make_sitk_image_from_xyz(vol, spacing=spacing, direction_mat=direction)

    bbox = {"x_start": 1, "x_end": 4, "y_start": 2, "y_end": 5, "z_start": 0, "z_end": 3}
    cropped = pu.crop_to_fg(img, bbox)

    assert cropped.GetSpacing() == pytest.approx(spacing)
    assert np.allclose(cropped.GetDirection(), direction.flatten())
