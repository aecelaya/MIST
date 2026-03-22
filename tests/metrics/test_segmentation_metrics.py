"""Unit tests for MIST metrics module."""
import numpy as np
import pytest

# MIST imports.
from mist.metrics import segmentation_metrics as metrics

# pylint: disable=protected-access, redefined-outer-name
@pytest.fixture
def create_synthetic_masks():
    """Fixture to create synthetic masks for testing."""
    mask_gt = np.zeros((10, 10, 10), dtype=bool)
    mask_pred = np.zeros((10, 10, 10), dtype=bool)
    mask_gt[2:5, 2:5, 2:5] = True
    mask_pred[3:6, 3:6, 3:6] = True
    spacing_mm = (1.0, 1.0, 1.0)
    return mask_gt, mask_pred, spacing_mm


def test_assert_is_numpy_array_pass():
    """Should pass when given a numpy array."""
    arr = np.array([1, 2, 3])
    metrics._assert_is_numpy_array("arr", arr) # No exception expected.


def test_assert_is_numpy_array_fail():
    """Should raise ValueError when input is not a numpy array."""
    with pytest.raises(ValueError, match="should be a numpy array"):
        metrics._assert_is_numpy_array("not_array", [1, 2, 3])  # List input.


def test_check_nd_numpy_array_correct_dims():
    """Should pass when array has correct number of dimensions."""
    arr_2d = np.zeros((3, 4))
    arr_3d = np.zeros((3, 4, 5))
    metrics._check_nd_numpy_array("arr_2d", arr_2d, 2)
    metrics._check_nd_numpy_array("arr_3d", arr_3d, 3)


def test_check_nd_numpy_array_wrong_dims():
    """Should raise ValueError when array has wrong number of dimensions."""
    arr = np.zeros((3, 4))
    with pytest.raises(ValueError, match="should be a 3D array"):
        metrics._check_nd_numpy_array("arr", arr, 3)


def test_check_2d_numpy_array_pass():
    """Should pass for a valid 2D numpy array."""
    arr = np.zeros((5, 5))
    metrics._check_2d_numpy_array("arr", arr) # No exception expected.


def test_check_2d_numpy_array_fail():
    """Should raise ValueError for a non-2D array."""
    arr = np.zeros((5, 5, 1))
    with pytest.raises(ValueError, match="should be a 2D array"):
        metrics._check_2d_numpy_array("arr", arr)


def test_crop_to_bounding_box_2d():
    """Should correctly crop and pad a 2D mask using bounding box."""
    # Create a 2D mask with a 2x2 square in the center.
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 3:5] = 1

    bbox_min = np.array([2, 3])
    bbox_max = np.array([3, 4])

    # Expected shape after cropping and adding +2 padding.
    expected_shape = bbox_max - bbox_min + 2

    crop = metrics._crop_to_bounding_box(mask, bbox_min, bbox_max)

    # Check shape is correct.
    assert crop.shape == tuple(expected_shape)

    # The main content (without padded last row/col) should match original.
    expected_cropped = np.zeros(expected_shape, dtype=np.uint8)
    expected_cropped[0, 0] = 1
    expected_cropped[0, 1] = 1
    expected_cropped[1, 0] = 1
    expected_cropped[1, 1] = 1
    assert np.array_equal(crop[0:2, 0:2], expected_cropped[0:2, 0:2])

    # Last row and column should be zero due to padding.
    assert np.all(crop[-1, :] == 0)
    assert np.all(crop[:, -1] == 0)


def test_crop_to_bounding_box_invalid_dims():
    """Should raise ValueError when input mask is not 2D or 3D."""
    # Create a dummy 4D boolean mask.
    mask = np.zeros((2, 2, 2, 2), dtype=bool)
    bbox_min = np.array([0, 0, 0, 0])
    bbox_max = np.array([1, 1, 1, 1])

    with pytest.raises(ValueError, match="Only 2D and 3D masks are supported"):
        metrics._crop_to_bounding_box(mask, bbox_min, bbox_max)


def test_assert_is_bool_numpy_array_pass():
    """Should pass when given a bool numpy array."""
    arr = np.zeros((3, 3), dtype=bool)
    metrics._assert_is_bool_numpy_array("arr", arr) # No exception expected.


def test_assert_is_bool_numpy_array_wrong_type():
    """Should raise ValueError when given a non-numpy input."""
    arr = [[True, False], [False, True]] # Not a numpy array.
    with pytest.raises(ValueError, match="should be a numpy array"):
        metrics._assert_is_bool_numpy_array("arr", arr)


def test_assert_is_bool_numpy_array_wrong_dtype():
    """Should raise ValueError when numpy array is not of type bool."""
    arr = np.zeros((3, 3), dtype=np.uint8) # Wrong dtype.
    with pytest.raises(
        ValueError, match="should be a numpy array of type bool"
    ):
        metrics._assert_is_bool_numpy_array("arr", arr)


def test_dice_coefficient(create_synthetic_masks):
    """Test that the Dice coefficient is computed correctly."""
    mask_gt, mask_pred, _ = create_synthetic_masks
    dice = metrics.compute_dice_coefficient(mask_gt, mask_pred)
    assert 0.0 < dice < 1.0


def test_dice_coefficient_empty():
    """Should return NaN when both masks are empty."""
    mask_gt = np.zeros((5, 5, 5), dtype=bool)
    mask_pred = np.zeros((5, 5, 5), dtype=bool)
    dice = metrics.compute_dice_coefficient(mask_gt, mask_pred)
    assert np.isnan(dice)


def test_compute_surface_distances_structure(create_synthetic_masks):
    """Should compute surface distances and return expected structure."""
    mask_gt, mask_pred, spacing = create_synthetic_masks
    distances = metrics.compute_surface_distances(mask_gt, mask_pred, spacing)
    keys = {
        "distances_gt_to_pred",
        "distances_pred_to_gt",
        "surfel_areas_gt",
        "surfel_areas_pred",
    }
    assert set(distances.keys()) == keys
    for key in keys:
        assert isinstance(distances[key], np.ndarray)
        assert distances[key].dtype == np.float64


def test_compute_surface_distances_2d():
    """Should compute surface distances correctly for 2D inputs."""
    # Create 2D binary masks with a square shape.
    mask_gt = np.zeros((10, 10), dtype=bool)
    mask_pred = np.zeros((10, 10), dtype=bool)
    mask_gt[3:7, 3:7] = 1  # Ground truth square.
    mask_pred[4:8, 4:8] = 1  # Slightly shifted square.

    spacing_mm = (1.0, 1.0)  # Uniform pixel spacing.

    # Run function under test.
    result = metrics.compute_surface_distances(mask_gt, mask_pred, spacing_mm)

    # Check that expected keys exist.
    expected_keys = {
        "distances_gt_to_pred",
        "distances_pred_to_gt",
        "surfel_areas_gt",
        "surfel_areas_pred",
    }
    assert set(result.keys()) == expected_keys

    # Check types and contents.
    for key in expected_keys:
        assert isinstance(result[key], np.ndarray)
        assert result[key].dtype == np.float64
        assert result[key].ndim == 1
        assert result[key].size > 0  # Should not be empty for valid inputs.


def test_surface_distance_metrics(create_synthetic_masks):
    """Test that surface distance metrics compute correctly."""
    mask_gt, mask_pred, spacing = create_synthetic_masks
    distances = metrics.compute_surface_distances(mask_gt, mask_pred, spacing)

    asd = metrics.compute_average_surface_distance(distances)
    assert isinstance(asd, float)
    assert asd > 0.0

    hd95 = metrics.compute_robust_hausdorff(distances, percent=95.0)
    assert isinstance(hd95, float)
    assert hd95 > 0.0

    overlap = metrics.compute_surface_overlap_at_tolerance(
        distances, tolerance_mm=2.0
    )
    assert isinstance(overlap, tuple)
    assert len(overlap) == 2
    assert all(0.0 <= val <= 1.0 for val in overlap)

    surface_dice = metrics.compute_surface_dice_at_tolerance(
        distances, tolerance_mm=2.0
    )
    assert isinstance(surface_dice, float)
    assert 0.0 <= surface_dice <= 1.0


def test_compute_surface_distances_invalid_dims():
    """Should raise ValueError for unsupported dimensionality (e.g., 4D)."""
    # Create a dummy 4D boolean mask.
    mask = np.zeros((2, 2, 2, 2), dtype=bool)
    spacing_mm = (1.0, 1.0, 1.0, 1.0)  # Match dimensionality

    with pytest.raises(ValueError, match="Only 2D and 3D masks are supported"):
        metrics.compute_surface_distances(mask, mask, spacing_mm) # type: ignore


def test_surface_distances_incompatible_shapes():
    """Should raise ValueError for incompatible mask shapes."""
    gt = np.zeros((5, 5, 5), dtype=bool)
    pred = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="compatible shape"):
        metrics.compute_surface_distances(gt, pred, (1.0, 1.0))


def test_crop_bounding_box_empty_raises():
    """Should raise ValueError when bounding box is empty."""
    empty = np.zeros((5, 5, 5), dtype=bool)
    with pytest.raises(ValueError, match="mask is empty"):
        metrics._compute_bounding_box(empty)


def test_compute_robust_hausdorff_empty_distances():
    """Should return inf when both distance arrays are empty."""
    surface_distances = {
        "distances_gt_to_pred": np.array([], dtype=np.float64),
        "distances_pred_to_gt": np.array([], dtype=np.float64),
        "surfel_areas_gt": np.array([], dtype=np.float64),
        "surfel_areas_pred": np.array([], dtype=np.float64),
    }
    hausdorff = metrics.compute_robust_hausdorff(
        surface_distances, percent=95.0
    )
    # Both inputs are empty, so result should be inf.
    assert np.isinf(hausdorff)


def test_surface_overlap_zero_when_disjoint():
    """Should return zero overlap when masks are disjoint."""
    mask_gt = np.zeros((5, 5, 5), dtype=bool)
    mask_pred = np.zeros((5, 5, 5), dtype=bool)
    mask_gt[1:3, 1:3, 1:3] = True
    mask_pred[4:5, 4:5, 4:5] = True
    spacing = (1.0, 1.0, 1.0)
    distances = metrics.compute_surface_distances(mask_gt, mask_pred, spacing)
    rel_gt, rel_pred = metrics.compute_surface_overlap_at_tolerance(
        distances, tolerance_mm=0.1
    )
    assert rel_gt == 0.0
    assert rel_pred == 0.0


def test_surface_dice_zero_when_disjoint():
    """Should return zero surface dice when masks are disjoint."""
    mask_gt = np.zeros((5, 5, 5), dtype=bool)
    mask_pred = np.zeros((5, 5, 5), dtype=bool)
    mask_gt[1:3, 1:3, 1:3] = True
    mask_pred[4:5, 4:5, 4:5] = True
    spacing = (1.0, 1.0, 1.0)
    distances = metrics.compute_surface_distances(mask_gt, mask_pred, spacing)
    surface_dice = metrics.compute_surface_dice_at_tolerance(
        distances, tolerance_mm=0.1
    )
    assert surface_dice == 0.0


def test_compute_surface_distances_empty_gt_borders():
    """GT all-zero: no GT surface voxels, distmap_gt falls back to all-inf."""
    mask_gt = np.zeros((8, 8, 8), dtype=bool)
    mask_pred = np.zeros((8, 8, 8), dtype=bool)
    mask_pred[2:6, 2:6, 2:6] = True
    distances = metrics.compute_surface_distances(mask_gt, mask_pred, (1.0, 1.0, 1.0))
    assert set(distances.keys()) == {
        "distances_gt_to_pred", "distances_pred_to_gt",
        "surfel_areas_gt", "surfel_areas_pred",
    }


def test_compute_surface_distances_empty_pred_borders():
    """Pred all-zero: no pred surface voxels, distmap_pred falls back to all-inf."""
    mask_gt = np.zeros((8, 8, 8), dtype=bool)
    mask_gt[2:6, 2:6, 2:6] = True
    mask_pred = np.zeros((8, 8, 8), dtype=bool)
    distances = metrics.compute_surface_distances(mask_gt, mask_pred, (1.0, 1.0, 1.0))
    assert set(distances.keys()) == {
        "distances_gt_to_pred", "distances_pred_to_gt",
        "surfel_areas_gt", "surfel_areas_pred",
    }
