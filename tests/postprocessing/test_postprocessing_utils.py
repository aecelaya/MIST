"""Tests for the MIST postprocessing utilities."""
import pytest
import numpy as np

# MIST imports.
from mist.postprocessing import postprocessing_utils as utils

# Tests for the group_labels_in_mask function.


def test_group_labels_with_valid_subset():
    """Test grouping with a valid subset of labels."""
    mask = np.array([
        [0, 1, 1],
        [2, 0, 2],
        [3, 3, 0]
    ], dtype=np.uint8)

    result = utils.group_labels_in_mask(mask, [1, 2])
    expected = np.array([
        [0, 1, 1],
        [2, 0, 2],
        [0, 0, 0]
    ], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_group_labels_with_all_labels():
    """Test grouping with all labels given as [-1]."""
    mask = np.array([
        [0, 1, 1],
        [2, 0, 2],
        [3, 3, 0]
    ], dtype=np.uint8)

    result = utils.group_labels_in_mask(mask, [-1])
    expected = np.array([
        [0, 1, 1],
        [2, 0, 2],
        [3, 3, 0]
    ], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_group_labels_with_empty_list_raises():
    """Test grouping with an empty list of labels raises ValueError."""
    mask = np.ones((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must not be empty"):
        utils.group_labels_in_mask(mask, [])


def test_group_labels_with_negative_label_raises():
    """Test grouping with a negative label raises ValueError."""
    mask = np.ones((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="strictly positive"):
        utils.group_labels_in_mask(mask, [-2])


# Tests for the remove_small_objects_binary function.
def test_remove_small_objects_binary_returns_original_if_empty():
    """Test that the function returns the original mask if empty."""
    binary_mask = np.zeros((5, 5), dtype=np.uint8)
    cleaned = utils.remove_small_objects_binary(binary_mask, threshold=5)
    np.testing.assert_array_equal(cleaned, np.zeros((5, 5), dtype=bool))


def test_remove_small_objects_binary_removes_objects_below_threshold():
    """Test that small objects are removed based on the threshold."""
    binary_mask = np.zeros((10, 10), dtype=np.uint8)
    binary_mask[1, 1] = 1  # Small component.
    binary_mask[5:9, 5:9] = 1  # Large component.

    cleaned = utils.remove_small_objects_binary(binary_mask, threshold=5)

    expected = np.zeros_like(binary_mask, dtype=bool)
    expected[5:9, 5:9] = 1
    np.testing.assert_array_equal(cleaned, expected)


def test_remove_small_objects_binary_retains_large_components():
    """Test that large components are retained."""
    binary_mask = np.zeros((10, 10), dtype=np.uint8)
    binary_mask[1, 1] = True  # Small component.
    binary_mask[5:9, 5:9] = True  # Large component.
    cleaned = utils.remove_small_objects_binary(binary_mask, threshold=10)

    expected = np.zeros_like(binary_mask, dtype=bool)
    expected[5:9, 5:9] = 1
    np.testing.assert_array_equal(cleaned, expected)


# Tests for the get_top_k_connected_components_binary function.
def create_mask_with_components():
    """Create a binary mask with three components of different sizes."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # Size 4.
    mask[4:7, 4:7] = 1  # Size 9.
    mask[8, 8] = 1      # Size 1.
    return mask


def test_top_k_connected_components_keeps_largest_k():
    """Test that the function keeps the top K largest components."""
    mask = create_mask_with_components()
    result = utils.get_top_k_connected_components_binary(mask, top_k=2)
    expected = np.zeros_like(mask, dtype=bool)
    expected[1:3, 1:3] = True
    expected[4:7, 4:7] = True
    np.testing.assert_array_equal(result, expected)


def test_top_k_connected_components_too_few_components():
    """Test that the function returns all components if fewer than K."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # One component.
    result = utils.get_top_k_connected_components_binary(mask, top_k=2)
    np.testing.assert_array_equal(result, mask.astype(bool))


def test_top_k_connected_components_empty_mask():
    """Test that the function returns an empty mask if input is empty."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = utils.get_top_k_connected_components_binary(mask, top_k=1)
    np.testing.assert_array_equal(result, mask.astype(bool))


def test_top_k_connected_components_with_morph_cleanup():
    """Test that the function applies morphological cleanup if specified."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:6, 1:6] = 1
    mask[3, 5] = 0  # Break a weak connection.
    result = utils.get_top_k_connected_components_binary(
        mask, top_k=1, morph_cleanup=True, morph_iterations=1
    )
    assert result.dtype == bool
    assert result.sum() <= mask.sum()


def test_top_k_connected_components_morph_cleanup_too_few_returns_original():
    """Fallback with morph_cleanup=True must return the pre-erosion mask."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 1  # One solid 4x4 component.

    # Requesting top_k=2 but only 1 component exists → fallback path.
    result = utils.get_top_k_connected_components_binary(
        mask, top_k=2, morph_cleanup=True, morph_iterations=1
    )

    # Result must equal the original (pre-erosion) mask, not the eroded one.
    np.testing.assert_array_equal(result, mask.astype(bool))


# Tests for replace_small_objects_binary function.
@pytest.mark.parametrize(
    "mask, original_label, replacement_label, min_size, expected_output",
    [
        # All below threshold -> replaced with replacement_label.
        (
            np.array([[False, False, False],
                      [False, True,  False],
                      [False, False, False]], dtype=bool),
            1, 99, 2,
            np.array([[0, 0, 0],
                      [0, 99, 0],
                      [0, 0, 0]], dtype=np.uint8)
        ),
        # All above threshold -> kept as original_label.
        (
            np.pad(np.ones((3, 3), dtype=bool), pad_width=1),
            2, 99, 5,
            np.pad(np.full((3, 3), 2, dtype=np.uint8), pad_width=1)
        )
    ]
)
def test_replace_small_objects_threshold_cases(
    mask, original_label, replacement_label, min_size, expected_output
):
    """Test replace_small_objects_binary with threshold cases."""
    result = utils.replace_small_objects_binary(
        binary_mask=mask,
        original_label=original_label,
        replacement_label=replacement_label,
        min_size=min_size
    )
    np.testing.assert_array_equal(result, expected_output)


def test_replace_small_objects_mixed():
    """Mix of small and large components → correct labeling per size."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True       # Area = 4 (large).
    mask[5, 5] = True           # Area = 1 (small).
    result = utils.replace_small_objects_binary(
        binary_mask=mask,
        original_label=3,
        replacement_label=77,
        min_size=3
    )
    expected = np.zeros_like(mask, dtype=np.uint8)
    expected[1:3, 1:3] = 3
    expected[5, 5] = 77
    np.testing.assert_array_equal(result, expected)


def test_replace_small_objects_empty_input():
    """Empty binary input → return unchanged."""
    mask = np.zeros((5, 5), dtype=bool)
    result = utils.replace_small_objects_binary(
        binary_mask=mask,
        original_label=5,
        replacement_label=9,
        min_size=2
    )
    np.testing.assert_array_equal(result, mask.astype(np.uint8))
