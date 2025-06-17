# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the transform registry in MIST postprocessing module."""
import pytest
import numpy as np

# MIST imports.
from mist.postprocessing import transform_registry as registry


# Test that the transform registry contains all expected transforms.
def test_transform_registry_contains_all_expected():
    """Ensure all expected transforms are registered."""
    assert "remove_small_objects" in registry.POSTPROCESSING_TRANSFORMS
    assert (
        "get_top_k_connected_components" in registry.POSTPROCESSING_TRANSFORMS
    )
    assert "fill_holes_with_label" in registry.POSTPROCESSING_TRANSFORMS
    assert (
        "replace_small_objects_with_label" in registry.POSTPROCESSING_TRANSFORMS
    )

def test_get_transform_returns_correct_function():
    """Check get_transform returns correct functions."""
    assert (
        registry.get_transform("remove_small_objects") is
        registry.remove_small_objects
    )
    assert (
        registry.get_transform("get_top_k_connected_components") is
        registry.get_top_k_connected_components
    )
    assert (
        registry.get_transform("fill_holes_with_label")
        is registry.fill_holes_with_label
    )
    assert (
        registry.get_transform("replace_small_objects_with_label") is
        registry.replace_small_objects_with_label
    )

def test_get_transform_raises_on_invalid_name():
    """Verify error raised when requesting unregistered transform."""
    with pytest.raises(
        ValueError, match="Transform 'non_existent' is not registered."
    ):
        registry.get_transform("non_existent")


# Tests for the remove_small_objects transform.
def test_remove_small_objects_returns_unchanged_if_empty():
    """Should return original mask if all entries are zero."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = registry.remove_small_objects(
        mask, labels_list=[1], apply_sequentially=False
    )
    np.testing.assert_array_equal(result, mask)


@pytest.mark.parametrize(
    "apply_sequentially",
    [False, True],
    ids=["grouped", "sequential"],
)
def test_remove_small_objects_removes_small_components(apply_sequentially):
    """Should remove small components from specified labels."""
    mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 1],
            [2, 2, 2, 0, 0, 0, 0, 1],
            [2, 2, 2, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 3, 3]
        ]
    )

    result = registry.remove_small_objects(
        mask,
        labels_list=[1, 2, 3],
        apply_sequentially=apply_sequentially,
        small_object_threshold=5,
    )

    expected = np.zeros_like(mask)
    expected[1:3, 0:3] = 2
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "threshold,label_size",
    [(5, 16), (10, 16)],
    ids=["small_thresh", "larger_thresh"]
)
def test_remove_small_objects_preserves_large_regions(threshold, label_size):
    """Should not remove large components if above threshold."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:2 + int(label_size ** 0.5), 2:2 + int(label_size ** 0.5)] = 4
    mask[1, 1] = 1  # Small component.

    result = registry.remove_small_objects(
        mask, labels_list=[4],
        apply_sequentially=True,
        small_object_threshold=threshold,
    )
    np.testing.assert_array_equal(result, mask)


def test_remove_small_objects_uses_default_threshold_without_kwargs():
    """Should use default threshold when none is specified."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 3  # Size = 4.

    result = registry.remove_small_objects(mask, labels_list=[3])
    expected = np.zeros_like(mask)
    np.testing.assert_array_equal(result, expected)


# Tests for the get_top_k_connected_components transform.
@pytest.mark.parametrize("sequential", [True, False])
def test_top_k_connected_components_keeps_largest(sequential):
    """Ensure only the largest connected components are retained."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # Label 1, small.
    mask[4:8, 4:8] = 1  # Label 1, large.

    result = registry.get_top_k_connected_components(
        mask=mask,
        labels_list=[1],
        apply_sequentially=sequential,
        top_k_connected_components=1,
        apply_morphological_cleaning=False,
        morphological_cleaning_iterations=1
    )

    assert result.dtype == np.uint8
    assert np.all(result[1:3, 1:3] == 0)  # Small component removed.
    assert np.all(result[4:8, 4:8] == 1)  # Large retained.


def test_top_k_connected_components_empty_input():
    """If the input is empty, return unchanged."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    result = registry.get_top_k_connected_components(
        mask=mask,
        labels_list=[1],
        apply_sequentially=False,
        top_k_connected_components=1
    )
    np.testing.assert_array_equal(result, mask)


@pytest.mark.parametrize("sequential", [True, False])
def test_top_k_connected_components_with_morph_cleanup(sequential):
    """Test with morph cleanup enabled — should retain connected structure."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:6, 1:6] = 2
    mask[3, 5] = 0  # Break small connection.

    result = registry.get_top_k_connected_components(
        mask=mask,
        labels_list=[2],
        apply_sequentially=sequential,
        top_k_connected_components=1,
        apply_morphological_cleaning=True,
        morphological_cleaning_iterations=1,
    )

    assert result.dtype == np.uint8
    assert result.sum() <= mask.sum()
    assert np.any(result == 2)


def test_top_k_connected_components_multiple_labels_grouped():
    """Test that grouping works across multiple labels."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # Small label 1.
    mask[4:7, 4:7] = 2  # Sarge label 2.

    result = registry.get_top_k_connected_components(
        mask=mask,
        labels_list=[1, 2],
        apply_sequentially=False,
        top_k_connected_components=1,
    )

    assert result[1:3, 1:3].sum() == 0  # Smaller removed.
    assert np.all(result[4:7, 4:7] == 2)  # Larger retained.


def test_top_k_connected_components_multiple_labels_sequential():
    """Test that top-k applies independently per label in sequential mode."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # Label 1.
    mask[4:7, 4:7] = 2  # Label 2.

    result = registry.get_top_k_connected_components(
        mask=mask,
        labels_list=[1, 2],
        apply_sequentially=True,
        top_k_connected_components=1,
    )

    assert np.all(result[1:3, 1:3] == 1)
    assert np.all(result[4:7, 4:7] == 2)


# Tests for the fill_holes_with_label transform.
def create_mask_with_hole(label=1):
    """Create a mask with a hole in the center."""
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:5, 1:5] = label
    mask[3, 3] = 0  # Introduce a hole.
    return mask


@pytest.mark.parametrize(
    "apply_sequentially, expected_value",
    [
        (False, 9),  # Grouped mode.
        (True, 9),   # Sequential mode.
    ]
)
def test_fill_holes_with_label_fills_expected_voxels(
    apply_sequentially, expected_value
):
    """Test function fills interior holes with the specified label."""
    mask = create_mask_with_hole(label=1)
    result = registry.fill_holes_with_label(
        mask=mask,
        labels_list=[1],
        apply_sequentially=apply_sequentially,
        fill_holes_label=expected_value,
    )
    assert result[3, 3] == expected_value
    assert result.dtype == np.uint8


def test_fill_holes_with_label_returns_original_if_empty():
    """Test that empty masks are returned unchanged."""
    empty_mask = np.zeros((5, 5), dtype=np.uint8)
    result = registry.fill_holes_with_label(
        mask=empty_mask,
        labels_list=[1],
        apply_sequentially=True,
        fill_holes_label=5,
    )
    np.testing.assert_array_equal(result, empty_mask.astype("uint8"))


# Tests for the replace_small_objects_with_label transform.
def test_replace_small_objects_one_large_one_small_component():
    """Test that one large and one small component are handled correctly."""
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1, 1] = 1      # Small component.
    mask[4:6, 4:6] = 2  # Large component.

    result = registry.replace_small_objects_with_label(
        mask=mask,
        labels_list=[1, 2],
        small_object_threshold=3,
        replacement_label=7,
    )

    expected = np.zeros_like(mask)
    expected[4:6, 4:6] = 2
    expected[1, 1] = 7  # Small component replaced.
    np.testing.assert_array_equal(result, expected)


def test_replace_small_objects_with_label_empty_input():
    """Test that an empty mask is returned unchanged."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    result = registry.replace_small_objects_with_label(
        mask=mask,
        labels_list=[1],
        small_object_threshold=3,
        replacement_label=7,
    )
    np.testing.assert_array_equal(result, mask)
