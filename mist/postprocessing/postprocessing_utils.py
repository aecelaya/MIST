"""Postprocessing utilities for MIST predictions."""
from typing import List, Any, Dict, TypedDict, cast
import numpy as np
import numpy.typing as npt
import skimage
from scipy import ndimage


class StrategyStep(TypedDict):
    """TypedDict for a single step in the postprocessing strategy."""
    transform: str
    apply_to_labels: list[int]
    per_label: bool
    kwargs: Dict[str, Any]


def group_labels_in_mask(
    mask_npy: npt.NDArray[Any],
    labels_list: List[int]
) -> npt.NDArray[Any]:
    """Extract a group of labels from a multi-label mask.

    Args:
        mask_npy: Input multi-label mask as a numpy array.
        labels_list: List of labels to group. If set to [-1],
            selects all labels > 0.

    Returns:
        grouped_labels: A mask with only the specified labels.
    """
    if not labels_list:
        raise ValueError("The labels_list argument must not be empty.")

    use_all_labels = len(labels_list) == 1 and labels_list[0] == -1

    if not use_all_labels:
        if any(label <= 0 for label in labels_list):
            raise ValueError(
                "All labels in labels_list must be strictly positive."
            )
        mask = np.isin(mask_npy, labels_list)
    else:
        mask = mask_npy > 0

    grouped_labels = mask_npy * mask
    return grouped_labels.astype(np.uint8)


def remove_small_objects_binary(
    binary_mask: npt.NDArray[Any],
    threshold: int,
) -> npt.NDArray[np.bool_]:
    """Remove small objects from a binary mask based on a size threshold.

    Args:
        binary_mask: Input binary mask (0/1 or boolean).
        threshold: Minimum size to retain.

    Returns:
        Cleaned binary mask as a boolean array.
    """
    # If the binary mask is completely empty, return it unchanged as a boolean
    # array.
    if binary_mask.max() == 0:
        return binary_mask.astype(bool)

    # Explicitly cast to ndarray because skimage.measure.label has multiple
    # return types, and we are using it with return_num=False, which always
    # returns an ndarray.
    labeled = cast(
        np.ndarray, skimage.measure.label(binary_mask, return_num=False)
    )

    # If there is only one non-zero label, pass a boolean array to
    # remove_small_objects. This helps avoid a warning from remove_small_objects
    # about not passing in a boolean array.
    if labeled.max() == 1:
        labeled = labeled.astype(bool)

    cleaned = skimage.morphology.remove_small_objects(
        labeled, min_size=threshold
    )
    return cleaned.astype(bool)


def get_top_k_connected_components_binary(
    binary_mask: npt.NDArray[Any],
    top_k: int,
    morph_cleanup: bool=False,
    morph_iterations: int=1,
) -> npt.NDArray[np.bool_]:
    """Extract the top K largest connected components from a binary mask.

    Args:
        binary_mask: Input binary mask (0/1 or bool).
        top_k: Number of components to keep.
        morph_cleanup: Whether to apply erosion/dilation cleanup.
        morph_iterations: Number of iterations for cleanup.

    Returns:
        A boolean mask with only the top K components retained.
    """
    # If the binary mask is completely empty, return it unchanged as a boolean
    # array.
    if binary_mask.max() == 0:
        return binary_mask.astype(bool)

    # Save original before any morphological operations so we can fall back to
    # it when there are too few components to select from.
    original_mask = binary_mask

    # Optionally apply binary erosion to clean up small connections before
    # labeling.
    if morph_cleanup:
        binary_mask = ndimage.binary_erosion(
            binary_mask, iterations=morph_iterations
        )

    # Label connected components in the mask.
    # Each connected component gets a unique integer label.
    # Explicitly cast to ndarray because skimage.measure.label has multiple
    # return types, and we are using it with return_num=False, which always
    # returns an ndarray.
    labeled = cast(
        np.ndarray, skimage.measure.label(binary_mask, return_num=False)
    )

    # Compute the size of each connected component.
    # We exclude the background (label 0) using [1:].
    component_sizes = np.bincount(labeled.flat)[1:]

    # If no components exist or there are fewer than top_k components, return
    # the original (pre-erosion) mask.
    if labeled.max() == 0 or len(component_sizes) < top_k:
        return original_mask.astype(bool)

    # Get the labels of the top K largest components based on size.
    # We add 1 to the indices because labels start at 1.
    top_k_labels = np.argsort(component_sizes)[-top_k:][::-1] + 1

    # Create a binary mask containing only the top K component labels.
    top_k_mask = np.isin(labeled, top_k_labels)

    # Optionally apply binary dilation to restore component size after erosion.
    if morph_cleanup:
        top_k_mask = ndimage.binary_dilation(
            top_k_mask, iterations=morph_iterations
        )

    # Return the final binary mask containing only the top K components.
    return top_k_mask.astype(bool)


def replace_small_objects_binary(
    binary_mask: npt.NDArray[np.bool_],
    original_label: int,
    replacement_label: int,
    min_size: int,
) -> npt.NDArray[np.uint8]:
    """Replace small connected components in a binary mask with a label.

    Replace connected components smaller than a threshold in a binary mask
    with a specified replacement label.

    Args:
        binary_mask: Binary mask (bool or 0/1) of the label to process.
        original_label: The label that the binary mask represents.
        replacement_label: Label to assign to small components.
        min_size: Minimum number of pixels required to retain a component.

    Returns:
        A labeled mask where small objects are relabeled.
    """
    # Check if the binary mask is empty. If so, return it unchanged as a uint8
    # array.
    if binary_mask.max() == 0:
        return binary_mask.astype(np.uint8)

    # Convert binary mask to labeled connected components.
    # Explicitly cast to ndarray because skimage.measure.label has multiple
    # return types, and we are using it with return_num=False, which always
    # returns an ndarray.
    labeled = cast(
        np.ndarray, skimage.measure.label(binary_mask, return_num=False)
    )

    # regionprops extracts stats for each component, including pixel
    # coordinates.
    regions = skimage.measure.regionprops(labeled)

    # Initialize the output mask where retained components remain as
    # original_label.
    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for region in regions:
        coords = tuple(region.coords.T) # Convert (N, D) to tuple for indexing.
        if region.area < min_size:
            # Assign replacement label to small components.
            output_mask[coords] = replacement_label
        else:
            # Keep large enough components as original label.
            output_mask[coords] = original_label

    return output_mask.astype(np.uint8)
