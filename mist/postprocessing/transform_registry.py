"""Registry and implementations of postprocessing transforms for MIST."""
from typing import Callable, Dict, Any, List
import numpy as np
import numpy.typing as npt
from scipy import ndimage

# MIST imports.
from mist.postprocessing import postprocessing_utils as utils
from mist.postprocessing.postprocessing_constants import (
    PostprocessingConstants as pc
)

# Registry dictionaries.
POSTPROCESSING_TRANSFORMS: Dict[str, Callable[..., npt.NDArray[Any]]] = {}
TRANSFORM_METADATA: Dict[str, Dict[str, Any]] = {}


def register_transform(name: str, metadata: Dict[str, Any]) -> Callable:
    """Decorator to register a postprocessing transform by name.

    Args:
        name: The name under which the transform is registered.
        metadata: A dict describing the transform for agent consumption.
            Expected keys:
                description (str): What the transform does.
                per_label (str): "both" if both True and False are supported,
                    or "per_label_only" if per_label must be True.
                kwargs (dict): Maps each kwarg name to a dict with keys:
                    type (str), description (str), default (Any), and
                    optionally min/max (numeric bounds).
    """
    def decorator(func: Callable[..., npt.NDArray[Any]]) -> Callable:
        POSTPROCESSING_TRANSFORMS[name] = func
        TRANSFORM_METADATA[name] = {"name": name, **metadata}
        return func
    return decorator


def get_transform(name: str) -> Callable:
    """Retrieve a postprocessing transform by name."""
    if name not in POSTPROCESSING_TRANSFORMS:
        raise ValueError(f"Transform '{name}' is not registered.")
    return POSTPROCESSING_TRANSFORMS[name]


def describe_transforms() -> List[Dict[str, Any]]:
    """Return structured metadata for all registered transforms.

    Intended for use in agent prompts. Each entry describes a transform's
    purpose, supported per_label modes, and configurable kwargs with types,
    defaults, and valid ranges.

    Returns:
        A list of metadata dicts, one per registered transform, in
        registration order.
    """
    return list(TRANSFORM_METADATA.values())


@register_transform(
    "remove_small_objects",
    metadata={
        "description": (
            "Removes connected components smaller than a voxel-count threshold "
            "from the specified labels. Pixels belonging to removed components "
            "are set to 0."
        ),
        "per_label": "both",
        "kwargs": {
            "small_object_threshold": {
                "type": "int",
                "description": "Minimum component size in voxels to retain.",
                "default": pc.SMALL_OBJECT_THRESHOLD,
                "min": 1,
            },
        },
    },
)
def remove_small_objects(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    per_label: bool = False,
    **kwargs
) -> npt.NDArray[Any]:
    """Multi-label wrapper around the binary remove_small_objects transform.

    Args:
        mask: Multi-label input mask.
        labels_list: Labels to clean.
        per_label: Apply transform per-label (True) or grouped (False).
        **kwargs: Requires 'small_object_threshold'.

    Returns:
        Cleaned multi-label mask.
    """
    if mask.max() == 0:
        return mask.astype(np.uint8)

    threshold = kwargs.get("small_object_threshold", pc.SMALL_OBJECT_THRESHOLD)

    cleaned_mask = mask.copy()

    if per_label:
        for label in labels_list:
            binary = mask == label
            cleaned = utils.remove_small_objects_binary(binary, threshold)
            cleaned_mask[binary & ~cleaned] = 0
    else:
        grouped = utils.group_labels_in_mask(mask, labels_list)
        binary = grouped > 0
        cleaned = utils.remove_small_objects_binary(binary, threshold)
        cleaned_mask[binary & ~cleaned] = 0

    return cleaned_mask.astype(np.uint8)


@register_transform(
    "get_top_k_connected_components",
    metadata={
        "description": (
            "Retains only the K largest connected components for the specified "
            "labels. All other components are zeroed out. Supports optional "
            "morphological erosion before component selection and dilation "
            "afterward to separate weakly connected structures."
        ),
        "per_label": "both",
        "kwargs": {
            "top_k_connected_components": {
                "type": "int",
                "description": "Number of largest components to keep.",
                "default": pc.TOP_K_CONNECTED_COMPONENTS,
                "min": 1,
            },
            "apply_morphological_cleaning": {
                "type": "bool",
                "description": (
                    "If True, applies binary erosion before labeling to "
                    "disconnect weakly joined components, then dilates the "
                    "retained components to restore their original size."
                ),
                "default": pc.APPLY_MORPHOLOGICAL_CLEANING,
            },
            "morphological_cleaning_iterations": {
                "type": "int",
                "description": (
                    "Number of erosion/dilation iterations. Only used when "
                    "apply_morphological_cleaning is True."
                ),
                "default": pc.MORPHOLOGICAL_CLEANING_ITERATIONS,
                "min": 1,
            },
        },
    },
)
def get_top_k_connected_components(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    per_label: bool = False,
    **kwargs
) -> npt.NDArray[Any]:
    """Keeps only top K connected components for specified labels in a mask.

    Args:
        mask: Input multi-label mask.
        labels_list: Labels to apply the transform to.
        per_label: Apply transform per-label (True) or grouped (False).
        **kwargs:
            top_k_connected_components: Number of components to keep.
            apply_morphological_cleaning: Whether to use erosion/dilation.
            morphological_cleaning_iterations: Number of cleanup iterations.

    Returns:
        Cleaned multi-label mask. With only top K components for each label
        or grouped labels.
    """
    # If the input mask is empty, return it as-is.
    if mask.max() == 0:
        return mask.astype(np.uint8)

    # Retrieve transformation parameters from kwargs or pc.
    top_k = kwargs.get(
        "top_k_connected_components", pc.TOP_K_CONNECTED_COMPONENTS
    )
    morph = kwargs.get(
        "apply_morphological_cleaning", pc.APPLY_MORPHOLOGICAL_CLEANING
    )
    morph_iters = kwargs.get(
        "morphological_cleaning_iterations",
        pc.MORPHOLOGICAL_CLEANING_ITERATIONS
    )

    # Make a copy of the original mask to apply modifications.
    cleaned_mask = mask.copy()

    if per_label:
        # Apply the transformation separately to each label in labels_list.
        for label in labels_list:
            # Create a binary mask for the current label.
            binary = mask == label

            # Apply the binary connected component filtering transform.
            cleaned = utils.get_top_k_connected_components_binary(
                binary,
                top_k,
                morph_cleanup=morph,
                morph_iterations=morph_iters
            )

            # Zero out any pixels in the original label mask that were not
            # retained.
            cleaned_mask[binary & ~cleaned] = 0
    else:
        # Group all specified labels together into one binary mask.
        grouped = utils.group_labels_in_mask(mask, labels_list)
        binary = grouped > 0

        # Apply the connected component transform to the grouped mask.
        cleaned = utils.get_top_k_connected_components_binary(
            binary,
            top_k,
            morph_cleanup=morph,
            morph_iterations=morph_iters
        )

        # Zero out any grouped label pixels that were not retained.
        cleaned_mask[binary & ~cleaned] = 0

    # Return the cleaned multi-label mask.
    return cleaned_mask.astype(np.uint8)


@register_transform(
    "fill_holes_with_label",
    metadata={
        "description": (
            "Fills interior holes in the binary mask of each specified label "
            "using a designated fill label. A hole is a background region "
            "completely enclosed by a foreground label."
        ),
        "per_label": "both",
        "kwargs": {
            "fill_holes_label": {
                "type": "int",
                "description": (
                    "Label value to assign to filled holes. Use 0 to fill with "
                    "background, or another label value to assign the hole to "
                    "an adjacent structure."
                ),
                "default": pc.FILL_HOLES_LABEL,
                "min": 0,
            },
        },
    },
)
def fill_holes_with_label(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    per_label: bool = False,
    **kwargs
) -> npt.NDArray[Any]:
    """
    Fill holes in a multi-label mask using a specified label.

    Args:
        mask: A multi-label mask as a numpy array.
        labels_list: List of labels to apply the transformation to.
        per_label: Apply transform per-label (True) or grouped (False).
        **kwargs:
            fill_holes_label: Label to use for filling the holes.

    Returns:
        Updated mask with holes filled using the specified label.
    """
    # If the input mask is empty, return it unchanged.
    if mask.max() == 0:
        return mask.astype(np.uint8)

    # Get the label used to fill holes. Defaults to constant.
    fill_label = kwargs.get("fill_holes_label", pc.FILL_HOLES_LABEL)

    # Make a copy of the original mask for modification.
    filled_mask = mask.copy()

    if per_label:
        # Process each label independently.
        for label in labels_list:
            # Create binary mask for this label.
            binary = mask == label

            # Identify holes inside this binary object.
            holes = ndimage.binary_fill_holes(binary) & ~binary

            # Add holes to the filled mask using the fill label.
            filled_mask[holes] = fill_label
    else:
        # Group all labels into a single binary mask.
        binary = utils.group_labels_in_mask(mask, labels_list) > 0
        binary = binary.astype(bool)

        # Identify holes in the grouped binary mask.
        holes = ndimage.binary_fill_holes(binary) & ~binary

        # Add holes to the filled mask using the fill label.
        filled_mask[holes] = fill_label

    # Return the modified mask.
    return filled_mask.astype(np.uint8)


@register_transform(
    "replace_small_objects_with_label",
    metadata={
        "description": (
            "Replaces connected components smaller than a voxel-count threshold "
            "with a specified replacement label instead of zeroing them out. "
            "Useful for reassigning small spurious components to an adjacent "
            "structure rather than discarding them entirely."
        ),
        "per_label": "per_label_only",
        "kwargs": {
            "small_object_threshold": {
                "type": "int",
                "description": "Maximum component size in voxels to replace.",
                "default": pc.SMALL_OBJECT_THRESHOLD,
                "min": 1,
            },
            "replacement_label": {
                "type": "int",
                "description": "Label value to assign to small components.",
                "default": pc.REPLACE_SMALL_OBJECTS_LABEL,
                "min": 0,
            },
        },
    },
)
def replace_small_objects_with_label(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    per_label: bool = True,
    **kwargs
) -> npt.NDArray[Any]:
    """Replace small objects belonging to certain labels with replacement label.

    This transform always operates per-label because replacement requires
    knowledge of each component's original label value. Grouped (non-sequential)
    application is not supported.

    Args:
        mask: A multi-label mask.
        labels_list: Labels to apply the transformation to.
        per_label: Must be True. Grouped mode is not supported for this
            transform because each component must retain its original label
            value.
        **kwargs:
            small_object_threshold: Threshold below which objects are replaced.
            replacement_label: Label to use for replacement.

    Returns:
        Updated mask with small components replaced.

    Raises:
        ValueError: If apply_sequentially is False.
    """
    if not per_label:
        raise ValueError(
            "replace_small_objects_with_label always operates per-label. "
            "Set per_label=True in the strategy."
        )

    # If the input mask is empty, return it unchanged.
    if mask.max() == 0:
        return mask.astype(np.uint8)

    # Get size threshold and replacement label from kwargs or use defaults.
    min_size = kwargs.get(
        "small_object_threshold", pc.SMALL_OBJECT_THRESHOLD
    )
    replacement = kwargs.get(
        "replacement_label", pc.REPLACE_SMALL_OBJECTS_LABEL
    )

    # Create a copy of the input mask to apply changes.
    updated_mask = mask.copy()

    # Apply the transformation to each label independently.
    # This is required to preserve the original label values for large
    # components while replacing only small components with the specified
    # replacement label.
    for label in labels_list:
        # Create a binary mask where the current label is present.
        binary = mask == label

        # Clear this label from the updated mask before replacement.
        updated_mask[binary] = 0

        # Replace small components with the replacement label.
        updated_mask += utils.replace_small_objects_binary(
            binary, label, replacement, min_size
        )

    # Return the updated mask as an unsigned 8-bit integer array.
    return updated_mask.astype(np.uint8)
