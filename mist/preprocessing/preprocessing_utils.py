"""Preprocessing utilities for MIST."""
import numpy as np
import skimage
import ants
import SimpleITK as sitk

# MIST imports.
from mist.preprocessing.preprocessing_constants import (
    PreprocessingConstants as pc
)


def ants_to_sitk(img_ants: ants.core.ants_image.ANTsImage) -> sitk.Image:
    """Convert ANTs image to SimpleITK image.

    Args:
        img_ants: ANTs image object.

    Returns:
        img_sitk: SimpleITK image object.
    """
    # Get spacing, origin, and direction from ANTs image.
    spacing = img_ants.spacing
    origin = img_ants.origin
    direction = tuple(img_ants.direction.flatten())

    # Convert ANTs image to numpy array and create SimpleITK image.
    img_sitk = sitk.GetImageFromArray(img_ants.numpy().T)

    # Set spacing, origin, and direction for SimpleITK image.
    img_sitk.SetSpacing(spacing)
    img_sitk.SetOrigin(origin)
    img_sitk.SetDirection(direction)
    return img_sitk


def sitk_to_ants(img_sitk: sitk.Image) -> ants.core.ants_image.ANTsImage:
    """Convert SimpleITK image to ANTs image.

    Args:
        img_sitk: SimpleITK image object.

    Returns:
        img_ants: ANTs image object.
    """
    # Get spacing, origin, and direction from SimpleITK image.
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction_sitk = img_sitk.GetDirection()
    dim = int(np.sqrt(len(direction_sitk)))
    direction = np.reshape(np.array(direction_sitk), (dim, dim))

    # Convert SimpleITK image to numpy array and create ANTs image.
    img_ants = ants.from_numpy(sitk.GetArrayFromImage(img_sitk).T)

    # Set spacing, origin, and direction for ANTs image.
    img_ants.set_spacing(spacing)
    img_ants.set_origin(origin)
    img_ants.set_direction(direction)
    return img_ants


def get_fg_mask_bbox(
    img_ants: ants.core.ants_image.ANTsImage,
) -> dict[str, int]:
    """Get the bounding box of the foreground mask.

    This function computes the bounding box of the foreground in a 3D image. It
    uses an Otsu threshold method to create a binary mask and then finds the
    coordinates of the non-zero elements in the mask to determine the bounding
    box.

    Args:
        img_ants: ANTs image object.

    Returns:
        fg_bbox: Dictionary containing the bounding box coordinates and original
        image size.
    """
    # Convert ANTs image to numpy array.
    image_npy = img_ants.numpy()

    # Clip image to remove outliers and improve foreground detection.
    lower, upper = np.percentile(
        image_npy,
        [pc.FOREGROUND_BBOX_PERCENTILE_LOW, pc.FOREGROUND_BBOX_PERCENTILE_HIGH]
    )
    image_npy = np.clip(image_npy, lower, upper)

    # Apply Otsu threshold to create a binary foreground mask.
    threshold = skimage.filters.threshold_otsu(image_npy)
    fg_mask = image_npy > threshold

    nz = np.nonzero(fg_mask)
    og_size = img_ants.shape

    # Create the bounding box based on non-zero values.
    if nz[0].size > 0:
        fg_bbox = {
            "x_start": int(np.min(nz[0])),
            "x_end": int(np.max(nz[0])),
            "y_start": int(np.min(nz[1])),
            "y_end": int(np.max(nz[1])),
            "z_start": int(np.min(nz[2])),
            "z_end": int(np.max(nz[2])),
        }
    else:
        # If no foreground is detected, use the entire image size as bbox.
        fg_bbox = {
            "x_start": 0,
            "x_end": og_size[0] - 1,
            "y_start": 0,
            "y_end": og_size[1] - 1,
            "z_start": 0,
            "z_end": og_size[2] - 1,
        }

    # Add original image size to the bbox dictionary.
    fg_bbox.update(
        {
            "x_og_size": og_size[0],
            "y_og_size": og_size[1],
            "z_og_size": og_size[2],
        }
    )
    return fg_bbox


def aniso_intermediate_resample(
    img_sitk: sitk.Image,
    new_size: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
    low_res_axis: int,
) -> sitk.Image:
    """Intermediate resampling step for anisotropic images.

    This function resamples an image along the low-resolution axis using nearest
    neighbor interpolation. This is an intermediate step in the resampling
    anisotropic images that we use to reduce resampling artifacts.

    Args:
        img_sitk: SimpleITK image object.
        new_size: New dimensions for the image after resampling.
        target_spacing: Target spacing for resampling.
        low_res_axis: Axis along which the image is low resolution.

    Returns:
        img_sitk: Resampled SimpleITK image object.
    """
    # Create temporary spacing and size for resampling. This spacing is the same
    # as the original image spacing, except for the low resolution axis.
    temp_spacing = list(img_sitk.GetSpacing())
    temp_spacing[low_res_axis] = target_spacing[low_res_axis]

    # Create temporary size for resampling. This new size is the same as the
    # original image size, except for the low resolution axis.
    temp_size = list(img_sitk.GetSize())
    temp_size[low_res_axis] = new_size[low_res_axis]

    # Use nearest neighbor interpolation only along the low resolution axis.
    img_sitk = sitk.Resample(
        img_sitk,
        size=np.array(temp_size).tolist(),
        transform=sitk.Transform(),
        interpolator=sitk.sitkNearestNeighbor,
        outputOrigin=img_sitk.GetOrigin(),
        outputSpacing=temp_spacing,
        outputDirection=img_sitk.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img_sitk.GetPixelID()
    )
    return img_sitk


def check_anisotropic(img_sitk: sitk.Image) -> dict:
    """Check if an image is anisotropic.

    Args:
        img_sitk: SimpleITK image object.

    Returns:
        Dictionary with the following keys:
            is_anisotropic: Boolean indicating if the image is anisotropic.
            low_resolution_axis: Axis along which the image is low resolution,
                if anisotropic.
    """
    spacing = img_sitk.GetSpacing()
    if np.max(spacing) / np.min(spacing) > 3:
        is_anisotropic = True
        low_resolution_axis = int(np.argmax(spacing))
    else:
        is_anisotropic = False
        low_resolution_axis = None

    return {
        "is_anisotropic": is_anisotropic,
        "low_resolution_axis": low_resolution_axis
    }


def make_onehot(
    mask_ants: ants.core.ants_image.ANTsImage,
    labels_list: list[int]
) -> list[sitk.Image]:
    """Convert a multi-class ANTs image into a list of binary sitk images.

    Args:
        mask_ants: ANTs image object.
        labels_list: List of unique labels to create binary masks for.

    Returns:
        masks_sitk: List of binary SimpleITK images corresponding to each label.
    """
    # Get spacing, origin, and direction from ANTs image.
    spacing = mask_ants.spacing
    origin = mask_ants.origin
    direction = tuple(mask_ants.direction.flatten())

    mask_npy = mask_ants.numpy()
    masks_sitk = []
    for current_label in labels_list:
        sitk_label_i = sitk.GetImageFromArray(
            (mask_npy == current_label).T.astype(np.float32)
        )
        sitk_label_i.SetSpacing(spacing)
        sitk_label_i.SetOrigin(origin)
        sitk_label_i.SetDirection(direction)
        masks_sitk.append(sitk_label_i)
    return masks_sitk


def sitk_get_min_max(image: sitk.Image) -> tuple[float, float]:
    """Get minimum and maximum voxel values from a SimpleITK image.

    Args:
        image: SimpleITK image object.

    Returns:
        min_val: Minimum voxel value in the image.
        max_val: Maximum voxel value in the image.
    """
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    return stats_filter.GetMinimum(), stats_filter.GetMaximum()


def sitk_get_sum(image: sitk.Image) -> float:
    """Get sum of voxels in SITK image.

    Args:
        image: SITK image object.

    Returns:
        image_sum: Sum of all voxel values in image.
    """
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    image_sum = stats_filter.GetSum()
    return image_sum


def crop_to_fg(
    img_ants: ants.core.ants_image.ANTsImage,
    fg_bbox: dict[str, int],
) -> ants.core.ants_image.ANTsImage:
    """Crop image to foreground bounding box.

    Args:
        img_ants: ANTs image object.
        fg_bbox: Foreground bounding box.

    Returns:
        Cropped ANTs image object.
    """
    return ants.crop_indices(
        img_ants,
        lowerind=[fg_bbox["x_start"], fg_bbox["y_start"], fg_bbox["z_start"]],
        upperind=[
            fg_bbox["x_end"] + 1, fg_bbox["y_end"] + 1, fg_bbox["z_end"] + 1
        ]
    )
