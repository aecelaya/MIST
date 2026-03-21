"""Utility functions for MIST inference modules."""
from typing import Any, Dict, Optional, Tuple, List, Union
from collections.abc import Callable
from pathlib import Path
import os
import ants
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

# MIST imports.
from mist.utils import io
from mist.analyze_data import analyzer_utils
from mist.preprocessing import preprocess
from mist.inference.inference_constants import InferenceConstants as ic
from mist.models import model_loader


def decrop_from_fg(
    ants_image: ants.core.ants_image.ANTsImage,
    fg_bbox: Dict[str, int],
) -> ants.core.ants_image.ANTsImage:
    """Decrop image to original size using foreground bounding box.

    Args:
        ants_image: ANTs image object.
        fg_bbox: Foreground bounding box.

    Returns:
        Decropped ANTs image object.
    """
    padding = [
        (
            np.max([0, fg_bbox["x_start"]]),
            np.max([0, fg_bbox["x_og_size"] - fg_bbox["x_end"]]) - 1
        ),
        (
            np.max([0, fg_bbox["y_start"]]),
            np.max([0, fg_bbox["y_og_size"] - fg_bbox["y_end"]]) - 1
        ),
        (
            np.max([0, fg_bbox["z_start"]]),
            np.max([0, fg_bbox["z_og_size"] - fg_bbox["z_end"]]) - 1
        )
    ]
    return ants.pad_image(ants_image, pad_width=padding, return_padvals=False)


def back_to_original_space(
    raw_prediction: npt.NDArray[Any],
    original_ants_image: ants.core.ants_image.ANTsImage,
    target_spacing: Tuple[float, float, float],
    training_labels: List[int],
    foreground_bounding_box: Optional[Dict[str, Any]],
) -> ants.core.ants_image.ANTsImage:
    """Place 3D prediction back into original image space.

    All predictions are natively in RAI orientation, possibly cropped to the
    foreground, and in the target spacing. This function will place the
    prediction back into the original image space by reorienting, resampling,
    possibly padding back to the original size, and copying the original image
    header to the prediction's header.

    Args:
        raw_prediction: The prediction to place back into the original image
            space. This should be a numpy array.
        original_ants_image: The original ANTs image.
        target_spacing: The spacing used for training. This can be found in the
            MIST configuration JSON file.
        training_labels: List of training labels in the dataset. This is used to
            resample the mask back to the original image space.
        foreground_bounding_box: The foreground bounding box. If we crop images
            as part of preprocessing, we need to pad back to the original size.
            The foreground bounding box is a dictionary with the following keys
            and values:
                - x_start: The starting x coordinate of the bounding box.
                - y_start: The starting y coordinate of the bounding box.
                - z_start: The starting z coordinate of the bounding box.
                - x_end: The ending x coordinate of the bounding box.
                - y_end: The ending y coordinate of the bounding box.
                - z_end: The ending z coordinate of the bounding box.
                - x_og_size: The original x size of the image.
                - y_og_size: The original y size of the image.
                - z_og_size: The original z size of the image.
            The bounding box dictionary has the necessary information to
            appropriately pad the prediction back to the original size.

    Returns:
        The prediction in the original image space. This will be an ANTs image.
    """
    # Convert prediction to ANTs image.
    prediction: ants.core.ants_image.ANTsImage = ants.from_numpy(
        data=raw_prediction, spacing=target_spacing
    )

    # Reorient prediction.
    prediction = ants.reorient_image2(
        prediction, ants.get_orientation(original_ants_image)
    )
    prediction.set_direction(original_ants_image.direction)

    # Enforce size for cropped images.
    if foreground_bounding_box is not None:
        # If we have a foreground bounding box, use that to determine the size.
        new_size = [
            foreground_bounding_box["x_end"] - foreground_bounding_box["x_start"] + 1,
            foreground_bounding_box["y_end"] - foreground_bounding_box["y_start"] + 1,
            foreground_bounding_box["z_end"] - foreground_bounding_box["z_start"] + 1,
        ]
    else:
        # Otherwise, use the original image size.
        new_size = original_ants_image.shape

    # Resample prediction to original image space.
    prediction = preprocess.resample_mask(
        prediction,
        labels=training_labels,
        target_spacing=original_ants_image.spacing,
        new_size=np.array(new_size, dtype="int").tolist(),
    )

    # Appropriately pad back to original size if necessary.
    if foreground_bounding_box is not None:
        prediction = decrop_from_fg(prediction, foreground_bounding_box)

    # Copy header from original image onto the prediction so they match. This
    # will take care of other details in the header like the origin and the
    # image bounding box.
    prediction = original_ants_image.new_image_like(prediction.numpy()) # type: ignore
    return prediction


def load_test_time_models(
    models_dir: str,
    mist_config: Dict,
    device: Optional[Union[str, torch.device]]=None,
) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """Load one or more models for test-time inference.

    This function loads all models matching the pattern `fold_*.pt` in the
    specified directory. The model configuration is read from the MIST
    configuration file under the "model" key.

    The function will validate the existence of the model directory and ensure
    that at least one model checkpoint file is found.

    Args:
        models_dir: Path to directory with `fold_*.pt` model weights.
        mist_config: MIST configuration dictionary.
        device: Device to load the models onto. If None, defaults to CUDA if
            available, otherwise CPU.

    Returns:
        List of loaded PyTorch models, ready for inference.

    Raises:
        FileNotFoundError: If model config or model files are missing.
        ValueError: If no model checkpoint files are found.
    """
    # Set the device if not provided.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the models directory and config file exist.
    models_path = Path(models_dir)
    if not models_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {models_path}")

    # Find all model checkpoint files matching fold_*.pt
    pt_files = sorted(models_path.glob("fold_*.pt"))
    pt_files = [f for f in pt_files if not f.name.startswith(".")]

    # Raise an error if no model files are found.
    if not pt_files:
        raise ValueError(
            f"No model checkpoints found in {models_path}, (expected fold_*.pt)"
        )

    models = []
    for model_path in pt_files:
        model_path = str(model_path)
        model = model_loader.load_model_from_config(model_path, mist_config)
        models.append(model.to(device).eval())
    return models


def remap_mask_labels(
    mask_npy: npt.NDArray[Any],
    original_labels: List[int],
) -> npt.NDArray[Any]:
    """Remap label indices in a predicted mask to their original values.

    Args:
        mask_npy: A numpy array containing class indices (i.e., 0, 1, 2, 3).
        original_labels: A list mapping each index to the true label value
            (i.e., [0, 1, 2, 4]).

    Returns:
        A new numpy array with label values remapped to match the original
            dataset.
    """
    remapped_mask = np.zeros_like(mask_npy)
    for i, label in enumerate(original_labels):
        remapped_mask[mask_npy == i] = label
    return remapped_mask


def validate_inference_images(
    patient_dict: Dict[str, str]
) -> Tuple[ants.core.ants_image.ANTsImage, List[str]]:
    """Validate all images listed in the patient dictionary.

    Ensures that each image file exists, is a valid 3D image, and that all
    images (if multiple) are spatially compatible with the anchor image.

    Args:
        patient_dict: Dictionary containing image paths, with optional metadata
            like 'id', 'mask', or 'fold' (ignored).

    Returns:
        anchor_image: The anchor image (first image in the list) as an ANTs
            image if all checks pass.
        image_paths: A list of image paths for further processing if all
            checks pass.

    Raises:
        FileNotFoundError: If any image file is missing.
        ValueError: If any image is not 3D or if headers do not match.
    """
    if "id" not in patient_dict:
        raise ValueError("Patient dictionary must contain an 'id' field.")

    image_paths = [
        v for k, v in patient_dict.items()
        if k not in ic.PATIENT_DF_IGNORED_COLUMNS
    ]

    if len(image_paths) == 0:
        raise ValueError(
            f"No image paths found for patient {patient_dict['id']}."
        )

    # Check all files exist.
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image file not found: {os.path.basename(image_path)}"
            )

    # Load anchor image. Check if it is 3D before proceeding.
    anchor_filename = os.path.basename(image_paths[0])
    anchor_header = ants.image_header_info(image_paths[0])
    if not analyzer_utils.is_image_3d(anchor_header):
        raise ValueError(f"Anchor image is not 3D: {anchor_filename}")
    anchor_image = ants.image_read(image_paths[0])

    # Check header compatibility for additional modalities.
    for image_path in image_paths[1:]:
        current_filename = os.path.basename(image_path)
        current_header = ants.image_header_info(image_path)
        if not analyzer_utils.is_image_3d(current_header):
            raise ValueError(f"Image is not 3D: {current_filename}")

        if not analyzer_utils.compare_headers(anchor_header, current_header):
            raise ValueError(
                f"Image headers do not match: {anchor_filename} and "
                f"{current_filename}"
            )
    return anchor_image, image_paths


def validate_paths_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate that the input dataframe for inference.

    We want to make sure that the dataframe contains an 'id' column and at least
    one other column with valid NIfTI file paths.

    Args:
        dataframe: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or criteria not met.
    """
    if "id" not in dataframe.columns:
        raise ValueError("The dataframe must contain an 'id' column.")

    nifti_columns = []

    for col in dataframe.columns:
        if col == "id":
            continue
        series = dataframe[col].astype(str)
        if series.apply(
            lambda x: x.endswith(".nii") or x.endswith(".nii.gz")
        ).all():
            nifti_columns.append(col)

    if not nifti_columns:
        raise ValueError(
            "The dataframe must contain at least one column with valid NIfTI "
            "file paths (.nii or .nii.gz). No such columns found among: "
            f"{list(dataframe.columns)}"
        )
