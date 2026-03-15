"""Utilities for the analyzer module."""
from typing import Dict, Any, Tuple, List
import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# MIST imports.
from mist.utils import io as io_utils


def compare_headers(header1: Dict[str, Any], header2: Dict[str, Any]) -> bool:
    """Compare two image headers to see if they match.

    We compare the dimensions, origin, spacing, and direction of the two images.

    Args:
        header1: Image header information from ants.image_header_info
        header2: Image header information from ants.image_header_info

    Returns:
        True if the dimensions, origin, spacing, and direction match.
    """
    if header1["dimensions"] != header2["dimensions"]:
        is_valid = False
    elif header1["origin"] != header2["origin"]:
        is_valid = False
    elif not np.allclose(
        np.array(header1["spacing"]), np.array(header2["spacing"])
    ):
        is_valid = False
    elif not np.allclose(header1["direction"], header2["direction"]):
        is_valid = False
    else:
        is_valid = True
    return is_valid


def is_image_3d(header: Dict[str, Any]) -> bool:
    """Check if image is 3D.

    Args:
        header: Image header information from ants.image_header_info

    Returns:
        True if the image is 3D.
    """
    return len(header["dimensions"]) == 3


def get_resampled_image_dimensions(
    dimensions: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Get new image dimensions after resampling.

    Args:
        original_dimensions: Original image dimensions.
        original_spacing: Original image spacing.
        target_spacing: Target image spacing.

    Returns:
        new_dimensions: New image dimensions after resampling.
    """
    new_dimensions = [
        int(np.round(dimensions[i] * spacing[i] / target_spacing[i]))
        for i in range(len(dimensions))
    ]
    return (new_dimensions[0], new_dimensions[1], new_dimensions[2])


def get_float32_example_memory_size(
    dimensions: Tuple[int, int, int],
    number_of_channels: int,
    number_of_labels: int,
) -> int:
    """Get memory size of float32 image-mask pair in bytes.

    Args:
        dimensions: Image dimensions.
        number_of_channels: Number of image channels.
        number_of_labels: Number of labels in mask.

    Returns:
        Memory size of image-mask pair in bytes.
    """
    _dims: np.ndarray = np.array(dimensions)
    return int(4 * (np.prod(_dims) * (number_of_channels + number_of_labels)))


def get_files_df(path_to_dataset_json: str, train_or_test: str) -> pd.DataFrame:
    """Get dataframe with file paths for each patient in the dataset.

    Args:
        path_to_dataset_json: Path to dataset json file with dataset
            information.
        train_or_test: "train" or "test". If "train", the dataframe will have
            columns for the mask and images. If "test", the dataframe
            will have columns for the images.

    Returns:
        DataFrame with file paths for each patient in the dataset.
    """
    # Read JSON file with dataset parameters.
    dataset_info = io_utils.read_json_file(path_to_dataset_json)

    # Determine columns based on the mode (train or test).
    columns = ["id"]
    if train_or_test == "train":
        columns.append("mask")
    columns.extend(dataset_info["images"].keys())

    # Base directory for the dataset.
    base_dir = os.path.abspath(dataset_info[f"{train_or_test}-data"])

    # Initialize an empty DataFrame with the determined columns.
    df = pd.DataFrame(columns=columns)

    # Get list of patient IDs.
    patient_ids = [
        f for f in os.listdir(base_dir) if not f.startswith('.')
    ]

    # Iterate over each patient and get the file paths for each patient.
    for patient_id in patient_ids:
        # Initialize row data with 'id' and empty values for other columns.
        row_data = {"id": patient_id}

        # Path to patient data.
        patient_dir = os.path.join(base_dir, patient_id)
        patient_files = glob.glob(
            os.path.join(patient_dir, '*')
        )

        # Map file paths to their respective columns.
        for image_type, identifying_strings in dataset_info["images"].items():
            matching_file = next(
                (file for file in patient_files
                 if any(s in file for s in identifying_strings)), None
            )
            if matching_file:
                row_data[image_type] = matching_file

        # Add the mask file if in training mode.
        if train_or_test == "train":
            mask_file = next(
                (file for file in patient_files
                 if any(s in file for s in dataset_info["mask"])), None
            )
            if mask_file:
                row_data["mask"] = mask_file

        # Append the row to the DataFrame.
        df = pd.concat(
            [df, pd.DataFrame([row_data], columns=columns)],
            ignore_index=True
        )
    return df


def add_folds_to_df(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Add folds to the dataframe for k-fold cross-validation.

    Args:
        df: Dataframe with file paths for each patient in the dataset.
        n_splits: Number of splits for k-fold cross-validation.

    Returns:
        df: Dataframe with folds added. The folds are added as a new column.
            The dataframe is sorted by the fold column. The fold next to each
            patient ID is the fold that the patient belongs to the test set
            for that given fold.
    """
    # Initialize KFold object
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize an empty 'folds' column.
    df.insert(loc=1, column="fold", value=[None] * len(df))

    # Assign fold numbers
    for fold_number, (_, test_indices) in enumerate(kfold.split(df)):
        df.loc[test_indices, "fold"] = fold_number

    # Sort the dataframe by the 'fold' column
    df = df.sort_values("fold").reset_index(drop=True)
    return df


def get_best_patch_size(med_img_size: List[int]) -> List[int]:
    """Get the best patch size based on median image size and maximum size.

    The best patch size is computed as the nearest power of two less than the
    median image size. In the future, we will consider other strategies, but
    for now this is a good starting point.

    Args:
        med_img_size: Median image size in the x y and z directions.

    Returns:
        patch_size: Selected patch size based on the input sizes.

    Raises:
        AssertionError: If the input sizes are invalid.
    """
    # Check that each dimension of the median image size is greater than 1.
    # Otherwise, we will get a negative or zero patch size.
    assert min(med_img_size) > 1, "Image size is too small"
    return [
        int(2 ** np.floor(np.log2(med_sz))) for med_sz in med_img_size
    ]


def build_evaluation_config(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Build evaluation field for the MIST configuration.

    Args:
        dataset: The dictionary containing the dataset description. This MUST
            contain the 'final_classes' field with the following format:
            {
                'final_classes': {
                    'final_class_1': [1, 3],
                    'final_class_2': [2, 4]
                }
            }

    Returns:
        A dictionary with the following format:
            {
                'evaluation': {
                    'final_class_1': {
                        'labels': [1, 3],
                        'metrics': {
                            'dice': {},
                            'haus95': {},
                        }
                    },
                    'final_class_2': {
                        'labels': [2, 4],
                        'metrics': {
                            'dice': {},
                            'haus95': {},
                        }
                    }
                }
            }
    """
    final_classes = dataset.get("final_classes", None)

    if final_classes is None:
        raise ValueError("Missing 'final_classes' in the dataset.")

    evaluation = {}
    for class_name, labels in final_classes.items():
        evaluation[class_name] = {
            'labels': labels,
            'metrics': {
                'dice': {},
                'haus95': {},
            }
        }
    return {'evaluation': evaluation}


def build_base_config() -> Dict[str, Any]:
    """Build base configuration dictionary.

    Returns:
        base_config: Base configuration dictionary.
    """
    return {
        "mist_version": None,
        "dataset_info": {
            "task": None,
            "modality": None,
            "images": None,
            "labels": None,
        },
        "preprocessing": {
            "skip": False,
            "target_spacing": None,
            "crop_to_foreground": None,
            "median_resampled_image_size": None,
            "normalize_with_nonzero_mask": None,
            "ct_normalization": {
                "window_min": None,
                "window_max": None,
                "z_score_mean": None,
                "z_score_std": None,
            },
            "compute_dtms": False,
            "normalize_dtms": True,
        },
        "model": {
            "architecture": "nnunet",
            "params": {
                "in_channels": None,
                "out_channels": None,
                "patch_size": None,
                "target_spacing": None,
                "use_deep_supervision": True,
                "use_residual_blocks": True,
                "use_pocket_model": False,
            },
        },
        "training": {
            "seed": 42,
            "nfolds": 5,
            "folds": None,
            "val_percent": 0.0,
            "epochs": 1000,
            "min_steps_per_epoch": 250,
            "batch_size_per_gpu": 2,
            "dali_foreground_prob": 0.6,
            "loss": {
                "name": "dice_ce",
                "params": {
                    "use_dtms": False,
                    "composite_loss_weighting": None
                },
            },
            "optimizer": "adam",
            "learning_rate": 0.001,
            "lr_scheduler": "cosine",
            "l2_penalty": 0.00001,
            "amp": True,
            "augmentation": {
                "enabled": True,
                "transforms": {
                    "flips": True,
                    "zoom": True,
                    "noise": True,
                    "blur": True,
                    "brightness": True,
                    "contrast": True,
                },
            },
            "hardware": {
                "num_gpus": None,
                "num_cpu_workers": 8,
                "master_addr": "localhost",
                "master_port": 12345,
                "communication_backend": "nccl",
            },
        },
        "inference": {
            "inferer": {
                "name": "sliding_window",
                "params": {
                    "patch_size": None,
                    "patch_blend_mode": "gaussian",
                    "patch_overlap": 0.5,
                },
            },
            "ensemble": {
                "strategy": "mean",
            },
            "tta": {
                "enabled": True,
                "strategy": "all_flips",
            },
        },
    }
