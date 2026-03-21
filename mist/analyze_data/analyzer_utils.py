"""Utilities for the analyzer module."""
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# MIST imports.
from mist.utils import io as io_utils


def compare_headers(
    header1: dict[str, Any], header2: dict[str, Any]
) -> bool:
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
        # Exact comparison is intentional: SimpleITK raises an error if image
        # origins don't match exactly, so floating-point tolerance here would
        # give a false sense of safety.
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


def is_image_3d(header: dict[str, Any]) -> bool:
    """Check if image is 3D.

    Args:
        header: Image header information from ants.image_header_info

    Returns:
        True if the image is 3D.
    """
    return len(header["dimensions"]) == 3


def get_resampled_image_dimensions(
    dimensions: tuple[int, int, int],
    spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Get new image dimensions after resampling.

    Args:
        dimensions: Original image dimensions.
        spacing: Original image spacing.
        target_spacing: Target image spacing.

    Returns:
        New image dimensions after resampling.
    """
    new_dimensions = [
        int(np.round(dimensions[i] * spacing[i] / target_spacing[i]))
        for i in range(len(dimensions))
    ]
    return (new_dimensions[0], new_dimensions[1], new_dimensions[2])


def get_float32_example_memory_size(
    dimensions: tuple[int, int, int],
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


def get_files_df(
    path_to_dataset_json: str,
    train_or_test: Literal["train", "test"],
) -> pd.DataFrame:
    """Get dataframe with file paths for each patient in the dataset.

    Args:
        path_to_dataset_json: Path to dataset json file with dataset
            information.
        train_or_test: "train" or "test". If "train", the dataframe will have
            columns for the mask and images. If "test", the dataframe
            will have columns for the images only.

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

    # Base directory for the dataset. Relative paths are resolved relative to
    # the dataset JSON file so the JSON and its data can be co-located and
    # moved together without adjusting the working directory.
    base_dir = (
        Path(path_to_dataset_json).resolve().parent
        / dataset_info[f"{train_or_test}-data"]
    ).resolve()

    # Get sorted list of patient IDs, skipping hidden files.
    # Sorting ensures deterministic ordering across platforms and runs.
    patient_ids = sorted(
        p.name for p in base_dir.iterdir() if not p.name.startswith(".")
    )

    # Build one row dict per patient, then create the DataFrame in one call.
    rows = []
    for patient_id in patient_ids:
        row_data: dict[str, Any] = {"id": patient_id}

        patient_dir = base_dir / patient_id
        patient_files = [str(p) for p in patient_dir.glob("*")]

        for image_type, identifying_strings in dataset_info["images"].items():
            matching_file = next(
                (
                    f for f in patient_files
                    if any(s in f for s in identifying_strings)
                ),
                None,
            )
            if matching_file:
                row_data[image_type] = matching_file
            else:
                logging.warning(
                    "Patient '%s': no file found for image type '%s' "
                    "(identifying strings: %s).",
                    patient_id,
                    image_type,
                    identifying_strings,
                )

        if train_or_test == "train":
            mask_file = next(
                (
                    f for f in patient_files
                    if any(s in f for s in dataset_info["mask"])
                ),
                None,
            )
            if mask_file:
                row_data["mask"] = mask_file
            else:
                logging.warning(
                    "Patient '%s': no mask file found "
                    "(identifying strings: %s).",
                    patient_id,
                    dataset_info["mask"],
                )

        rows.append(row_data)

    return pd.DataFrame(rows, columns=columns)


def add_folds_to_df(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Add folds to the dataframe for k-fold cross-validation.

    Args:
        df: Dataframe with file paths for each patient in the dataset.
        n_splits: Number of splits for k-fold cross-validation.

    Returns:
        Dataframe with folds added as a new column, sorted by fold. The fold
        value next to each patient ID indicates the fold in which that patient
        belongs to the test set.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    df.insert(loc=1, column="fold", value=[None] * len(df))

    for fold_number, (_, test_indices) in enumerate(kfold.split(df)):
        df.loc[test_indices, "fold"] = fold_number

    return df.sort_values("fold").reset_index(drop=True)


def get_best_patch_size(med_img_size: list[int]) -> list[int]:
    """Get the best patch size based on median image size.

    The best patch size is computed as the nearest power of two less than the
    median image size along each axis.

    Args:
        med_img_size: Median image size in the x, y, and z directions.

    Returns:
        Selected patch size based on the input sizes.

    Raises:
        ValueError: If any dimension of med_img_size is less than or equal to 1.
    """
    if min(med_img_size) <= 1:
        raise ValueError(
            f"All image dimensions must be greater than 1 to compute a valid "
            f"patch size. Got: {med_img_size}."
        )
    return [
        int(2 ** np.floor(np.log2(med_sz))) for med_sz in med_img_size
    ]


def build_evaluation_config(dataset: dict[str, Any]) -> dict[str, Any]:
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
            "labels": labels,
            "metrics": {
                "dice": {},
                "haus95": {},
            },
        }
    return {"evaluation": evaluation}


def build_base_config() -> dict[str, Any]:
    """Build base configuration dictionary.

    Returns:
        Base configuration dictionary.
    """
    return {
        "mist_version": None,
        "dataset_info": {
            "task": None,
            "modality": None,
            "images": None,
            "labels": None,
        },
        "spatial_config": {
            "patch_size": None,
            "target_spacing": None,
        },
        "preprocessing": {
            "skip": False,
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
                    "composite_loss_weighting": None,
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
