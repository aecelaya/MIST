"""Utilities for the analyzer module."""

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# MIST imports.
from mist.analyze_data.analyzer_constants import AnalyzeConstants as constants
from mist.utils import io as io_utils


def compare_headers(header1: dict[str, Any], header2: dict[str, Any]) -> bool:
    """Compare two image headers to see if they match.

    We compare the dimensions, origin, spacing, and direction of the two images.

    Args:
        header1: Image header information from sitk_io.read_image_header
        header2: Image header information from sitk_io.read_image_header

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
    elif not np.allclose(np.array(header1["spacing"]), np.array(header2["spacing"])):
        is_valid = False
    elif not np.allclose(header1["direction"], header2["direction"]):
        is_valid = False
    else:
        is_valid = True
    return is_valid


def is_image_3d(header: dict[str, Any]) -> bool:
    """Check if image is 3D.

    Args:
        header: Image header information from sitk_io.read_image_header

    Returns:
        True if the image is 3D.
    """
    return len(header["dimensions"]) == 3


def compute_target_spacing(spacings: np.ndarray, sizes: np.ndarray) -> list[float]:
    """Choose a dataset's target spacing with nnU-Net's rule.

    The target is the median spacing of each axis. If the coarsest axis has both
    much larger spacing and many fewer voxels than the other axes, its median
    would interpolate the thinner-sliced cases heavily, so that axis uses a low
    percentile of its spacings instead, but stays coarser than the other axes.

    Args:
        spacings: Spacing of each image, with shape (number of images, 3).
        sizes: Size in voxels of each image, with the same shape and axis order.

    Returns:
        The target spacing of each axis.
    """
    target = np.median(spacings, axis=0)
    median_sizes = np.median(sizes, axis=0)

    coarsest = int(np.argmax(target))
    others = [axis for axis in range(len(target)) if axis != coarsest]
    next_coarsest_spacing = target[others].max()

    threshold = constants.ANISOTROPY_THRESHOLD
    has_anisotropic_spacing = target[coarsest] > threshold * next_coarsest_spacing
    has_anisotropic_voxels = median_sizes[coarsest] * threshold < median_sizes[others].min()

    if has_anisotropic_spacing and has_anisotropic_voxels:
        percentile = np.percentile(
            spacings[:, coarsest], constants.ANISOTROPIC_LOW_RESOLUTION_AXIS_PERCENTILE
        )
        if percentile < next_coarsest_spacing:
            percentile = next_coarsest_spacing + constants.ANISOTROPIC_SPACING_MARGIN
        target[coarsest] = percentile

    return [float(spacing) for spacing in target]


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
        Path(path_to_dataset_json).resolve().parent / dataset_info[f"{train_or_test}-data"]
    ).resolve()

    # Get sorted list of patient IDs, skipping hidden files.
    # Sorting ensures deterministic ordering across platforms and runs.
    patient_ids = sorted(p.name for p in base_dir.iterdir() if not p.name.startswith("."))

    # Build one row dict per patient, then create the DataFrame in one call.
    rows = []
    for patient_id in patient_ids:
        row_data: dict[str, Any] = {"id": patient_id}

        patient_dir = base_dir / patient_id
        patient_files = [str(p) for p in patient_dir.glob("*")]

        for image_type, identifying_strings in dataset_info["images"].items():
            matching_file = next(
                (f for f in patient_files if any(s in f for s in identifying_strings)),
                None,
            )
            if matching_file:
                row_data[image_type] = matching_file
            else:
                logging.warning(
                    "Patient '%s': no file found for image type '%s' (identifying strings: %s).",
                    patient_id,
                    image_type,
                    identifying_strings,
                )

        if train_or_test == "train":
            mask_file = next(
                (f for f in patient_files if any(s in f for s in dataset_info["mask"])),
                None,
            )
            if mask_file:
                row_data["mask"] = mask_file
            else:
                logging.warning(
                    "Patient '%s': no mask file found (identifying strings: %s).",
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
            "patch_size": "auto",
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
            "architecture": "nnunet-plain",
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
            "validate_every": 1,
            "epochs": 1000,
            "min_steps_per_epoch": 250,
            "batch_size_per_gpu": "auto",
            "dali_foreground_prob": 0.6,
            "loss": {
                "name": "dice_ce",
                "composite_loss_weighting": None,
            },
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "lr_scheduler": "cosine",
            "warmup_epochs": 20,
            "l2_penalty": 0.0001,
            "grad_clip_norm": 1.0,
            # BF16 or FP32, chosen by a speed test at the first mist_train. See
            # mist.training.precision_test.
            "amp": "auto",
            # Compile the model with torch.compile during training, on GPUs that
            # support it. Configs from before this setting train without it.
            "compile": True,
            "augmentation": {
                "enabled": True,
                "transforms": {
                    "flips": True,
                    "rotation": True,
                    "scaling": True,
                    "noise": True,
                    "blur": True,
                    "brightness": True,
                    "contrast": True,
                    "low_resolution": True,
                    "gamma": True,
                },
            },
            "hardware": {
                "num_gpus": None,
                "num_cpu_workers": 8,
                "master_addr": "localhost",
                "master_port": 12345,
                # Resolved against the current hardware at train time (mirrors
                # "amp" above), not here at analyze time, since the machine
                # that runs analyze may not be the machine that trains --
                # see hardware.resolve_communication_backend(). A config.json
                # from before this key existed already has a literal "nccl"
                # here rather than "auto", so it will keep using NCCL even if
                # training later moves to CPU-only hardware; edit it by hand
                # (or re-run analyze) to pick up automatic resolution.
                "communication_backend": "auto",
                # Same resolve-and-persist story as communication_backend
                # above: "auto" becomes "dali" on CUDA or "generic" elsewhere,
                # resolved at train time via hardware.resolve_data_loader().
                # No CLI flag -- a user who wants to force one edits this
                # field directly, the same way patch_size or grad_clip_norm
                # are hand-edited rather than exposed as flags.
                "data_loader": "auto",
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
