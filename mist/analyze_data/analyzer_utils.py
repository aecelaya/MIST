"""Utilities for the analyzer module."""
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

# MIST imports.
from mist.analyze_data.analyzer_constants import AnalyzeConstants as constants
from mist.models.nnunet.nnunet_utils import get_unet_params
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


def _get_voxel_budget(batch_size_per_gpu: int = 2) -> int:
    """Return a per-patch voxel budget for patch size selection.

    Queries the minimum total memory across all available CUDA devices and
    scales linearly from a 16 GB / batch-size-2 reference (128^3 voxels per
    patch). The budget scales inversely with batch_size_per_gpu so that total
    memory per step (batch_size × patch_voxels × network overhead) stays
    roughly constant. Falls back to the default constant when no GPU is
    present.

    Args:
        batch_size_per_gpu: Number of samples per GPU per step. Defaults to 2,
            matching the MIST default training configuration.

    Returns:
        Per-patch voxel budget as a positive integer.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        min_mem = min(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        )
        return int(
            min_mem
            / constants.PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES
            * constants.PATCH_BUDGET_REFERENCE_VOXELS
            * constants.PATCH_BUDGET_REFERENCE_BATCH_SIZE
            / batch_size_per_gpu
        )
    return constants.PATCH_BUDGET_DEFAULT_VOXELS


def _largest_multiple_of_32_leq(value: float, minimum: int = 32) -> int:
    """Return the largest multiple of 32 that is ≤ value, floored to minimum.

    The nnUNet encoder downsamples each isotropic axis by stride 2 for up to
    MAX_DEPTH=5 levels, so patch dimensions must be divisible by 2^5 = 32 for
    exact decoder reconstruction.

    Args:
        value: Upper bound (inclusive, may be a float).
        minimum: Smallest allowed return value. Defaults to 32.

    Returns:
        Largest multiple of 32 that does not exceed value, at least minimum.
    """
    snapped = int(value // 32) * 32
    return max(snapped, minimum)


def _snap_lr_to_nnunet_compatible(
    lr_patch: int,
    low_res_axis: int,
    median_ip: int,
    median_lr: int,
    min_lr: int,
    target_spacing: list[float],
) -> int:
    """Snap lr_patch to a value compatible with the nnUNet decoder.

    ConvTranspose3d upsamples by the exact stride factor. If the encoder
    halved a dimension from an odd value (e.g. 9 → 4), the decoder produces
    8 rather than 9, causing a skip-connection size mismatch at runtime.

    This function queries the nnUNet architecture planner with a trial patch
    to find the cumulative stride on the low-res axis (z_divisor), then snaps
    lr_patch up to the nearest compatible multiple ≤ median_lr. If snapping
    up would exceed median_lr, it snaps down instead.

    Args:
        lr_patch: Initial low-res axis patch size (from budget calculation).
        low_res_axis: Index of the low-resolution axis (0, 1, or 2).
        median_ip: Median image size on the in-plane axes.
        median_lr: Median image size on the low-res axis.
        min_lr: Minimum allowed patch size on the low-res axis.
        target_spacing: Target voxel spacing in mm, [x, y, z].

    Returns:
        Snapped lr_patch that is divisible by z_divisor.
    """
    trial = [median_ip, median_ip, median_ip]
    trial[low_res_axis] = lr_patch
    _, strides, _ = get_unet_params(trial, target_spacing)

    # Compute the product of all strides > 1 on the low-res axis.
    # strides[0] is always [1,1,1] (the input block); skip it.
    z_divisor = 1
    for s in strides[1:]:
        if s[low_res_axis] > 1:
            z_divisor *= s[low_res_axis]

    if z_divisor <= 1:
        return lr_patch

    # Prefer snapping up (preserves more coverage along the low-res axis).
    snapped_up = ((lr_patch + z_divisor - 1) // z_divisor) * z_divisor
    if snapped_up <= median_lr:
        return snapped_up

    # Fall back to snapping down if snapping up would exceed the median.
    snapped_down = (lr_patch // z_divisor) * z_divisor
    return max(snapped_down, min_lr)


def get_best_patch_size(
    median_resampled_size: list[int],
    target_spacing: list[float],
    batch_size_per_gpu: int = 2,
) -> list[int]:
    """Select a patch size from the median resampled image size and spacing.

    Uses a GPU-memory-derived voxel budget and the target spacing to choose
    between two strategies:

    **Quasi-2D mode** (triggered when the target spacing is still anisotropic
    after analysis, i.e. max/min spacing > MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD):
    The low-resolution axis is identified as the axis with the largest spacing.
    Its patch size is chosen as the largest value that leaves the full voxel
    budget available for the in-plane axes (clamped to
    MIN_LOW_RES_AXIS_PATCH_SIZE … median_lr), then snapped to the nearest
    multiple of the nnUNet cumulative stride on that axis so the decoder can
    reconstruct without skip-connection size mismatches. Both in-plane axes
    receive the same patch size (largest multiple of 32 that fits within the
    budget).

    **3D isotropic mode** (all other cases):
    A physically isotropic target patch extent in mm is computed from the
    budget, then converted to per-axis voxel counts. Axes whose raw voxel
    count would exceed the median image size are clamped and the remaining
    budget is redistributed to the unclamped axes. Each axis is then snapped
    down to the nearest multiple of 32 (minimum 32).

    Args:
        median_resampled_size: Median image size after resampling, [x, y, z].
        target_spacing: Target voxel spacing in mm, [x, y, z].
        batch_size_per_gpu: Number of samples per GPU per step. The voxel
            budget scales inversely with this value so that total memory per
            step stays constant. Defaults to 2 (the MIST default).

    Returns:
        Patch size as a list of three integers.
    """
    budget = _get_voxel_budget(batch_size_per_gpu)

    anisotropy_ratio = max(target_spacing) / min(target_spacing)
    low_res_axis = int(np.argmax(target_spacing))
    in_plane_axes = [i for i in range(3) if i != low_res_axis]

    if anisotropy_ratio > constants.MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD:
        # Quasi-2D: maximize in-plane resolution; keep low-res axis small.
        # Use min() so the square in-plane patch fits both in-plane axes without
        # requiring padding on the smaller one (using max() could select a patch
        # larger than one of the axes).
        median_lr = median_resampled_size[low_res_axis]
        median_ip = min(median_resampled_size[i] for i in in_plane_axes)

        # Largest low-res patch that leaves the full budget for in-plane.
        # Enforce MIN_LOW_RES_AXIS_PATCH_SIZE unless the image itself is
        # smaller, and never exceed the median image depth.
        min_lr = min(constants.MIN_LOW_RES_AXIS_PATCH_SIZE, median_lr)
        lr_patch = int(np.clip(budget / (median_ip ** 2), min_lr, median_lr))

        # Snap to the nearest multiple of the nnUNet cumulative low-res stride
        # so the decoder skip connections always match in spatial size.
        lr_patch = _snap_lr_to_nnunet_compatible(
            lr_patch, low_res_axis, median_ip, median_lr, min_lr, target_spacing
        )

        # In-plane: both axes get the same patch (largest multiple of 32).
        # Clamp to median_ip before snapping so the snap-to-32 always operates
        # on a value ≤ median_ip, guaranteeing the result is a valid multiple
        # of 32 (min(512, 491) = 491 is not a multiple of 32).
        ip_raw = min(int(round(np.sqrt(budget / lr_patch))), median_ip)
        ip_patch = _largest_multiple_of_32_leq(ip_raw)

        patch = [ip_patch, ip_patch, ip_patch]
        patch[low_res_axis] = lr_patch
        return patch

    # 3D isotropic mode: distribute budget proportionally in physical space,
    # iteratively clamping axes that would exceed the median image size and
    # redistributing the freed budget to the remaining axes.
    free_axes = list(range(3))
    fixed: dict[int, int] = {}

    while free_axes:
        free_spacings = [target_spacing[i] for i in free_axes]
        remaining_budget = float(budget)
        for v in fixed.values():
            remaining_budget /= v

        n = len(free_axes)
        target_mm = (remaining_budget * float(np.prod(free_spacings))) ** (1.0 / n)

        new_fixed = [
            i for i in free_axes
            if (target_mm / target_spacing[i]) >= median_resampled_size[i]
        ]
        if not new_fixed:
            break

        for i in new_fixed:
            # Store the snapped value so budget redistribution reflects what
            # will actually be placed in the patch, not the raw median.
            fixed[i] = _largest_multiple_of_32_leq(median_resampled_size[i])
            free_axes.remove(i)

    patch = []
    for i in range(3):
        if i in fixed:
            raw = float(fixed[i])
        else:
            raw = target_mm / target_spacing[i]
        # Round to nearest int before snapping to avoid floating-point
        # precision causing e.g. 127.9999 to snap to 96 instead of 128.
        clamped = min(int(round(raw)), median_resampled_size[i])
        patch.append(_largest_multiple_of_32_leq(clamped))
    return patch


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
                "composite_loss_weighting": None,
            },
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "lr_scheduler": "cosine",
            "warmup_epochs": 20,
            "l2_penalty": 0.0001,
            "grad_clip_norm": 1.0,
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
