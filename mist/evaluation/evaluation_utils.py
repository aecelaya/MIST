"""Utilities for evaluating predictions."""

import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import ants
import numpy as np
import pandas as pd


def validate_mask(
    mask_path: Union[str, Path],
    evaluation_config: Dict[str, Any],
    mask_type: str = "mask",
) -> Optional[str]:
    """Validate a mask (ground truth or prediction) for evaluation.

    Checks that the mask can be read, is 3D, has an integer or boolean dtype,
    and only contains labels defined in the evaluation config.

    Note: This function calls ants.image_read, which loads the full image into
    memory. When called from build_evaluation_dataframe, this is a sequential
    operation that runs before the parallel evaluation pipeline. For large
    datasets with large images, consider disabling validation via the
    validate=False flag in build_evaluation_dataframe if you trust your data.

    Args:
        mask_path: Path to the mask.
        evaluation_config: Evaluation config mapping class names to their
            labels and metrics configurations.
        mask_type: Label used in error messages to identify the mask type,
            e.g., "ground truth mask" or "prediction". Defaults to "mask".

    Returns:
        An error message string if validation fails, otherwise None.
    """
    try:
        # Check 3D first using only the header — cheap and avoids loading the
        # full image if it fails.
        header = ants.image_header_info(str(mask_path))
        if len(header["dimensions"]) != 3:
            return f"{mask_type} is not a 3D image."

        mask_np = ants.image_read(str(mask_path)).numpy()

        # Reject only if the mask contains fractional values, which indicates a
        # probability map rather than a label mask. Integer-valued floats
        # (e.g., float32 masks from BraTS or FSL) are accepted.
        if not (
            np.issubdtype(mask_np.dtype, np.integer)
            or np.issubdtype(mask_np.dtype, np.bool_)
        ) and not np.all(mask_np == np.floor(mask_np)):
            return (
                f"{mask_type} has dtype '{mask_np.dtype}' with non-integer "
                "values. Expected a label mask, not a probability map."
            )

        # Collect all valid labels from the evaluation config (background
        # label 0 is always valid).
        expected_labels = {0}
        for class_info in evaluation_config.values():
            expected_labels.update(class_info["labels"])

        unexpected = set(mask_np.astype(int).flat) - expected_labels
        if unexpected:
            return f"{mask_type} contains unexpected labels: {unexpected}."

    except RuntimeError as e:
        return f"Could not read {mask_type}: {e}"

    return None


def build_evaluation_dataframe(
    train_paths_csv: Union[str, Path],
    prediction_folder: Union[str, Path],
    evaluation_config: Optional[Dict[str, Any]] = None,
    validate: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Get DataFrame with columns 'id', 'mask', and 'prediction' for evaluation.

    This function matches predictions to ground truth masks based on the patient
    ID, skipping any rows where either file is missing. Optionally validates
    each mask pair before including them in the output.

    Args:
        train_paths_csv: Filepath for train_paths.csv in the MIST results
            folder.
        prediction_folder: Folder containing the predictions from either
            the MIST training pipeline or from the MIST Postprocessor.
        evaluation_config: Evaluation config mapping class names to their
            labels and metrics configurations. Required if validate=True.
        validate: If True, validate each mask pair before including them.
            Validation loads the full image to check dtype and labels, which
            adds sequential I/O overhead before the parallel evaluation
            pipeline. Disable if you trust your data. Defaults to False.

    Returns:
        Tuple:
            Dataframe with valid entries for evaluation.
            Any warning messages related to missing or invalid files.
    """
    if validate and evaluation_config is None:
        raise ValueError(
            "evaluation_config must be provided when validate=True."
        )

    # Initialize the error messages list.
    error_messages = []

    # Convert inputs to Path objects for robust handling.
    train_paths_csv = Path(train_paths_csv)
    prediction_folder = Path(prediction_folder)

    # Check if the train_paths.csv file exists.
    if not train_paths_csv.exists():
        error_messages.append(
            f"No train_paths.csv at {train_paths_csv}"
        )
        return pd.DataFrame(), "\n".join(error_messages)

    # Otherwise, read the CSV file and extract the patient IDs and file paths.
    train_paths_df = pd.read_csv(train_paths_csv)

    # Initialize an empty list to store the file paths for valid entries.
    filepaths = []

    for _, row in train_paths_df.iterrows():
        patient_id = str(row["id"])

        # Create Path objects.
        # We assume 'mask' in CSV is relative or absolute. Path() handles both.
        mask_path = Path(row["mask"])

        # Use the '/' operator for cleaner path joining.
        prediction_path = prediction_folder / f"{patient_id}.nii.gz"

        missing = []
        if not mask_path.exists():
            missing.append("mask")
        if not prediction_path.exists():
            missing.append("prediction")

        if missing:
            error_messages.append(
                f"Skipping ID '{patient_id}' due to missing "
                f"{', '.join(missing)} file(s)."
            )
            continue

        if validate:
            gt_error = validate_mask(
                mask_path, evaluation_config, mask_type="ground truth mask"
            )
            pred_error = validate_mask(
                prediction_path, evaluation_config, mask_type="prediction"
            )

            validation_errors = [e for e in (gt_error, pred_error) if e]
            if validation_errors:
                error_messages.append(
                    f"Skipping ID '{patient_id}': "
                    + " | ".join(validation_errors)
                )
                continue

        # Convert back to string for compatibility with ANTsPy/Evaluator.
        filepaths.append({
            "id": patient_id,
            "mask": str(mask_path),
            "prediction": str(prediction_path)
        })

    # Explicitly define columns if no valid files are found.
    if not filepaths:
        filepaths_df = pd.DataFrame(columns=["id", "mask", "prediction"])
    else:
        filepaths_df = pd.DataFrame(filepaths)

    return filepaths_df, "\n".join(error_messages) if error_messages else None


def initialize_results_dataframe(
    evaluation_config: Dict[str, Any]
) -> pd.DataFrame:
    """Initialize results dataframe from the evaluation configuration.

    Args:
        evaluation_config: Nested dictionary mapping class names to their 
            labels and metrics configurations.

    Returns:
        results_df: Initialized results dataframe with exact class_metric
            columns.
    """
    results_cols = ["id"]
    for class_name, class_info in evaluation_config.items():
        metrics = class_info.get("metrics", {})
        for metric_name in metrics.keys():
            results_cols.append(f"{class_name}_{metric_name}")

    results_df = pd.DataFrame(columns=results_cols)
    return results_df


def compute_results_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for results dataframe.

    Args:
        results_df: Dataframe containing the metrics for each class for each
            patient.

    Returns:
        results_df: Updated results dataframe with statistics added at the 
            bottom five rows.
    """
    stats_labels = [
        "Mean", "Std", "25th Percentile", "Median", "75th Percentile"
    ]

    def safe_stat(func, array):
        """Helper to compute stats safely ignoring all-NaN warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # If the column is entirely empty/NaN, return NaN instead of
            # crashing.
            if array.isna().all():
                return np.nan
            return func(array)

    stats_functions = [
        np.nanmean,
        np.nanstd,
        partial(np.nanpercentile, q=25),
        partial(np.nanpercentile, q=50),
        partial(np.nanpercentile, q=75),
    ]

    # Compute statistics for each column and create corresponding rows.
    # We skip the 'id' column (results_df.columns[1:]).
    stats_rows = [
        {
            "id": label, **{
                col: safe_stat(func, results_df[col])
                for col in results_df.columns[1:]
            }
        }
        for label, func in zip(stats_labels, stats_functions)
    ]

    results_df = pd.concat(
        [results_df, pd.DataFrame(stats_rows)], ignore_index=True
    )
    return results_df


def crop_to_union(
    mask: np.ndarray,
    prediction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop both arrays to the bounding box of their non-zero union.

    This significantly speeds up surface distance metrics (Hausdorff) by
    removing empty background space.

    Args:
        mask: Ground truth segmentation mask.
        prediction: Predicted segmentation mask.

    Returns:
        Cropped versions of mask and prediction, where the cropping is based on
        the bounding box that contains all non-zero voxels from both arrays.
    """
    # 1. Identify all non-zero voxels in either array.
    # Uses bitwise OR | to combine the boolean masks.
    coords = np.argwhere((mask > 0) | (prediction > 0))

    # 2. Handle edge case: Both arrays are empty.
    if coords.size == 0:
        return mask, prediction

    # 3. Find the bounding box corners.
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1  # +1 because slicing is exclusive

    # 4. Generate slices dynamically for N-dimensions (works for 2D or 3D).
    slices = tuple(
        slice(min_c, max_c)
        for min_c, max_c in zip(min_coords, max_coords)
    )

    return mask[slices], prediction[slices]
