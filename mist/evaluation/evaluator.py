"""Evaluation class for computing segmentation accuracy metrics."""

import concurrent.futures
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import ants
import numpy as np
import pandas as pd
from mist.analyze_data import analyzer_utils
from mist.evaluation import evaluation_utils
from mist.metrics.metrics_registry import get_metric
from mist.utils import progress_bar
from mist.utils.console import print_warning, print_success, print_error


class Evaluator:
    """Evaluator class for computing segmentation accuracy metrics.

    This class evaluates segmentation predictions against ground truth masks
    using various metrics defined in a class-specific configuration format.
    Evaluation is parallelized to significantly speed up processing over large
    datasets while maintaining memory safety.

    Attributes:
        filepaths_dataframe: DataFrame indexed by 'id', containing 'mask' and
            'prediction' columns with paths to files.
        evaluation_config: Dictionary containing class names as keys and
            their respective 'labels' and 'metrics' configuration as values.
        output_csv_path: Path to save the evaluation results as a CSV file.
        results_dataframe: DataFrame storing the evaluation results.
    """

    def __init__(
        self,
        filepaths_dataframe: pd.DataFrame,
        evaluation_config: Dict[str, Any],
        output_csv_path: Union[str, Path],
        validate_masks: bool = False,
    ):
        """Initialize the Evaluator.

        Args:
            filepaths_dataframe: DataFrame containing columns 'id', 'mask', and
                'prediction'.
            evaluation_config: Dictionary mapping class names to their labels
                and metrics configurations.
            output_csv_path: Path where the CSV results will be saved.
            validate_masks: Opt-in flag to validate ground truth and prediction
                masks before evaluation. This will perform basic checks like
                verifying that images are 3D, checking if there are no
                unexpected labels, and that the dtype of the data is integer
                or boolean.
        """
        # 1. Validate columns first.
        df = self._validate_filepaths_dataframe(filepaths_dataframe)

        # 2. Check for duplicates and set index to 'id' for O(1) lookups.
        if df["id"].duplicated().any():
            raise ValueError(
                "Duplicate patient IDs found in DataFrame. IDs must be unique."
            )

        self.filepaths_dataframe = df.set_index("id")

        # 3. Validate and store the evaluation configuration.
        self.evaluation_config = self._validate_evaluation_config(
            evaluation_config
        )
        self.output_csv_path = Path(output_csv_path)

        # 4. Initialize the results DataFrame structure using the validated
        # config.
        self.results_dataframe = evaluation_utils.initialize_results_dataframe(
            self.evaluation_config
        )

        # Set opt-in validation flag. This is not necessary for MIST pipelines,
        # but should be used for evaluation on external data.
        self.validate_masks = validate_masks

    @staticmethod
    def _validate_filepaths_dataframe(
        filepaths_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Check if the filepaths DataFrame has the required columns."""
        required_columns = ["id", "mask", "prediction"]
        if not all(
            col in filepaths_dataframe.columns for col in required_columns
        ):
            raise ValueError(
                f"DataFrame must contain columns: {', '.join(required_columns)}"
            )
        return filepaths_dataframe.copy()

    @staticmethod
    def _validate_evaluation_config(
        evaluation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the evaluation configuration dictionary.

        Args:
            evaluation_config: Nested config mapping class names to labels and
                metrics.

        Returns:
            The validated configuration dictionary.

        Raises:
            ValueError: If structure is incorrect, or class labels are empty
                or non-positive.
        """
        # Validate the new format.
        for class_name, class_info in evaluation_config.items():
            if "labels" not in class_info or "metrics" not in class_info:
                raise ValueError(
                    f"Class '{class_name}' must contain both 'labels' and "
                    "'metrics' keys."
                )

            class_labels = class_info["labels"]
            if not isinstance(class_labels, list) or not class_labels:
                raise ValueError(
                    f"Class '{class_name}' must have a non-empty list of class "
                    "labels."
                )
            if any(label <= 0 for label in class_labels):
                raise ValueError(
                    f"Class labels for '{class_name}' must be greater than 0."
                )

            if not isinstance(class_info["metrics"], dict):
                raise ValueError(
                    f"Metrics for '{class_name}' must be a dictionary."
                )

        return evaluation_config

    @staticmethod
    def _compute_diagonal_distance(
        shape: Tuple[int, ...],
        spacing: Tuple[float, ...],
    ) -> float:
        """Compute the Euclidean diagonal distance of an image in mm."""
        dims_mm = np.multiply(shape, spacing)
        return np.linalg.norm(dims_mm).item()

    @staticmethod
    def _handle_edge_cases(
        num_mask_voxels: int,
        num_prediction_voxels: int,
        best_case_value: float,
        worst_case_value: float
    ) -> Union[float, None]:
        """Return best/worst case values for empty masks."""
        if num_mask_voxels == 0 and num_prediction_voxels == 0:
            return best_case_value
        if (num_mask_voxels == 0) ^ (num_prediction_voxels == 0):
            return worst_case_value
        return None

    def _load_patient_data(
        self,
        patient_id: str
    ) -> Dict[str, ants.core.ants_image.ANTsImage]:
        """Load the ground truth and prediction paths for a given patient ID."""
        try:
            row = self.filepaths_dataframe.loc[patient_id]
        except KeyError as e:
            raise ValueError(
                f"No data found for patient ID: {patient_id}. "
                f"See the following exception: {e}."
            ) from e

        row_data = row.to_dict()

        if not Path(row_data['mask']).exists():
            raise FileNotFoundError(f"Mask not found: {row_data['mask']}")
        if not Path(row_data['prediction']).exists():
            raise FileNotFoundError(
                f"Prediction not found: {row_data['prediction']}")

        # Optional validation: checks 3D shape, integer dtype, and valid labels.
        # Adds I/O overhead (extra image read) but catches bad data early.
        if self.validate_masks:
            mask_error = evaluation_utils.validate_mask(
                row_data['mask'],
                self.evaluation_config,
                mask_type="ground truth mask",
            )
            pred_error = evaluation_utils.validate_mask(
                row_data['prediction'],
                self.evaluation_config,
                mask_type="prediction",
            )
            errors = [e for e in (mask_error, pred_error) if e]
            if errors:
                raise ValueError(
                    f"Mask validation failed for {patient_id}: "
                    + " | ".join(errors)
                )

        # Validate headers before loading heavy image data.
        mask_header = ants.image_header_info(row_data['mask'])
        pred_header = ants.image_header_info(row_data['prediction'])

        if not analyzer_utils.compare_headers(mask_header, pred_header):
            raise ValueError(
                f"Header mismatch for {patient_id}. Ensure mask and prediction "
                "have identical geometry."
            )

        return {
            "mask": ants.image_read(row_data['mask']),
            "prediction": ants.image_read(row_data['prediction'])
        }

    def _compute_metrics(
        self,
        patient_id: str,
        mask: np.ndarray,
        prediction: np.ndarray,
        spacing: Tuple[float, ...],
        class_metrics_config: Dict[str, Dict[str, Any]],
        diagonal_distance_override: Optional[float] = None
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Compute metrics for a binary mask pair."""
        result = {}
        error_messages = []
        sum_of_mask = mask.sum()
        sum_of_prediction = prediction.sum()

        # Calculate worst-case distance (diagonal of the FOV).
        if diagonal_distance_override is not None:
            diagonal_distance_mm = diagonal_distance_override
        else:
            diagonal_distance_mm = self._compute_diagonal_distance(
                mask.shape, spacing
            )

        for metric_name, metric_kwargs in class_metrics_config.items():
            metric = get_metric(metric_name)

            # Determine worst-case value for this metric (e.g. inf or diagonal).
            worst = (
                diagonal_distance_mm if metric.worst == float("inf")
                else metric.worst
            )

            # Check for edge cases (empty masks).
            metric_value = self._handle_edge_cases(
                sum_of_mask, sum_of_prediction, metric.best, worst
            )

            if metric_value is not None:
                result[metric_name] = metric_value
            else:
                try:
                    # Unpack the specific kwargs mapped to this metric from the
                    # config.
                    val = metric(
                        mask, prediction, spacing, **metric_kwargs
                    )

                    # Sanity check for NaNs or Infs.
                    if np.isnan(val) or np.isinf(val):
                        error_messages.append(
                            f"{metric_name} returned NaN/Inf for {patient_id}. "
                            "Using worst-case value."
                        )
                        result[metric_name] = worst
                    else:
                        result[metric_name] = val

                except Exception as e:  # pylint: disable=broad-except
                    error_messages.append(
                        f"Error in {metric_name} for {patient_id}: {e}"
                    )
                    result[metric_name] = worst

        return result, "\n".join(error_messages) if error_messages else None

    def _evaluate_single_patient(
        self,
        patient_id: str,
        mask: np.ndarray,
        prediction: np.ndarray,
        spacing: Tuple[float, ...],
    ) -> Tuple[Dict[str, Union[str, float]], Optional[str]]:
        """Evaluate single patient and compute metrics for each class."""
        results: Dict[str, Union[str, float]] = {"id": patient_id}
        patient_errors = []

        full_diagonal_distance = self._compute_diagonal_distance(
            mask.shape, spacing
        )

        for class_name, class_info in self.evaluation_config.items():
            class_labels = class_info["labels"]
            class_metrics_config = class_info["metrics"]

            if len(class_labels) == 1:
                binary_sub_mask = mask == class_labels[0]
                binary_sub_prediction = prediction == class_labels[0]
            else:
                binary_sub_mask = np.isin(mask, class_labels)
                binary_sub_prediction = np.isin(prediction, class_labels)

            binary_sub_mask, binary_sub_prediction = (
                evaluation_utils.crop_to_union(
                    binary_sub_mask, binary_sub_prediction
                )
            )

            class_metrics, class_errs = self._compute_metrics(
                patient_id=patient_id,
                mask=binary_sub_mask,
                prediction=binary_sub_prediction,
                spacing=spacing,
                class_metrics_config=class_metrics_config,
                diagonal_distance_override=full_diagonal_distance,
            )

            # Flatten metrics into result dict (i.e., "Tumor_dice").
            for metric_name, value in class_metrics.items():
                results[f"{class_name}_{metric_name}"] = value

            if class_errs:
                patient_errors.append(f"[{class_name}] {class_errs}")

        return (
            results, "\n".join(patient_errors) if patient_errors else None
        )

    def _evaluate_patient_pipeline(
        self,
        patient_id: str
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Complete evaluation for a single patient to be run in parallel."""
        patient_data = None
        result = None
        patient_errors = None

        try:
            # 1. Load data.
            patient_data = self._load_patient_data(patient_id)
            spacing = patient_data["mask"].spacing

            # 2. Convert to numpy.
            mask_np = patient_data["mask"].numpy()
            pred_np = patient_data["prediction"].numpy()

            # 3. Compute metrics.
            result, errors = self._evaluate_single_patient(
                patient_id=patient_id,
                mask=mask_np,
                prediction=pred_np,
                spacing=spacing,
            )

            if errors:
                patient_errors = f"Patient {patient_id}: {errors}"

        except Exception as e:  # pylint: disable=broad-except
            patient_errors = f"CRITICAL FAILURE for {patient_id}: {str(e)}"

        finally:
            # Explicit memory cleanup is critical in multiprocessing.
            if patient_data:
                del patient_data
            gc.collect()

        return result, patient_errors

    def run(self, max_workers: int = 1) -> None:
        """Run evaluation over all patients and write results to CSV.

        Args:
            max_workers: The maximum number of parallel processes to use.
                Defaults to 1. Increase for faster evaluation on machines
                with many CPUs, but reduce if you encounter OOM errors.
        """
        all_error_messages = []
        results_list = []

        # Iterate over index (which is now 'id').
        patient_ids = self.filepaths_dataframe.index.tolist()

        # Execute in parallel
        with (
            concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            as executor
        ):
            # Submit all patient tasks to the executor.
            future_to_patient = {
                executor.submit(self._evaluate_patient_pipeline, pid): pid
                for pid in patient_ids
            }

            with progress_bar.get_progress_bar("Evaluating predictions") as pb:
                # Track them as they complete (order doesn't matter).
                for future in pb.track(
                    concurrent.futures.as_completed(future_to_patient),
                    total=len(patient_ids)
                ):
                    result, patient_errors = future.result()

                    if result is not None:
                        results_list.append(result)
                    if patient_errors:
                        all_error_messages.append(patient_errors)

        # Report errors.
        if all_error_messages:
            print_warning("\n".join(all_error_messages))

        # Create DataFrame.
        if results_list:
            self.results_dataframe = pd.DataFrame(
                results_list,
                columns=self.results_dataframe.columns
            )

        # Compute summary stats (Mean, Std, etc.).
        if not self.results_dataframe.empty:
            self.results_dataframe = evaluation_utils.compute_results_stats(
                self.results_dataframe
            )

        # Save to disk.
        self.results_dataframe.to_csv(self.output_csv_path, index=False)
        if self.results_dataframe.empty:
            print_error(
                "Evaluation produced no results. All patients failed. "
                f"Empty CSV saved to {self.output_csv_path}"
            )
        else:
            print_success(
                f"Evaluation complete. Results saved to {self.output_csv_path}"
            )
