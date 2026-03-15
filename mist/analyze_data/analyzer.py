"""Analyzer class for MIST.

This module contains the Analyzer class, which is responsible for analyzing
the dataset and preparing the configuration file for MIST. It checks the
dataset information, validates the dataset, checks if cropping to the
foreground bounding box is beneficial, checks the non-zero ratio of the images,
determines the target spacing, checks the resampled dimensions, and computes
the normalization parameters for CT images if applicable. It also saves the
configuration file and the paths dataframe to the results directory, which will
be used for preprocessing and training models.
"""
import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from importlib import metadata

import ants
import pandas as pd
import numpy as np
import rich

# MIST imports.
from mist.utils import io, progress_bar
from mist.preprocessing import preprocessing_utils
from mist.analyze_data import analyzer_utils
from mist.analyze_data.analyzer_constants import AnalyzeConstants as constants
from mist.analyze_data.data_dumper import DataDumper


def _welford_merge(
    stats: Iterable[tuple[int, float, float]],
) -> tuple[int, float, float]:
    """Merge per-group statistics using the parallel Welford algorithm.

    Each group contributes a tuple of (n, mean, M2) where M2 is the sum of
    squared deviations from that group's mean. Groups with n == 0 are skipped.
    The returned standard deviation uses ddof=0 (population std), consistent
    with numpy's default.

    Args:
        stats: Iterable of (n, mean, M2) tuples.

    Returns:
        Tuple of (total_n, total_mean, total_std).
    """
    total_n, total_mean, total_M2 = 0, 0.0, 0.0
    for n_i, mean_i, M2_i in stats:
        if n_i == 0:
            continue
        if total_n == 0:
            total_n, total_mean, total_M2 = n_i, float(mean_i), float(M2_i)
        else:
            delta = float(mean_i) - total_mean
            total_M2 = (
                total_M2
                + float(M2_i)
                + delta ** 2 * total_n * n_i / (total_n + n_i)
            )
            total_mean = (
                (total_mean * total_n + float(mean_i) * n_i)
                / (total_n + n_i)
            )
            total_n += n_i
    total_std = float(np.sqrt(total_M2 / total_n)) if total_n > 0 else 0.0
    return total_n, total_mean, total_std


def _percentile_from_histogram(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    percentile: float,
) -> float:
    """Estimate a percentile value from a pre-built histogram.

    Locates the first bin whose cumulative count reaches the requested
    percentile threshold and returns the left edge of that bin. With
    CT_HU_HIST_BINS = 4096 bins over the full HU range this gives ~1 HU
    resolution, which is more than sufficient for windowing parameters.

    Args:
        hist: Integer array of bin counts, length N.
        bin_edges: Float array of bin edges, length N + 1.
        percentile: Percentile to estimate, in [0, 100].

    Returns:
        Estimated percentile value as float. Returns 0.0 if hist is empty.
    """
    total = int(hist.sum())
    if total == 0:
        return 0.0
    cumsum = np.cumsum(hist)
    threshold = percentile / 100.0 * total
    idx = int(np.searchsorted(cumsum, threshold))
    idx = min(idx, len(bin_edges) - 2)
    return float(bin_edges[idx])


class Analyzer:
    """Analyzer class for getting config.json file for MIST.

    Attributes:
        mist_arguments: MIST arguments.
        dataset_info: Dataset information from MIST arguments.
        config: Configuration dictionary.
        paths_df: Dataframe containing paths to images and masks.
        console: Rich console for printing messages.
    """

    def __init__(self, mist_arguments: argparse.Namespace) -> None:
        # Initialize the rich console for printing messages.
        self.console = rich.console.Console()

        # Get MIST command line arguments.
        self.mist_arguments = mist_arguments

        # Read the dataset information from the JSON file and validate it.
        self.dataset_info = io.read_json_file(self.mist_arguments.data)
        self._check_dataset_info()

        # Load the base configuration file.
        self.config = analyzer_utils.build_base_config()

        # Initialize the dataframe with the file paths for the images and masks.
        self.paths_df = analyzer_utils.get_files_df(
            self.mist_arguments.data, "train"
        )

        # Set file paths for saving files like the training paths,
        # foreground bounding boxes, and configuration file. Resolve so that
        # relative paths passed via the CLI become absolute Path objects.
        self.results_dir = Path(self.mist_arguments.results).resolve()
        self.paths_csv = self.results_dir / "train_paths.csv"
        self.fg_bboxes_csv = self.results_dir / "fg_bboxes.csv"
        self.config_json = self.results_dir / "config.json"

        # If the config.json file already exists, we will overwrite it and
        # print a warning to the console. This is to ensure that the user
        # is aware that the configuration file is being overwritten and that
        # they should check the new configuration file for any changes.
        if self.config_json.exists() and self.mist_arguments.overwrite:
            self.console.print(
                "[yellow]Overwriting existing configuration at "
                f"{self.config_json}[/yellow]"
            )

        # Number of parallel workers for analysis. Stored as an instance
        # attribute rather than in config so it is not persisted to
        # config.json — it is system-dependent and not needed to reproduce
        # analysis results.
        num_workers = getattr(
            self.mist_arguments, "num_workers_analyze", None
        )
        self.n_workers = int(num_workers) if num_workers is not None else 1

    def _check_dataset_info(self) -> None:
        """Check if the dataset description file is in the correct format.

        This function checks that the dataset description JSON file contains
        all the required fields and that they are in the correct format. It
        raises an error if any of the required fields are missing or if they
        are in the wrong format. The required fields are:
            - task: The name of the task to be performed, this could be
                something like 'brats-2025' or 'lits-tumor'.
            - modality: The modality of the images, i.e., 'ct', 'mr',
                or 'other'.
            - train-data: The path to the training data folder.
            - mask: The list of strings that identify the masks files in the
                training data folder.
            - images: A dictionary of the format {'image_type': [list of
                strings that identify the image files in the training data
                folder]}. The image_type can be anything like 'ct', 'mr',
                't1', 't2', etc.
            - labels: A list of integers that represent the labels in the
                dataset. This list must contain zero as a label.
            - final_classes: A dictionary of the format {class_name: [list of
                labels]}. The class_name can be anything like 'tumor', 'edema',
                'necrosis', etc. The list of labels can contain multiple labels
                for a single class, i.e., {'tumor': [1, 2, 3], 'edema': [4]}.
        """
        required_fields = [
            "task",
            "modality",
            "train-data",
            "mask",
            "images",
            "labels",
            "final_classes",
        ]
        for field in required_fields:
            # Check that the required fields are in the JSON file.
            if field not in self.dataset_info:
                raise KeyError(
                    f"Dataset description JSON file must contain a "
                    f"entry '{field}'. There is no '{field}' in the "
                    "JSON file."
                )

            # Check that the required fields are not None.
            if self.dataset_info[field] is None:
                raise ValueError(
                    f"Dataset description JSON file must contain a "
                    f"entry '{field}'. Got None for '{field}' in the "
                    "JSON file."
                )

            # Check that the train data folder exists and is not empty.
            # Relative paths are resolved relative to the dataset JSON file so
            # that the JSON and its data can be moved together without needing
            # to adjust the working directory.
            if field == "train-data":
                train_data_path = (
                    Path(self.mist_arguments.data).resolve().parent
                    / self.dataset_info[field]
                ).resolve()
                if not train_data_path.exists():
                    raise FileNotFoundError(
                        "In the 'train-data' entry, the directory does "
                        "not exist. No such file or directory: "
                        f"{self.dataset_info[field]}"
                    )

                if not any(train_data_path.iterdir()):
                    raise FileNotFoundError(
                        "In the 'train-data' entry, the directory is empty: "
                        f"{self.dataset_info[field]}"
                    )

            # Check that the mask entry is a list and not empty.
            if field == "mask":
                if not isinstance(self.dataset_info[field], list):
                    raise TypeError(
                        "The 'mask' entry must be a list of mask names in "
                        "the dataset description JSON file. Found the "
                        "following entry instead: "
                        f"{self.dataset_info[field]}."
                    )

                if not self.dataset_info[field]:
                    raise ValueError(
                        "The 'mask' entry is empty. Please provide a list of "
                        "mask names in the dataset description JSON file."
                    )

            # Check that the images entry is a dictionary and not empty.
            if field == "images":
                if not isinstance(self.dataset_info[field], dict):
                    raise TypeError(
                        "The 'images' entry must be a dictionary of the "
                        "format 'image_type': [list of image names] in "
                        "the dataset description JSON file."
                    )

                if not self.dataset_info[field]:
                    raise ValueError(
                        "The 'images' entry is empty. Please provide a "
                        "dictionary of the format "
                        "{'image_type': [list of image names]} in the "
                        "dataset description JSON file."
                    )

            # Check that the labels entry is a list and not empty. Also check
            # that zero is an entry in the labels list.
            if field == "labels":
                if not isinstance(self.dataset_info[field], list):
                    raise TypeError(
                        "The 'labels' entry must be a list of labels in "
                        "the dataset. This list must contain zero as a "
                        "label. Found the following entry instead: "
                        f"{self.dataset_info[field]}."
                    )

                if not self.dataset_info[field]:
                    raise ValueError(
                        "The 'labels' entry must be a list of labels in the "
                        "dataset. This list must contain zero as a label. The "
                        "list is empty."
                    )

                if 0 not in self.dataset_info[field]:
                    raise ValueError(
                        "The 'labels' entry must be a list of labels in the "
                        "dataset. This list must contain zero as a label. No "
                        "zero label found in the list."
                    )

            # Check that the modality is one of the allowed values.
            if field == "modality":
                allowed_modalities = {"ct", "mr", "other"}
                if self.dataset_info[field].lower() not in allowed_modalities:
                    raise ValueError(
                        f"The 'modality' entry must be one of "
                        f"{sorted(allowed_modalities)}. Got: "
                        f"'{self.dataset_info[field]}'."
                    )

            # Check that the final classes entry is a dictionary and not empty.
            if field == "final_classes":
                if not isinstance(self.dataset_info[field], dict):
                    raise TypeError(
                        "The 'final_classes' entry must be a dictionary "
                        "of the format {class_name: [list of labels]}. "
                        "Found the following entry instead: "
                        f"{self.dataset_info[field]}."
                    )

                if not self.dataset_info[field]:
                    raise ValueError(
                        "The 'final_classes' entry must be a dictionary "
                        "of the format {class_name: [list of labels]}. "
                        "The dictionary is empty."
                    )

        # Cross-validate: every label referenced in final_classes must appear
        # in labels. This is checked after the loop so both fields are known
        # to be valid lists/dicts before we compare them.
        all_labels = set(self.dataset_info["labels"])
        for class_name, class_labels in (
            self.dataset_info["final_classes"].items()
        ):
            unknown = [l for l in class_labels if l not in all_labels]
            if unknown:
                raise ValueError(
                    f"In 'final_classes', class '{class_name}' contains "
                    f"label(s) {unknown} that are not present in 'labels' "
                    f"{self.dataset_info['labels']}."
                )

    def check_crop_fg(self) -> tuple[bool, np.ndarray]:
        """Check if cropping to FG reduces image volume by at least 20%.

        This function checks if cropping the images to the foreground
        bounding box reduces the image volume by at least 20%. It computes
        the bounding box for the foreground mask of each image and calculates
        the volume reduction. If the average volume reduction is greater than
        or equal to 20%, it returns True, indicating that cropping to the
        foreground is beneficial. It also saves the bounding box information
        to a CSV file.

        To compute the foreground bounding box, it uses the
        `get_fg_mask_bbox`, which uses an Otsu threshold method to find the
        foreground mask and then computes the bounding box around the
        non-zero voxels of the foreground mask.
        """
        def _process(patient):
            try:
                image_list = list(patient.values())[3:]
                image = ants.image_read(image_list[0])
                fg_bbox = preprocessing_utils.get_fg_mask_bbox(image)
                cropped_dims_i = [
                    fg_bbox["x_end"] - fg_bbox["x_start"] + 1,
                    fg_bbox["y_end"] - fg_bbox["y_start"] + 1,
                    fg_bbox["z_end"] - fg_bbox["z_start"] + 1,
                ]
                vol_reduction_i = (
                    1.0 - (np.prod(cropped_dims_i) / np.prod(image.shape))
                )
                fg_bbox["id"] = patient["id"]
                return fg_bbox, cropped_dims_i, vol_reduction_i
            except Exception as e:
                raise RuntimeError(
                    f"Error processing patient '{patient['id']}': {e}"
                ) from e

        n_workers = self.n_workers
        patients = [
            self.paths_df.iloc[i].to_dict()
            for i in range(len(self.paths_df))
        ]
        fg_bbox_records = [None] * len(patients)
        cropped_dims = np.zeros((len(patients), 3))
        vol_reduction = [0.0] * len(patients)

        progress = progress_bar.get_progress_bar("Checking FG vol. reduction")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process, p): i
                for i, p in enumerate(patients)
            }
            with progress as pb:
                for future in pb.track(
                    as_completed(futures), total=len(patients)
                ):
                    i = futures[future]
                    fg_bbox_records[i], cropped_dims[i, :], vol_reduction[i] = (
                        future.result()
                    )

        pd.DataFrame(fg_bbox_records).to_csv(self.fg_bboxes_csv, index=False)
        crop_to_fg = (
            np.mean(vol_reduction)
            >= constants.MIN_AVERAGE_VOLUME_REDUCTION_FRACTION
        )
        return crop_to_fg, cropped_dims

    def check_nz_ratio(self) -> bool:
        """Check if 20% or less of the image is non-zero.

        This function checks the fraction of non-zero voxels in the images
        to zero-valued voxels. If, on average, less than 20% of the voxels
        in an image are non-zero, then the dataset is considered sparse, and
        preprocessing will compute normalization parameters (for non CT
        cases) and apply the normalization scheme only to the non-zero
        voxels.
        """
        def _process(patient):
            try:
                image_list = list(patient.values())[3:]
                image = ants.image_read(image_list[0])
                return float(np.sum(image.numpy() != 0) / np.prod(image.shape))
            except Exception as e:
                raise RuntimeError(
                    f"Error processing patient '{patient['id']}': {e}"
                ) from e

        n_workers = self.n_workers
        patients = [
            self.paths_df.iloc[i].to_dict()
            for i in range(len(self.paths_df))
        ]
        nz_ratio = [0.0] * len(patients)

        progress = progress_bar.get_progress_bar("Checking non-zero ratio")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process, p): i
                for i, p in enumerate(patients)
            }
            with progress as pb:
                for future in pb.track(
                    as_completed(futures), total=len(patients)
                ):
                    nz_ratio[futures[future]] = future.result()

        use_nz_mask = (
            (1.0 - np.mean(nz_ratio)) >= constants.MIN_SPARSITY_FRACTION
        )
        return use_nz_mask

    def get_target_spacing(self) -> list[float]:
        """Get target spacing for preprocessing.

        Compute the target spacing for the dataset based on the median
        spacing along each axis for the images in the dataset. If this
        median-based spacing is anisotropic (i.e., the ratio of the maximum
        to minimum spacing is greater than a threshold), then adjust the
        coarsest resolution to bring the ratio down. This is done to ensure
        that we still have a reasonable resolution when we preprocess the
        data.
        """
        def _process(patient):
            try:
                mask = ants.image_read(patient["mask"])
                mask = ants.reorient_image2(mask, "RAI")
                mask.set_direction(constants.RAI_ANTS_DIRECTION)
                return tuple(mask.spacing)
            except Exception as e:
                raise RuntimeError(
                    f"Error processing patient '{patient['id']}': {e}"
                ) from e

        n_workers = self.n_workers
        patients = [
            self.paths_df.iloc[i].to_dict()
            for i in range(len(self.paths_df))
        ]
        spacings = [None] * len(patients)

        progress = progress_bar.get_progress_bar("Getting target spacing")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process, p): i
                for i, p in enumerate(patients)
            }
            with progress as pb:
                for future in pb.track(
                    as_completed(futures), total=len(patients)
                ):
                    spacings[futures[future]] = future.result()

        original_spacings = np.array(spacings, dtype=float)

        # Initialize target spacing.
        target_spacing = list(np.median(original_spacings, axis=0))

        # If anisotropic, adjust the coarsest resolution to bring ratio down.
        if (
            np.max(target_spacing) / np.min(target_spacing)
            > constants.MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD
        ):
            low_res_axis = np.argmax(target_spacing)
            target_spacing[low_res_axis] = np.percentile(
                original_spacings[:, low_res_axis],
                constants.ANISOTROPIC_LOW_RESOLUTION_AXIS_PERCENTILE,
            )
        return target_spacing

    def check_resampled_dims(self, cropped_dims: np.ndarray) -> list[float]:
        """Determine dimensions of resampled data.

        After we've determined the target spacing, we can compute the
        dimensions of the resampled data. This gives us a median image size
        which we can use to determine the patch size for training. We also
        check if the resampled image size is larger than the recommended
        memory size. If it is, we warn the user and suggest that they
        coarsen the resolution or remove the example from the dataset. This
        is done to avoid running out of memory during training, as the
        resampled image size can be quite large, especially for 3D images
        with multiple channels.

        Additionally, if we determine that we crop to the foreground
        bounding box, we use the cropped dimensions to compute the resampled
        dimensions. If we do not crop to the foreground, we use the original
        mask dimensions to compute the resampled dimensions. This is because
        cropping to the foreground can significantly reduce the size of the
        resampled image, which can help with memory usage during training.
        """
        # Guard: these keys are written by analyze_dataset. A missing key means
        # check_resampled_dims was called out of order.
        required_keys = ("crop_to_foreground", "target_spacing")
        preprocessing = self.config.get("preprocessing", {})
        missing = [k for k in required_keys if k not in preprocessing]
        if missing:
            raise RuntimeError(
                f"check_resampled_dims requires config keys "
                f"{list(required_keys)} under 'preprocessing', but "
                f"{missing} are missing. Ensure analyze_dataset has been "
                "called before check_resampled_dims."
            )

        # Capture config values in local variables so the closure does not
        # hold a reference to self (keeps the worker lightweight).
        crop_to_fg = bool(self.config["preprocessing"]["crop_to_foreground"])
        tgt_spacing = self.config["preprocessing"]["target_spacing"]
        n_labels = len(self.dataset_info["labels"])

        def _process(patient, cropped_dims_i):
            try:
                mask_header = ants.image_header_info(patient["mask"])
                image_list = list(patient.values())[3:]
                current_dims = (
                    cropped_dims_i if crop_to_fg else mask_header["dimensions"]
                )
                current_spacing = mask_header["spacing"]
                new_dims = analyzer_utils.get_resampled_image_dimensions(
                    current_dims, current_spacing, tgt_spacing
                )
                image_memory_size = (
                    analyzer_utils.get_float32_example_memory_size(
                        new_dims, len(image_list), n_labels
                    )
                )
                msg = None
                if image_memory_size > constants.MAX_RECOMMENDED_MEMORY_SIZE:
                    msg = (
                        f"[yellow][Warning] In {patient['id']}: Resampled "
                        "example is larger than the recommended memory size "
                        f"of {constants.MAX_RECOMMENDED_MEMORY_SIZE / 1e9} "
                        "GB. Consider coarsening or removing this "
                        "example.[/yellow]"
                    )
                return new_dims, msg
            except Exception as e:
                raise RuntimeError(
                    f"Error processing patient '{patient['id']}': {e}"
                ) from e

        n_workers = self.n_workers
        patients = [
            self.paths_df.iloc[i].to_dict()
            for i in range(len(self.paths_df))
        ]
        resampled_dims = np.zeros((len(patients), 3))
        messages = [None] * len(patients)

        progress = progress_bar.get_progress_bar(
            "Checking resampled dimensions"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process, patients[i], cropped_dims[i, :]): i
                for i in range(len(patients))
            }
            with progress as pb:
                for future in pb.track(
                    as_completed(futures), total=len(patients)
                ):
                    i = futures[future]
                    resampled_dims[i, :], messages[i] = future.result()

        for msg in messages:
            if msg:
                self.console.print(msg)

        return list(np.median(resampled_dims, axis=0))

    def get_ct_normalization_parameters(self) -> dict[str, float]:
        """Get windowing and normalization parameters for CT images.

        CT images are treated differently than other modalities since the
        voxel intensities in CT images are physically meaningful (i.e.,
        Hounsfield units). Therefore, we compute the normalization
        parameters (global z-score mean and standard deviation) based on
        the foreground intensities in the CT images. We also compute the
        global window range based on the foreground intensities.

        Rather than accumulating all subsampled voxel intensities across the
        dataset (which can reach several GB for large CT datasets), each
        worker returns lightweight per-patient summary statistics:
          - (n, mean, M2) for exact mean/std via parallel Welford's algorithm.
          - A histogram over the clinical HU range for percentile estimation.
        These are merged in the main thread with no per-voxel storage.
        """
        _hist_bin_edges = np.linspace(
            constants.CT_HU_HIST_MIN,
            constants.CT_HU_HIST_MAX,
            constants.CT_HU_HIST_BINS + 1,
        )

        def _process(patient):
            try:
                image_list = list(patient.values())[3:]
                image = ants.image_read(image_list[0])
                mask = ants.image_read(patient["mask"])
                arr = np.asarray(
                    (image[mask != 0]).tolist()[  # type: ignore
                        ::constants.CT_GATHER_EVERY_ITH_VOXEL_VALUE
                    ],
                    dtype=np.float64,
                )
                if len(arr) == 0:
                    return (
                        0, 0.0, 0.0,
                        np.zeros(constants.CT_HU_HIST_BINS, dtype=np.int64),
                        0,
                    )
                n = len(arr)
                mean = float(np.mean(arr))
                M2 = float(np.sum((arr - mean) ** 2))
                hist, _ = np.histogram(arr, bins=_hist_bin_edges)
                n_out_of_range = int(np.sum(
                    (arr < constants.CT_HU_HIST_MIN)
                    | (arr > constants.CT_HU_HIST_MAX)
                ))
                return n, mean, M2, hist.astype(np.int64), n_out_of_range
            except Exception as e:
                raise RuntimeError(
                    f"Error processing patient '{patient['id']}': {e}"
                ) from e

        n_workers = self.n_workers
        patients = [
            self.paths_df.iloc[i].to_dict()
            for i in range(len(self.paths_df))
        ]
        per_patient = [None] * len(patients)

        progress = progress_bar.get_progress_bar("Getting CT norm. params.")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process, p): i
                for i, p in enumerate(patients)
            }
            with progress as pb:
                for future in pb.track(
                    as_completed(futures), total=len(patients)
                ):
                    per_patient[futures[future]] = future.result()

        # Merge per-patient statistics — no per-voxel list required.
        welford_stats = [
            (n_i, mean_i, M2_i)
            for n_i, mean_i, M2_i, _, _ in per_patient
        ]
        combined_hist = np.sum(
            [hist_i for _, _, _, hist_i, _ in per_patient], axis=0
        )
        total_out_of_range = sum(r[4] for r in per_patient)
        if total_out_of_range > 0:
            self.console.print(
                f"[yellow][Warning] {total_out_of_range:,} foreground voxels "
                "had HU values outside the histogram range "
                f"[{constants.CT_HU_HIST_MIN:.0f}, "
                f"{constants.CT_HU_HIST_MAX:.0f}]. These voxels are included "
                "in mean/std but excluded from window bound "
                "estimation.[/yellow]"
            )
        _, global_z_score_mean, global_z_score_std = _welford_merge(
            welford_stats
        )
        global_window_range_min = _percentile_from_histogram(
            combined_hist, _hist_bin_edges, constants.CT_GLOBAL_CLIP_MIN_PERCENTILE
        )
        global_window_range_max = _percentile_from_histogram(
            combined_hist, _hist_bin_edges, constants.CT_GLOBAL_CLIP_MAX_PERCENTILE
        )
        return {
            "window_min": float(global_window_range_min),
            "window_max": float(global_window_range_max),
            "z_score_mean": float(global_z_score_mean),
            "z_score_std": float(global_z_score_std),
        }

    def analyze_dataset(self) -> None:
        """Analyze dataset and prepare configuration file.

        This function analyzes the dataset to prepare the configuration
        file for training. It checks if the dataset is sparse, if cropping
        to the foreground bounding box is beneficial, and which target
        spacing to use, and the normalization parameters for CT images if
        applicable. It updates the configuration dictionary with these
        parameters and saves it to a JSON file. It also sets the number of
        channels and classes based on the dataset information. The
        configuration file is used by for preprocessing and training the
        model.
        """
        # Add current MIST version to the configuration.
        self.config["mist_version"] = metadata.version("mist-medical")

        # Update the dataset information in the configuration.
        self.config["dataset_info"]["task"] = self.dataset_info["task"]
        self.config["dataset_info"]["modality"] = (
            self.dataset_info["modality"].lower()
        )
        # Store the names of the images for more robust inference later. This
        # allows us to ensure that the order of the images is correct for new
        # test data.
        self.config["dataset_info"]["images"] = (
            list(self.dataset_info["images"].keys())
        )
        self.config["dataset_info"]["labels"] = self.dataset_info["labels"]

        # Get the target spacing for the dataset.
        target_spacing = self.get_target_spacing()
        self.config["preprocessing"]["target_spacing"] = [
            float(spacing) for spacing in target_spacing
        ]

        # Check if cropping to the foreground bounding box is beneficial.
        crop_to_fg, cropped_dims = self.check_crop_fg()
        self.config["preprocessing"]["crop_to_foreground"] = bool(crop_to_fg)
        # Store cropped dims so run() can pass them to DataDumper for accurate
        # vol-fraction-of-image computation when crop_to_foreground is True.
        self.cropped_dims = cropped_dims

        # Get the resampled and possible cropped dimensions of the images.
        median_dims = self.check_resampled_dims(cropped_dims)
        self.config["preprocessing"]["median_resampled_image_size"] = [
            int(dim) for dim in median_dims
        ]

        # Check if the images are sparse, i.e., if 20% or less of the image is
        # non-zero.
        normalize_with_nz_mask = self.check_nz_ratio()
        self.config["preprocessing"]["normalize_with_nonzero_mask"] = (
            bool(normalize_with_nz_mask)
        )

        # If we are using CT images, we need to get the normalization
        # parameters for CT images and update the configuration.
        if self.config["dataset_info"]["modality"] == "ct":
            # Get CT normalization parameters.
            ct_normalization_parameters = (
                self.get_ct_normalization_parameters()
            )
            self.config["preprocessing"]["ct_normalization"].update(
                ct_normalization_parameters
            )

        # Update the number of channels and classes in the model section of the
        # configuration.
        self.config["model"]["params"]["in_channels"] = len(
            self.config["dataset_info"]["images"]
        )
        self.config["model"]["params"]["out_channels"] = len(
            self.config["dataset_info"]["labels"]
        )

        # Set a default patch size based on the median resampled image size.
        # The patch size can be overridden by the user in the config file or in
        # the command line arguments for the training pipeline.
        patch_size = analyzer_utils.get_best_patch_size(median_dims)
        self.config["model"]["params"]["patch_size"] = [
            int(size) for size in patch_size
        ]

        # Set the patch size for inference to be the same as training.
        self.config["inference"]["inferer"]["params"]["patch_size"] = (
            self.config["model"]["params"]["patch_size"]
        )

        # Add the target spacing to the model parameters in the model section
        # of the configuration. This is already in the preprocessing section
        # of the configuration, but this makes loading models and keeping track
        # of model-specific parameters easier.
        self.config["model"]["params"]["target_spacing"] = (
            self.config["preprocessing"]["target_spacing"]
        )

        # Build and add the evaluation metrics to the evaluation section of the
        # configuration.
        evaluation_config = analyzer_utils.build_evaluation_config(
            self.dataset_info
        )
        self.config.update(evaluation_config)

    def validate_dataset(self) -> None:
        """Check if headers match, images are 3D, and create paths dataframe.

        This runs basic checks on the dataset to ensure that the dataset is
        valid for training. It checks if the headers of the images and masks
        match according to their dimensions, origin, and spacing. It also
        checks that all images are 3D and that the mask is 3D. If there are
        multiple images, it checks that they have the same header
        information. If any of these checks fail, the patient is excluded
        from training.
        """
        dataset_labels_set = set(self.dataset_info["labels"])
        data_path = self.mist_arguments.data

        def _validate(patient):
            """Validate one patient. Returns (is_bad, message_or_None)."""
            try:
                # Patient values are ["id", "mask", "image_1", "image_2", ...].
                image_list = list(patient.values())[2:]
                mask = ants.image_read(patient["mask"])
                mask_labels = set(mask.unique().astype(int))
                mask_header = ants.image_header_info(patient["mask"])
                image_header = ants.image_header_info(image_list[0])  # noqa: F841

                if not mask_labels.issubset(dataset_labels_set):
                    return True, (
                        f"[red]In {patient['id']}: Labels in mask do not "
                        f"match  those specified in {data_path}[/red]"
                    )

                if not analyzer_utils.is_image_3d(mask_header):
                    return True, (
                        f"[red]In {patient['id']}: Got 4D mask, make sure all"
                        "images are 3D[/red]"
                    )

                for image_path in image_list:
                    image_header = ants.image_header_info(image_path)
                    if not analyzer_utils.compare_headers(
                        mask_header, image_header
                    ):
                        return True, (
                            f"[red]In {patient['id']}: Mismatch between image "
                            " and mask header information[/red]"
                        )
                    if not analyzer_utils.is_image_3d(image_header):
                        return True, (
                            f"[red]In {patient['id']}: Got 4D image, make"
                            " sure all images are 3D[/red]"
                        )

                if len(image_list) > 1:
                    anchor_header = ants.image_header_info(image_list[0])
                    for image_path in image_list[1:]:
                        image_header = ants.image_header_info(image_path)
                        if not analyzer_utils.compare_headers(
                            anchor_header, image_header
                        ):
                            return True, (
                                f"[red]In {patient['id']}: Mismatch between "
                                "images' header information[/red]"
                            )

                return False, None

            except Exception as e:
                return True, (
                    f"[red]In {patient['id']}: {e}[/red]"
                )

        n_workers = self.n_workers
        patients = [
            self.paths_df.iloc[i].to_dict()
            for i in range(len(self.paths_df))
        ]
        is_bad = [False] * len(patients)
        messages = [None] * len(patients)

        progress = progress_bar.get_progress_bar("Verifying dataset")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_validate, p): i
                for i, p in enumerate(patients)
            }
            with progress as pb:
                for future in pb.track(
                    as_completed(futures), total=len(patients)
                ):
                    i = futures[future]
                    is_bad[i], messages[i] = future.result()

        bad_data = {i for i, bad in enumerate(is_bad) if bad}

        # Print any validation messages and report excluded patients.
        for msg in messages:
            if msg:
                self.console.print(msg)
        if bad_data:
            self.console.print(
                f"[bold red]Excluding {len(bad_data)} example(s) from "
                "training.[/bold red]"
            )

        # If all of the data is bad, then raise an error.
        if len(bad_data) >= len(self.paths_df):
            raise RuntimeError(
                "All examples were excluded from training. "
                "Please check your data."
            )

        # Drop bad data from paths dataframe and reset index.
        rows_to_drop = self.paths_df.index[list(bad_data)]
        self.paths_df.drop(rows_to_drop, inplace=True)
        self.paths_df.reset_index(drop=True, inplace=True)

    def run(self) -> None:
        """Run the analyzer to get configuration file."""
        text = rich.text.Text("\nAnalyzing dataset\n")  # type: ignore
        text.stylize("bold")
        self.console.print(text)

        # Step 1: Run the dataset validation checks to clean up the paths
        # dataframe and ensure that the dataset is valid for training.
        self.validate_dataset()

        # Step 2: Add folds to the paths dataframe and update the
        # configuration with the number of folds that we are using.
        if self.mist_arguments.nfolds is not None:
            self.config["training"]["nfolds"] = int(
                self.mist_arguments.nfolds
            )
        self.paths_df = analyzer_utils.add_folds_to_df(
            self.paths_df, n_splits=self.config["training"]["nfolds"]
        )

        # By default, we assume that we are running all folds for training.
        # This can be overridden by the user in the config file or in the
        # command line arguments for the training pipeline.
        self.config["training"]["folds"] = (
            list(range(self.config["training"]["nfolds"]))
        )

        # Step 3: Analyze the dataset to prepare the configuration file.
        self.analyze_dataset()

        # Step 4: Save the configuration file and the paths dataframe.
        self.paths_df.to_csv(self.paths_csv, index=False)
        io.write_json_file(self.config_json, self.config)

        # Step 5: Build and save the rich data dump (data_dump.json and
        # data_dump.md) alongside the configuration file.
        data_dumper = DataDumper(
            paths_df=self.paths_df,
            dataset_info=self.dataset_info,
            config=self.config,
            results_dir=self.results_dir,
            cropped_dims=self.cropped_dims,
        )
        data_dumper.run()

        # Step 6: If the user specified test data in the dataset JSON file,
        # create a test paths dataframe and save it as CSV.
        if self.dataset_info.get("test-data"):
            test_data_dir = (
                Path(self.mist_arguments.data).resolve().parent
                / self.dataset_info["test-data"]
            ).resolve()
            if not test_data_dir.exists():
                raise FileNotFoundError(
                    f"Test data directory does not exist: {test_data_dir}"
                )

            # Create a test paths dataframe from the test data directory.
            test_paths_df = analyzer_utils.get_files_df(
                self.mist_arguments.data, "test"
            )

            test_paths_csv = self.results_dir / "test_paths.csv"
            test_paths_df.to_csv(test_paths_csv, index=False)
