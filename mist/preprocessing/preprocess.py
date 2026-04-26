"""Preprocessing functions for medical images and masks."""
import argparse
import concurrent.futures
from pathlib import Path
from typing import Any
import ants
import numpy as np
import numpy.typing as npt
import pandas as pd
import SimpleITK as sitk

# MIST imports.
from mist.utils import io, progress_bar
from mist.utils.console import print_section_header, print_warning, print_success
from mist.analyze_data import analyzer_utils
from mist.preprocessing import preprocessing_utils
from mist.preprocessing.preprocessing_constants import (
    PreprocessingConstants as pc
)


def resample_image(
    img_ants: ants.core.ants_image.ANTsImage,
    target_spacing: tuple[float, float, float],
    new_size: tuple[int, int, int] | None = None,
) -> ants.core.ants_image.ANTsImage:
    """Resample an image to a target spacing.

    Args:
        img_ants: Image as ANTs image.
        target_spacing: Target spacing as a tuple.
        new_size: New size of the image as a tuple or None.

    Returns:
        Resampled image as ANTs image.

    Raises:
        ValueError: If the low resolution axis is not an integer when
            resampling an anisotropic image.
    """
    # Convert ants image to sitk image. We do this because the resampling
    # function in SimpleITK is more robust and faster than the one in ANTs.
    img_sitk = preprocessing_utils.ants_to_sitk(img_ants)

    # Get new size if not provided. This is done to ensure that the image
    # is resampled to the correct dimensions.
    if new_size is None:
        new_size = analyzer_utils.get_resampled_image_dimensions(
            img_sitk.GetSize(), img_sitk.GetSpacing(), target_spacing
        )

    # Check if the image is anisotropic.
    anisotropic_results = preprocessing_utils.check_anisotropic(img_sitk)

    # If the image is anisotropic, we need to use an intermediate resampling
    # step to avoid artifacts. This step uses nearest neighbor interpolation
    # to resample the image to its new size along the low resolution axis.
    if anisotropic_results["is_anisotropic"]:
        if not isinstance(anisotropic_results["low_resolution_axis"], int):
            raise ValueError(
                "The low resolution axis must be an integer."
            )
        img_sitk = preprocessing_utils.aniso_intermediate_resample(
            img_sitk,
            new_size,
            target_spacing,
            int(anisotropic_results["low_resolution_axis"])
        )

    # Resample the image to the target spacing using B-spline interpolation.
    img_sitk = sitk.Resample(
        img_sitk,
        size=np.array(new_size).tolist(),
        transform=sitk.Transform(),
        interpolator=sitk.sitkBSpline,
        outputOrigin=img_sitk.GetOrigin(),
        outputSpacing=target_spacing,
        outputDirection=img_sitk.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img_sitk.GetPixelID(),
    )

    # Convert the resampled image back to ANTs image.
    return preprocessing_utils.sitk_to_ants(img_sitk)


def resample_mask(
    mask_ants: ants.core.ants_image.ANTsImage,
    labels: list[int],
    target_spacing: tuple[float, float, float],
    new_size: tuple[int, int, int] | None = None,
) -> ants.core.ants_image.ANTsImage:
    """Resample a mask to a target spacing.

    Args:
        mask_ants: Mask as ANTs image.
        labels: List of labels in the dataset.
        target_spacing: Target spacing as a tuple.
        new_size: New size of the mask as a tuple or None.

    Returns:
        Resampled mask as ANTs image.

    Raises:
        ValueError: If the low resolution axis is not an integer when
            resampling an anisotropic mask.
    """
    # Get mask as a series of onehot encoded series of sitk images.
    masks_sitk = preprocessing_utils.make_onehot(mask_ants, labels)
    if new_size is None:
        new_size = analyzer_utils.get_resampled_image_dimensions(
            masks_sitk[0].GetSize(), masks_sitk[0].GetSpacing(), target_spacing
        )

    # Check if the mask is anisotropic. Only do this for the first mask.
    anisotropic_results = preprocessing_utils.check_anisotropic(masks_sitk[0])

    # Resample each mask in the series. If the mask is anisotropic, we
    # need to use an intermediate resampling step to avoid artifacts. This
    # step uses nearest neighbor interpolation to resample the mask to its
    # new size along the low resolution axis.
    for i in range(len(labels)):
        if anisotropic_results["is_anisotropic"]:
            if not isinstance(anisotropic_results["low_resolution_axis"], int):
                raise ValueError(
                    "The low resolution axis must be an integer."
                )
            masks_sitk[i] = preprocessing_utils.aniso_intermediate_resample(
                masks_sitk[i],
                new_size,
                target_spacing,
                anisotropic_results["low_resolution_axis"]
            )

        # Use linear interpolation for each mask in the series. We use
        # linear interpolation to avoid artifacts in the mask.
        masks_sitk[i] = sitk.Resample(
            masks_sitk[i],
            size=np.array(new_size).tolist(),
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            outputOrigin=masks_sitk[i].GetOrigin(),
            outputSpacing=target_spacing,
            outputDirection=masks_sitk[i].GetDirection(),
            defaultPixelValue=0,
            outputPixelType=masks_sitk[i].GetPixelID()
        )

    # Use the argmax function to join the masks into a single mask.
    mask = preprocessing_utils.sitk_to_ants(sitk.JoinSeries(masks_sitk))
    mask = mask.numpy()
    mask = np.argmax(mask, axis=-1)

    # Set the target spacing, origin, and direction for the mask.
    mask = ants.from_numpy(data=mask.astype(np.float32))
    mask.set_spacing(target_spacing)
    mask.set_origin(mask_ants.origin)
    mask.set_direction(mask_ants.direction)
    return mask


def window_and_normalize(
    image: npt.NDArray[Any],
    config: dict[str, Any],
) -> npt.NDArray[Any]:
    """Window and normalize an image.

    Args:
        image: Image as a numpy array.
        config: Dictionary with information from the MIST configuration file
            (config.json). This file contains information about the modality,
            window range, and normalization parameters.

    Returns:
        Normalized image as a numpy array.
    """
    # Get the nonzero values in the image if necessary. In the case that the
    # images in a dataset are sparse enough, we may want to only normalize
    # the nonzero values. Additionally, we will only use the nonzero values
    # to compute the mean and standard deviation if not already given as
    # global parameters (i.e, for CT images).
    if config["preprocessing"]["normalize_with_nonzero_mask"]:
        nonzero_mask = (image != 0).astype(np.float32)
        nonzeros = image[nonzero_mask != 0]

    # Normalize the image based on the modality.
    # For CT images, we use precomputed window ranges and normalization
    # parameters.
    if config["dataset_info"]["modality"] == "ct":
        # Get window range and normalization parameters from config file.
        lower = config["preprocessing"]["ct_normalization"]["window_min"]
        upper = config["preprocessing"]["ct_normalization"]["window_max"]

        # Get mean and standard deviation from config file if given.
        mean = config["preprocessing"]["ct_normalization"]["z_score_mean"]
        std = config["preprocessing"]["ct_normalization"]["z_score_std"]
    else:
        # For all other modalities, we clip with the 0.5 and 99.5 percentiles
        # values of either the entire image or the nonzero values.
        if config["preprocessing"]["normalize_with_nonzero_mask"] and len(nonzeros) > 0:
            # Compute the window range based on the 0.5 and 99.5 percentiles
            # of the nonzero values.
            lower = np.percentile(nonzeros, pc.WINDOW_PERCENTILE_LOW)
            upper = np.percentile(nonzeros, pc.WINDOW_PERCENTILE_HIGH)

            # Compute the mean and standard deviation of the nonzero values.
            mean = np.mean(nonzeros)
            std = np.std(nonzeros)
        else:
            # Window image based on the 0.5 and 99.5 percentiles of the entire
            # image.
            lower = np.percentile(image, pc.WINDOW_PERCENTILE_LOW)
            upper = np.percentile(image, pc.WINDOW_PERCENTILE_HIGH)

            # Get mean and standard deviation of the entire image.
            mean = np.mean(image)
            std = np.std(image)

    # Window the image based on the lower and upper values.
    image = np.clip(image, lower, upper)

    # Normalize the image based on the mean and standard deviation.
    # Guard against std=0 (e.g. constant or all-zero images).
    if std == 0:
        return np.zeros_like(image, dtype=np.float32)
    image = (image - mean) / std

    # Apply nonzero mask if necessary.
    if config["preprocessing"]["normalize_with_nonzero_mask"]:
        image *= nonzero_mask

    return image.astype(np.float32)


def compute_dtm(
    mask_ants: ants.core.ants_image.ANTsImage,
    labels: list[int],
    normalize_dtm: bool,
) -> npt.NDArray[Any]:
    """Compute distance transform map (DTM) for a mask.

    Args:
        mask_ants: Mask as ANTs image.
        labels: List of labels in the dataset.
        normalize_dtm: Normalize the output DTM to be between -1 and 1.

    Returns:
        dtm: DTM for each label in mask as a 4D numpy array.
    """
    # Initialize the list of distance transform maps.
    dtms_sitk = []

    # Get the one-hot encoded masks as a list of SimpleITK images.
    masks_sitk = preprocessing_utils.make_onehot(mask_ants, labels)

    for mask in masks_sitk:
        # Start with case that the mask for the label is non-empty.
        if preprocessing_utils.sitk_get_sum(mask) != 0:
            # Compute the DTM for the current mask.
            dtm_i = sitk.SignedMaurerDistanceMap(
                sitk.Cast(mask, sitk.sitkUInt8),
                squaredDistance=False,
                useImageSpacing=False
            )

            # Normalize the DTM if necessary.
            if normalize_dtm:
                # Separate the negative (interior) and positive (exterior)
                # parts.
                dtm_int = sitk.Cast((dtm_i < 0), sitk.sitkFloat32)
                dtm_int *= dtm_i
                int_min, _ = preprocessing_utils.sitk_get_min_max(dtm_int)

                dtm_ext = sitk.Cast((dtm_i > 0), sitk.sitkFloat32)
                dtm_ext *= dtm_i
                _, ext_max = preprocessing_utils.sitk_get_min_max(dtm_ext)

                # Safeguard against division by zero.
                # If ext_max is zero, then there are no positive distances.
                # Similarly, if int_min is zero, then there are no negative
                # distances.
                if ext_max == 0:
                    # Avoid division by zero; this effectively leaves dtm_ext
                    # unchanged.
                    ext_max = 1
                if int_min == 0:
                    # Avoid division by zero; this effectively leaves dtm_int
                    # unchanged.
                    int_min = -1

                dtm_i = (dtm_ext / ext_max) - (dtm_int / int_min)
        else:
            # Handle the case of an empty mask.
            mask_depth = mask.GetDepth()
            mask_width = mask.GetWidth()
            mask_height = mask.GetHeight()
            all_ones_mask = np.ones((mask_depth, mask_height, mask_width))
            if normalize_dtm:
                dtm_i = sitk.GetImageFromArray(all_ones_mask)
            else:
                diagonal_distance = np.sqrt(
                    mask_depth**2 + mask_width**2 + mask_height**2
                )
                dtm_i = sitk.GetImageFromArray(
                    diagonal_distance * all_ones_mask
                )

        # Set the pixel type and spacing for the DTM.
        dtm_i = sitk.Cast(dtm_i, sitk.sitkFloat32)
        dtm_i.SetSpacing(mask.GetSpacing())
        dtm_i.SetOrigin(mask.GetOrigin())
        dtm_i.SetDirection(mask.GetDirection())

        # Append the current DTM to the final list.
        dtms_sitk.append(dtm_i)

    # Join the DTMs into a single 4D image and return as a numpy array.
    dtm = preprocessing_utils.sitk_to_ants(sitk.JoinSeries(dtms_sitk))
    dtm = dtm.numpy()
    return dtm


def preprocess_example(
    config: dict,
    image_paths_list: list[str],
    mask_path: str | None = None,
    fg_bbox: dict | None = None,
) -> dict:
    """Preprocessing function for a single example.

    If config['preprocessing']['skip'] is True:
      - Images (and mask) are read as-is and converted directly to numpy.
      - NO reorientation, NO cropping, NO resampling, NO normalization.
      - fg_bbox in the returned dict is always None.
      - (Optionally) compute DTMs if requested and in training mode.

    Else:
      - Reorient to RAI
      - (Optionally) crop to foreground
      - Resample to target spacing
      - Window & normalize images
      - (Optionally) compute DTMs

    This function is used for both training (mask_path provided) and
    inference (mask_path=None). In inference mode mask and dtm are always
    None in the returned dict.

    Args:
        config: Dictionary with information from config.json.
        image_paths_list: List containing paths to images for the example.
        mask_path: Path to segmentation mask. Pass None for inference.
        fg_bbox: Foreground bounding box. Used only when skip=False and
            crop_to_foreground=True. Ignored when skip=True.
    Returns:
        preprocessed_output: Dictionary containing the following keys:
            image: Preprocessed image(s) as a numpy array.
            mask: Segmentation mask as one-hot encoded numpy array, or None
                in inference mode.
            fg_bbox: Foreground bounding box, or None if skip=True or no
                cropping was performed.
            dtm: DTM(s) as a numpy array, or None in inference mode.
    """
    # Set the training flag based on the presence of a mask.
    training = bool(mask_path)

    # Set preprocessing flags.
    skip = config["preprocessing"]["skip"]
    crop = config["preprocessing"]["crop_to_foreground"]
    target_spacing = config["spatial_config"]["target_spacing"]
    compute_dtms = config["preprocessing"]["compute_dtms"]
    normalize_dtms = config["preprocessing"]["normalize_dtms"]
    labels = config["dataset_info"]["labels"]

    # Read all images.
    images = []
    for i, image_path in enumerate(image_paths_list):
        # Load image as ants image.
        image_i = ants.image_read(image_path)

        if not skip:
            # Get foreground mask if necessary.
            if i == 0 and crop and fg_bbox is None:
                fg_bbox = preprocessing_utils.get_fg_mask_bbox(image_i)

            # If cropping is requested, but the foreground bounding box is not
            # provided, raise an error. Otherwise, crop the image to the
            # foreground bounding box.
            if crop:
                if fg_bbox is None:
                    raise ValueError(
                        "Foreground bounding box is required for cropping, but "
                        "none was provided."
                    )
                image_i = preprocessing_utils.crop_to_fg(image_i, fg_bbox)

            # Put image into standard space.
            image_i = ants.reorient_image2(image_i, "RAI")
            image_i.set_direction(pc.RAI_ANTS_DIRECTION)
            if not np.allclose(image_i.spacing, target_spacing):
                image_i = resample_image(image_i, target_spacing=target_spacing)

        images.append(image_i)

    if training:
        # Read mask if we are in training mode.
        mask = ants.image_read(mask_path)

        if not skip:
            # Crop to foreground.
            if crop:
                mask = preprocessing_utils.crop_to_fg(mask, fg_bbox)

            # Put mask into standard space.
            mask = ants.reorient_image2(mask, "RAI")
            mask.set_direction(pc.RAI_ANTS_DIRECTION)
            mask = resample_mask(
                mask, labels=labels, target_spacing=target_spacing
            )

        # Compute DTM if requested and cast to float32.
        if compute_dtms:
            dtm = compute_dtm(mask, labels=labels, normalize_dtm=normalize_dtms)
            dtm = dtm.astype(np.float32)
        else:
            dtm = None

        # Add channel axis to mask and cast to uint8.
        mask = np.expand_dims(mask.numpy(), axis=-1).astype(np.uint8)
    else:
        mask = None
        dtm = None

    # Build the image array from all channels.
    image = np.zeros((*images[0].shape, len(images)), dtype=np.float32)
    for i, image_i in enumerate(images):
        if not skip:
            # Apply windowing and normalization if not skipping preprocessing.
            image[..., i] = window_and_normalize(image_i.numpy(), config)
        else:
            # skip=True: pure pass-through, just convert to numpy.
            image[..., i] = image_i.numpy()

    # Cast image to float32 for consistency.
    image = image.astype(np.float32)
    return {
        "image": image,
        "mask": mask,
        "fg_bbox": fg_bbox if not skip else None,
        "dtm": dtm,
    }


def _preprocess_single_patient(
    config: dict,
    patient: dict,
    image_columns: list[str],
    fg_bbox: dict | None,
    output_dirs: dict[str, Path],
    compute_dtms: bool,
) -> str | None:
    """Preprocess a single patient and save outputs to disk.

    Args:
        config: MIST configuration dictionary.
        patient: Row from train_paths.csv as a dictionary.
        image_columns: List of image column names from config.
        fg_bbox: Foreground bounding box dict for this patient, or None.
        output_dirs: Mapping of output type to destination directory.
        compute_dtms: Whether to compute and save DTMs.

    Returns:
        An error message string if processing fails, otherwise None.
    """
    try:
        patient_id = str(patient["id"])
        image_list = [patient[col] for col in image_columns]
        mask = patient["mask"]

        result = preprocess_example(
            config=config,
            image_paths_list=image_list,
            mask_path=mask,
            fg_bbox=fg_bbox,
        )

        patient_npy = f"{patient_id}.npy"
        np.save(output_dirs["images"] / patient_npy, result["image"])
        np.save(output_dirs["labels"] / patient_npy, result["mask"])
        if compute_dtms and result["dtm"] is not None:
            np.save(output_dirs["dtms"] / patient_npy, result["dtm"])

        return None
    except Exception as exc:  # pylint: disable=broad-except
        return f"Patient {patient['id']}: {exc}"


def preprocess_dataset(args: argparse.Namespace) -> None:
    """Preprocess a MIST compatible dataset.

    Args:
        args: Namespace object with MIST arguments.

    Raises:
        FileNotFoundError: If configuration file, training paths file, or
            foreground bounding box file is not found.
    """
    results_dir = Path(args.results)

    # Check if configuration file exists and read it.
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found in {args.results}."
        )
    config = io.read_json_file(config_path)

    # Check if training paths file exists and read it.
    train_paths_file = results_dir / "train_paths.csv"
    if not train_paths_file.exists():
        raise FileNotFoundError(
            f"Training paths file not found in {args.results}."
        )
    df = pd.read_csv(train_paths_file)

    # Create output directories for preprocessed images and labels.
    output_dirs = {
        "images": Path(args.numpy) / "images",
        "labels": Path(args.numpy) / "labels",
        "dtms": Path(args.numpy) / "dtms",
    }
    output_dirs["images"].mkdir(parents=True, exist_ok=True)
    output_dirs["labels"].mkdir(parents=True, exist_ok=True)

    # Apply CLI overrides to config and write back once, so downstream stages
    # (training, inference) see the correct flags regardless of which
    # combination of --compute-dtms / --no-preprocess the user passed.
    if args.compute_dtms:
        output_dirs["dtms"].mkdir(parents=True, exist_ok=True)
        config["preprocessing"]["compute_dtms"] = True
    if args.no_preprocess:
        config["preprocessing"]["skip"] = True
    if args.compute_dtms or args.no_preprocess:
        io.write_json_file(config_path, config)

    # Print preprocessing message and get progress bar.
    print_section_header("Preprocessing dataset")

    if config["preprocessing"]["skip"]:
        progress = progress_bar.get_progress_bar("Converting nifti to npy")
    else:
        progress = progress_bar.get_progress_bar("Preprocessing")

    # Check if foreground bounding box file exists and read it.
    fg_bboxes_file = results_dir / "fg_bboxes.csv"
    if not fg_bboxes_file.exists():
        raise FileNotFoundError(
            "Foreground bounding box (fg_bboxes.csv) file not found in "
            f"{args.results}."
        )
    fg_bboxes = pd.read_csv(fg_bboxes_file)

    # Pre-build a per-patient fg_bbox lookup for O(1) access in workers.
    fg_bboxes_by_id = {
        row["id"]: {k: v for k, v in row.items() if k != "id"}
        for row in fg_bboxes.to_dict(orient="records")
    }

    image_columns = config["dataset_info"]["images"]
    crop = config["preprocessing"]["crop_to_foreground"]
    patients = df.to_dict(orient="records")
    max_workers = getattr(args, "num_workers_preprocess", 1)

    error_messages = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_patient = {
            executor.submit(
                _preprocess_single_patient,
                config,
                patient,
                image_columns,
                fg_bboxes_by_id.get(patient["id"]) if crop else None,
                output_dirs,
                args.compute_dtms,
            ): patient
            for patient in patients
        }
        with progress as pb:
            for future in pb.track(
                concurrent.futures.as_completed(future_to_patient),
                total=len(patients),
            ):
                err = future.result()
                if err:
                    error_messages.append(err)

    if error_messages:
        print_warning("\n".join(error_messages))
        print_warning(
            f"{len(error_messages)} of {len(patients)} patient(s) had errors "
            "and were skipped."
        )
    else:
        print_success("Preprocessing complete.")
