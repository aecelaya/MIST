"""Converts medical segmentation decathlon dataset to MIST dataset."""
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pprint
import rich
import numpy as np
import SimpleITK as sitk

# MIST imports.
from mist.utils import io, progress_bar
from mist.conversion_tools import conversion_utils

console = rich.console.Console()


def _copy_single_patient_msd(
    patient_entry: Any,
    source: Path,
    dest: Path,
    modalities: Dict[int, str],
    is_training: bool,
    image_source_dir: str,
    dest_mode_dir: str,
) -> Optional[str]:
    """Copy a single MSD patient to the destination in MIST format.

    Args:
        patient_entry: Dict with "image" key (training) or path string (test).
        source: Root source directory of the MSD dataset.
        dest: Root destination directory for the converted dataset.
        modalities: Mapping of modality index to modality name.
        is_training: True for training data, False for test data.
        image_source_dir: Source subdirectory name ("imagesTr" or "imagesTs").
        dest_mode_dir: Destination subdirectory name ("train" or "test").

    Returns:
        An error message string if a required file is missing, otherwise None.
    """
    # Extract patient ID from the MSD JSON entry.
    raw_name = patient_entry["image"] if is_training else patient_entry
    patient_id = Path(raw_name).name.split(".")[0]

    # Verify image exists.
    image_path = source / image_source_dir / f"{patient_id}.nii.gz"
    if not image_path.exists():
        return f"Image {image_path} does not exist!"

    # Verify mask exists (training only).
    if is_training:
        mask_path = source / "labelsTr" / f"{patient_id}.nii.gz"
        if not mask_path.exists():
            return f"Mask {mask_path} does not exist!"

    # Create patient directory in destination.
    patient_directory = dest / "raw" / dest_mode_dir / patient_id
    patient_directory.mkdir(parents=True, exist_ok=True)

    # Handle multi-modality: split 4D image into per-modality 3D images.
    if len(modalities) > 1:
        image_sitk = sitk.ReadImage(str(image_path))
        image_npy = sitk.GetArrayFromImage(image_sitk)

        direction = (
            np.array(image_sitk.GetDirection()).reshape((4, 4))[0:3, 0:3].ravel()
        )
        spacing = image_sitk.GetSpacing()[:-1]
        origin = image_sitk.GetOrigin()[:-1]

        for j, modality in modalities.items():
            img_j = sitk.GetImageFromArray(image_npy[j])
            img_j.SetDirection(direction)
            img_j.SetSpacing(spacing)
            img_j.SetOrigin(origin)
            sitk.WriteImage(
                img_j, str(patient_directory / f"{modality}.nii.gz")
            )
    else:
        conversion_utils.copy_image_from_source_to_dest(
            image_path, patient_directory / f"{modalities[0]}.nii.gz"
        )

    if is_training:
        conversion_utils.copy_image_from_source_to_dest(
            mask_path, patient_directory / "mask.nii.gz"
        )

    return None


def copy_msd_data(
    source: Union[str, Path],
    dest: Union[str, Path],
    msd_json: Dict[str, Any],
    modalities: Dict[int, str],
    mode: str,
    progress_bar_message: str,
    max_workers: Optional[int] = None,
) -> None:
    """Copy MSD data to destination in MIST format.

    Args:
        source: Path to the source MSD directory.
        dest: Path to the destination directory.
        msd_json: Dictionary containing the MSD dataset information.
        modalities: Mapping of modality index to modality name.
        mode: Mode of the data — "training" or "test".
        progress_bar_message: Message displayed on left side of progress bar.
        max_workers: Maximum number of parallel threads. Defaults to None.

    Returns:
        None. The data is copied to the destination directory.
    """
    source = Path(source)
    dest = Path(dest)

    is_training = mode == "training"
    image_source_dir = "imagesTr" if is_training else "imagesTs"
    dest_mode_dir = "train" if is_training else "test"

    error_messages = []
    patients = msd_json[mode]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_patient = {
            executor.submit(
                _copy_single_patient_msd,
                p, source, dest, modalities, is_training,
                image_source_dir, dest_mode_dir,
            ): p
            for p in patients
        }
        with progress_bar.get_progress_bar(progress_bar_message) as pb:
            for future in pb.track(
                concurrent.futures.as_completed(future_to_patient),
                total=len(patients),
            ):
                err = future.result()
                if err:
                    error_messages.append(err)

    if error_messages:
        console.print(rich.text.Text("\n".join(error_messages)))  # type: ignore


def convert_msd(
    source: Union[str, Path],
    dest: Union[str, Path],
    max_workers: Optional[int] = None,
) -> None:
    """Converts medical segmentation decathlon dataset to MIST dataset.

    Args:
        source: Path to the source MSD directory.
        dest: Path to the destination directory.
        max_workers: Maximum number of parallel threads for file copying.

    Returns:
        None. The data is copied to the destination directory.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        FileNotFoundError: If the MSD dataset.json file does not exist.
    """
    source = Path(source).resolve()
    dest = Path(dest).resolve()

    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist!")

    # Create destination directories.
    (dest / "raw" / "train").mkdir(parents=True, exist_ok=True)
    test_data_exists = (source / "imagesTs").exists()
    if test_data_exists:
        (dest / "raw" / "test").mkdir(parents=True, exist_ok=True)

    # Load the MSD dataset JSON.
    dataset_json_path = source / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"{dataset_json_path} does not exist!")
    msd_json = io.read_json_file(dataset_json_path)

    # Extract modalities.
    modalities = {int(idx): mod for idx, mod in msd_json["modality"].items()}

    # Copy training data.
    copy_msd_data(
        source=source,
        dest=dest,
        msd_json=msd_json,
        modalities=modalities,
        mode="training",
        progress_bar_message="Converting training data to MIST format",
        max_workers=max_workers,
    )

    # Copy test data if it exists.
    if test_data_exists:
        copy_msd_data(
            source=source,
            dest=dest,
            msd_json=msd_json,
            modalities=modalities,
            mode="test",
            progress_bar_message="Converting test data to MIST format",
            max_workers=max_workers,
        )

    # Build MIST dataset JSON. Paths are relative to the output directory so
    # that the dataset remains portable across machines.
    modalities_lower = [mod.lower() for mod in modalities.values()]
    labels_list = list(map(int, msd_json["labels"].keys()))
    dataset_json = {
        "task": msd_json["name"],
        "modality": (
            "ct" if "ct" in modalities_lower
            else "mr" if "mri" in modalities_lower
            else "other"
        ),
        "train-data": "raw/train",
        "test-data": "raw/test" if test_data_exists else None,
        "mask": ["mask.nii.gz"],
        "images": {
            mod: [f"{mod}.nii.gz"] for mod in modalities.values()
        },
        "labels": labels_list,
        "final_classes": {
            msd_json["labels"][str(label)].replace(" ", "_"): [label]
            for label in labels_list if label != 0
        },
    }

    if not test_data_exists:
        dataset_json.pop("test-data")

    # Write MIST dataset description to json file.
    output_json_path = dest / "dataset.json"
    console.print(rich.text.Text(  # type: ignore
        f"MIST dataset parameters written to {output_json_path}\n"
    ))
    pprint.pprint(dataset_json, sort_dicts=False)
    console.print(rich.text.Text("\n"))  # type: ignore
    console.print(rich.text.Text(  # type: ignore
        "Please add task, modality, labels, and final classes to parameters.\n"
    ))

    io.write_json_file(output_json_path, dataset_json)
