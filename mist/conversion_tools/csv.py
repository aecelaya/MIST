"""Converts data from csv files to MIST format."""
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pprint
import rich
import pandas as pd

# MIST imports.
from mist.utils import io, progress_bar
from mist.conversion_tools import conversion_utils

# Set up console for rich text.
console = rich.console.Console()


def _copy_single_patient_csv(
    patient_dict: Dict[str, Any],
    dest: Path,
    mode: str,
) -> Optional[str]:
    """Copy a single patient's files to the destination directory.

    Args:
        patient_dict: Row from the CSV as a dictionary (id, mask, images...).
        dest: Root destination directory for this split.
        mode: "training" or "test".

    Returns:
        An error message string if any file is missing, otherwise None.
    """
    errors = []
    image_start_idx = 2 if mode == "training" else 1

    patient_dest = dest / str(patient_dict["id"])
    patient_dest.mkdir(parents=True, exist_ok=True)

    if mode == "training":
        mask_source = Path(patient_dict["mask"]).resolve()
        if not mask_source.exists():
            return f"Mask {mask_source} does not exist!"
        conversion_utils.copy_image_from_source_to_dest(
            mask_source, patient_dest / "mask.nii.gz"
        )

    image_keys = list(patient_dict.keys())[image_start_idx:]
    image_list = list(patient_dict.values())[image_start_idx:]

    for image_key, image_path in zip(image_keys, image_list):
        image_source = Path(image_path).resolve()
        if not image_source.exists():
            errors.append(f"Image {image_source} does not exist!")
            continue
        conversion_utils.copy_image_from_source_to_dest(
            image_source, patient_dest / f"{image_key}.nii.gz"
        )

    return "\n".join(errors) if errors else None


def copy_csv_data(
    df: pd.DataFrame,
    dest: Union[str, Path],
    mode: str,
    progress_bar_message: str,
    max_workers: Optional[int] = None,
) -> None:
    """Copy data from csv file to a MIST-compatible directory structure.

    Args:
        df: Dataframe containing the csv file data.
        dest: Destination directory to save the data.
        mode: "training" or "test". If "training", the mask will be copied.
        progress_bar_message: Message displayed on left side of progress bar.
        max_workers: Maximum number of parallel threads. Defaults to None
            (uses the ThreadPoolExecutor default).

    Returns:
        None. The data is copied to the destination directory.
    """
    dest = Path(dest)
    error_messages = []
    patients = [row._asdict() for row in df.itertuples(index=False)]  # type: ignore

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_patient = {
            executor.submit(_copy_single_patient_csv, p, dest, mode): p
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


def convert_csv(
    train_csv: Union[str, Path],
    dest: Union[str, Path],
    test_csv: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Converts train and test data from csv files to MIST format.

    Args:
        train_csv: Path to the training csv file.
        dest: Destination directory to save the data.
        test_csv: Optional path to the testing csv file.
        max_workers: Maximum number of parallel threads for file copying.

    Returns:
        None. The data is copied to the destination directory.

    Raises:
        FileNotFoundError: If train_csv or test_csv does not exist.
    """
    dest = Path(dest).resolve()
    train_csv = Path(train_csv).resolve()

    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} does not exist!")

    if test_csv is not None:
        test_csv = Path(test_csv).resolve()
        if not test_csv.exists():
            raise FileNotFoundError(f"{test_csv} does not exist!")

    # Create destination directories for train (and test if provided).
    train_dest = dest / "raw" / "train"
    train_dest.mkdir(parents=True, exist_ok=True)
    if test_csv is not None:
        test_dest = dest / "raw" / "test"
        test_dest.mkdir(parents=True, exist_ok=True)

    # Convert training data to MIST-compatible format.
    train_df = pd.read_csv(train_csv)
    copy_csv_data(
        train_df,
        train_dest,
        "training",
        "Converting training data to MIST format",
        max_workers=max_workers,
    )

    # Convert testing data to MIST-compatible format.
    if test_csv is not None:
        test_df = pd.read_csv(test_csv)
        copy_csv_data(
            test_df,
            test_dest,
            "test",
            "Converting test data to MIST format",
            max_workers=max_workers,
        )

    # Build MIST dataset JSON. Paths are relative to the output directory so
    # that the dataset remains portable across machines.
    dataset_json = {
        "task": None,
        "modality": None,
        "train-data": "raw/train",
        "test-data": "raw/test" if test_csv is not None else None,
        "mask": ["mask.nii.gz"],
        "images": {
            modality: [f"{modality}.nii.gz"]
            for modality in list(train_df.columns)[2:]
        },
        "labels": None,
        "final_classes": None,
    }

    if test_csv is None:
        dataset_json.pop("test-data")

    # Write MIST dataset description to json file.
    dataset_json_path = dest / "dataset.json"
    console.print(rich.text.Text(  # type: ignore
        f"MIST dataset parameters written to {dataset_json_path}\n"
    ))
    pprint.pprint(dataset_json, sort_dicts=False)
    console.print(rich.text.Text("\n"))  # type: ignore
    console.print(rich.text.Text(  # type: ignore
        "Please add task, modality, labels, and final classes to parameters.\n"
    ))

    io.write_json_file(dataset_json_path, dataset_json)
