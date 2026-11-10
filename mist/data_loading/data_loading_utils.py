"""Utility functions for data loading."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nvidia.dali import ops


def get_numpy_reader(
    files: list[str],
    shard_id: int,
    num_shards: int,
    seed: int,
    shuffle: bool,
) -> ops.readers.Numpy:
    """Creates and returns a DALI Numpy reader operator that reads numpy files.

    Args:
        files: List with file paths to numpy files (i.e., /path/to/file.npy)
        shard_id: The ID of the current shard, used for distributed data
            loading.
        num_shards: Total number of shards for splitting the data among workers.
        seed: Random seed for shuffling or any other randomness in the reader.
        shuffle: Whether to shuffle the data after each epoch.

    Returns:
        A DALI numpy reader operator configured with the provided parameters.
    """
    return ops.readers.Numpy(
        seed=seed,
        files=files,
        device="cpu",  # Reading happens on the CPU.
        read_ahead=True,  # Preload the data to speed up the reading process.
        shard_id=shard_id,  # Which shard of the data this instance will read.
        pad_last_batch=True,  # Pad last batch to for consistent batch sizes.
        num_shards=num_shards,  # Number of shards to split the dataset.
        dont_use_mmap=True,  # Disable memory mapping for reading files.
        shuffle_after_epoch=shuffle,  # Shuffle the data after every epoch.
    )


def validate_train_and_eval_inputs(
    imgs: list[str],
    lbls: list[str],
    dtms: list[str] | None = None,
) -> None:
    """Validate that the input data is correct.

    Ensures that images, labels, and optional DTM data are provided and that
    the lengths of the image, label, and DTM lists match.

    Args:
        imgs: List of image file paths.
        lbls: List of label file paths.
        dtms: Optional list of DTM data file paths. Defaults to None.

    Raises:
        ValueError: If the number of images, labels, or DTMs are incorrect.
    """
    if not imgs:
        raise ValueError("No images found!")

    if not lbls:
        raise ValueError("No labels found!")

    if len(imgs) != len(lbls):
        raise ValueError("Number of images and labels do not match!")

    if dtms is not None:
        if not dtms:
            raise ValueError("No DTM data found!")
        if len(imgs) != len(dtms):
            raise ValueError("Number of images and DTMs do not match!")


def is_valid_generic_pipeline_input(input_data: Any) -> bool:
    """Check if the input data is a valid generic pipeline input.

    Args:
        input_data: The input data to check.

    Returns:
        True if the input data is a valid generic pipeline input, False otherwise.
    """
    if not isinstance(input_data, Sequence) or isinstance(input_data, str):
        return False  # Must be a sequence but not a single string.

    if len(input_data) == 0:
        return False  # Empty lists are not valid.

    return all(
        isinstance(item, str) and item.endswith(".npy") and Path(item).is_file()
        for item in input_data
    )
