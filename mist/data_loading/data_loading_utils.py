"""Utility functions for data loading."""

from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Any

from nvidia.dali import fn
from nvidia.dali import math
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.tensors import TensorGPU

from mist.data_loading.data_loading_constants import DataLoadingConstants as constants


def get_numpy_reader(
        files: List[str],
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
        shuffle_after_epoch=shuffle  # Shuffle the data after every epoch.
    )


def random_augmentation(
        probability: float,
        augmented_data: TensorGPU,
        original_data: TensorGPU,
) -> TensorGPU:
    """Apply random augmentation to the data based on a given probability.

    This function returns the augmented version of the original data with a
    user defined probability.

    Args:
        probability: The probability of applying the augmentation.
        augmented_data: The augmented version of the data.
        original_data: The original data.

    Returns:
        The augmented version of the data if the flip_coin function returns true
        with the user defined probability.
    """
    # Generate a condition using a coin flip based on the provided probability.
    condition = fn.cast(
        fn.random.coin_flip(probability=probability),
        dtype=types.DALIDataType.BOOL
    )

    # Invert the condition (negation) for the alternative case.
    neg_condition = condition ^ True

    # Return augmented data if condition is true.
    return condition * augmented_data + neg_condition * original_data


def noise_fn(img: TensorGPU) -> TensorGPU:
    """Apply random noise to the image data.

    This function applies random noise to the image data using a Gaussian
    noise function. The standard deviation of the noise is randomly selected
    from a range of 0.0 to 0.33.

    Args:
        img: The image data to apply noise to.

    Returns:
        The image data with random noise applied with a probability of 0.15.
    """
    # Generate random noise with a standard deviation between 0.0 and 0.33,
    # then clamp to the original image range to avoid out-of-range values.
    img_noised = math.clamp(
        img + fn.random.normal(img, stddev=fn.random.uniform(
            range=(
                constants.NOISE_FN_RANGE_MIN,
                constants.NOISE_FN_RANGE_MAX,
            )
        )),
        fn.reductions.min(img),
        fn.reductions.max(img),
    )

    # Return the augmented image data with a probability of 0.15.
    return random_augmentation(
        constants.NOISE_FN_PROBABILITY, img_noised, img
    )


def blur_fn(img: TensorGPU) -> TensorGPU:
    """Apply random Gaussian blur to the image data.

    This function applies random Gaussian blur to the image data. The sigma
    value for the Gaussian blur is randomly selected from a range of 0.5 to
    1.5.

    Args:
        img: The image data to apply Gaussian blur to.

    Returns:
        The image data with random Gaussian blur applied with a probability
        of 0.15.
    """
    # Apply random Gaussian blur with a sigma between 0.5 and 1.5,
    # then clamp to the original image range to avoid out-of-range values.
    img_blurred = math.clamp(
        fn.gaussian_blur(
            img, sigma=fn.random.uniform(
                range=(
                    constants.BLUR_FN_RANGE_MIN,
                    constants.BLUR_FN_RANGE_MAX,
                )
            )
        ),
        fn.reductions.min(img),
        fn.reductions.max(img),
    )

    # Return the augmented image data with a probability of 0.15.
    return random_augmentation(
        constants.BLUR_FN_PROBABILITY, img_blurred, img
    )


def brightness_fn(img: TensorGPU) -> TensorGPU:
    """Apply random brightness scaling to the image data.

    This function applies random brightness scaling to the image data. The
    brightness scale is randomly selected from a range of 0.7 to 1.3.

    Args:
        img: The image data to apply brightness scaling to.

    Returns:
        The image data with random brightness scaling applied with a
        probability of 0.15.
    """
    # Generate a random brightness scale between 0.7 and 1.3 with a
    # probability of 0.15. Otherwise, the brightness scale is 1.0.
    brightness_scale = random_augmentation(
        constants.BRIGHTNESS_FN_PROBABILITY,
        fn.random.uniform(
            range=(
                constants.BRIGHTNESS_FN_RANGE_MIN,
                constants.BRIGHTNESS_FN_RANGE_MAX,
            )
        ),
        1.0,
    )

    # Return the image data with the random brightness scale applied.
    return img * brightness_scale


def contrast_fn(img: TensorGPU) -> TensorGPU:
    """Apply random contrast scaling to the image data.

    This function applies random contrast scaling to the image data. The
    scaling factor is randomly selected from a range of 0.65 to 1.5. The
    minimum and maximum values of the image data are used to clamp the
    contrast scaling. This function is applied with a probability of 0.15.

    Args:
        img: The image data to apply contrast scaling to.

    Returns:
        The image data with random contrast scaling applied with a
        probability of 0.15.
    """
    # Get the minimum and maximum values of the image data.
    min_, max_ = fn.reductions.min(img), fn.reductions.max(img)

    # Generate a random contrast scaling factor between 0.65 and 1.5 with
    # a probability of 0.15. Otherwise, the scaling factor is 1.0.
    scale = random_augmentation(
        constants.CONTRAST_FN_PROBABILITY,
        fn.random.uniform(
            range=(
                constants.CONTRAST_FN_RANGE_MIN,
                constants.CONTRAST_FN_RANGE_MAX,
            )
        ),
        1.0,
    )

    # Scale the image data and clamp the values between the minimum and
    # maximum values of the original image data.
    img = math.clamp(img * scale, min_, max_)
    return img


def validate_train_and_eval_inputs(
        imgs: List[str],
        lbls: List[str],
        dtms: Optional[List[str]] = None,
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
        isinstance(item, str) and
        item.endswith(".npy") and
        Path(item).is_file() for item in input_data
    )
