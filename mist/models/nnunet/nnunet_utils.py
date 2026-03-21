"""Utility functions for nnUNet."""
from collections.abc import Sequence
from typing import Union, Tuple
import numpy as np

# MIST imports.
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants


def get_padding(
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> Union[int, Sequence[int]]:
    """Get padding for a convolution layer based on kernel size and stride.

    Args:
        kernel_size: Size of the kernel for convolution layer as an integer or
            a sequence of integers.
        stride: Stride of the convolution layer as an integer or a sequence of
            integers.

    Returns:
        Padding for convolution layer as an integer or a sequence of integers.

    Raises:
        ValueError: If padding value is negative.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise ValueError(
            "Padding value should not be negative, please change the kernel "
            "size and/or stride."
        )
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0] # type: ignore


def get_output_padding(
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
) -> Union[int, Sequence[int]]:
    """Get output padding of a convolution layer.

    This is used used for transposed convolutional layers, where we pad the
    output to match the input size.

    Args:
        kernel_size: Size of the kernel for convolutional layer as an integer or
            a sequence of integers.
        stride: Stride of the convolutional layer as an integer or a sequence of
            integers.
        padding: Padding for the input of the convolutional layer as an integer
            or a sequence of integers.

    Returns:
        out_padding: Output padding for convolutional layer as an integer or a
            sequence of integers.

    Raises:
        ValueError: If out_padding value is negative.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise ValueError(
            "The value of out_padding should not be negative, please change "
            "the kernel size and/or stride."
        )
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0] # type: ignore


def get_unet_params(
    patch_size: Sequence[int],
    spacings: Sequence[float],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Get parameters for UNet architecture based on patch size and spacings.

    Args:
        patch_size: Size of the patch as a sequence of integers.
        spacings: The voxel spacings of the input image as a sequence of floats.
        max_depth: Maximum depth of the UNet. The UNet will have at most
            max_depth + 1 convolutional layers along the encoder path.

    Returns:
        kernels: Kernel sizes for each convolutional layer in the UNet.
        strides: Strides for each convolutional layer in the UNet.
        final_encoded_dimension: Size of the output of the bottleneck layer in
            the UNet. We don't use this in the current implementation, but it
            could be useful later.

    Raises:
        ValueError: If the patch size and spacings have different lengths.
    """
    # Check that the patch size and spacings have the same number of dimensions.
    if len(patch_size) != len(spacings):
        raise ValueError(
            "The patch size and the spacings must be the same length, "
            f"but got {len(patch_size)} and {len(spacings)}."
        )

    strides, kernels, final_encoded_dimension = [], [], patch_size
    while True:
        # Compute ratio of spacing to minimum spacing. This is used to determine
        # if the input is anisotropic.
        spacing_ratio = [spacing / min(spacings) for spacing in spacings]

        # If the spacing ratio is less than or equal to 2 for a given dimension,
        # and the size of the dimension is greater than 8, we use a stride of 2.
        # Otherwise, we use a stride of 1.
        stride = [
            constants.DEFAULT_STRIDE if
            (
                ratio <= constants.ANISOTROPIC_MAX_RATIO and
                size >= constants.MIN_SIZE_FOR_STRIDE
            )
            else constants.ANISOTROPIC_STRIDE
            for (ratio, size) in zip(spacing_ratio, final_encoded_dimension)
        ]

        # If the spacing ratio is less than or equal to 2 for a given dimension,
        # we use a kernel size of 3. Otherwise, we use a kernel size of 1.
        kernel = [
            constants.DEFAULT_KERNEL_SIZE if
            (
                ratio <= constants.ANISOTROPIC_MAX_RATIO
            )
            else constants.ANISOTROPIC_KERNEL_SIZE for ratio in spacing_ratio
        ]

        # If all strides are 1, we break out of the loop.
        if all(s == constants.ANISOTROPIC_STRIDE for s in stride):
            break

        # Update the final encoded dimension and spacings based on the stride.
        final_encoded_dimension = [
            i // j for i, j in zip(final_encoded_dimension, stride)
        ]

        # Update the spacings based on the stride.
        spacings = [i * j for i, j in zip(spacings, stride)]

        # Append the kernel and stride to the list.
        kernels.append(kernel)
        strides.append(stride)

        # If the number of strides is equal to the maximum depth, we break out
        # of the loop.
        if len(strides) == constants.MAX_DEPTH:
            break

    # Add a stride of 1 for the input layer.
    strides.insert(0, len(spacings) * [1])

    # Add a kernel size of 3 to the last convolutional layer for the bottleneck.
    kernels.append(len(spacings) * [3])
    return kernels, strides, final_encoded_dimension
