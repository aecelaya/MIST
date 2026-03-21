"""Tests for nnUNet utility functions."""
import pytest

# MIST imports.
from mist.models.nnunet import nnunet_utils

# Tests for get_padding
@pytest.mark.parametrize("kernel, stride, expected", [
    (3, 1, 1),
    ((3, 3), (1, 1), (1, 1)),
    ((5, 3, 3), (1, 1, 1), (2, 1, 1)),
])
def test_get_padding_valid(kernel, stride, expected):
    """Covers the case where kernel and stride are valid."""
    assert nnunet_utils.get_padding(kernel, stride) == expected


def test_get_padding_negative_raises():
    """Covers the case where padding would be negative."""
    with pytest.raises(
        ValueError, match="Padding value should not be negative"
    ):
        nnunet_utils.get_padding(2, 4)  # Will cause negative padding


# Tests for get_output_padding.
@pytest.mark.parametrize("kernel, stride, padding, expected", [
    (3, 1, 1, 0),
    ((3, 3), (1, 1), (1, 1), (0, 0)),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
])
def test_get_output_padding_valid(kernel, stride, padding, expected):
    """Covers the case where kernel, stride, and padding are valid."""
    assert nnunet_utils.get_output_padding(kernel, stride, padding) == expected


def test_get_output_padding_negative_raises():
    """Covers the case where output padding would be negative."""
    with pytest.raises(
        ValueError, match="out_padding should not be negative"
    ):
        nnunet_utils.get_output_padding(5, 1, 1)


# Tests for get_unet_params.
def test_get_unet_params_valid():
    """Covers the case where patch_size and spacings are valid."""
    patch_size = [128, 128, 128]
    spacings = [1.0, 1.0, 1.0]
    kernels, strides, final_dim = nnunet_utils.get_unet_params(
        patch_size, spacings
    )

    assert isinstance(kernels, list)
    assert isinstance(strides, list)
    assert isinstance(final_dim, list)

    assert all(isinstance(k, list) for k in kernels)
    assert all(isinstance(s, list) for s in strides)

    # Final stride should always be [1, 1, 1].
    assert strides[0] == [1, 1, 1]


def test_get_unet_params_mismatched_input_lengths():
    """Covers the case where patch_size and spacings have different lengths."""
    with pytest.raises(ValueError, match="must be the same length"):
        nnunet_utils.get_unet_params([128, 128], [1.0, 1.0, 1.0])


def test_get_unet_params_early_exit_on_anisotropic_stride():
    """Covers early exit when all strides are ANISOTROPIC_STRIDE (i.e., 1)."""
    # Force all stride values to be 1 by using large spacing ratios.
    patch_size = [4, 4, 4]
    spacings = [1.0, 1.0, 100.0]  # spacing_ratio >> 2

    kernels, strides, _ = nnunet_utils.get_unet_params(patch_size, spacings)

    # Should only have one stride (the initial [1, 1, 1]).
    assert strides == [[1, 1, 1]]

    # The loop should break early, so kernels should have only one entry.
    assert len(kernels) == 1
    assert kernels[0] == [3, 3, 3]  # default bottleneck kernel.
