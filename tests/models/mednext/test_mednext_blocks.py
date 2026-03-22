"""Unit tests for MedNeXt block components."""
import pytest
import torch

# MIST imports.
from mist.models.mednext.mednext_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    MedNeXtOutBlock,
)


def test_mednext_block_forward():
    """Test forward pass of MedNeXtBlock preserves shape."""
    x = torch.randn(1, 4, 32, 64, 64)
    block = MedNeXtBlock(
        in_channels=4,
        out_channels=4,
        use_residual_connection=True,
        global_resp_norm=True,
    )
    y = block(x)
    assert y.shape == x.shape


def test_mednext_down_block_forward():
    """Test downsampling block halves spatial dimensions."""
    x = torch.randn(1, 4, 32, 64, 64)
    block = MedNeXtDownBlock(
        in_channels=4,
        out_channels=4,
        use_residual_connection=True,
    )
    y = block(x)
    assert y.shape == (1, 4, 16, 32, 32)


def test_mednext_up_block_forward():
    """Test upsampling block doubles spatial dimensions."""
    x = torch.randn(1, 4, 16, 32, 32)
    block = MedNeXtUpBlock(
        in_channels=4,
        out_channels=4,
        use_residual_connection=True,
    )
    y = block(x)
    assert y.shape == (1, 4, 32, 64, 64)


def test_mednext_out_block_forward():
    """Test output block maps to correct number of classes."""
    x = torch.randn(1, 4, 32, 64, 64)
    block = MedNeXtOutBlock(in_channels=4, n_classes=3)
    y = block(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 3


def test_mednext_block_layer_norm():
    """MedNeXtBlock with layer norm works for arbitrary spatial sizes."""
    x = torch.randn(1, 4, 32, 64, 64)
    block = MedNeXtBlock(
        in_channels=4,
        out_channels=4,
        kernel_size=7,
        norm_type="layer",
        use_residual_connection=True,
    )
    y = block(x)
    assert y.shape == x.shape


def test_mednext_block_invalid_norm_type():
    """MedNeXtBlock raises ValueError for an unsupported norm_type."""
    with pytest.raises(ValueError, match="norm_type must be 'group' or 'layer'"):
        MedNeXtBlock(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            norm_type="batch",
        )
