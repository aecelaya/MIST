"""Unit tests for MedNeXt block components."""
import pytest
import torch

# MIST imports.
from mist.models.mednext.mednext_blocks import (
    get_conv_layer,
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    MedNeXtOutBlock,
)


@pytest.mark.parametrize("spatial_dim,transpose,expected", [
    (2, False, torch.nn.Conv2d),
    (2, True, torch.nn.ConvTranspose2d),
    (3, False, torch.nn.Conv3d),
    (3, True, torch.nn.ConvTranspose3d),
])
def test_get_conv_layer(spatial_dim, transpose, expected):
    """Test correct layer returned with spatial dimension and transpose flag."""
    conv_layer = get_conv_layer(spatial_dim, transpose)
    assert conv_layer == expected


def test_get_conv_layer_invalid_dim():
    """Test that an invalid spatial dimension raises ValueError."""
    with pytest.raises(ValueError, match="Invalid spatial dimension"):
        get_conv_layer(spatial_dim=4)


@pytest.mark.parametrize("dim,shape", [
    ("2d", (1, 4, 64, 64)),
    ("3d", (1, 4, 32, 64, 64)),
])
def test_mednext_block_forward(dim, shape):
    """Test forward pass of MedNeXtBlock."""
    x = torch.randn(shape)
    block = MedNeXtBlock(
        in_channels=4,
        out_channels=4,
        dim=dim,
        use_residual_connection=True,
        global_resp_norm=True
    )
    y = block(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("dim,shape,out_shape", [
    ("2d", (1, 4, 64, 64), (1, 4, 32, 32)),
    ("3d", (1, 4, 32, 64, 64), (1, 4, 16, 32, 32)),
])
def test_mednext_down_block_forward(dim, shape, out_shape):
    """Test downsampling block reduces spatial dimensions."""
    x = torch.randn(shape)
    block = MedNeXtDownBlock(
        in_channels=4,
        out_channels=4,
        dim=dim,
        use_residual_connection=True
    )
    y = block(x)
    assert y.shape == out_shape


@pytest.mark.parametrize("dim,shape,out_shape", [
    ("2d", (1, 4, 32, 32), (1, 4, 64, 64)),
    ("3d", (1, 4, 16, 32, 32), (1, 4, 32, 64, 64)),
])
def test_mednext_up_block_forward(dim, shape, out_shape):
    """Test upsampling block increases spatial dimensions."""
    x = torch.randn(shape)
    block = MedNeXtUpBlock(
        in_channels=4,
        out_channels=4,
        dim=dim,
        use_residual_connection=True
    )
    y = block(x)
    assert y.shape == out_shape


@pytest.mark.parametrize("dim,shape,n_classes", [
    ("2d", (1, 4, 64, 64), 3),
    ("3d", (1, 4, 32, 64, 64), 2),
])
def test_mednext_out_block_forward(dim, shape, n_classes):
    """Test output block converts to number of classes."""
    x = torch.randn(shape)
    block = MedNeXtOutBlock(in_channels=4, n_classes=n_classes, dim=dim)
    y = block(x)
    assert y.shape[1] == n_classes
    assert y.shape[0] == x.shape[0]  # batch size unchanged


@pytest.mark.parametrize("dim,shape", [
    ("2d", (1, 4, 64, 64)),
    ("3d", (1, 4, 32, 64, 64)),
])
def test_mednext_block_layer_norm(dim, shape):
    """Test MedNeXtBlock with layer norm works for arbitrary spatial sizes.

    Previously used spatial_size==kernel_size which masked a bug where
    nn.LayerNorm was constructed with a fixed spatial shape. Now uses
    GroupNorm(1, C) which is equivalent and spatial-size agnostic.
    """
    x = torch.randn(shape)
    block = MedNeXtBlock(
        in_channels=4,
        out_channels=4,
        dim=dim,
        kernel_size=7,
        norm_type="layer",
        use_residual_connection=True
    )
    y = block(x)
    assert y.shape == x.shape
