"""Tests for the Residual Encoder UNet building blocks."""

import pytest
import torch
from torch import nn

# MIST imports.
from mist.models.resenc.resenc_blocks import (
    BasicBlockD,
    ConvNormAct,
    ResidualEncoder,
    StackedConvBlocks,
    StackedResidualBlocks,
    UNetDecoder,
)


def _encoder(in_channels=1):
    """Build a small three-stage residual encoder."""
    return ResidualEncoder(
        in_channels=in_channels,
        features_per_stage=[8, 16, 32],
        kernel_sizes=[[3, 3, 3]] * 3,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 1]],
        blocks_per_stage=[1, 2, 2],
    )


# ConvNormAct.
def test_conv_norm_act_with_activation():
    """The conv is registered both directly and inside all_modules."""
    layer = ConvNormAct(2, 4, 3, 2)
    keys = layer.state_dict().keys()
    assert "conv.weight" in keys
    assert "all_modules.0.weight" in keys
    assert isinstance(layer.all_modules[-1], nn.LeakyReLU)
    assert layer(torch.randn(1, 2, 8, 8, 8)).shape == (1, 4, 4, 4, 4)


def test_conv_norm_act_without_activation():
    """Without activation the sequence ends with the norm and has no bias."""
    layer = ConvNormAct(2, 4, [1, 1, 1], 1, use_activation=False, conv_bias=False)
    assert not hasattr(layer, "nonlin")
    assert isinstance(layer.all_modules[-1], nn.InstanceNorm3d)
    assert layer.conv.bias is None
    assert layer.conv.padding == (0, 0, 0)


def test_stacked_conv_blocks_only_first_is_strided():
    """Only the first conv applies the stride."""
    blocks = StackedConvBlocks(3, 2, 4, 3, 2)
    assert len(blocks.convs) == 3
    assert blocks.convs[0].conv.stride == (2, 2, 2)
    assert blocks.convs[1].conv.stride == (1, 1, 1)
    assert blocks(torch.randn(1, 2, 8, 8, 8)).shape == (1, 4, 4, 4, 4)


# BasicBlockD.
def test_basic_block_identity_skip():
    """Same channels and no stride uses an identity skip."""
    block = BasicBlockD(4, 4, 3, 1)
    assert isinstance(block.skip, nn.Identity)
    assert block(torch.randn(1, 4, 8, 8, 8)).shape == (1, 4, 8, 8, 8)


def test_basic_block_strided_projection_skip():
    """Stride and channel change give an avgpool followed by a 1x1x1 projection."""
    block = BasicBlockD(4, 8, 3, [2, 2, 1])
    assert isinstance(block.skip[0], nn.AvgPool3d)
    assert isinstance(block.skip[1], ConvNormAct)
    assert "skip.1.conv.weight" in block.state_dict()
    assert block(torch.randn(1, 4, 8, 8, 8)).shape == (1, 8, 4, 4, 8)


def test_basic_block_projection_only_skip():
    """A channel change alone projects without pooling."""
    block = BasicBlockD(4, 8, 3, 1)
    assert len(block.skip) == 1
    assert "skip.0.conv.weight" in block.state_dict()


def test_basic_block_stride_only_skip():
    """A stride alone pools without projecting."""
    block = BasicBlockD(4, 4, 3, 2)
    assert len(block.skip) == 1
    assert isinstance(block.skip[0], nn.AvgPool3d)


def test_stacked_residual_blocks():
    """Only the first residual block downsamples."""
    blocks = StackedResidualBlocks(3, 4, 8, 3, 2)
    assert len(blocks.blocks) == 3
    assert blocks(torch.randn(1, 4, 8, 8, 8)).shape == (1, 8, 4, 4, 4)


# ResidualEncoder.
def test_residual_encoder_returns_one_output_per_stage():
    """The encoder returns every stage's output, from full to low resolution."""
    encoder = _encoder()
    skips = encoder(torch.randn(1, 1, 16, 16, 16))
    assert [s.shape for s in skips] == [
        (1, 8, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 32, 4, 4, 8),
    ]
    assert encoder.strides[2] == [2, 2, 1]


def test_residual_encoder_mismatched_lengths_raise():
    """Per-stage settings must all have one entry per stage."""
    with pytest.raises(ValueError, match="one entry per stage"):
        ResidualEncoder(
            in_channels=1,
            features_per_stage=[8, 16],
            kernel_sizes=[[3, 3, 3]] * 2,
            strides=[[1, 1, 1], [2, 2, 2]],
            blocks_per_stage=[1],
        )


# UNetDecoder.
def test_unet_decoder_outputs_and_layers():
    """The decoder returns features per stage and builds a seg layer for each."""
    encoder = _encoder()
    decoder = UNetDecoder(encoder, out_channels=3, convs_per_stage=1)
    assert len(decoder.stages) == len(decoder.transpconvs) == len(decoder.seg_layers) == 2
    outputs = decoder(encoder(torch.randn(1, 1, 16, 16, 16)))
    assert [o.shape for o in outputs] == [(1, 8, 16, 16, 16), (1, 16, 8, 8, 8)]
