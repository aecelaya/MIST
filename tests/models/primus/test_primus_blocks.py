"""Tests for the Primus tokenizers and patch decoder."""

import torch
from torch import nn

# MIST imports.
from mist.models.primus.primus_blocks import (
    LayerNormNd,
    PatchDecode,
    PatchEmbed,
    PatchEmbedDeeper,
)


def test_layer_norm_nd_normalizes_channels():
    """Each voxel's channel vector has zero mean and unit variance."""
    out = LayerNormNd(6)(torch.randn(2, 6, 3, 3, 3) * 5 + 2)
    assert torch.allclose(out.mean(1), torch.zeros(2, 3, 3, 3), atol=1e-5)
    assert torch.allclose(out.var(1, unbiased=False), torch.ones(2, 3, 3, 3), atol=1e-3)


def test_patch_embed_shape():
    """One token per 8x8x8 patch."""
    out = PatchEmbed((8, 8, 8), 2, 48)(torch.randn(1, 2, 16, 24, 32))
    assert out.shape == (1, 48, 2, 3, 4)


def test_patch_embed_deeper_shape_and_relu():
    """The PrimusV2 tokenizer also downsamples by 8 and uses ReLU."""
    tokenizer = PatchEmbedDeeper(2, 48)
    out = tokenizer(torch.randn(1, 2, 16, 16, 16))
    assert out.shape == (1, 48, 2, 2, 2)
    assert isinstance(tokenizer.stem.blocks[0].nonlin2, nn.ReLU)
    assert tokenizer.stem.blocks[0].conv1.conv.bias is not None
    assert tokenizer.stages[0].blocks[0].conv1.conv.bias is None


def test_patch_decode_shape():
    """The decoder restores full resolution with the requested channels."""
    decoder = PatchDecode((8, 8, 8), 48, 3)
    assert len(decoder.decode) == 3
    out = decoder(torch.randn(1, 48, 2, 2, 2))
    assert out.shape == (1, 3, 16, 16, 16)
