"""Tokenizers and the patch decoder for Primus.

Adapted from MIC-DKFZ's dynamic-network-architectures (Apache License 2.0).
Module and attribute names mirror the original so that state dict keys match
nnU-Net and nnssl checkpoints.
"""

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

# MIST imports.
from mist.models.primus.primus_constants import PrimusConstants as constants
from mist.models.resenc.resenc_blocks import StackedResidualBlocks


def _relu() -> nn.Module:
    """Return the activation used by the PrimusV2 tokenizer."""
    return nn.ReLU(inplace=True)


class LayerNormNd(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, ...) tensor."""

    def __init__(self, num_channels: int, eps: float = constants.DECODER_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each voxel's channel vector."""
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return self.weight.view(shape) * x + self.bias.view(shape)


class PatchEmbed(nn.Module):
    """Primus tokenizer: one strided convolution per token."""

    def __init__(self, token_size: Sequence[int], in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, token_size, token_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return token features of shape (B, embed_dim, D', H', W')."""
        return self.proj(x)


class PatchEmbedDeeper(nn.Module):
    """PrimusV2 tokenizer: a residual stem and three downsampling stages.

    The stem keeps full resolution; each stage halves it, so tokens cover
    8x8x8 voxels as in Primus, but neighboring tokens' receptive fields overlap.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        base = constants.TOKENIZER_BASE_FEATURES
        self.stem = StackedResidualBlocks(
            1, in_channels, base, 3, 1, conv_bias=True, activation=_relu
        )
        stages = []
        stage_in = base
        for level, num_blocks in enumerate(constants.TOKENIZER_BLOCKS_PER_LEVEL):
            stage_out = base * 2**level
            stages.append(
                StackedResidualBlocks(
                    num_blocks, stage_in, stage_out, 3, 2, conv_bias=False, activation=_relu
                )
            )
            stage_in = stage_out
        self.stages = nn.ModuleList(stages)
        self.final_proj = nn.Conv3d(stage_in, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return token features of shape (B, embed_dim, D/8, H/8, W/8)."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.final_proj(x)


class PatchDecode(nn.Module):
    """Lightweight decoder from tokens back to full resolution.

    A stack of 2x transposed convolutions with channel-first LayerNorm and
    GELU between them; the channel count shrinks geometrically from
    embed_dim towards out_channels.
    """

    def __init__(self, token_size: Sequence[int], embed_dim: int, out_channels: int):
        super().__init__()

        def _round_to_8(value: float) -> int:
            return int(max(8, np.round((value + 1e-6) / 8) * 8))

        num_stages = int(math.log2(max(token_size)))
        strides = [
            [2 if (p / 2**n) % 2 == 0 else 1 for p in token_size] for n in range(num_stages)
        ][::-1]
        reduction = (embed_dim / (2 * out_channels)) ** (1 / num_stages)
        channels = [embed_dim] + [
            _round_to_8(embed_dim / reduction ** (i + 1)) for i in range(num_stages)
        ]
        channels[-1] = out_channels

        stages: list[nn.Module] = [
            nn.Sequential(
                nn.ConvTranspose3d(channels[s], channels[s + 1], strides[s], strides[s]),
                LayerNormNd(channels[s + 1]),
                nn.GELU(),
            )
            for s in range(num_stages - 1)
        ]
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], strides[-1], strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample token features of shape (B, embed_dim, D', H', W')."""
        return self.decode(x)
