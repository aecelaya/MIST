"""Building blocks for the Residual Encoder UNet (ResEnc).

Adapted from MIC-DKFZ's dynamic-network-architectures (Apache License 2.0),
specialized to 3D with instance normalization and LeakyReLU (or another
activation, for reuse in Primus's tokenizer). Module and
attribute names deliberately mirror the original so that state dict keys match
nnU-Net and nnssl checkpoints exactly. In particular, ConvNormAct registers its
convolution both as ``conv`` and inside ``all_modules``, so every convolution
weight appears under two keys, as it does upstream.
"""

from collections.abc import Callable, Sequence

import torch
from torch import nn

# MIST imports.
from mist.models.resenc.resenc_constants import ResEncConstants as constants


def leaky_relu() -> nn.Module:
    """Return the default ResEnc activation."""
    return nn.LeakyReLU(negative_slope=constants.NEGATIVE_SLOPE, inplace=True)


def _to_3d(value: int | Sequence[int]) -> list[int]:
    """Convert a scalar or sequence to a three-element list."""
    if isinstance(value, int):
        return [value] * 3
    return list(value)


class ConvNormAct(nn.Module):
    """Convolution, instance normalization, and optional activation.

    Attributes:
        conv: The 3D convolution.
        norm: Instance normalization.
        nonlin: Activation (LeakyReLU by default), present only if
            use_activation is True.
        all_modules: The above layers applied in sequence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
        use_activation: bool = True,
        conv_bias: bool = constants.CONV_BIAS,
        activation: Callable[[], nn.Module] = leaky_relu,
    ):
        super().__init__()
        kernel_size = _to_3d(kernel_size)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            _to_3d(stride),
            padding=[(k - 1) // 2 for k in kernel_size],
            bias=conv_bias,
        )
        self.norm = nn.InstanceNorm3d(out_channels, eps=constants.NORM_EPS, affine=True)
        ops = [self.conv, self.norm]
        if use_activation:
            self.nonlin = activation()
            ops.append(self.nonlin)
        self.all_modules = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    """A sequence of ConvNormAct layers; only the first one is strided."""

    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        initial_stride: int | Sequence[int],
    ):
        super().__init__()
        self.convs = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size, initial_stride),
            *[ConvNormAct(out_channels, out_channels, kernel_size, 1) for _ in range(1, num_convs)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stacked convolutions."""
        return self.convs(x)


class BasicBlockD(nn.Module):
    """Residual block following ResNet-D.

    Two 3x3x3 convolutions on the main path. When the block downsamples or
    changes the channel count, the skip path average-pools (if strided) and
    then projects with a 1x1x1 convolution instead of using a strided 1x1x1
    convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
        conv_bias: bool = constants.CONV_BIAS,
        activation: Callable[[], nn.Module] = leaky_relu,
    ):
        super().__init__()
        stride = _to_3d(stride)
        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            conv_bias=conv_bias,
            activation=activation,
        )
        self.conv2 = ConvNormAct(
            out_channels, out_channels, kernel_size, 1, use_activation=False, conv_bias=conv_bias
        )
        self.nonlin2 = activation()

        has_stride = any(s != 1 for s in stride)
        requires_projection = in_channels != out_channels
        if has_stride or requires_projection:
            ops: list[nn.Module] = []
            if has_stride:
                ops.append(nn.AvgPool3d(stride, stride))
            if requires_projection:
                ops.append(
                    ConvNormAct(
                        in_channels,
                        out_channels,
                        1,
                        1,
                        use_activation=False,
                        conv_bias=False,
                    )
                )
            self.skip: nn.Module = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        out += residual
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    """A sequence of BasicBlockD blocks; only the first one is strided."""

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        initial_stride: int | Sequence[int],
        conv_bias: bool = constants.CONV_BIAS,
        activation: Callable[[], nn.Module] = leaky_relu,
    ):
        super().__init__()
        block_kwargs = {"conv_bias": conv_bias, "activation": activation}
        self.blocks = nn.Sequential(
            BasicBlockD(in_channels, out_channels, kernel_size, initial_stride, **block_kwargs),
            *[
                BasicBlockD(out_channels, out_channels, kernel_size, 1, **block_kwargs)
                for _ in range(1, num_blocks)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stacked residual blocks."""
        return self.blocks(x)


class ResidualEncoder(nn.Module):
    """Residual encoder: a non-strided stem followed by residual stages.

    Attributes:
        stem: A single ConvNormAct layer with stride 1.
        stages: One StackedResidualBlocks per resolution stage. The first block
            of each stage applies that stage's stride.
        output_channels: Number of output channels of each stage.
        strides: Stride of each stage.
        kernel_sizes: Kernel size of each stage.
    """

    def __init__(
        self,
        in_channels: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        blocks_per_stage: Sequence[int],
    ):
        super().__init__()
        n_stages = len(features_per_stage)
        if not len(kernel_sizes) == len(strides) == len(blocks_per_stage) == n_stages:
            raise ValueError(
                "features_per_stage, kernel_sizes, strides, and blocks_per_stage "
                "must all have one entry per stage, but got lengths "
                f"{len(features_per_stage)}, {len(kernel_sizes)}, {len(strides)}, "
                f"and {len(blocks_per_stage)}."
            )

        self.stem = StackedConvBlocks(1, in_channels, features_per_stage[0], kernel_sizes[0], 1)

        stages = []
        stage_in_channels = features_per_stage[0]
        for s in range(n_stages):
            stages.append(
                StackedResidualBlocks(
                    blocks_per_stage[s],
                    stage_in_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    strides[s],
                )
            )
            stage_in_channels = features_per_stage[s]
        self.stages = nn.Sequential(*stages)

        self.output_channels = list(features_per_stage)
        self.strides = [_to_3d(s) for s in strides]
        self.kernel_sizes = [_to_3d(k) for k in kernel_sizes]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the output of every stage, from full to lowest resolution."""
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    """Plain UNet decoder for the residual encoder.

    Each decoder stage upsamples with a transposed convolution, concatenates the
    matching skip connection, and applies StackedConvBlocks. A 1x1x1
    segmentation layer is built for every decoder stage so that weights load
    the same way with or without deep supervision.

    Attributes:
        stages: Convolution blocks, ordered from lowest to highest resolution.
        transpconvs: Transposed convolutions, in the same order as stages.
        seg_layers: Segmentation layers, in the same order as stages.
    """

    def __init__(
        self,
        encoder: ResidualEncoder,
        out_channels: int,
        convs_per_stage: int,
    ):
        super().__init__()
        n_stages_encoder = len(encoder.output_channels)
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            features_below = encoder.output_channels[-s]
            features_skip = encoder.output_channels[-(s + 1)]
            stride = encoder.strides[-s]
            transpconvs.append(
                nn.ConvTranspose3d(
                    features_below,
                    features_skip,
                    stride,
                    stride,
                    bias=constants.CONV_BIAS,
                )
            )
            stages.append(
                StackedConvBlocks(
                    convs_per_stage,
                    2 * features_skip,
                    features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                )
            )
            seg_layers.append(nn.Conv3d(features_skip, out_channels, 1, 1, 0, bias=True))
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode the encoder skips.

        Args:
            skips: Encoder stage outputs, from full to lowest resolution.

        Returns:
            Decoder stage outputs (features, not segmentations), ordered from
            full to lowest resolution.
        """
        x = skips[-1]
        outputs = []
        for s, (transpconv, stage) in enumerate(zip(self.transpconvs, self.stages, strict=True)):
            x = transpconv(x)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = stage(x)
            outputs.append(x)
        return outputs[::-1]
