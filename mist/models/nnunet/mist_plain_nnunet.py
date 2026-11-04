"""MIST implementation of nnU-Net's plain convolutional UNet."""

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# MIST imports.
from mist.models.base_model import MISTModel
from mist.models.nnunet import nnunet_utils
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants
from mist.models.resenc.resenc_blocks import StackedConvBlocks, UNetDecoder


class PlainConvEncoder(nn.Module):
    """UNet encoder of plain convolutions; each stage's first one is strided.

    Module and attribute names mirror nnU-Net's PlainConvEncoder so that state
    dict keys match its checkpoints.

    Attributes:
        stages: One stack of convolutions per resolution, from full to lowest.
        output_channels: Number of output channels per stage.
        strides: Stride of each stage.
        kernel_sizes: Kernel size of each stage.
    """

    def __init__(
        self,
        in_channels: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        convs_per_stage: int,
    ):
        super().__init__()
        stages = []
        for features, kernel_size, stride in zip(
            features_per_stage, kernel_sizes, strides, strict=True
        ):
            stages.append(
                nn.Sequential(
                    StackedConvBlocks(convs_per_stage, in_channels, features, kernel_size, stride)
                )
            )
            in_channels = features
        self.stages = nn.Sequential(*stages)
        self.output_channels = list(features_per_stage)
        self.strides = [list(s) for s in strides]
        self.kernel_sizes = [list(k) for k in kernel_sizes]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return each stage's output, from full to lowest resolution."""
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class PlainNNUNet(MISTModel):
    """nnU-Net's default network, PlainConvUNet (Isensee et al., 2021).

    Each stage has two plain convolutions in both the encoder and decoder, and
    the topology follows nnU-Net's planner (see
    nnunet_utils.get_plain_unet_params) rather than MIST's other adaptive
    networks. Parameter names match nnU-Net's PlainConvUNet, except that the
    decoder keeps no reference to the encoder, so its keys omit the duplicate
    "decoder.encoder." entries found upstream.

    Attributes:
        encoder: The plain convolutional encoder.
        decoder: The plain UNet decoder.
        use_deep_supervision: Whether to return deep supervision outputs during
            training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Sequence[int],
        target_spacing: Sequence[float],
        use_deep_supervision: bool,
    ):
        super().__init__()
        if len(patch_size) != 3:
            raise ValueError(
                f"PlainNNUNet requires a 3D patch_size, but got {len(patch_size)} "
                "dimensions. For anisotropic data use a thin Z slice "
                "(e.g., 256×256×5) rather than a 2D patch."
            )

        kernel_sizes, strides, divisors = nnunet_utils.get_plain_unet_params(
            patch_size, target_spacing
        )
        incompatible = [
            (size, divisor)
            for size, divisor in zip(patch_size, divisors, strict=True)
            if size % divisor != 0
        ]
        if incompatible:
            raise ValueError(
                f"Patch size {list(patch_size)} doesn't fit the network planned for "
                f"it: each axis must be divisible by {divisors}, or skip connections "
                "won't match in size. Choose a patch size whose axes are multiples of "
                "these values."
            )

        features_per_stage = [
            min(2**i * constants.INITIAL_FILTERS, constants.MAX_FILTERS_3D)
            for i in range(len(strides))
        ]
        self.encoder = PlainConvEncoder(
            in_channels=in_channels,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            convs_per_stage=constants.PLAIN_CONVS_PER_STAGE,
        )
        self.decoder = UNetDecoder(
            encoder=self.encoder,
            out_channels=out_channels,
            convs_per_stage=constants.PLAIN_CONVS_PER_STAGE,
        )
        self.use_deep_supervision = use_deep_supervision

        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """Initialize convolutions with Kaiming normal weights and zero bias."""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=constants.NEGATIVE_SLOPE)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return the encoder weights."""
        return OrderedDict({k: v for k, v in self.state_dict().items() if k.startswith("encoder.")})

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        """Forward pass.

        Returns:
            In eval mode, the full-resolution prediction. In training mode, a
            dict with the prediction under "prediction" and, if deep supervision
            is enabled, the lower-resolution predictions interpolated to full
            resolution under "deep_supervision" (otherwise None).
        """
        decoder_outputs = self.decoder(self.encoder(x))
        seg_layers = self.decoder.seg_layers
        prediction = seg_layers[-1](decoder_outputs[0])

        if not self.training:
            return prediction

        deep_supervision = None
        if self.use_deep_supervision:
            # decoder_outputs[k] is k levels below full resolution and pairs with
            # seg_layers[-(k + 1)].
            num_heads = min(constants.PLAIN_DEEP_SUPERVISION_HEADS, len(decoder_outputs) - 1)
            deep_supervision = [
                F.interpolate(seg_layers[-(k + 1)](decoder_outputs[k]), prediction.shape[2:])
                for k in range(1, num_heads + 1)
            ]

        return {"prediction": prediction, "deep_supervision": deep_supervision}
