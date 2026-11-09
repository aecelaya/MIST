"""MIST implementation of the Residual Encoder UNet (ResEnc)."""

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# MIST imports.
from mist.models.base_model import MISTModel
from mist.models.nnunet import nnunet_utils
from mist.models.resenc.resenc_blocks import BasicBlockD, ResidualEncoder, UNetDecoder
from mist.models.resenc.resenc_constants import ResEncConstants as constants


class ResEncUNet(MISTModel):
    """Residual Encoder UNet from nnU-Net Revisited (Isensee et al., 2024).

    The encoder stacks ResNet-D residual blocks, with more blocks in deeper
    stages, and the decoder is a lightweight plain UNet decoder. The topology
    (number of stages, kernel sizes, and strides) adapts to patch_size and
    target_spacing exactly as it does for MIST's nnUNet, so anisotropic axes use
    stride 1.

    The encoder's parameter names match nnU-Net's ResidualEncoderUNet, so
    encoder weights from nnU-Net or nnssl checkpoints (e.g., nnFoundationCNN)
    load by key. The decoder does not keep a reference to the encoder, so its
    keys omit the duplicate "decoder.encoder." entries found upstream.

    Attributes:
        encoder: The residual encoder.
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
                f"ResEncUNet requires a 3D patch_size, but got {len(patch_size)} "
                "dimensions. For anisotropic data use a thin Z slice "
                "(e.g., 256×256×5) rather than a 2D patch."
            )
        if len(patch_size) != len(target_spacing):
            raise ValueError(
                "patch_size and target_spacing must have the same number of "
                f"dimensions, but got {len(patch_size)} and "
                f"{len(target_spacing)}."
            )

        kernel_sizes, strides, _ = nnunet_utils.get_unet_params(patch_size, target_spacing)
        n_stages = len(strides)
        features_per_stage = [
            min(2**i * constants.INITIAL_FILTERS, constants.MAX_FILTERS_3D) for i in range(n_stages)
        ]

        self.encoder = ResidualEncoder(
            in_channels=in_channels,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            blocks_per_stage=constants.BLOCKS_PER_STAGE[:n_stages],
        )
        self.decoder = UNetDecoder(
            encoder=self.encoder,
            out_channels=out_channels,
            convs_per_stage=constants.CONVS_PER_DECODER_STAGE,
        )
        self.use_deep_supervision = use_deep_supervision

        self.apply(self.initialize_weights)
        self.decoder.freeze_unused_seg_layers(
            1 + constants.NUM_DEEP_SUPERVISION_HEADS if use_deep_supervision else 1
        )

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """Initialize weights as nnU-Net does for residual encoders.

        Convolutions use Kaiming normal initialization with zero bias. The last
        normalization layer of each residual block is zeroed so that every block
        starts as an identity mapping.
        """
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=constants.NEGATIVE_SLOPE)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, BasicBlockD):
            nn.init.constant_(module.conv2.norm.weight, 0)
            nn.init.constant_(module.conv2.norm.bias, 0)

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder weights: stem and residual stages."""
        return OrderedDict({k: v for k, v in self.state_dict().items() if k.startswith("encoder.")})

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        """Forward pass for the Residual Encoder UNet.

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
            num_heads = min(constants.NUM_DEEP_SUPERVISION_HEADS, len(decoder_outputs) - 1)
            deep_supervision = [
                F.interpolate(seg_layers[-(k + 1)](decoder_outputs[k]), prediction.shape[2:])
                for k in range(1, num_heads + 1)
            ]

        return {"prediction": prediction, "deep_supervision": deep_supervision}
