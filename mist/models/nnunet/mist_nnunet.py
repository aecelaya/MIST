"""MIST implementation of nnUNet."""
from collections import OrderedDict
from collections.abc import Sequence
from typing import Union, Dict
import torch

# MIST imports.
from mist.models.base_model import MISTModel
from mist.models.nnunet import dynamic_unet
from mist.models.nnunet import nnunet_utils
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants


class NNUNet(MISTModel):
    """MIST implementation of nnUNet.

    This is essentially a wrapper around the DynamicUNet class that we implement
    in dynamic_unet.py. It is used to create a nnUNet model with the specified
    parameters. These parameters include the number of input channels, number of
    output channels, the image size (i.e., ROI size), the image spacing, whether
    to use residual blocks, whether to use deep supervision, and whether to use
    the pocket version of the model.

    All inputs are expected to be 3D volumes. For highly anisotropic datasets
    a quasi-2D approach can be achieved by using a thin patch in one dimension
    (e.g., 256×256×5); the adaptive topology will automatically use stride-1 in
    the thin axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Sequence[int],
        target_spacing: Sequence[float],
        use_residual_blocks: bool,
        use_deep_supervision: bool,
        use_pocket_model: bool,
    ):
        super().__init__()
        if len(patch_size) != 3:
            raise ValueError(
                f"NNUNet requires a 3D patch_size, but got {len(patch_size)} "
                "dimensions. For anisotropic data use a thin Z slice "
                "(e.g., 256×256×5) rather than a 2D patch."
            )
        if len(patch_size) != len(target_spacing):
            raise ValueError(
                "patch_size and target_spacing must have the same number of "
                f"dimensions, but got {len(patch_size)} and "
                f"{len(target_spacing)}."
            )

        kernel_sizes, strides, _ = (
            nnunet_utils.get_unet_params(patch_size, target_spacing)
        )

        if use_pocket_model:
            filters = [constants.INITIAL_FILTERS] * len(strides)
        else:
            filters = [
                min(2 ** i * constants.INITIAL_FILTERS, constants.MAX_FILTERS_3D)
                for i in range(len(strides))
            ]

        self.unet = dynamic_unet.DynamicUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=filters,
            act_name=constants.ACTIVATION,
            norm_name=constants.NORMALIZATION,
            use_residual_blocks=use_residual_blocks,
            use_deep_supervision=use_deep_supervision,
            trans_bias=True,
        )

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder weights: input block, encoder layers, bottleneck."""
        encoder_prefixes = (
            "unet.input_block.",
            "unet.encoder_layers.",
            "unet.bottleneck.",
        )
        return OrderedDict(
            {k: v for k, v in self.state_dict().items()
             if k.startswith(encoder_prefixes)}
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict]:
        """Forward pass for nnUNet."""
        return self.unet(x)
