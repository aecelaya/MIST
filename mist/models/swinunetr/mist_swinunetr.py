"""MIST wrapper for SwinUNETR-V2."""

from collections import OrderedDict
from typing import Any, Dict, Union

import torch
from torch import nn
from monai.networks.nets import SwinUNETR

from mist.models.base_model import MISTModel


class MistSwinUNETR(MISTModel):
    """MIST wrapper for SwinUNETR-V2.

    Wraps MONAI's SwinUNETR with use_v2=True to conform to MIST's training
    interface: training mode returns a dict with 'prediction' and
    'deep_supervision' keys; eval mode returns a plain tensor.

    SwinUNETR-V2 adds residual convolutional blocks at the start of each Swin
    Transformer stage, improving generalization on smaller medical imaging
    datasets compared to the original SwinUNETR.

    Note: SwinUNETR does not support deep supervision natively. When
    use_deep_supervision=True, the 'deep_supervision' key in the training
    output will always be None. Input spatial dimensions must be divisible
    by 32 (patch_size=2 × 2^4 downsampling stages).

    Attributes:
        use_deep_supervision: Whether deep supervision was requested.
        model: The underlying MONAI SwinUNETR-V2 instance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 24,
        use_deep_supervision: bool = True,
        **kwargs: Any,
    ):
        """Initialize MistSwinUNETR.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output segmentation classes.
            feature_size: Base feature dimension. Controls model capacity:
                24 (small), 48 (base), 96 (large). Defaults to 24.
            use_deep_supervision: Accepted for interface compatibility.
                SwinUNETR does not support deep supervision natively;
                the 'deep_supervision' output is always None. Defaults to True.
            **kwargs: Additional keyword arguments (ignored). Accepts
                patch_size, target_spacing, use_residual_blocks, and
                use_pocket_model for MIST interface compatibility.
        """
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_v2=True,
            spatial_dims=3,
        )

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder weights: Swin Transformer backbone only."""
        return OrderedDict(
            {k: v for k, v in self.state_dict().items()
             if k.startswith("model.swinViT.")}
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W). Spatial dimensions
                must be divisible by 32.

        Returns:
            In eval mode: segmentation tensor of shape (B, out_channels, D, H, W).
            In training mode: dict with keys:
                - 'prediction': segmentation tensor.
                - 'deep_supervision': None (not supported by SwinUNETR).
        """
        output = self.model(x)
        if self.training:
            return {"prediction": output, "deep_supervision": None}
        return output
