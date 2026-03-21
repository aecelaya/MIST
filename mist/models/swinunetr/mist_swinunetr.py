"""MIST wrapper for SwinUNETR-V2."""

from collections import OrderedDict
from typing import Any, Dict, Union

import torch
from monai.networks.nets import SwinUNETR

from mist.models.base_model import MISTModel


class MistSwinUNETR(MISTModel):
    """MIST wrapper for SwinUNETR-V2.

    Wraps MONAI's SwinUNETR with use_v2=True to conform to MIST's training
    interface: training mode returns a dict with 'prediction' key; eval mode
    returns a plain tensor.

    SwinUNETR-V2 adds residual convolutional blocks at the start of each Swin
    Transformer stage, improving generalization on smaller medical imaging
    datasets compared to the original SwinUNETR.

    Input spatial dimensions must be divisible by 32 (patch_size=2 x 2^4
    downsampling stages).

    Attributes:
        model: The underlying MONAI SwinUNETR-V2 instance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 24,
        **kwargs: Any,
    ):
        """Initialize MistSwinUNETR.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output segmentation classes.
            feature_size: Base feature dimension. Controls model capacity:
                24 (small), 48 (base), 96 (large). Defaults to 24.
            **kwargs: Additional keyword arguments (ignored). Accepts
                patch_size and target_spacing for MIST interface
                compatibility.
        """
        super().__init__()
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
            In training mode: dict with 'prediction' key containing the
                segmentation tensor.
        """
        output = self.model(x)
        if self.training:
            return {"prediction": output}
        return output
