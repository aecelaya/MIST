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
        patch_size: Any = None,
        feature_size: int = 24,
        **kwargs: Any,
    ):
        """Initialize MistSwinUNETR.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output segmentation classes.
            patch_size: Expected input spatial dimensions. All values must be
                divisible by 32 (patch_size=2 tokenizer × 2^4 downsampling
                stages). Ignored at forward time — used only for validation.
            feature_size: Base feature dimension. Controls model capacity:
                24 (small), 48 (base), 96 (large). Defaults to 24.
            **kwargs: Additional keyword arguments forwarded from the MIST
                interface (e.g. target_spacing). Unused.

        Raises:
            ValueError: If any dimension of patch_size is not divisible
                by 32.
        """
        super().__init__()

        if patch_size is not None:
            bad_dims = [d for d in patch_size if d % 32 != 0]
            if bad_dims:
                raise ValueError(
                    f"SwinUNETR requires all patch_size dimensions to be "
                    f"divisible by 32. Got patch_size={list(patch_size)}, "
                    f"offending dimensions: {bad_dims}."
                )

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
                segmentation tensor and 'deep_supervision' key set to None
                (SwinUNETR does not support deep supervision).
        """
        output = self.model(x)
        if self.training:
            return {"prediction": output, "deep_supervision": None}
        return output
