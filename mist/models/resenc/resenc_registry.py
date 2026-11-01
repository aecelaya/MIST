"""Factory method to create the Residual Encoder UNet (ResEnc)."""

# MIST imports.
from mist.models.model_registry import register_model
from mist.models.resenc.mist_resenc import ResEncUNet


@register_model("resenc")
def create_resenc(**kwargs) -> ResEncUNet:
    """Create a Residual Encoder UNet with deep supervision.

    nnU-Net's ResEnc M/L/XL presets share this network; they differ only in the
    planned patch and batch size, which MIST sets through its own analysis.

    Args:
        **kwargs: Model configuration, including in_channels, out_channels,
            patch_size, and target_spacing.

    Returns:
        An instance of ResEncUNet.
    """
    required_keys = ["in_channels", "out_channels", "patch_size", "target_spacing"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required key '{key}' in model configuration.")

    return ResEncUNet(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        patch_size=kwargs["patch_size"],
        target_spacing=kwargs["target_spacing"],
        use_deep_supervision=True,
    )
