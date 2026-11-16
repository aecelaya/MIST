"""Factory method to create the Residual Encoder UNet (ResEnc)."""

# MIST imports.
from mist.models.model_registry import register_model
from mist.models.resenc.mist_resenc import ResEncUNet


def _create_resenc_base(use_pocket_model: bool, **kwargs) -> ResEncUNet:
    """Shared factory logic for resenc and resenc-pocket.

    Args:
        use_pocket_model: Whether every stage keeps 32 filters.
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
        use_pocket_model=use_pocket_model,
    )


@register_model("resenc")
def create_resenc(**kwargs) -> ResEncUNet:
    """Create a Residual Encoder UNet with deep supervision.

    nnU-Net's ResEnc M/L/XL presets share this network; they differ only in the
    planned patch and batch size, which MIST sets through its own analysis.
    """
    return _create_resenc_base(use_pocket_model=False, **kwargs)


@register_model("resenc-pocket")
def create_resenc_pocket(**kwargs) -> ResEncUNet:
    """Create a pocket ResEnc UNet with constant 32-filter width across all stages."""
    return _create_resenc_base(use_pocket_model=True, **kwargs)
