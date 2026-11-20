"""Create variants of nnUNet model."""

# MIST imports.
from mist.models.model_registry import register_model
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.nnunet.mist_plain_nnunet import PlainNNUNet


def _create_nnunet_base(use_pocket_model: bool, **kwargs) -> NNUNet:
    """Shared factory logic for nnunet-res and nnunet-res-pocket.

    Args:
        use_pocket_model: Whether to use constant-width filters (pocket mode).
        **kwargs: Model configuration, must include in_channels, out_channels,
            patch_size, and target_spacing.

    Returns:
        An instance of the NNUNet model.
    """
    required_keys = ["in_channels", "out_channels", "patch_size", "target_spacing"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required key '{key}' in model configuration.")

    return NNUNet(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        patch_size=kwargs["patch_size"],
        target_spacing=kwargs["target_spacing"],
        use_residual_blocks=True,
        use_deep_supervision=True,
        use_pocket_model=use_pocket_model,
    )


def _create_nnunet_plain_base(use_pocket_model: bool, **kwargs) -> PlainNNUNet:
    """Shared factory logic for nnunet-plain and nnunet-plain-pocket.

    Args:
        use_pocket_model: Whether to use a constant 32 filters per stage.
        **kwargs: Model configuration, must include in_channels, out_channels,
            patch_size, and target_spacing.

    Returns:
        An instance of the PlainNNUNet model.
    """
    required_keys = ["in_channels", "out_channels", "patch_size", "target_spacing"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required key '{key}' in model configuration.")

    return PlainNNUNet(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        patch_size=kwargs["patch_size"],
        target_spacing=kwargs["target_spacing"],
        use_deep_supervision=True,
        use_pocket_model=use_pocket_model,
    )


@register_model("nnunet-plain")
def create_nnunet_plain(**kwargs) -> PlainNNUNet:
    """Create nnU-Net's PlainConvUNet with deep supervision, MIST's default network."""
    return _create_nnunet_plain_base(use_pocket_model=False, **kwargs)


@register_model("nnunet-plain-pocket")
def create_nnunet_plain_pocket(**kwargs) -> PlainNNUNet:
    """Create nnunet-plain with a constant 32 filters in every stage."""
    return _create_nnunet_plain_base(use_pocket_model=True, **kwargs)


@register_model("nnunet-res")
def create_nnunet_res(**kwargs) -> NNUNet:
    """Create a residual nnUNet with deep supervision."""
    return _create_nnunet_base(use_pocket_model=False, **kwargs)


@register_model("nnunet")
def create_nnunet(**kwargs) -> NNUNet:
    """Create the same network as nnunet-res; this name is kept for compatibility."""
    return create_nnunet_res(**kwargs)


@register_model("nnunet-res-pocket")
def create_nnunet_res_pocket(**kwargs) -> NNUNet:
    """Create nnunet-res with a constant 32 filters in every stage."""
    return _create_nnunet_base(use_pocket_model=True, **kwargs)


@register_model("nnunet-pocket")
def create_nnunet_pocket(**kwargs) -> NNUNet:
    """Create the same network as nnunet-res-pocket; this name is kept for compatibility."""
    return create_nnunet_res_pocket(**kwargs)
