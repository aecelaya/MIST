"""Create variants of nnUNet model."""
# MIST imports.
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.model_registry import register_model


def _create_nnunet_base(use_pocket_model: bool, **kwargs) -> NNUNet:
    """Shared factory logic for nnUNet and nnunet-pocket.

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
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    return NNUNet(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        patch_size=kwargs["patch_size"],
        target_spacing=kwargs["target_spacing"],
        use_residual_blocks=True,
        use_deep_supervision=True,
        use_pocket_model=use_pocket_model,
    )


@register_model("nnunet")
def create_nnunet(**kwargs) -> NNUNet:
    """Create a standard nnUNet model with residual blocks and deep supervision."""
    return _create_nnunet_base(use_pocket_model=False, **kwargs)


@register_model("nnunet-pocket")
def create_nnunet_pocket(**kwargs) -> NNUNet:
    """Create a pocket nnUNet with constant 32-filter width across all depths."""
    return _create_nnunet_base(use_pocket_model=True, **kwargs)
