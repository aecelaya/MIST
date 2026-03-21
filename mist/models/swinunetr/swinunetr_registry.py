"""Factory functions for SwinUNETR-V2 model variants."""

from mist.models.swinunetr.mist_swinunetr import MistSwinUNETR
from mist.models.model_registry import register_model


def create_swinunetr(variant: str, **kwargs) -> MistSwinUNETR:
    """Factory method to create SwinUNETR-V2 model variants.

    All variants use SwinUNETR-V2 (residual convolutional blocks in the
    encoder). Size is controlled by the feature_size parameter:
      - small: feature_size=24  — lightweight, good for limited GPU memory
      - base:  feature_size=48  — standard configuration
      - large: feature_size=96  — highest capacity

    Note: Input spatial dimensions must be divisible by 32.

    Args:
        variant: Size variant. Options: 'small', 'base', 'large'.
        **kwargs: Additional keyword arguments including:
            - in_channels: Number of input channels.
            - out_channels: Number of output classes.
            - patch_size, target_spacing: Accepted but not used.

    Returns:
        An instance of MistSwinUNETR.
    """
    required_keys = ["in_channels", "out_channels"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    feature_size_map = {"small": 24, "base": 48, "large": 96}
    variant = variant.lower()
    if variant not in feature_size_map:
        raise ValueError(
            f"Unknown SwinUNETR variant: '{variant}'. "
            f"Choose from: {list(feature_size_map)}."
        )

    return MistSwinUNETR(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        feature_size=feature_size_map[variant],
    )


@register_model("swinunetr-small")
def create_swinunetr_small(**kwargs) -> MistSwinUNETR:
    """Create a small SwinUNETR-V2 model (feature_size=24)."""
    return create_swinunetr(variant="small", **kwargs)


@register_model("swinunetr-base")
def create_swinunetr_base(**kwargs) -> MistSwinUNETR:
    """Create a base SwinUNETR-V2 model (feature_size=48)."""
    return create_swinunetr(variant="base", **kwargs)


@register_model("swinunetr-large")
def create_swinunetr_large(**kwargs) -> MistSwinUNETR:
    """Create a large SwinUNETR-V2 model (feature_size=96)."""
    return create_swinunetr(variant="large", **kwargs)
