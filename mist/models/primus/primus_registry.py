"""Factory methods to create Primus and PrimusV2 model variants."""

# MIST imports.
from mist.models.model_registry import register_model
from mist.models.primus.mist_primus import Primus
from mist.models.primus.primus_constants import PrimusConstants as constants


def create_primus(variant: str, use_deeper_tokenizer: bool, **kwargs) -> Primus:
    """Factory method to create Primus model variants.

    Args:
        variant: Size variant: 'small', 'base', 'medium', or 'large'.
        use_deeper_tokenizer: If True, create PrimusV2.
        **kwargs: Model configuration, including in_channels, out_channels,
            and patch_size. target_spacing is accepted but not used.

    Returns:
        An instance of Primus.
    """
    required_keys = ["in_channels", "out_channels", "patch_size"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required key '{key}' in model configuration.")

    variant = variant.lower()
    if variant not in constants.VARIANTS:
        raise ValueError(
            f"Unknown Primus variant: '{variant}'. Choose from: {list(constants.VARIANTS)}."
        )
    depth, num_heads, embed_dim = constants.VARIANTS[variant]

    return Primus(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        patch_size=kwargs["patch_size"],
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        use_deeper_tokenizer=use_deeper_tokenizer,
    )


@register_model("primus-small")
def create_primus_small(**kwargs) -> Primus:
    """Create a small Primus model."""
    return create_primus("small", use_deeper_tokenizer=False, **kwargs)


@register_model("primus-base")
def create_primus_base(**kwargs) -> Primus:
    """Create a base Primus model."""
    return create_primus("base", use_deeper_tokenizer=False, **kwargs)


@register_model("primus-medium")
def create_primus_medium(**kwargs) -> Primus:
    """Create a medium Primus model (the paper's recommended default)."""
    return create_primus("medium", use_deeper_tokenizer=False, **kwargs)


@register_model("primus-large")
def create_primus_large(**kwargs) -> Primus:
    """Create a large Primus model."""
    return create_primus("large", use_deeper_tokenizer=False, **kwargs)


@register_model("primusv2-small")
def create_primusv2_small(**kwargs) -> Primus:
    """Create a small PrimusV2 model."""
    return create_primus("small", use_deeper_tokenizer=True, **kwargs)


@register_model("primusv2-base")
def create_primusv2_base(**kwargs) -> Primus:
    """Create a base PrimusV2 model."""
    return create_primus("base", use_deeper_tokenizer=True, **kwargs)


@register_model("primusv2-medium")
def create_primusv2_medium(**kwargs) -> Primus:
    """Create a medium PrimusV2 model."""
    return create_primus("medium", use_deeper_tokenizer=True, **kwargs)


@register_model("primusv2-large")
def create_primusv2_large(**kwargs) -> Primus:
    """Create a large PrimusV2 model."""
    return create_primus("large", use_deeper_tokenizer=True, **kwargs)
