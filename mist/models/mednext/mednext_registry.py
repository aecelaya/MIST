"""Factory method to create MedNeXt model variants."""
# MIST imports.
from mist.models.mednext.mist_mednext import MedNeXt
from mist.models.model_registry import register_model


def create_mednext(variant: str, **kwargs) -> MedNeXt:
    """Factory method to create MedNeXt model variants.

    Args:
        variant: The MedNeXt variant to create. Options are small 'S', base 'B',
            medium 'M', and large 'L'.
        **kwargs: Additional keyword arguments for model configuration,
            including:
            - in_channels: Number of input channels.
            - out_channels: Number of output classes for segmentation.
            - use_residual_blocks: Turn on residual connections in the model.
            - use_deep_supervision: Turn on deep supervision in the model.
            - use_pocket_model: Use pocket version of the model.

    Returns:
        An instance of the requested MedNeXt variant.
    """
    # Validate presence of required keys to avoid obscure KeyErrors.
    required_keys = [
        "in_channels", "out_channels", "use_residual_blocks",
        "use_deep_supervision", "use_pocket_model"
    ]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    common_args = {
        "spatial_dims": 3,
        "in_channels": kwargs["in_channels"],
        "out_channels": kwargs["out_channels"],
        # Larger depthwise kernels (5, 7) are MedNeXt's distinguishing feature
        # over nnUNet. Defaults to 3 for memory efficiency; set to 5 or 7 for
        # higher-capacity configurations on well-resourced hardware.
        "kernel_size": kwargs.get("kernel_size", 3),
        "use_deep_supervision": kwargs["use_deep_supervision"],
        "use_residual_blocks": kwargs["use_residual_blocks"],
        "norm_type": "group",
        "global_resp_norm": False,
        "init_filters": 32,
        "use_pocket_model": kwargs["use_pocket_model"],
    }

    variant = variant.upper()
    if variant == "S":
        return MedNeXt(
            encoder_expansion_ratio=2,
            decoder_expansion_ratio=2,
            bottleneck_expansion_ratio=2,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            **common_args,
        )
    elif variant == "B":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            **common_args,
        )
    elif variant == "M":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(3, 4, 4, 4),
            blocks_bottleneck=4,
            blocks_up=(4, 4, 4, 3),
            **common_args,
        )
    elif variant == "L":
        return MedNeXt(
            encoder_expansion_ratio=(3, 4, 8, 8),
            decoder_expansion_ratio=(8, 8, 4, 3),
            bottleneck_expansion_ratio=8,
            blocks_down=(3, 4, 8, 8),
            blocks_bottleneck=8,
            blocks_up=(8, 8, 4, 3),
            **common_args,
        )
    else:
        raise ValueError(f"Invalid MedNeXt variant: '{variant}'")


@register_model("mednext-small")
def create_mednext_small(**kwargs) -> MedNeXt:
    """Create a small MedNeXt model."""
    return create_mednext(variant="S", **kwargs)


@register_model("mednext-base")
def create_mednext_base(**kwargs) -> MedNeXt:
    """Create a base MedNeXt model."""
    return create_mednext(variant="B", **kwargs)


@register_model("mednext-medium")
def create_mednext_medium(**kwargs) -> MedNeXt:
    """Create a medium MedNeXt model."""
    return create_mednext(variant="M", **kwargs)


@register_model("mednext-large")
def create_mednext_large(**kwargs) -> MedNeXt:
    """Create a large MedNeXt model."""
    return create_mednext(variant="L", **kwargs)
