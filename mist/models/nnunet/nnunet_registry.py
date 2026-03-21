"""Create variants of nnUNet model."""
# MIST imports.
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.model_registry import register_model


@register_model("nnunet")
def create_nnunet(**kwargs) -> NNUNet:
    """Factory method to create nnUNet model.

    There is only one variant of nnUNet, so this function serves as a
    factory method to instantiate the model with the given parameters. In the
    future, if more variants are added, this function can be extended to handle
    those cases.

    Args:
        **kwargs: Additional keyword arguments for model configuration,
            including:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels (classes).
            - patch_size: Size of the region of interest (ROI).
            - target_spacing: Image spacing of the input.
            - use_residual_blocks: Whether to use residual connections.
            - use_deep_supervision: Whether to use deep supervision.
            - use_pocket_model: Whether to use the pocket model variant.

    Returns:
        An instance of the NNUNet model.
    """
    # Validate presence of required keys to avoid obscure KeyErrors.
    required_keys = [
        "in_channels", "out_channels", "patch_size", "target_spacing",
        "use_residual_blocks", "use_deep_supervision", "use_pocket_model"
    ]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    common_args = {
        "in_channels": kwargs["in_channels"],
        "out_channels": kwargs["out_channels"],
        "patch_size": kwargs["patch_size"],
        "target_spacing": kwargs["target_spacing"],
        "use_residual_blocks": kwargs["use_residual_blocks"],
        "use_deep_supervision": kwargs["use_deep_supervision"],
        "use_pocket_model": kwargs["use_pocket_model"],
        "spatial_dims": 3,
    }
    return NNUNet(**common_args)
