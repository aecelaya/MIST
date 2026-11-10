"""Constants for nnUNet."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class NNUnetConstants:
    """Data class storing constants for nnUNet."""

    MAX_DEPTH = 5  # Maximum depth of the network.
    DEFAULT_STRIDE = 2  # Default stride for convolutional layers.
    ANISOTROPIC_MAX_RATIO = 2.0  # Maximum of spacing[i] / min(spacing).
    ANISOTROPIC_STRIDE = 1  # Stride for anisotropic dimension.
    DEFAULT_KERNEL_SIZE = 3  # Default kernel size for convolutional layers.
    ANISOTROPIC_KERNEL_SIZE = 1  # Kernel size for anisotropic dimension.
    MIN_SIZE_FOR_STRIDE = 8  # Minimum size for default stride to be applied.
    INITIAL_FILTERS = 32  # Initial number of filters for the network.
    MAX_FILTERS_3D = 320  # Maximum number of filters for 3D network.

    # Activation function for nnUNet.
    ACTIVATION = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})

    # Normalization layer for nnUNet.
    NORMALIZATION = ("INSTANCE", {"affine": True})

    # Negative slope parameter for leaky ReLU activation function.
    NEGATIVE_SLOPE = 0.01

    # Initial value for bias in convolutional layers.
    INITIAL_BIAS_VALUE = 0

    # nnunet-plain: nnU-Net's planner stops downsampling an axis once halving it
    # would leave fewer than this many voxels in the bottleneck, and has no
    # depth limit.
    PLAIN_MIN_FEATURE_MAP_SIZE = 4
    PLAIN_CONVS_PER_STAGE = 2  # Convolutions per encoder and decoder stage.
    PLAIN_DEEP_SUPERVISION_HEADS = 2  # Heads below the full-resolution output.
