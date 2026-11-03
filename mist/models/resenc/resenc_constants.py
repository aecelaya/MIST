"""Constants for the Residual Encoder UNet (ResEnc)."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ResEncConstants:
    """Data class storing constants for the Residual Encoder UNet.

    The block counts, filter widths, and layer settings match nnU-Net's ResEnc
    presets (M/L/XL share the same network; the presets only differ in the
    planned patch and batch size). Matching them exactly keeps encoder state
    dict keys and shapes compatible with nnU-Net and nnssl checkpoints, such as
    nnFoundationCNN.
    """

    # Residual blocks per encoder stage, truncated to the number of stages.
    BLOCKS_PER_STAGE = (1, 3, 4, 6, 6, 6)

    # Convolutions per decoder stage.
    CONVS_PER_DECODER_STAGE = 1

    INITIAL_FILTERS = 32  # Filters in the first stage; doubled at each stage.
    MAX_FILTERS_3D = 320  # Cap on the number of filters per stage.

    CONV_BIAS = True
    NORM_EPS = 1e-5
    NEGATIVE_SLOPE = 0.01  # LeakyReLU slope, also used for He initialization.

    # Number of deep supervision heads below the full-resolution output.
    NUM_DEEP_SUPERVISION_HEADS = 2
