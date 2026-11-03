"""Constants for the Primus transformer architectures."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class PrimusConstants:
    """Data class storing constants for Primus and PrimusV2.

    Sizes and settings follow the Primus paper (Wald et al., 2025) and the
    official implementation in MIC-DKFZ's dynamic-network-architectures, so
    state dict keys and shapes match nnU-Net and nnssl checkpoints.
    """

    # Transformer size per variant: (depth, num_heads, embed_dim).
    VARIANTS = {
        "small": (12, 6, 396),
        "base": (12, 12, 792),
        "medium": (16, 12, 864),
        "large": (24, 16, 1056),
        # PrimusX: large's width with 40 layers. Not in the Primus paper; it is
        # the encoder of nnFoundationViT (nnssl), so its checkpoint loads in full.
        "x": (40, 16, 1056),
    }

    TOKEN_SIZE = 8  # Edge length of the 3D patch mapped to one token.
    MLP_RATIO = 4 * 2 / 3  # SwiGLU hidden width relative to embed_dim.
    LAYER_SCALE_INIT = 0.1  # LayerScale initial value.
    DROP_PATH_RATE = 0.2  # Stochastic depth rate of the last block.
    INIT_STD = 0.02  # Std of the truncated normal init for linear layers.
    ROPE_TEMPERATURE = 10000.0
    LAYER_NORM_EPS = 1e-5  # Transformer LayerNorm (torch default).
    DECODER_NORM_EPS = 1e-6  # Channel-first LayerNorm in the patch decoder.
    HE_NEGATIVE_SLOPE = 1e-2  # Kaiming init slope for the conv layers.

    # PrimusV2 iterative tokenizer: stem width and residual blocks per level.
    TOKENIZER_BASE_FEATURES = 32
    TOKENIZER_BLOCKS_PER_LEVEL = (1, 1, 1)
