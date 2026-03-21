"""Constants for MGNet architectures."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class MGNetConstants:
    """Constants specific to MGNet (FMGNet / WNet)."""

    # Maximum number of input channels before applying a 1x1 projection to
    # reduce memory. Channels above this threshold are projected down before
    # the residual/basic block to avoid excessive memory usage at wide layers.
    REDUCTION_THRESHOLD = 512
