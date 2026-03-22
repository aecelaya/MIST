"""Constants for learning rate configurations in MIST."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class LRSchedulerConstants:
    """Configuration constants for optimizers."""
    # SGD optimizer defaults.
    POLYNOMIAL_DECAY: float = 0.9
    CONSTANT_LR_FACTOR: float = 1.0
    # Warmup starting LR fraction (1% of target LR).
    WARMUP_START_FACTOR: float = 0.01
