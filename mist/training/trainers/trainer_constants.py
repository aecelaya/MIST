"""Constants for base trainer configurations in MIST."""

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class TrainerConstants:
    """Configuration constants for base trainer."""

    # Epsilon value for numerical stability in optimizers.
    OPTIMIZER_EPS: float = 1e-8

    # Gradient clipping value.
    GRAD_CLIP_VALUE: float = 1.0

    # Loss names that require sddl_spacing_xyz at construction.
    SPACING_AWARE_LOSSES: frozenset[str] = frozenset({"volumetric_sddl", "vessel_sddl"})

    # Loss names that require precomputed distance transform maps (DTMs).
    DTM_AWARE_LOSSES: frozenset[str] = frozenset({"bl", "hdos", "gsl"})

    # Loss names that blend two terms via a schedulable alpha weight.
    COMPOSITE_LOSSES: frozenset[str] = frozenset(
        {"bl", "hdos", "gsl", "cldice", "volumetric_sddl", "vessel_sddl"}
    )

    # SDDL defaults.
    DEFAULT_SDDL_SPACING: Sequence[float] = (1.0, 1.0, 1.0)
    DEFAULT_SDDL_TAU_MM: str = "auto"
    DEFAULT_SDDL_TAU_MM_SAFETY_FACTOR: float = 1.25

    # Model selection. With a validation split carved from the training data
    # (val_percent > 0), the saved model is the one with the lowest validation
    # loss after exponential smoothing with this decay, which damps
    # epoch-to-epoch noise (nnU-Net smooths its pseudo-Dice the same way).
    VAL_LOSS_EMA_DECAY: float = 0.9

    # Fewer validation cases than this make best-model selection noisy.
    MIN_VAL_CASES_FOR_SELECTION: int = 5
