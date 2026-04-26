"""Constants for base trainer configurations in MIST."""

from collections.abc import Sequence
import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainerConstants:
    """Configuration constants for base trainer."""
    # Epsilon value for numerical stability in optimizers.
    AMP_EPS: float = 1e-4
    NO_AMP_EPS: float = 1e-8

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
