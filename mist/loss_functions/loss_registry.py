"""Registry for segmentation loss functions in MIST."""

from collections.abc import Callable

from mist.loss_functions.base import SegmentationLoss

LOSS_REGISTRY: dict[str, Callable[..., SegmentationLoss]] = {}


def register_loss(name: str):
    """Decorator to register a loss function class.

    Automatically normalizes the name to lowercase to avoid config case errors.

    Args:
        name: String identifier for the loss (i.e., "dice", "gsl").

    Usage:
        @register_loss("dice")
        class DiceLoss(SegmentationLoss): ...
    """
    def decorator(cls: type[SegmentationLoss]) -> type[SegmentationLoss]:
        # Normalize to lowercase for robustness
        normalized_name = name.lower()

        if normalized_name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{normalized_name}' is already registered.")

        LOSS_REGISTRY[normalized_name] = cls
        return cls
    return decorator


def get_loss(name: str) -> Callable[..., SegmentationLoss]:
    """Retrieve a registered loss function by name.

    Args:
        name: The string name of the registered loss (case-insensitive).

    Returns:
        A callable class or factory function that returns a loss module.

    Raises:
        ValueError: If the loss is not registered.
    """
    normalized_name = name.lower()

    if normalized_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{name}' is not registered. "
            f"Available: {list_registered_losses()}"
        )
    return LOSS_REGISTRY[normalized_name]


def list_registered_losses() -> list[str]:
    """List all registered loss function names."""
    return sorted(LOSS_REGISTRY.keys())
