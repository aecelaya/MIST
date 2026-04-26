"""Test time augmentation (TTA) strategies for MIST.

This module defines the TTA strategies used in MIST. Each strategy is
a list of TTA transforms that are applied to the input image. The predictions
then undergo an inverse transformation to obtain the final prediction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar
from collections.abc import Callable

from mist.inference.tta.transforms import AbstractTransform, get_transform

# Registry for strategy classes. Stores classes; instances are created on
# demand by get_strategy() to avoid shared mutable state.
T = TypeVar("T", bound="TTAStrategy")
TTA_STRATEGY_REGISTRY: dict[str, type[TTAStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a TTA strategy class by name."""
    def decorator(cls: type[T]) -> type[T]:
        if not issubclass(cls, TTAStrategy):
            raise TypeError(f"{cls.__name__} must inherit from TTAStrategy.")
        if name in TTA_STRATEGY_REGISTRY:
            raise KeyError(f"Strategy '{name}' is already registered.")
        TTA_STRATEGY_REGISTRY[name] = cls  # Register class, not an instance
        return cls
    return decorator


def list_strategies() -> list[str]:
    """List all registered TTA strategies."""
    return list(TTA_STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> TTAStrategy:
    """Retrieve a fresh instance of a registered TTA strategy by name."""
    if name not in TTA_STRATEGY_REGISTRY:
        raise KeyError(
            f"TTA strategy '{name}' is not registered. "
            f"Available: [{', '.join(list_strategies())}]."
        )
    return TTA_STRATEGY_REGISTRY[name]()


class TTAStrategy(ABC):
    """Abstract base class for all TTA strategies.

    This simply generates a list of transforms to apply at inference time.
    This class does not execute the transforms. The transforms are executed in
    core inference logic.
    """

    def __init__(self):
        """Initialize the transform with a name."""
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def get_transforms(self) -> list[AbstractTransform]:
        """Return a list of forward/inverse transforms to apply at inference."""
        pass  # pylint: disable=unnecessary-pass # pragma: no cover

    def __call__(self) -> list[AbstractTransform]:
        return self.get_transforms()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TTAStrategy) and self.name == other.name


@register_strategy("none")
class NoTTAStrategy(TTAStrategy):
    """No TTA strategy.

    This strategy does not apply any transformations to the input image.
    """

    def get_transforms(self) -> list[AbstractTransform]:
        return [get_transform("identity")]


@register_strategy("all_flips")
class AllFlipsStrategy(TTAStrategy):
    """All flips TTA strategy.

    This strategy applies all possible flips to the input image.
    """

    def get_transforms(self) -> list[AbstractTransform]:
        return [
            get_transform("identity"),
            get_transform("flip_x"),
            get_transform("flip_y"),
            get_transform("flip_z"),
            get_transform("flip_xy"),
            get_transform("flip_xz"),
            get_transform("flip_yz"),
            get_transform("flip_xyz"),
        ]
