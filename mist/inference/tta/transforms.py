"""Test time augmentation (TTA) transforms for MIST."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, TypeVar, Callable
import torch

# Registry for transform classes. Stores classes; instances are created on
# demand by get_transform() to avoid shared mutable state.
T = TypeVar("T", bound="AbstractTransform")
TTA_TRANSFORM_REGISTRY: Dict[str, Type["AbstractTransform"]] = {}


def register_transform(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a TTA transform class by name."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractTransform):
            raise TypeError(f"{cls.__name__} must subclass AbstractTransform.")
        if name in TTA_TRANSFORM_REGISTRY:
            raise KeyError(f"Transform '{name}' is already registered.")
        TTA_TRANSFORM_REGISTRY[name] = cls  # Register the class, not an instance.
        return cls
    return decorator


def list_transforms() -> List[str]:
    """List all registered TTA transforms."""
    return list(TTA_TRANSFORM_REGISTRY.keys())


def get_transform(name: str) -> "AbstractTransform":
    """Retrieve a fresh instance of a registered TTA transform by name."""
    if name not in TTA_TRANSFORM_REGISTRY:
        raise KeyError(
            f"TTA transform '{name}' is not registered. "
            f"Available: [{', '.join(list_transforms())}]."
        )
    return TTA_TRANSFORM_REGISTRY[name]()


class AbstractTransform(ABC):
    """Abstract base class for all TTA transforms.

    All TTA transforms should have a `forward` method that applies the
    transformation to the input image and an `inverse` method that inverts
    the transformation on the prediction. The `__call__` method is an alias
    for the `forward` method, allowing the transform to be called like a
    function.
    """

    def __init__(self):
        """Initialize the transform with a name."""
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input image."""
        pass # pylint:disable=unnecessary-pass # pragma: no cover

    @abstractmethod
    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        """Invert the transformation on the prediction."""
        pass # pylint:disable=unnecessary-pass # pragma: no cover

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(image)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractTransform) and self.name == other.name


@register_transform("identity")
class IdentityTransform(AbstractTransform):
    """Identity transform for TTA.

    This transform does not apply any changes to the input image. It is
    useful for testing the performance of the model without any TTA.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction


@register_transform("flip_x")
class FlipXTransform(AbstractTransform):
    """Flip the input image along the X axis.

    This transform flips the input image along the X axis (horizontal flip).
    It is useful for augmenting the training data and improving model
    robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2,))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2,))


@register_transform("flip_y")
class FlipYTransform(AbstractTransform):
    """Flip the input image along the Y axis.

    This transform flips the input image along the Y axis (vertical flip).
    It is useful for augmenting the training data and improving model
    robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(3,))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(3,))


@register_transform("flip_z")
class FlipZTransform(AbstractTransform):
    """Flip the input image along the Z axis.

    This transform flips the input image along the Z axis (depth flip).
    It is useful for augmenting the training data and improving model
    robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(4,))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(4,))


@register_transform("flip_xy")
class FlipXYTransform(AbstractTransform):
    """Flip the input image along the X and Y axes.

    This transform flips the input image along both the X and Y axes
    (horizontal and vertical flip). It is useful for augmenting the
    training data and improving model robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2, 3))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2, 3))


@register_transform("flip_xz")
class FlipXZTransform(AbstractTransform):
    """Flip the input image along the X and Z axes.

    This transform flips the input image along both the X and Z axes
    (horizontal and depth flip). It is useful for augmenting the training
    data and improving model robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2, 4))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2, 4))


@register_transform("flip_yz")
class FlipYZTransform(AbstractTransform):
    """Flip the input image along the Y and Z axes.

    This transform flips the input image along both the Y and Z axes
    (vertical and depth flip). It is useful for augmenting the training
    data and improving model robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(3, 4))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(3, 4))


@register_transform("flip_xyz")
class FlipXYZTransform(AbstractTransform):
    """Flip the input image along the X, Y, and Z axes.

    This transform flips the input image along all three axes (horizontal,
    vertical, and depth flip). It is useful for augmenting the training
    data and improving model robustness.
    """
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2, 3, 4))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2, 3, 4))
