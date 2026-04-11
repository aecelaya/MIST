"""Registry for inference strategies in MIST."""
from typing import Dict, Type, List, Callable, TypeVar

from mist.inference.inferers.base import AbstractInferer

# Global registry for inferers.
T = TypeVar("T", bound=AbstractInferer)
INFERER_REGISTRY: Dict[str, Type[AbstractInferer]] = {}


def register_inferer(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a new inferer class."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractInferer):
            raise TypeError(
                f"{cls.__name__} must inherit from AbstractInferer."
            )
        if name in INFERER_REGISTRY:
            raise KeyError(f"Inferer '{name}' is already registered.")
        INFERER_REGISTRY[name] = cls  # Register the class, not an instance
        return cls
    return decorator


def list_inferers() -> List[str]:
    """List all registered inferers."""
    return list(INFERER_REGISTRY.keys())


def get_inferer(name: str) -> Type[AbstractInferer]:
    """Retrieve a registered inferer class by name."""
    if name not in INFERER_REGISTRY:
        raise KeyError(
            f"Inferer '{name}' is not registered. "
            f"Available: [{', '.join(list_inferers())}]"
        )
    return INFERER_REGISTRY[name]
