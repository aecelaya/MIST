"""Registry for ensembling strategies in MIST."""
from typing import Dict, Type, List, TypeVar, Callable

# MIST imports.
from mist.inference.ensemblers.base import AbstractEnsembler

# Global registry for ensemblers. Stores classes; instances are created on
# demand by get_ensembler() to avoid shared mutable state.
T = TypeVar("T", bound=AbstractEnsembler)
ENSEMBLER_REGISTRY: Dict[str, Type[AbstractEnsembler]] = {}


def register_ensembler(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a new ensembler class."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractEnsembler):
            raise TypeError(
                f"{cls.__name__} must inherit from AbstractEnsembler."
            )
        if name in ENSEMBLER_REGISTRY:
            raise KeyError(f"Ensembler '{name}' is already registered.")
        ENSEMBLER_REGISTRY[name] = cls  # Register the class, not an instance.
        return cls
    return decorator


def list_ensemblers() -> List[str]:
    """List all registered ensemblers."""
    return list(ENSEMBLER_REGISTRY.keys())


def get_ensembler(name: str) -> AbstractEnsembler:
    """Retrieve a fresh instance of a registered ensembler by name."""
    if name not in ENSEMBLER_REGISTRY:
        raise KeyError(
            f"Ensembler '{name}' is not registered. "
            f"Available: [{', '.join(list_ensemblers())}]"
        )
    return ENSEMBLER_REGISTRY[name]()
