"""Registry for probability-space ensemblers in MIST."""

from collections.abc import Callable
from typing import TypeVar

from mist.inference.probability_ensemblers.base import AbstractProbabilityEnsembler

T = TypeVar("T", bound=AbstractProbabilityEnsembler)
PROBABILITY_ENSEMBLER_REGISTRY: dict[str, type[AbstractProbabilityEnsembler]] = {}


def register_probability_ensembler(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a new probability ensembler class."""

    def decorator(cls: type[T]) -> type[T]:
        if not issubclass(cls, AbstractProbabilityEnsembler):
            raise TypeError(f"{cls.__name__} must inherit from AbstractProbabilityEnsembler.")
        if name in PROBABILITY_ENSEMBLER_REGISTRY:
            raise KeyError(f"Probability ensembler '{name}' is already registered.")
        PROBABILITY_ENSEMBLER_REGISTRY[name] = cls
        return cls

    return decorator


def list_probability_ensemblers() -> list[str]:
    """List all registered probability ensemblers."""
    return list(PROBABILITY_ENSEMBLER_REGISTRY.keys())


def get_probability_ensembler(name: str) -> AbstractProbabilityEnsembler:
    """Retrieve a fresh instance of a registered probability ensembler by name."""
    if name not in PROBABILITY_ENSEMBLER_REGISTRY:
        raise KeyError(
            f"Probability ensembler '{name}' is not registered. "
            f"Available: [{', '.join(list_probability_ensemblers())}]"
        )
    return PROBABILITY_ENSEMBLER_REGISTRY[name]()
