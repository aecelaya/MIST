"""Abstract base class for all prediction ensemblers in MIST."""
from abc import ABC, abstractmethod
from typing import Any

import torch


class AbstractEnsembler(ABC):
    """Abstract base class for ensembling soft prediction outputs.

    Implementations of this class define how a list of soft model outputs
    (e.g., probabilities or logits) are aggregated into a single output.
    """

    def __init__(self):
        """Initialize the ensembler."""
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def combine(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate a list of predictions into a single output.

        Args:
            predictions: List of soft prediction tensors, each of shape
                (1, C, D, H, W).

        Returns:
            Aggregated prediction tensor of the same shape.
        """
        pass  # pylint: disable=unnecessary-pass # pragma: no cover

    def __call__(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        return self.combine(predictions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractEnsembler) and self.name == other.name
