"""Abstract base class for probability-space ensemblers in MIST."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class AbstractProbabilityEnsembler(ABC):
    """Abstract base class for ensembling per-voxel class probabilities.

    Implementations define how a list of continuous, pre-argmax probability
    volumes (one per model, already resampled into a common original-image
    space) are combined into a single consensus probability volume. This is
    distinct from the label-space ensemblers in mist.inference.label_ensemblers,
    which operate on discrete, post-argmax label maps.
    """

    def __init__(self):
        """Initialize the probability ensembler."""
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def combine(
        self, probabilities: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Combine a list of probability volumes into a single consensus.

        Args:
            probabilities: List of numpy arrays of shape (D, H, W, C). All
                arrays must have the same shape.

        Returns:
            Consensus probability volume of shape (D, H, W, C).
        """
        pass  # pylint: disable=unnecessary-pass # pragma: no cover

    def __call__(
        self, probabilities: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Call combine directly."""
        return self.combine(probabilities)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractProbabilityEnsembler) and self.name == other.name
