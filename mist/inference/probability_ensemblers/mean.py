"""Mean probability ensembler for MIST."""

from typing import Any

import numpy as np
import numpy.typing as npt

from mist.inference.probability_ensemblers.base import AbstractProbabilityEnsembler
from mist.inference.probability_ensemblers.probability_ensembler_registry import (
    register_probability_ensembler,
)


@register_probability_ensembler("mean")
class MeanProbabilityEnsembler(AbstractProbabilityEnsembler):
    """Simple element-wise averaging ensembler over probability volumes."""

    def combine(
        self, probabilities: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Average a list of probability volumes element-wise.

        Args:
            probabilities: List of probability arrays, each of shape
                (D, H, W, C). All arrays must have the same shape.

        Returns:
            Element-wise mean array of shape (D, H, W, C).

        Raises:
            ValueError: If probabilities is empty or shapes disagree.
        """
        if not probabilities:
            raise ValueError("MeanProbabilityEnsembler requires at least one probability volume.")
        reference_shape = probabilities[0].shape
        for p in probabilities:
            if p.shape != reference_shape:
                raise ValueError(
                    "All probability volumes must have the same shape. Got "
                    f"{reference_shape} and {p.shape}."
                )
        mean_probability = np.zeros_like(probabilities[0], dtype=np.float64)
        for p in probabilities:
            mean_probability += p
        mean_probability /= len(probabilities)
        return mean_probability.astype(probabilities[0].dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
