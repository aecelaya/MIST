"""Mean-based softmax ensembler for MIST inference."""
from typing import List
import torch

# MIST imports.
from mist.inference.ensemblers.base import AbstractEnsembler
from mist.inference.ensemblers.ensembler_registry import register_ensembler


@register_ensembler("mean")
class MeanEnsembler(AbstractEnsembler):
    """Simple averaging ensembler over softmax outputs."""
    def combine(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Average a list of predictions element-wise.

        Args:
            predictions: List of soft prediction tensors, each of shape
                (1, C, D, H, W).

        Returns:
            Element-wise mean tensor of shape (1, C, D, H, W).
        """
        if not predictions:
            raise ValueError("MeanEnsembler requires at least one prediction.")

        stacked = torch.stack(predictions, dim=0)     # Shape: (N, 1, C, D, H, W)
        mean_prediction = torch.mean(stacked, dim=0)  # Shape: (1, C, D, H, W)
        return mean_prediction

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
