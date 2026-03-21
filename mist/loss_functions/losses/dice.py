"""Soft Dice loss function for segmentation tasks."""

from typing import Any

import torch

from mist.loss_functions.base import SegmentationLoss
from mist.loss_functions.loss_registry import register_loss


@register_loss(name="dice")
class DiceLoss(SegmentationLoss):
    """Soft Dice loss function for segmentation tasks.

    For each class, the Dice loss is defined as:
        L(x, y) = ||x - y||² / (||x||² + ||y||² + ε)

    We then take the mean of the Dice loss across all classes. By default, the
    Dice loss function includes the background class.
    """
    def __init__(self, exclude_background: bool = False, **kwargs: Any):
        """Initialize Dice loss.

        Args:
            exclude_background: If True, the background class (class 0) is
                excluded from the loss computation.
            kwargs: Additional keyword arguments for future extensions.
        """
        super().__init__(exclude_background = exclude_background, **kwargs)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the Dice loss.

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Raw model output tensor of shape (B, C, H, W, D).

        Returns:
            Dice loss as a scalar tensor.
        """
        y_true, y_pred = self.preprocess(y_true, y_pred)

        numerator = torch.sum(
            torch.square(y_true - y_pred), dim=self.spatial_dims_3d
        )
        denominator = (
            torch.sum(torch.square(y_true), dim=self.spatial_dims_3d) +
            torch.sum(torch.square(y_pred), dim=self.spatial_dims_3d) +
            self.avoid_division_by_zero
        )

        loss = numerator / denominator  # Per class.
        loss = torch.mean(loss, dim=1)  # Mean over classes.
        return torch.mean(loss)  # Mean over batch.
