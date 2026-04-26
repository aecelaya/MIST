"""Wrapper for deep supervision loss in segmentation tasks."""

from typing import Any
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from mist.loss_functions.base import SegmentationLoss


class DeepSupervisionLoss(nn.Module):
    """Loss function wrapper for deep supervision in 3D segmentation tasks.

    This wrapper handles deep supervision by upsampling lower-resolution
    auxiliary predictions to match the ground truth resolution using trilinear
    interpolation before computing the loss.

    Attributes:
        loss_fn: The base loss function to apply.
        scaling_fn: A function to scale the loss for each supervision head.
            Defaults to geometric scaling by 0.5 ** k.
    """

    def __init__(
        self,
        loss_fn: SegmentationLoss,
        scaling_fn: Callable[[int], float] | None = None,
    ):
        """Initialize the DeepSupervisionLoss wrapper.

        Args:
            loss_fn: The base loss function (e.g., DiceLoss, DiceCELoss).
            scaling_fn: Function to calculate weight for k-th supervision head.
                If None, defaults to 0.5^k.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.scaling_fn = scaling_fn or (lambda k: 0.5**k)

    def apply_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Applies the loss, upsampling y_pred via trilinear interpolation.

        Args:
            y_true: Ground truth tensor (Full Resolution). Expected shape is
                (B, 1, H, W, D).
            y_pred: Predicted tensor (Potentially Low Resolution). Expected
                shape is (B, C, H, W, D).
            **kwargs: Extra arguments (e.g., 'dtm', 'alpha').

        Returns:
            Computed loss for this specific prediction head.
        """
        # 1. Determine target spatial size (H, W, D) from ground truth.
        # Assumes y_true is strictly (batch, channel, height, width, depth).
        target_size = y_true.shape[2:]

        # 2. Check if prediction needs upsampling.
        # y_pred is (batch, channel, height, width, depth).
        if y_pred.shape[2:] != target_size:
            y_pred = F.interpolate(
                y_pred, size=target_size, mode="trilinear", align_corners=False
            )

        # 3. Compute loss.
        # y_true and kwargs (like dtm) are used at full resolution.
        return self.loss_fn(y_true, y_pred, **kwargs)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_supervision: tuple[torch.Tensor, ...] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Computes the total weighted loss.

        Args:
            y_true: Ground truth mask of shape (B, 1, H, W, D).
            y_pred: Main prediction of shape (B, C, H, W, D).
            y_supervision: Tuple of auxiliary predictions.
            **kwargs: Additional arguments for the base loss.

        Returns:
            Total weighted loss.
        """
        # Collect main prediction and deep supervision outputs.
        # Main output is always index 0.
        all_preds = [y_pred, *(y_supervision or ())]

        # Compute weighted losses for each head using list comprehension.
        # apply_loss handles the upsampling internally.
        weighted_losses = [
            self.scaling_fn(k) * self.apply_loss(y_true, pred, **kwargs)
            for k, pred in enumerate(all_preds)
        ]

        # Calculate normalization factor (sum of weights).
        total_weight = sum(
            self.scaling_fn(k) for k in range(len(all_preds))
        )

        return sum(weighted_losses) / total_weight
