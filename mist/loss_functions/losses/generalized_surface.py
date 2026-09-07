"""Generalized surface loss function for segmentation tasks."""

from typing import Any

import torch

from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss


@register_loss(name="gsl")
class GenSurfLoss(DiceCELoss):
    """Generalized Surface Loss (GSL) with DiceCE region term.

    This loss combines a region-based DiceCE term with a boundary-aware surface
    loss that penalizes misalignment between prediction and ground truth
    surfaces.

    Attributes:
        smooth: Small constant for numerical stability in division.
        exclude_background: Whether to exclude the background class (channel 0)
            from the Dice+CrossEntropy term.
            Note: Background is ALWAYS excluded from the Surface term.
    """

    def __init__(self, exclude_background: bool = False, **kwargs: Any):
        """Initialize Generalized Surface Loss.

        Args:
            exclude_background: If True, class 0 is excluded from Dice/CE.
                If False (Default), Dice/CE includes background.
                (Surface term always excludes background).
            kwargs: Additional keyword arguments for future extensions.
        """
        super().__init__(exclude_background=exclude_background, **kwargs)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the Generalized Surface Loss (GSL).

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Logits tensor of shape (B, C, H, W, D).
            **kwargs: Extra arguments.
                - 'dtm' (torch.Tensor): Precomputed distance transform map.
                  Required. Shape must match y_pred (B, C, H, W, D).
                - 'alpha' (float): Weight for region term (0 <= alpha <= 1).
                  Defaults to 0.5.

        Returns:
            Scalar loss combining region and boundary terms.

        Raises:
            ValueError: If 'dtm' is not provided in kwargs.
        """
        dtm = kwargs.get("dtm")
        if dtm is None:
            raise ValueError(
                "GenSurfLoss requires 'dtm' (distance transform map) to be "
                "provided in the arguments."
            )
        alpha = kwargs.get("alpha", 0.5)

        # 1. Compute Dice + CE region loss.
        # This respects self.exclude_background.
        region_loss = super().forward(y_true, y_pred, **kwargs)

        # 2. Compute Surface Loss.
        # Preprocess inputs: y_true -> OneHot, y_pred -> Softmax.
        y_true_onehot, y_pred_softmax = self.preprocess(y_true, y_pred)

        # CRITICAL LOGIC: Enforce background exclusion for the surface term.
        # We assume input DTM includes the background channel (C channels).

        # Always slice DTM to remove background channel (0).
        dtm = dtm[:, 1:]

        # If exclude_background is False, predictions still have background.
        # We must slice them out to match the sliced DTM.
        if not self.exclude_background:
            y_true_onehot = y_true_onehot[:, 1:]
            y_pred_softmax = y_pred_softmax[:, 1:]

        # Generalized surface loss calculation (Foreground only).
        # We want to measure agreement relative to the boundary complexity.
        diff = 1.0 - (y_true_onehot + y_pred_softmax)

        # Numerator: Weighted variance of the difference.
        numerator = torch.sum((dtm * diff) ** 2, dim=self.spatial_dims_3d)

        # Denominator: Total weighted variance of the DTM.
        denominator = torch.sum(dtm**2, dim=self.spatial_dims_3d) + self.avoid_division_by_zero

        # Surface loss: 1 - ratio.
        surface_loss = torch.mean(1.0 - numerator / denominator)

        # Weighted sum.
        return alpha * region_loss + (1.0 - alpha) * surface_loss
