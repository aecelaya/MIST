"""clDice loss function."""

from typing import Any

import torch

from mist.loss_functions import loss_utils
from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss


@register_loss(name="cldice")
class CLDice(DiceCELoss):
    """Combined soft Dice + cross entropy + clDice loss.

    Computes a weighted sum of:
        alpha * DiceCELoss + (1 - alpha) * clDice.

    Attributes:
        iterations: Number of iterations for soft skeletonization.
        smooth: Small constant for numerical stability.
        soft_skeletonize: Module for soft skeletonization.
    """

    def __init__(
        self,
        iterations: int = 10,
        smooth: float = 1.0,
        exclude_background: bool = False,
        **kwargs: Any,
    ):
        """Initialize the CLDice loss.

        Args:
            iterations: Number of iterations for soft skeletonization.
            smooth: Small constant for numerical stability.
            exclude_background: If True, excludes background from Dice/CE.
                (Note: Background is ALWAYS excluded from clDice component).
            kwargs: Additional keyword arguments for future extensions.
        """
        # Initialize parent DiceCELoss.
        super().__init__(exclude_background=exclude_background, **kwargs)

        self.iterations = iterations
        self.smooth = smooth
        self.soft_skeletonize = loss_utils.SoftSkeletonize(num_iter=iterations)

    def cldice(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the soft clDice loss component.

        Args:
            y_true: Ground truth tensor (B, C, H, W, D).
            y_pred: Prediction tensor (B, C, H, W, D).

        Returns:
            The calculated clDice loss (scalar).
        """
        # CRITICAL LOGIC:
        # If exclude_background is False, the inputs y_true/y_pred still
        # contain channel 0 (background). We MUST remove it for clDice
        # to work correctly, even if we kept it for the Cross Entropy part.
        if not self.exclude_background:
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        # Skeletonize (channel-wise operation).
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        # Compute precision and sensitivity PER CLASS.
        tprec = (torch.sum(skel_pred * y_true, dim=self.spatial_dims_3d) + self.smooth) / (
            torch.sum(skel_pred, dim=self.spatial_dims_3d) + self.smooth
        )

        tsens = (torch.sum(skel_true * y_pred, dim=self.spatial_dims_3d) + self.smooth) / (
            torch.sum(skel_true, dim=self.spatial_dims_3d) + self.smooth
        )

        # Compute clDice score PER CLASS.
        cldice_score = 2.0 * (tprec * tsens) / (tprec + tsens)

        # Loss = 1 - clDice (average over classes).
        return 1.0 - torch.mean(cldice_score)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Logits tensor of shape (B, C, H, W, D).
            **kwargs: Extra args. Must contain 'alpha' to control weighting.
                Defaults to alpha=0.5 if missing.

        Returns:
            Combined scalar loss.
        """
        # Extract alpha from kwargs, default to 0.5.
        alpha = kwargs.get("alpha", 0.5)

        # 1. Compute Dice + Cross Entropy (using parent logic).
        # Parent handles the logic for (B, 1, H, W, D) input (squeezing for CE).
        dice_ce_loss = super().forward(y_true, y_pred, **kwargs)

        # 2. Compute clDice.
        # We manually preprocess here to get probabilities for skeletonization.
        # preprocess does one-hot conversion of y_true and softmax of y_pred.
        y_true_onehot, y_pred_softmax = self.preprocess(y_true, y_pred)

        cldice_loss = self.cldice(y_true_onehot, y_pred_softmax)

        # 3. Combine.
        return alpha * dice_ce_loss + (1.0 - alpha) * cldice_loss
