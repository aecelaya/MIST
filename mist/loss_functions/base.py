"""Base class for segmentation loss functions."""

from abc import ABC, abstractmethod
from typing import Tuple, Any


import torch
from torch import nn
from torch.nn import functional as F

from mist.loss_functions import loss_utils
from mist.loss_functions.loss_constants import LossConstants as constants


class SegmentationLoss(nn.Module, ABC):
    """Base class for segmentation loss functions.

    This class provides common preprocessing steps for segmentation losses,
    such as one-hot encoding of ground truth labels and applying softmax to
    predictions. It also allows for optional exclusion of the background class
    from the loss computation.

    Attributes:
        exclude_background: If True, the background class (class 0) is
            excluded from the loss computation.
        spatial_dims_3d: Tuple indicating the spatial dimensions for 3D data.
        avoid_division_by_zero: A small constant to prevent division by zero
            in loss computations.
    """

    def __init__(self, exclude_background: bool = False, **kwargs: Any):
        """Initialize the SegmentationLoss.

        Args:
            exclude_background: If True, the background class (class 0) is
                excluded from the loss computation.
            kwargs: Additional keyword arguments for future extensions.
        """
        super().__init__()
        self.exclude_background = exclude_background
        self.spatial_dims_3d = constants.SPATIAL_DIMS_3D
        self.avoid_division_by_zero = constants.AVOID_DIVISION_BY_ZERO_CONSTANT

    def preprocess(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocesses loss inputs for segmentation tasks.

        This method converts the ground truth labels to one-hot encoding and
        applies softmax to the predictions. If `exclude_background` is set to
        True, it removes the background class from both the ground truth and
        predictions. This method also validates the inputs to ensure they have
        the correct dimensions and are compatible with each other.

        Args:
            y_true: Ground truth labels, a tensor of shape
                (batch_size, height, width, depth) with integer class labels.
            y_pred: The raw output from the model, a tensor of shape
                (batch_size, num_classes, height, width, depth).

        Returns:
            A tuple containing the preprocessed ground truth and predictions,
            both of shape (batch_size, num_classes, height, width, depth).

        Raises:
            ValueError: If the input tensors do not have the expected shapes or
                if their dimensions are incompatible.
        """
        loss_utils.check_loss_inputs(y_true, y_pred)
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1])
        y_pred = F.softmax(y_pred, dim=1)

        if self.exclude_background:
            y_true = y_true[:, 1:]
            y_pred = y_pred[:, 1:]

        return y_true, y_pred

    @abstractmethod
    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the segmentation loss.

        This method must be implemented by subclasses to compute the specific
        loss value based on the preprocessed ground truth and predictions.

        Args:
            y_true: Preprocessed ground truth tensor of shape
                (batch_size, num_classes, height, width, depth).
            y_pred: Preprocessed prediction tensor of shape
                (batch_size, num_classes, height, width, depth).
            *args: Additional positional arguments for specific loss functions.
            **kwargs: Additional keyword arguments for specific loss functions.

        Returns:
            The computed loss as a scalar tensor.
        """
