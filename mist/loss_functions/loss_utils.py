"""Loss function utilities."""

import torch
from torch import nn
from torch.nn import functional as F


def get_one_hot(y_true: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Converts a 3D label tensor to one-hot encoding.

    Args:
        y_true: 3D tensor of shape (batch_size, 1, height, width, depth).
        n_classes: Number of classes.

    Returns:
        One-hot encoding of the input segmentation mask.
    """
    y_true = y_true.long()
    y_true = F.one_hot(y_true, num_classes=n_classes)
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    return y_true.float()


class SoftSkeletonize(nn.Module):
    """Soft skeletonization of a binary mask.

    Iteratively erodes and opens the image to isolate the centerline.
    """
    def __init__(self, num_iter=40):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        """Soft erosion with cross-shaped structuring element approximation."""
        if len(img.shape) != 5:
            raise ValueError(f"Expected 5D input, got {len(img.shape)}.")

        # Separable min-pooling (approximates erosion)
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))

        return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        """Soft dilation using a 3x3x3 max-pooling kernel."""
        if len(img.shape) != 5:
            raise ValueError(f"Expected 5D input, got {len(img.shape)}.")

        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        """Soft opening (Erosion followed by Dilation)."""
        return self.soft_dilate(self.soft_erode(img))

    def soft_skeletonize(self, img: torch.Tensor) -> torch.Tensor:
        """Perform iterative soft skeletonization."""
        soft_opened_image = self.soft_open(img)
        skeleton = F.relu(img - soft_opened_image)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            soft_opened_image = self.soft_open(img)
            delta = F.relu(img - soft_opened_image)

            # Use out-of-place addition to preserve gradient flow
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the soft skeleton."""
        return self.soft_skeletonize(img)


def check_loss_inputs(y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Check that the input tensors for 3D segmentation losses are valid.

    This function ensures that:
    - Both inputs are 5D tensors (batch_size, channels, height, width, depth).
    - y_pred has at least 2 channels (for multi-class segmentation).
    - y_true has exactly 1 channel.
    - Batch size, height, width, and depth match between y_true and y_pred.

    Args:
        y_true: Ground truth segmentation mask tensor of shape
            (batch_size, 1, height, width, depth).
        y_pred: Predicted segmentation tensor of shape
            (batch_size, num_classes, height, width, depth).

    Raises:
        ValueError: If any of the checks fail.
    """
    if len(y_true.shape) != 5 or len(y_pred.shape) != 5:
        raise ValueError(
            f"For 3D data, the input tensors must be 5D. "
            f"Got shapes {y_true.shape} and {y_pred.shape}."
        )

    if y_pred.shape[1] < 2:
        raise ValueError(
            f"The number of classes in the prediction must be at least 2. "
            f"Got {y_pred.shape[1]}."
        )

    if y_true.shape[1] != 1:
        raise ValueError(
            f"The number of channels in the ground truth mask must be 1. "
            f"Got {y_true.shape[1]}."
        )

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"The batch sizes must match. "
            f"Got batch sizes {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    if y_true.shape[2:] != y_pred.shape[2:]:
        raise ValueError(
            f"The spatial dimensions (height, width, depth) must match. "
            f"Got {y_true.shape[2:]} and {y_pred.shape[2:]}."
        )
