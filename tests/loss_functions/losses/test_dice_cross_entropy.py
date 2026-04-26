"""Unit tests for DiceCELoss in MIST."""


import pytest
import torch

from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss


def _make_mock_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates valid 3D input tensors for DiceCE loss testing.

    Args:
        n_classes: Number of classes.
        batch_size: Batch size.
        size: Spatial size (cubic).

    Returns:
        Tuple containing:
            - y_true: Ground truth labels (Batch, 1, Height, Width, Depth).
            - y_pred: Logits (Batch, Class, Height, Width, Depth).
    """
    # Ground truth: Integer labels with channel dim (B, 1, H, W, D).
    y_true = torch.randint(
        0, n_classes, size=(batch_size, 1, size, size, size)
    )
    # Predictions: Logits (B, C, H, W, D).
    y_pred = torch.randn((batch_size, n_classes, size, size, size))
    return y_true, y_pred


class TestDiceCELoss:
    """Tests for the DiceCELoss class."""

    def test_forward_runs_and_returns_scalar(self):
        """Test that forward pass returns a non-negative scalar."""
        loss_fn = DiceCELoss()
        y_true, y_pred = _make_mock_data()

        loss = loss_fn(y_true, y_pred)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("exclude_bg", [True, False])
    def test_exclude_background_configuration(self, exclude_bg):
        """Test that CE ignore_index is always default (-100).

        We deliberately want CrossEntropy to include background pixels (to 
        suppress false positives) even if the Dice term excludes them.
        """
        loss_fn = DiceCELoss(exclude_background=exclude_bg)

        # Verify Dice parent class configuration
        assert loss_fn.exclude_background == exclude_bg

        # Verify Cross Entropy configuration (should always process background)
        assert loss_fn.cross_entropy.ignore_index == -100

    def test_forward_with_exclude_background(self):
        """Test forward pass runs successfully when excluding background."""
        loss_fn = DiceCELoss(exclude_background=True)
        y_true, y_pred = _make_mock_data(n_classes=4)

        loss = loss_fn(y_true, y_pred)
        assert loss.item() >= 0.0

    @pytest.mark.parametrize(
        "pred_shape, true_shape",
        [
            # Case 1: Wrong number of channels in target (2 instead of 1).
            ((1, 3, 4, 4, 4), (1, 2, 4, 4, 4)),
            # Case 2: Mismatched batch size.
            ((2, 3, 4, 4, 4), (1, 1, 4, 4, 4)),
            # Case 3: Mismatched spatial dimensions.
            ((1, 3, 4, 4, 4), (1, 1, 5, 5, 5)),
        ]
    )
    def test_invalid_shape_raises_error(self, pred_shape, true_shape):
        """Ensure invalid input shapes raise ValueError (via base class)."""
        loss_fn = DiceCELoss()
        y_pred = torch.randn(*pred_shape)
        y_true = torch.randint(0, 2, size=true_shape).float()

        with pytest.raises(ValueError):
            loss_fn(y_true, y_pred)

    def test_perfect_prediction(self):
        """Test for perfect prediction (should give loss close to 0)."""
        # Shape: (B, 1, H, W, D).
        y_true = torch.zeros((1, 1, 5, 5, 5), dtype=torch.long)
        y_pred = torch.empty(1, 2, 5, 5, 5)

        # Logits: Make class 0 very high (correct), class 1 very low (correct).
        y_pred[:, 0, ...] = 100.0
        y_pred[:, 1, ...] = -100.0

        loss_fn = DiceCELoss()
        loss = loss_fn(y_true, y_pred)

        # 0.5 * (Dice(0) + CE(0)) = 0.
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_worst_prediction(self):
        """Test for worst-case prediction (should give very high loss)."""
        # Shape: (B, 1, H, W, D).
        y_true = torch.zeros((1, 1, 5, 5, 5), dtype=torch.long)
        y_pred = torch.empty(1, 2, 5, 5, 5)

        # Logits: Make class 0 very low (Wrong), class 1 very high (Wrong).
        y_pred[:, 0, ...] = -100.0
        y_pred[:, 1, ...] = 100.0

        loss_fn = DiceCELoss()
        loss = loss_fn(y_true, y_pred)

        # Dice component ~ 1.0.
        # CE component ~ 100 (huge).
        # We just assert it is appropriately massive (> 1.0).
        assert loss.item() > 1.0
