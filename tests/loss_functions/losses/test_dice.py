"""Unit tests for DiceLoss in MIST."""


import torch

from mist.loss_functions.losses.dice import DiceLoss


def _make_mock_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates valid input tensors for Dice loss testing.

    Args:
        n_classes: Number of classes.
        batch_size: Batch size.
        size: Spatial size (cubic).

    Returns:
        Tuple containing:
            - y_true: Ground truth labels (B, 1, H, W, D).
            - y_pred: Logits (B, C, H, W, D).
    """
    # Ground truth: Integer labels with channel dim (B, 1, H, W, D).
    y_true = torch.randint(
        0, n_classes, size=(batch_size, 1, size, size, size)
    )
    # Predictions: Logits (B, C, H, W, D).
    y_pred = torch.randn((batch_size, n_classes, size, size, size))
    return y_true, y_pred


class TestDiceLoss:
    """Tests for the DiceLoss class."""

    def test_forward_runs_and_returns_scalar(self):
        """Test that Dice loss computes without error and returns a scalar."""
        loss_fn = DiceLoss()
        y_true, y_pred = _make_mock_data()

        loss = loss_fn(y_true, y_pred)

        assert loss.ndim == 0
        assert 0.0 <= loss.item() <= 1.0

    def test_excludes_background(self):
        """Test that exclude_background=True removes the first channel."""
        loss_fn = DiceLoss(exclude_background=True)
        y_true, y_pred = _make_mock_data(n_classes=3)

        # We manually call preprocess to verify the internal tensors.
        # Note: In the actual loss flow, this happens inside forward().
        # preprocess() handles the one-hot conversion of y_true.
        y_true_proc, y_pred_proc = loss_fn.preprocess(y_true, y_pred)

        # Original was 3 classes. Result should have 2 classes.
        assert y_true_proc.shape[1] == 2
        assert y_pred_proc.shape[1] == 2

    def test_perfect_prediction(self):
        """Test that the loss is near 0.0 when predictions match labels."""
        # Create a single pixel batch for simplicity.
        # Shape: (B, 1, H, W, D).
        y_true = torch.zeros((1, 1, 5, 5, 5), dtype=torch.long)
        y_pred = torch.zeros((1, 2, 5, 5, 5))

        # Make class 0 (background) logits very high, class 1 very low.
        # This matches y_true=0.
        y_pred[:, 0, ...] = 100.0
        y_pred[:, 1, ...] = -100.0

        loss_fn = DiceLoss()
        loss = loss_fn(y_true, y_pred)

        # Loss should be exactly 0.0 because numerator (x-y)^2 is 0.
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_worst_prediction(self):
        """Test that the loss is near 1.0 when predictions are opposite."""
        # Shape: (B, 1, H, W, D).
        y_true = torch.zeros((1, 1, 5, 5, 5), dtype=torch.long)
        y_pred = torch.zeros((1, 2, 5, 5, 5))

        # Make class 1 logits very high, class 0 very low.
        # This is the exact opposite of y_true=0.
        y_pred[:, 0, ...] = -100.0
        y_pred[:, 1, ...] = 100.0

        loss_fn = DiceLoss()
        loss = loss_fn(y_true, y_pred)

        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)
