"""Unit tests for the clDice loss function."""

from unittest import mock

import torch
import pytest

from mist.loss_functions.losses.cl_dice import CLDice


def _make_mock_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates valid input tensors for CLDice loss testing.

    Args:
        n_classes: Number of classes.
        batch_size: Batch size.
        size: Spatial size (cubic).

    Returns:
        Tuple containing:
            - y_true: Ground truth labels (Batch, 1, Height, Width, Depth).
            - y_pred: Logits (Batch, Class, Height, Width, Depth).
    """
    # Ground truth: Integer labels (B, 1, H, W, D).
    y_true = torch.randint(
        0, n_classes, size=(batch_size, 1, size, size, size)
    )
    # Predictions: Logits (B, C, H, W, D).
    y_pred = torch.randn((batch_size, n_classes, size, size, size))
    return y_true, y_pred


class TestCLDice:
    """Tests for the CLDice loss class."""

    def test_forward_runs_and_returns_scalar(self):
        """Test that forward pass returns a non-negative scalar."""
        loss_fn = CLDice(iterations=3)
        y_true, y_pred = _make_mock_data()

        loss = loss_fn(y_true, y_pred)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_alpha_weighting_stability(self, alpha):
        """Test that loss runs successfully with different alpha values."""
        loss_fn = CLDice(iterations=1)
        y_true, y_pred = _make_mock_data()

        loss = loss_fn(y_true, y_pred, alpha=alpha)
        assert loss.item() >= 0.0

    def test_alpha_weighting_impact(self):
        """Test that changing alpha actually changes the loss value."""
        loss_fn = CLDice(iterations=1)
        y_true, y_pred = _make_mock_data()

        loss_high_alpha = loss_fn(y_true, y_pred, alpha=0.9)
        loss_low_alpha = loss_fn(y_true, y_pred, alpha=0.1)

        # Unless inputs are perfect, these should differ.
        assert not torch.isclose(loss_high_alpha, loss_low_alpha)

    def test_background_always_excluded_from_skeletonization(self):
        """Verify clDice excludes background even if exclude_background=False.

        This ensures the CRITICAL LOGIC block in the cldice method works.
        """
        # Initialize with exclude_background=False (Parent keeps background).
        loss_fn = CLDice(exclude_background=False, iterations=1)
        y_true, y_pred = _make_mock_data(n_classes=3)

        # We spy on the soft_skeletonize module's forward pass.
        # We want to check the shape of the tensor passed to it.
        with mock.patch.object(
            loss_fn.soft_skeletonize,
            "forward",
            wraps=loss_fn.soft_skeletonize.forward,
        ) as mock_skel:
            _ = loss_fn(y_true, y_pred)

            # Get the arguments passed to the first call of soft_skeletonize.
            # args[0] is the image tensor.
            args, _ = mock_skel.call_args
            input_tensor = args[0]

            # Input was 3 classes.
            # If logic works, skeletonize should see 2 channels (Foreground).
            # If logic fails, it would see 3 channels.
            assert input_tensor.shape[1] == 2

    def test_background_excluded_when_flag_true(self):
        """Verify behavior when exclude_background=True."""
        # Initialize with exclude_background=True (Parent removes background).
        loss_fn = CLDice(exclude_background=True, iterations=1)
        y_true, y_pred = _make_mock_data(n_classes=3)

        with mock.patch.object(
            loss_fn.soft_skeletonize,
            "forward",
            wraps=loss_fn.soft_skeletonize.forward,
        ) as mock_skel:
            _ = loss_fn(y_true, y_pred)

            args, _ = mock_skel.call_args
            input_tensor = args[0]

            # Input was 3 classes. Parent removed channel 0.
            # cldice logic sees flag is True, so it does NOT slice again.
            # Result should be 2 channels.
            assert input_tensor.shape[1] == 2

    def test_perfect_prediction_with_vessel(self):
        """Test that loss is ~0 when predicting a vessel-like structure."""
        # Setup dimensions: (Batch, Channel, H, W, D).
        # We use slightly larger dimensions (16) to allow space for the vessel.
        b, c, s = 1, 2, 16
        y_true = torch.zeros((b, 1, s, s, s), dtype=torch.long)
        y_pred = torch.empty((b, c, s, s, s))

        # 1. Create a "Vessel" (A line running through the center depth-wise).
        # Coordinates: Center of H/W, running from D=2 to D=14.
        center = s // 2
        y_true[..., center, center, 2:-2] = 1

        # 2. Create Perfect Logits.
        # Initialize: Strong prediction for Background (Class 0).
        # Logits: Class 0 = 20.0, Class 1 = -20.0.
        y_pred[:, 0] = 20.0
        y_pred[:, 1] = -20.0

        # Overwrite: Strong prediction for Vessel (Class 1) where y_true == 1.
        # Logits: Class 0 = -20.0, Class 1 = 20.0.
        mask = (y_true == 1).squeeze(1)
        y_pred[:, 0][mask] = -20.0
        y_pred[:, 1][mask] = 20.0

        # 3. Compute Loss.
        # We use fewer iterations because the structure is thin (1 px).
        loss_fn = CLDice(iterations=3, smooth=1e-5)
        loss = loss_fn(y_true, y_pred)

        # 4. Assert.
        # Since prediction is perfect, DiceCE is 0 and clDice is 0.
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)

    def test_kwargs_passthrough(self):
        """Test that kwargs are accepted without error."""
        loss_fn = CLDice(iterations=1)
        y_true, y_pred = _make_mock_data()

        # Pass random kwarg, should be ignored or handled by parent.
        loss = loss_fn(y_true, y_pred, alpha=0.5, dummy_arg=123)
        assert loss.item() >= 0
