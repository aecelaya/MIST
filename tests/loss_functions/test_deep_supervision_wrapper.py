"""Unit tests for DeepSupervisionLoss wrapper."""

from typing import Any

import torch

# MIST imports.
from mist.loss_functions.base import SegmentationLoss
from mist.loss_functions.deep_supervision_wrapper import DeepSupervisionLoss


class ShapeSpyLoss(SegmentationLoss):
    """A dummy loss that records input shapes for verification.

    Attributes:
        history: A list of tuples containing (y_true shape, y_pred shape).
    """

    def __init__(self):
        super().__init__()
        self.history: list[tuple[torch.Size, torch.Size]] = []

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Records shapes and returns a dummy scalar."""
        self.history.append((y_true.shape, y_pred.shape))
        return y_pred.sum()


class DummySumLoss(SegmentationLoss):
    """Returns simple sum of predictions for testing scaling behavior."""

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return y_pred.sum()


def _make_mock_data(
    num_supervisions: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Creates mock data for supervision testing.

    Args:
        num_supervisions: Number of auxiliary heads to generate.

    Returns:
        Tuple of (y_true, y_pred, y_supervision).
    """
    batch_size, channels, size = 1, 2, 4
    # Ground truth is 5D: (Batch, 1, Height, Width, Depth).
    y_true = torch.zeros((batch_size, 1, size, size, size), dtype=torch.int64)
    # Prediction is 5D: (Batch, C, Height, Width, Depth).
    y_pred = torch.ones((batch_size, channels, size, size, size))

    # Create supervision heads with increasing values to verify weighting.
    y_supervision = tuple(
        torch.ones_like(y_pred) * (i + 2) for i in range(num_supervisions)
    )
    return y_true, y_pred, y_supervision


class TestDeepSupervisionScaling:
    """Tests the weighting and scaling logic of the wrapper."""

    def test_no_supervision(self):
        """Test deep supervision with only the main prediction."""
        loss_fn = DeepSupervisionLoss(DummySumLoss())
        y_true, y_pred, _ = _make_mock_data()

        loss = loss_fn(y_true, y_pred)
        expected = y_pred.sum()  # scale = 1.0

        assert torch.isclose(loss, expected)

    def test_geometric_scaling(self):
        """Test that default geometric scaling (0.5^k) is applied."""
        loss_fn = DeepSupervisionLoss(DummySumLoss())
        y_true, y_pred, y_sup = _make_mock_data(num_supervisions=2)

        # Expected Math:
        # Main head (k=0): weight 1.0 * sum(1s)
        # Sup head 1 (k=1): weight 0.5 * sum(2s)
        # Sup head 2 (k=2): weight 0.25 * sum(3s)
        # Normalization: 1.0 + 0.5 + 0.25 = 1.75

        term_main = 1.0 * y_pred.sum()
        term_sup1 = 0.5 * y_sup[0].sum()
        term_sup2 = 0.25 * y_sup[1].sum()

        expected = (term_main + term_sup1 + term_sup2) / 1.75

        loss = loss_fn(y_true, y_pred, y_supervision=y_sup)
        assert torch.isclose(loss, expected)

    def test_custom_scaling_function(self):
        """Test using a user-defined scaling function (constant weights)."""
        # Force all weights to 1.0.
        loss_fn = DeepSupervisionLoss(
            DummySumLoss(), scaling_fn=lambda k: 1.0
        )
        y_true, y_pred, y_sup = _make_mock_data(num_supervisions=2)

        # (Sum(1s) + Sum(2s) + Sum(3s)) / 3.0.
        expected = (y_pred.sum() + y_sup[0].sum() + y_sup[1].sum()) / 3.0

        loss = loss_fn(y_true, y_pred, y_supervision=y_sup)
        assert torch.isclose(loss, expected)


class TestDeepSupervisionUpsampling:
    """Tests the trilinear upsampling logic for resolution mismatch."""

    def test_upsampling_behavior(self):
        """Test that small predictions are upsampled to match y_true."""
        spy_loss = ShapeSpyLoss()
        loss_fn = DeepSupervisionLoss(spy_loss)

        # Setup: Ground Truth is 4x4x4 (plus batch/channel dims).
        # Shape: (B, 1, H, W, D).
        batch_size, channels = 1, 2
        y_true = torch.zeros((batch_size, 1, 4, 4, 4))
        y_pred_main = torch.randn((batch_size, channels, 4, 4, 4))

        # Deep supervision head is half resolution (2x2x2).
        y_pred_aux = torch.randn((batch_size, channels, 2, 2, 2))

        loss_fn(y_true, y_pred_main, y_supervision=(y_pred_aux,))

        # Verify calls recorded by the spy.
        assert len(spy_loss.history) == 2

        # Call 1 (Main Head):
        # Input was 4x4x4, Target was 4x4x4. No resizing needed.
        true_shape_0, pred_shape_0 = spy_loss.history[0]
        assert true_shape_0 == (batch_size, 1, 4, 4, 4)
        assert pred_shape_0 == (batch_size, channels, 4, 4, 4)

        # Call 2 (Aux Head):
        # Input was 2x2x2. Wrapper should have upsampled it to 4x4x4.
        true_shape_1, pred_shape_1 = spy_loss.history[1]
        assert true_shape_1 == (batch_size, 1, 4, 4, 4)
        # The spy receives the UPSAMPLED prediction.
        assert pred_shape_1 == (batch_size, channels, 4, 4, 4)

    def test_kwargs_passthrough(self):
        """Test that kwargs (like alpha/dtm) are passed through untouched."""

        class ArgCheckLoss(SegmentationLoss):
            """Dummy loss to verify received kwargs."""

            def forward(self, y_true, y_pred, **kwargs):
                assert kwargs["alpha"] == 0.75
                # Verify DTM is passed through without resizing.
                # (Since we assume DTM matches y_true/full resolution).
                assert kwargs["dtm"].shape == (1, 2, 4, 4, 4)
                return y_pred.sum()

        loss_fn = DeepSupervisionLoss(ArgCheckLoss())

        y_true = torch.zeros((1, 1, 4, 4, 4))
        y_pred = torch.randn((1, 2, 4, 4, 4))
        dtm_full = torch.randn((1, 2, 4, 4, 4))

        loss_fn(y_true, y_pred, alpha=0.75, dtm=dtm_full)
