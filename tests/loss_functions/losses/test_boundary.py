"""Unit tests for the BoundaryLoss function."""


import torch
import pytest

from mist.loss_functions.losses.boundary import BoundaryLoss


def _make_mock_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates valid inputs + DTM for BoundaryLoss testing."""
    y_true = torch.randint(
        0, n_classes, size=(batch_size, 1, size, size, size)
    )
    y_pred = torch.randn((batch_size, n_classes, size, size, size))
    # Mock DTM: Positive values (simple random)
    dtm = torch.abs(torch.randn((batch_size, n_classes, size, size, size)))
    return y_true, y_pred, dtm


class TestBoundaryLoss:
    """Tests for the BoundaryLoss class."""

    def test_default_init(self):
        """Test default initialization values."""
        loss_fn = BoundaryLoss()
        # Default should now be False (include background in Dice/CE)
        assert loss_fn.exclude_background is False

    def test_forward_runs_and_returns_scalar(self):
        """Test that forward pass runs with valid DTM."""
        loss_fn = BoundaryLoss()
        y_true, y_pred, dtm = _make_mock_data()

        loss = loss_fn(y_true, y_pred, dtm=dtm)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_exclude_background_false_slices_boundary_correctly(self):
        """Test that boundary loss excludes background even if DiceCE keeps it."""
        # 1. Setup: exclude_background=False (Default).
        loss_fn = BoundaryLoss(exclude_background=False)
        y_true, y_pred, dtm = _make_mock_data(n_classes=3)

        # 2. Run loss with alpha=0 (Pure Boundary Loss).
        loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)

        # 3. Manual Verification.
        # Even though exclude_background=False, the boundary term should
        # IGNORE channel 0.
        y_pred_soft = torch.softmax(y_pred, dim=1)

        # Manually slice foreground (channels 1 and 2).
        y_pred_fg = y_pred_soft[:, 1:]
        dtm_fg = dtm[:, 1:]

        expected_loss = torch.mean(dtm_fg * y_pred_fg)

        assert torch.isclose(loss, expected_loss, atol=1e-6)

    def test_exclude_background_true_slices_dtm_correctly(self):
        """Test that DTM is sliced to match predictions when bg is excluded."""
        # 1. Setup: exclude_background=True.
        loss_fn = BoundaryLoss(exclude_background=True)
        y_true, y_pred, dtm = _make_mock_data(n_classes=3)

        # 2. Run loss (pure boundary).
        loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)

        # 3. Manual verification.
        # y_pred_soft (from preprocess) already has channel 0 removed.
        y_pred_soft_fg = torch.softmax(y_pred, dim=1)[:, 1:]

        # DTM must be manually sliced to match.
        dtm_fg = dtm[:, 1:]

        expected_loss = torch.mean(dtm_fg * y_pred_soft_fg)

        assert torch.isclose(loss, expected_loss, atol=1e-6)

    def test_raises_error_missing_dtm(self):
        """Test that missing 'dtm' kwarg raises ValueError."""
        loss_fn = BoundaryLoss()
        y_true, y_pred, _ = _make_mock_data()

        with pytest.raises(ValueError, match="requires 'dtm'"):
            loss_fn(y_true, y_pred)
