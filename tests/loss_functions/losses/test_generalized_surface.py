"""Unit tests for the Generalized Surface Loss (GSL)."""


import torch
import pytest

from mist.loss_functions.losses.generalized_surface import GenSurfLoss


def _make_mock_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates valid inputs + DTM for GenSurfLoss testing.

    Returns:
        y_true: (B, 1, H, W, D)
        y_pred: (B, C, H, W, D)
        dtm:    (B, C, H, W, D)
    """
    y_true = torch.randint(
        0, n_classes, size=(batch_size, 1, size, size, size)
    )
    y_pred = torch.randn((batch_size, n_classes, size, size, size))
    dtm = torch.abs(torch.randn((batch_size, n_classes, size, size, size)))
    return y_true, y_pred, dtm


class TestGenSurfLoss:
    """Tests for the GenSurfLoss class."""

    def test_default_init(self):
        """Test default initialization."""
        loss_fn = GenSurfLoss()
        # Default should be False (Dice/CE includes background).
        assert loss_fn.exclude_background is False

    def test_forward_runs_successfully(self):
        """Test that forward pass returns a scalar."""
        loss_fn = GenSurfLoss()
        y_true, y_pred, dtm = _make_mock_data()

        loss = loss_fn(y_true, y_pred, dtm=dtm)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_surface_term_excludes_background_when_flag_false(self):
        """Test surface math ignores background even if exclude_background=False.

        This verifies that the manual slicing inside forward() works.
        """
        # 1. Setup: exclude_background=False (Default).
        # We use alpha=0 to isolate the Surface term.
        loss_fn = GenSurfLoss(exclude_background=False)
        y_true, y_pred, dtm = _make_mock_data(n_classes=3)

        # Run loss (Pure Surface Loss).
        loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)

        # 2. Manual Verification (Foreground Only).
        # Retrieve constants from the instance (inherited from DiceCELoss).
        eps = loss_fn.avoid_division_by_zero
        dims = loss_fn.spatial_dims_3d

        # Manually convert to OneHot/Softmax.
        n_classes = 3
        y_true_oh = torch.nn.functional.one_hot(
            y_true.squeeze(1), num_classes=n_classes
        ).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        y_pred_soft = torch.softmax(y_pred, dim=1)

        # Slice to remove background (channel 0) from ALL tensors.
        y_true_fg = y_true_oh[:, 1:]
        y_pred_fg = y_pred_soft[:, 1:]
        dtm_fg = dtm[:, 1:]

        # Replicate math.
        diff = 1.0 - (y_true_fg + y_pred_fg)

        numerator = torch.sum((dtm_fg * diff) ** 2, dim=dims)
        denominator = torch.sum(dtm_fg ** 2, dim=dims) + eps

        expected_surf_loss = torch.mean(1.0 - numerator / denominator)

        assert torch.isclose(loss, expected_surf_loss, atol=1e-5)

    def test_surface_term_excludes_background_when_flag_true(self):
        """Test surface math works correctly when exclude_background=True.

        In this case, preprocess() removes the background from predictions
        automatically, so the manual slice for predictions is skipped, 
        but DTM must still be sliced.
        """
        # 1. Setup: exclude_background=True.
        loss_fn = GenSurfLoss(exclude_background=True)
        y_true, y_pred, dtm = _make_mock_data(n_classes=3)

        loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)

        # 2. Manual Verification.
        # Retrieve constants from the instance.
        eps = loss_fn.avoid_division_by_zero
        dims = loss_fn.spatial_dims_3d

        # Simulate preprocess() result (which slices background).
        y_true_oh = torch.nn.functional.one_hot(
            y_true.squeeze(1), num_classes=3
        ).permute(0, 4, 1, 2, 3)[:, 1:]
        y_pred_soft = torch.softmax(y_pred, dim=1)[:, 1:]

        # DTM must be manually sliced to match.
        dtm_fg = dtm[:, 1:]

        diff = 1.0 - (y_true_oh + y_pred_soft)

        numerator = torch.sum((dtm_fg * diff) ** 2, dim=dims)
        denominator = torch.sum(dtm_fg ** 2, dim=dims) + eps

        expected_surf_loss = torch.mean(1.0 - numerator / denominator)

        assert torch.isclose(loss, expected_surf_loss, atol=1e-5)

    def test_missing_dtm_raises_error(self):
        """Test that missing DTM argument raises error."""
        loss_fn = GenSurfLoss()
        y_true, y_pred, _ = _make_mock_data()

        with pytest.raises(ValueError, match="requires 'dtm'"):
            loss_fn(y_true, y_pred)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_alpha_weighting_stability(self, alpha):
        """Test that loss runs successfully with different alpha values."""
        loss_fn = GenSurfLoss()
        y_true, y_pred, dtm = _make_mock_data()

        loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=alpha)
        assert not torch.isnan(loss)
