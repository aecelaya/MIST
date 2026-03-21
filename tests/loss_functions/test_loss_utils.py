"""Unit tests for mist.loss_functions.loss_utils."""

import pytest
import torch

# MIST imports.
from mist.loss_functions import loss_utils


class TestGetOneHot:
    """Tests for the get_one_hot utility function."""

    def test_valid_input_encoding(self):
        """Test one-hot encoding for correct shape and values."""
        # Input shape: (Batch, Channel, Height, Width, Depth) -> (1, 1, 1, 2, 2)
        # Values: Class 0, 1, and 2 are present.
        y_true = torch.tensor([[[[[1, 0], [2, 1]]]]], dtype=torch.float32)

        # Expected Output Shape: (1, 3, 1, 2, 2)
        one_hot = loss_utils.get_one_hot(y_true, n_classes=3)

        assert one_hot.shape == (1, 3, 1, 2, 2)
        assert one_hot.dtype == torch.float32

        # Check mapping correctness.
        # Channel 0 should mark where y_true == 0.
        assert torch.all(one_hot[:, 0] == (y_true == 0))
        # Channel 1 should mark where y_true == 1.
        assert torch.all(one_hot[:, 1] == (y_true == 1))
        # Channel 2 should mark where y_true == 2.
        assert torch.all(one_hot[:, 2] == (y_true == 2))


class TestCheckLossInputs:
    """Tests for the check_loss_inputs validation utility."""

    def test_valid_inputs(self):
        """Test valid 5D input with matching shapes does not raise error."""
        y_true = torch.zeros((2, 1, 32, 32, 32))
        y_pred = torch.zeros((2, 3, 32, 32, 32))
        loss_utils.check_loss_inputs(y_true, y_pred)

    @pytest.mark.parametrize(
        "y_true_shape,y_pred_shape,error_match",
        [
            (
                (2, 1, 32, 32, 32),
                (2, 1, 32, 32, 32),
                "number of classes.*at least 2",
            ),
            (
                (2, 2, 32, 32, 32),
                (2, 3, 32, 32, 32),
                "number of channels.*must be 1",
            ),
            (
                (2, 1, 32, 32),
                (2, 3, 32, 32, 32),
                "must be 5D",
            ),
            (
                (2, 1, 32, 32, 32),
                (1, 3, 32, 32, 32),
                "batch sizes must match",
            ),
            (
                (2, 1, 32, 32, 32),
                (2, 3, 64, 64, 64),
                "spatial dimensions.*must match",
            ),
        ],
    )
    def test_invalid_inputs_raise_error(
        self, y_true_shape, y_pred_shape, error_match
    ):
        """Test that invalid shapes raise ValueError with correct messages."""
        y_true = torch.zeros(y_true_shape)
        y_pred = torch.zeros(y_pred_shape)
        with pytest.raises(ValueError, match=error_match):
            loss_utils.check_loss_inputs(y_true, y_pred)

    def test_5d_check_fires_before_channel_check(self):
        """4D input with 1 prediction channel raises 5D error, not channel error."""
        y_true = torch.zeros((2, 1, 32, 32))
        y_pred = torch.zeros((2, 1, 32, 32))
        with pytest.raises(ValueError, match="must be 5D"):
            loss_utils.check_loss_inputs(y_true, y_pred)


class TestSoftSkeletonize:
    """Tests for the SoftSkeletonize module and its morphological ops."""

    def test_output_shape(self):
        """Test output shape matches input shape."""
        model = loss_utils.SoftSkeletonize(num_iter=5)
        img = torch.rand((1, 1, 8, 8, 8))
        out = model(img)
        assert out.shape == img.shape
        # Probabilities should be non-negative.
        assert (out >= 0).all()

    def test_erode_dilate_properties(self):
        """Test basic mathematical properties of erosion and dilation."""
        model = loss_utils.SoftSkeletonize(num_iter=5)
        # Create a volume with values 0.5.
        img = torch.ones((1, 1, 8, 8, 8)) * 0.5

        # Add a "hole" (0.0) and a "peak" (1.0) to test min/max pooling.
        img[0, 0, 4, 4, 4] = 0.0
        img[0, 0, 2, 2, 2] = 1.0

        eroded = model.soft_erode(img)
        dilated = model.soft_dilate(img)

        # Erosion should be <= Original.
        assert torch.all(eroded <= img + 1e-6)
        # Dilation should be >= Original.
        assert torch.all(dilated >= img - 1e-6)

    def test_soft_open_behavior(self):
        """Test that soft opening (erode -> dilate) works as expected."""
        model = loss_utils.SoftSkeletonize()
        img = torch.rand((1, 1, 8, 8, 8))

        # Opening generally suppresses bright features smaller than kernel.
        opened = model.soft_open(img)

        # Result should be same shape.
        assert opened.shape == img.shape

        # Result should be non-negative.
        assert (opened >= 0).all()

    def test_numerical_stability(self):
        """Test repeated iterations do not produce NaNs."""
        model = loss_utils.SoftSkeletonize(num_iter=10)
        img = torch.rand((1, 1, 16, 16, 16))
        skel = model(img)
        assert not torch.isnan(skel).any()
        assert not torch.isinf(skel).any()

    def test_raises_on_invalid_dims(self):
        """Test that operations raise ValueError on non-5D input."""
        model = loss_utils.SoftSkeletonize(num_iter=3)
        bad_input = torch.rand((1, 1, 16, 16))  # 4D input.

        # Check main forward pass.
        with pytest.raises(ValueError, match="Expected 5D input"):
            model(bad_input)

        # Check individual methods.
        with pytest.raises(ValueError, match="Expected 5D input"):
            model.soft_dilate(bad_input)

        with pytest.raises(ValueError, match="Expected 5D input"):
            model.soft_erode(bad_input)
