"""Unit tests for MIST-compatible MedNeXt implementation."""
import torch
import pytest

# MIST imports.
from mist.models.mednext.mist_mednext import MedNeXt


def test_mednext_forward_eval_mode():
    """Test MedNeXt forward pass in eval mode (no deep supervision)."""
    model = MedNeXt(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=False,
    )
    model.eval()
    x = torch.randn(1, 1, 32, 64, 64)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 3


def test_mednext_forward_train_mode():
    """Test MedNeXt forward pass in train mode without deep supervision."""
    model = MedNeXt(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=False,
    )
    model.train()
    x = torch.randn(1, 1, 32, 64, 64)
    output = model(x)
    assert isinstance(output, dict)
    assert "prediction" in output
    assert output["prediction"].shape[0] == x.shape[0]
    assert output["prediction"].shape[1] == 3
    assert output["deep_supervision"] is None


def test_mednext_forward_with_deep_supervision():
    """Test MedNeXt forward pass in train mode with deep supervision."""
    model = MedNeXt(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=True,
        blocks_up=(1, 1),
        blocks_down=(1, 1),
        blocks_bottleneck=1,
    )
    model.train()
    x = torch.randn(1, 1, 32, 64, 64)
    output = model(x)

    assert isinstance(output, dict)
    assert "prediction" in output
    assert "deep_supervision" in output
    assert isinstance(output["deep_supervision"], list)
    assert all(
        ds.shape == output["prediction"].shape for
        ds in output["deep_supervision"]
    )
