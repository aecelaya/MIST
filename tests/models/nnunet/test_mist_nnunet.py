"""Tests for the MIST nnUNet model class."""
import torch
import pytest

# MIST imports.
from mist.models.nnunet.mist_nnunet import NNUNet


# Fixtures.
@pytest.fixture
def base_kwargs():
    """Base keyword arguments for constructing the NNUNet."""
    return {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "patch_size": [32, 32, 32],
        "target_spacing": [1.0, 1.0, 1.0],
        "use_residual_blocks": True,
        "use_deep_supervision": False,
        "use_pocket_model": False,
    }


# Construction tests.
def test_nnunet_initialization_full_model(base_kwargs):
    """Covers construction of a standard nnUNet model."""
    model = NNUNet(**base_kwargs)
    assert isinstance(model.unet, torch.nn.Module)


def test_nnunet_initialization_pocket_model(base_kwargs):
    """Covers construction of a pocket nnUNet model."""
    base_kwargs["use_pocket_model"] = True
    model = NNUNet(**base_kwargs)
    assert isinstance(model.unet, torch.nn.Module)


def test_nnunet_mismatched_dimensions_raises(base_kwargs):
    """Covers the case where patch size and spacing do not match spatial dims."""
    base_kwargs["patch_size"] = [32, 32]
    with pytest.raises(
        ValueError, match="must have the same number of dimensions"
    ):
        NNUNet(**base_kwargs)


def test_nnunet_initialization_2d_filters_branch(base_kwargs):
    """Covers the 2D filters configuration path in model initialization."""
    base_kwargs.update({
        "spatial_dims": 2,
        "patch_size": [32, 32],
        "target_spacing": [1.0, 1.0],
        "use_pocket_model": False,
    })
    model = NNUNet(**base_kwargs)
    assert isinstance(model.unet, torch.nn.Module)


# Forward pass tests.
def test_nnunet_forward_output_shape(base_kwargs):
    """Covers basic forward pass and output shape."""
    model = NNUNet(**base_kwargs)
    model.eval()
    input_tensor = torch.randn(1, base_kwargs["in_channels"], 32, 32, 32)
    output = model(input_tensor)

    # Deep supervision is off; expect tensor output.
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1
    assert output.shape[1] == base_kwargs["out_channels"]


def test_nnunet_forward_with_deep_supervision(base_kwargs):
    """Covers forward pass with deep supervision enabled."""
    base_kwargs.update({"use_deep_supervision": True})
    model = NNUNet(**base_kwargs)
    input_tensor = torch.randn(1, base_kwargs["in_channels"], 32, 32, 32)
    output = model(input_tensor)

    # Expect dictionary output with deep supervision keys.
    assert isinstance(output, dict)
    assert "prediction" in output
    assert "deep_supervision" in output
    assert isinstance(output["deep_supervision"], list)
