"""Tests for the DynamicUNet model in the MIST package."""
from typing import Dict
import pytest
import torch

from mist.models.nnunet.dynamic_unet import DynamicUNet


def create_valid_params(
        use_deep_supervision: bool = False,
        num_deep_supervision_heads: int = 1
) -> Dict:
    """Helper to generate a valid set of parameters for a 3D UNet.

    In this helper, we define a network with three levels (input, decoder,
    and bottleneck) to ensure the kernel_size and strides have a length of 3.
    For the decoder layers, the upsample_kernel_size must have a length of
    `len(strides)-1` (which is 2 in this case). We use batch normalization and
    ReLU activation.
    """
    params = {
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": [3, 3, 3],
        "strides": [1, 2, 2],
        "upsample_kernel_size": [2, 2],
        "filters": [16, 32, 64],
        "norm_name": "batch",
        "act_name": "relu",
        "dropout": None,
        "use_deep_supervision": use_deep_supervision,
        "num_deep_supervision_heads": num_deep_supervision_heads,
        "use_residual_blocks": False,
        "trans_bias": False,
    }
    return params


def test_forward_train_and_eval():
    """Test basic functionality of the DynamicUNet model.

    Test that the model instantiates correctly and that the forward pass
    returns a dictionary in training mode (with deep supervision outputs)
    and a tensor in evaluation mode.
    """
    params = create_valid_params(
        use_deep_supervision=True, num_deep_supervision_heads=1
    )
    model = DynamicUNet(**params)

    # Create a dummy input tensor for 3D volumes:
    # (batch, channels, depth, height, width).
    x = torch.randn(1, params["in_channels"], 32, 64, 64)

    # Test training mode: output should be a dictionary with both keys.
    model.train()
    out_train = model(x)
    assert isinstance(out_train, dict)
    assert "prediction" in out_train
    # deep_supervision should be a list when deep supervision is enabled.
    assert out_train["deep_supervision"] is not None
    assert isinstance(out_train["deep_supervision"], list)
    # Check that the prediction is a tensor.
    assert torch.is_tensor(out_train["prediction"])

    # Test evaluation mode: output should be a tensor.
    model.eval()
    out_eval = model(x)
    assert torch.is_tensor(out_eval)


def test_invalid_kernel_stride():
    """Test that a ValueError is raised when kernel_size and strides mismatch.

    Test that a ValueError is raised when the lengths of kernel_size and strides
    do not match or are less than 3.
    """
    params = create_valid_params()
    params["kernel_size"] = [3, 3]
    with pytest.raises(ValueError) as excinfo:
        DynamicUNet(**params)
    assert (
        "Length of kernel_size and strides should be the same"
        in str(excinfo.value)
    )


def test_invalid_deep_supervision_heads():
    """
    Test that a ValueError is raised if the number of deep supervision heads is
    not less than the number of upsampling layers.
    Note: with strides length=3, there are 2 upsampling layers.
    """
    params = create_valid_params(use_deep_supervision=True,
                                 num_deep_supervision_heads=2)
    with pytest.raises(ValueError) as excinfo:
        DynamicUNet(**params)
    assert ("num_deep_supervision_heads should be less than the number of" in
            str(excinfo.value))


def test_negative_deep_supervision_heads():
    """Test that a ValueError is raised for negative deep supervision heads.

    Test that a ValueError is raised if the number of deep supervision heads is
    not less than the number of upsampling layers.
    """
    params = create_valid_params(
        use_deep_supervision=True, num_deep_supervision_heads=-1
    )
    with pytest.raises(ValueError) as excinfo:
        DynamicUNet(**params)
    assert (
        "num_deep_supervision_heads should be larger than 0."
        in str(excinfo.value)
    )


def test_invalid_filters_length():
    """Test for ValueError when filters length is less than strides length.

    Test that a ValueError is raised when the length of filters is less than the
    length of strides.
    """
    params = create_valid_params()
    params["filters"] = [16, 32]
    with pytest.raises(ValueError) as excinfo:
        DynamicUNet(**params)
    assert (
        "The length of filters should be no less than the length of strides"
        in str(excinfo.value)
    )


def test_kernel_sequence_length_mismatch():
    """ValueError when a kernel_size element has wrong number of dimensions."""
    params = create_valid_params()
    params["kernel_size"] = [(3, 3), 3, 3]  # (3,3) has 2 dims, not 3
    with pytest.raises(ValueError) as excinfo:
        DynamicUNet(**params)
    assert (
        "Length of kernel_size in block 0 should be 3"
        in str(excinfo.value)
    )


def test_stride_sequence_length_mismatch():
    """ValueError when a strides element has wrong number of dimensions."""
    params = create_valid_params()
    params["strides"] = [(1, 1), 2, 2]  # (1,1) has 2 dims, not 3
    with pytest.raises(ValueError) as excinfo:
        DynamicUNet(**params)
    assert (
        "Length of stride in block 0 should be 3"
        in str(excinfo.value)
    )
