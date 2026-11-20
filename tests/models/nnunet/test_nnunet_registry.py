"""Unit tests for the nnUNet model factory function."""

import pytest

from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.nnunet.mist_plain_nnunet import PlainNNUNet

# MIST imports.
from mist.models.nnunet.nnunet_registry import (
    create_nnunet,
    create_nnunet_plain,
    create_nnunet_plain_pocket,
    create_nnunet_pocket,
    create_nnunet_res_pocket,
)


@pytest.fixture
def valid_config():
    """Returns a valid configuration dictionary for the NNUNet model."""
    return {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": (128, 128, 128),
        "target_spacing": (1.0, 1.0, 1.0),
    }


def test_create_nnunet_success(valid_config):
    """Test that create_nnunet returns an NNUNet instance for valid config."""
    model = create_nnunet(**valid_config)
    assert isinstance(model, NNUNet)


@pytest.mark.parametrize("factory", [create_nnunet_pocket, create_nnunet_res_pocket])
def test_create_nnunet_pocket_success(valid_config, factory):
    """nnunet-pocket and nnunet-res-pocket both build the residual NNUNet."""
    model = factory(**valid_config)
    assert isinstance(model, NNUNet)


@pytest.mark.parametrize("factory", [create_nnunet_plain, create_nnunet_plain_pocket])
def test_create_nnunet_plain_success(valid_config, factory):
    """nnunet-plain and nnunet-plain-pocket both build PlainNNUNet."""
    model = factory(**valid_config)
    assert isinstance(model, PlainNNUNet)


@pytest.mark.parametrize(
    "missing_key",
    [
        "in_channels",
        "out_channels",
        "patch_size",
        "target_spacing",
    ],
)
def test_create_nnunet_missing_keys(valid_config, missing_key):
    """Test create_nnunet raises ValueError if a required key is missing."""
    config = valid_config.copy()
    config.pop(missing_key)
    with pytest.raises(ValueError, match=f"Missing required key '{missing_key}'"):
        create_nnunet(**config)


@pytest.mark.parametrize(
    "missing_key", ["in_channels", "out_channels", "patch_size", "target_spacing"]
)
def test_create_nnunet_plain_pocket_missing_keys(valid_config, missing_key):
    """The plain factories check their configuration too."""
    config = valid_config.copy()
    config.pop(missing_key)
    with pytest.raises(ValueError, match=f"Missing required key '{missing_key}'"):
        create_nnunet_plain_pocket(**config)
