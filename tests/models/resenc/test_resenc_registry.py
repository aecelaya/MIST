"""Unit tests for the Residual Encoder UNet factory function."""

import pytest

# MIST imports.
from mist.models.model_registry import get_model_from_registry
from mist.models.resenc.mist_resenc import ResEncUNet
from mist.models.resenc.resenc_registry import create_resenc


@pytest.fixture
def valid_config():
    """Returns a valid configuration dictionary for the ResEnc model."""
    return {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": (32, 32, 32),
        "target_spacing": (1.0, 1.0, 1.0),
    }


def test_create_resenc_success(valid_config):
    """create_resenc returns a ResEncUNet with deep supervision enabled."""
    model = create_resenc(**valid_config)
    assert isinstance(model, ResEncUNet)
    assert model.use_deep_supervision


def test_resenc_is_registered(valid_config):
    """The model is available from the registry as 'resenc'."""
    assert isinstance(get_model_from_registry("resenc", **valid_config), ResEncUNet)


@pytest.mark.parametrize(
    "missing_key",
    ["in_channels", "out_channels", "patch_size", "target_spacing"],
)
def test_create_resenc_missing_key_raises(valid_config, missing_key):
    """A missing required key raises a ValueError naming the key."""
    del valid_config[missing_key]
    with pytest.raises(ValueError, match=f"Missing required key '{missing_key}'"):
        create_resenc(**valid_config)
