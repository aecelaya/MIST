"""Unit tests for the Primus factory functions."""

import pytest

# MIST imports.
from mist.models.model_registry import get_model_from_registry
from mist.models.primus.mist_primus import Primus
from mist.models.primus.primus_constants import PrimusConstants as constants
from mist.models.primus.primus_registry import create_primus

VARIANT_NAMES = ["small", "base", "medium", "large"]


@pytest.fixture
def valid_config():
    """Returns a valid configuration dictionary for Primus models."""
    return {
        "in_channels": 2,
        "out_channels": 3,
        "patch_size": [16, 16, 16],
        "target_spacing": [1.0, 1.0, 1.0],
    }


@pytest.mark.parametrize("version", ["primus", "primusv2"])
@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_registered_variants(valid_config, version, variant):
    """Every size of Primus and PrimusV2 is registered with the paper's sizes."""
    model = get_model_from_registry(f"{version}-{variant}", **valid_config)
    depth, num_heads, embed_dim = constants.VARIANTS[variant]
    assert isinstance(model, Primus)
    assert len(model.eva.blocks) == depth
    assert model.eva.blocks[0].attn.num_heads == num_heads
    assert model.eva.pos_embed.shape[-1] == embed_dim
    assert hasattr(model.down_projection, "stem") == (version == "primusv2")


def test_create_primus_unknown_variant_raises(valid_config):
    """An unknown size raises a ValueError listing the options."""
    with pytest.raises(ValueError, match="Unknown Primus variant: 'huge'"):
        create_primus("huge", use_deeper_tokenizer=False, **valid_config)


@pytest.mark.parametrize("missing_key", ["in_channels", "out_channels", "patch_size"])
def test_create_primus_missing_key_raises(valid_config, missing_key):
    """A missing required key raises a ValueError naming the key."""
    del valid_config[missing_key]
    with pytest.raises(ValueError, match=f"Missing required key '{missing_key}'"):
        create_primus("small", use_deeper_tokenizer=False, **valid_config)
