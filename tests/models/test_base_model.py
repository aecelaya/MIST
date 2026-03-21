"""Tests for the MISTModel base class and encoder interface."""
from collections import OrderedDict
import pytest

# MIST imports.
from mist.models.base_model import MISTModel
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.mednext.mist_mednext import MedNeXt
from mist.models.mgnets.mist_mgnets import MGNet
from mist.models.swinunetr.mist_swinunetr import MistSwinUNETR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nnunet_model():
    """Minimal pocket NNUNet for fast construction."""
    return NNUNet(
        in_channels=1,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )


@pytest.fixture
def mednext_model():
    """Minimal MedNeXt for fast construction."""
    return MedNeXt(
        in_channels=1,
        out_channels=2,
        use_residual_blocks=False,
        use_deep_supervision=False,
    )


@pytest.fixture
def mgnet_model():
    """Minimal MGNet for fast construction."""
    return MGNet(
        in_channels=1,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
    )


@pytest.fixture
def swinunetr_model():
    """Minimal SwinUNETR for fast construction."""
    return MistSwinUNETR(in_channels=1, out_channels=2)


# ---------------------------------------------------------------------------
# isinstance tests — all wrappers must inherit MISTModel
# ---------------------------------------------------------------------------

def test_nnunet_is_mist_model(nnunet_model):
    assert isinstance(nnunet_model, MISTModel)


def test_mednext_is_mist_model(mednext_model):
    assert isinstance(mednext_model, MISTModel)


def test_mgnet_is_mist_model(mgnet_model):
    assert isinstance(mgnet_model, MISTModel)


def test_swinunetr_is_mist_model(swinunetr_model):
    assert isinstance(swinunetr_model, MISTModel)


# ---------------------------------------------------------------------------
# Return type — all models must return a non-empty OrderedDict
# ---------------------------------------------------------------------------

def test_nnunet_encoder_state_dict_is_ordered_dict(nnunet_model):
    enc = nnunet_model.get_encoder_state_dict()
    assert isinstance(enc, OrderedDict)
    assert len(enc) > 0


def test_mednext_encoder_state_dict_is_ordered_dict(mednext_model):
    enc = mednext_model.get_encoder_state_dict()
    assert isinstance(enc, OrderedDict)
    assert len(enc) > 0


def test_mgnet_encoder_state_dict_is_ordered_dict(mgnet_model):
    enc = mgnet_model.get_encoder_state_dict()
    assert isinstance(enc, OrderedDict)
    assert len(enc) > 0


def test_swinunetr_encoder_state_dict_is_ordered_dict(swinunetr_model):
    enc = swinunetr_model.get_encoder_state_dict()
    assert isinstance(enc, OrderedDict)
    assert len(enc) > 0


# ---------------------------------------------------------------------------
# Key correctness — encoder keys present, decoder keys absent
# ---------------------------------------------------------------------------

def test_nnunet_encoder_keys_contain_only_encoder_prefixes(nnunet_model):
    enc = nnunet_model.get_encoder_state_dict()
    encoder_prefixes = (
        "unet.input_block.", "unet.encoder_layers.", "unet.bottleneck."
    )
    decoder_prefixes = (
        "unet.decoder_layers.", "unet.output_block.", "unet.deep_supervision_heads."
    )
    assert all(k.startswith(encoder_prefixes) for k in enc)
    assert not any(k.startswith(decoder_prefixes) for k in enc)


def test_mednext_encoder_keys_contain_only_encoder_prefixes(mednext_model):
    enc = mednext_model.get_encoder_state_dict()
    encoder_prefixes = ("stem.", "enc_stages.", "down_blocks.", "bottleneck.")
    decoder_prefixes = ("up_blocks.", "dec_stages.", "out_0.", "out_blocks.")
    assert all(k.startswith(encoder_prefixes) for k in enc)
    assert not any(k.startswith(decoder_prefixes) for k in enc)


def test_mgnet_encoder_keys_contain_only_main_encoder(mgnet_model):
    enc = mgnet_model.get_encoder_state_dict()
    excluded_prefixes = (
        "spikes.", "main_decoder_blocks.", "main_decoder_upsamples.",
        "final_output_conv.", "deep_supervision_heads.",
    )
    assert all(k.startswith("main_encoder.") for k in enc)
    assert not any(k.startswith(excluded_prefixes) for k in enc)


def test_swinunetr_encoder_keys_contain_only_swin_vit(swinunetr_model):
    enc = swinunetr_model.get_encoder_state_dict()
    assert all(k.startswith("model.swinViT.") for k in enc)


# ---------------------------------------------------------------------------
# Encoder keys are a strict subset of the full state dict
# ---------------------------------------------------------------------------

def test_nnunet_encoder_keys_subset_of_full_state_dict(nnunet_model):
    full_keys = set(nnunet_model.state_dict().keys())
    enc_keys = set(nnunet_model.get_encoder_state_dict().keys())
    assert enc_keys.issubset(full_keys)
    assert enc_keys != full_keys


def test_mednext_encoder_keys_subset_of_full_state_dict(mednext_model):
    full_keys = set(mednext_model.state_dict().keys())
    enc_keys = set(mednext_model.get_encoder_state_dict().keys())
    assert enc_keys.issubset(full_keys)
    assert enc_keys != full_keys


def test_mgnet_encoder_keys_subset_of_full_state_dict(mgnet_model):
    full_keys = set(mgnet_model.state_dict().keys())
    enc_keys = set(mgnet_model.get_encoder_state_dict().keys())
    assert enc_keys.issubset(full_keys)
    assert enc_keys != full_keys


def test_swinunetr_encoder_keys_subset_of_full_state_dict(swinunetr_model):
    full_keys = set(swinunetr_model.state_dict().keys())
    enc_keys = set(swinunetr_model.get_encoder_state_dict().keys())
    assert enc_keys.issubset(full_keys)
    assert enc_keys != full_keys
