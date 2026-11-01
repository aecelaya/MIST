"""Tests for the MIST Primus model class."""

import pytest
import torch

# MIST imports.
from mist.models.model_loader import load_pretrained_encoder
from mist.models.primus.mist_primus import Primus


def _tiny(patch_size=(16, 16, 16), in_channels=1, **kwargs):
    """Build a tiny Primus for fast CPU tests."""
    return Primus(
        in_channels=in_channels,
        out_channels=2,
        patch_size=patch_size,
        embed_dim=48,
        depth=2,
        num_heads=2,
        **kwargs,
    )


# Construction tests.
def test_primus_non_3d_patch_size_raises():
    """2D patch_size raises a clear 3D-only error."""
    with pytest.raises(ValueError, match="3D patch_size"):
        _tiny(patch_size=(16, 16))


def test_primus_patch_size_not_divisible_by_8_raises():
    """Every patch dimension must be divisible by the 8-voxel token size."""
    with pytest.raises(ValueError, match="offending dimensions: \\[12\\]"):
        _tiny(patch_size=(16, 12, 16))


def test_primus_v2_uses_deeper_tokenizer():
    """PrimusV2 has a residual stem; Primus has a single projection."""
    assert hasattr(_tiny().down_projection, "proj")
    assert hasattr(_tiny(use_deeper_tokenizer=True).down_projection, "stem")


def test_primus_conv_init():
    """Convolutions use He init with zero bias."""
    model = _tiny()
    assert torch.all(model.down_projection.proj.bias == 0)
    assert model.down_projection.proj.weight.std() > 0


# Forward pass tests.
@pytest.mark.parametrize("use_deeper_tokenizer", [False, True])
def test_primus_eval_returns_tensor(use_deeper_tokenizer):
    """Eval mode returns the full-resolution prediction for a non-cubic patch."""
    model = _tiny(patch_size=(16, 24, 32), use_deeper_tokenizer=use_deeper_tokenizer)
    model.eval()
    output = model(torch.randn(1, 1, 16, 24, 32))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2, 16, 24, 32)


def test_primus_train_returns_dict_without_deep_supervision():
    """Training mode returns a dict with deep_supervision set to None."""
    model = _tiny()
    model.train()
    output = model(torch.randn(2, 1, 16, 16, 16))
    assert output["prediction"].shape == (2, 2, 16, 16, 16)
    assert output["deep_supervision"] is None


# Encoder state dict and transfer tests.
def test_primus_encoder_state_dict_keys():
    """Encoder weights are the tokenizer and transformer, named as in nnU-Net."""
    encoder_sd = _tiny().get_encoder_state_dict()
    assert "down_projection.proj.weight" in encoder_sd
    assert "eva.pos_embed" in encoder_sd
    assert "eva.blocks.1.attn.q_proj.weight" in encoder_sd
    assert all(k.startswith(("down_projection.", "eva.")) for k in encoder_sd)


def test_adapt_pretrained_state_dict_resizes_cubic_pos_embed():
    """A cubic position embedding from another patch size is resized."""
    source = _tiny(patch_size=(32, 32, 32)).state_dict()
    target = _tiny(patch_size=(16, 24, 32))
    adapted = target.adapt_pretrained_state_dict(source)
    assert adapted["eva.pos_embed"].shape == target.eva.pos_embed.shape
    assert source["eva.pos_embed"].shape == (1, 64, 48)  # Input is not modified.


def test_adapt_pretrained_state_dict_constant_embedding_stays_constant():
    """Resizing a constant embedding keeps it constant."""
    source = {"eva.pos_embed": torch.full((1, 64, 48), 0.5)}
    adapted = _tiny().adapt_pretrained_state_dict(source)
    assert torch.allclose(adapted["eva.pos_embed"], torch.full((1, 8, 48), 0.5))


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"eva.pos_embed": torch.zeros(1, 8, 48)},  # Already the right size.
        {"eva.pos_embed": torch.zeros(1, 12, 48)},  # Not a cubic grid.
        {"eva.pos_embed": torch.zeros(1, 64, 96)},  # Different embed_dim.
    ],
)
def test_adapt_pretrained_state_dict_leaves_unresizable_inputs(source):
    """Missing, matching, non-cubic, or differently wide embeddings are left alone."""
    assert _tiny().adapt_pretrained_state_dict(source) is source


def test_primus_pretrained_transfer_across_patch_size_and_channels(tmp_path):
    """Weights from a 1-channel 32^3 model fully load into a 4-channel 16^3 model."""
    source = _tiny(patch_size=(32, 32, 32))
    path = tmp_path / "source.pt"
    torch.save(source.state_dict(), path)

    target = _tiny(patch_size=(16, 16, 16), in_channels=4)
    _, summary = load_pretrained_encoder(target, str(path), "average")
    assert summary["skipped"] == []
    assert summary["channel_strategy_applied"] == ["down_projection.proj.weight"]
    assert "eva.pos_embed" in summary["loaded"]
    key = "eva.blocks.1.mlp.fc2.weight"
    assert torch.equal(target.state_dict()[key], source.state_dict()[key])
