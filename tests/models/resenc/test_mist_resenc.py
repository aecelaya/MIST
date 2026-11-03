"""Tests for the MIST Residual Encoder UNet model class."""

import pytest
import torch

# MIST imports.
from mist.models.model_loader import load_pretrained_encoder
from mist.models.resenc.mist_resenc import ResEncUNet


@pytest.fixture
def base_kwargs():
    """Base keyword arguments for constructing the ResEncUNet."""
    return {
        "in_channels": 1,
        "out_channels": 2,
        "patch_size": [32, 32, 32],
        "target_spacing": [1.0, 1.0, 1.0],
        "use_deep_supervision": False,
    }


# Construction tests.
def test_resenc_topology_follows_patch_size(base_kwargs):
    """A 32^3 patch yields four stages with the ResEnc block counts."""
    model = ResEncUNet(**base_kwargs)
    assert model.encoder.output_channels == [32, 64, 128, 256]
    assert [len(stage.blocks) for stage in model.encoder.stages] == [1, 3, 4, 6]


def test_resenc_full_depth_matches_nnfoundation_cnn_size():
    """A 192^3 single-channel model has six stages and ~102M parameters."""
    model = ResEncUNet(1, 1, [192, 192, 192], [1.0, 1.0, 1.0], use_deep_supervision=False)
    assert model.encoder.output_channels == [32, 64, 128, 256, 320, 320]
    n_params = sum(p.numel() for p in model.parameters())
    assert 102.0e6 < n_params < 102.5e6


# SHA-256 of the sorted "name:shape" lines of the nnFoundationCNN checkpoint's
# network_weights (huggingface.co/MIC-DKFZ/nnFoundationCNN, checkpoint_final.pth),
# excluding the duplicate "decoder.encoder.*" names nnU-Net's decoder registers.
# The checkpoint is 410 MB, so the fingerprint stands in for it. With the
# weights loaded, MIST's network was verified to match nnU-Net's own
# ResidualEncoderUNet output exactly.
NNFOUNDATION_CNN_TENSORS = 508
NNFOUNDATION_CNN_FINGERPRINT = "e6587bdf1a95a2d3bb66a675714a6f48d301f0dd0360e3594d1f87da30d34892"


def test_resenc_layout_matches_nnfoundation_cnn_checkpoint():
    """At 192^3 and 1 mm, every name and shape matches nnFoundationCNN exactly.

    Encoder, decoder, and head all line up, so the checkpoint loads in full:
    MIST transfers the encoder, and MISFIT can start MAE pretraining from the
    whole network. A failure here means a change to the ResEnc layout broke
    compatibility with nnFoundationCNN and nnssl ResEnc-L checkpoints.
    """
    import hashlib

    model = ResEncUNet(1, 1, [192, 192, 192], [1.0, 1.0, 1.0], use_deep_supervision=False)
    lines = sorted(f"{k}:{tuple(v.shape)}" for k, v in model.state_dict().items())
    assert len(lines) == NNFOUNDATION_CNN_TENSORS
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == (NNFOUNDATION_CNN_FINGERPRINT)


def test_resenc_non_3d_patch_size_raises(base_kwargs):
    """2D patch_size raises a clear 3D-only error."""
    base_kwargs["patch_size"] = [32, 32]
    base_kwargs["target_spacing"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="3D patch_size"):
        ResEncUNet(**base_kwargs)


def test_resenc_mismatched_patch_spacing_raises(base_kwargs):
    """Mismatched patch_size / target_spacing lengths raises ValueError."""
    base_kwargs["target_spacing"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="must have the same number of dimensions"):
        ResEncUNet(**base_kwargs)


def test_resenc_initialization_zeroes_last_norm(base_kwargs):
    """Each residual block's last norm starts at zero."""
    model = ResEncUNet(**base_kwargs)
    block = model.encoder.stages[1].blocks[0]
    assert torch.all(block.conv2.norm.weight == 0)
    assert torch.all(block.conv2.norm.bias == 0)
    assert torch.all(model.decoder.transpconvs[0].bias == 0)


# Forward pass tests.
def test_resenc_eval_returns_tensor(base_kwargs):
    """Eval mode returns the full-resolution prediction."""
    model = ResEncUNet(**base_kwargs)
    model.eval()
    output = model(torch.randn(1, 1, 32, 32, 32))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2, 32, 32, 32)


def test_resenc_train_without_deep_supervision(base_kwargs):
    """Training mode returns a dict with no deep supervision outputs."""
    model = ResEncUNet(**base_kwargs)
    model.train()
    output = model(torch.randn(1, 1, 32, 32, 32))
    assert output["prediction"].shape == (1, 2, 32, 32, 32)
    assert output["deep_supervision"] is None


def test_resenc_train_with_deep_supervision(base_kwargs):
    """Deep supervision gives two heads interpolated to full resolution."""
    base_kwargs["use_deep_supervision"] = True
    model = ResEncUNet(**base_kwargs)
    model.train()
    output = model(torch.randn(1, 1, 32, 32, 32))
    assert len(output["deep_supervision"]) == 2
    for head in output["deep_supervision"]:
        assert head.shape == (1, 2, 32, 32, 32)


def test_resenc_deep_supervision_capped_by_depth():
    """A three-stage network has two decoder stages, so only one head."""
    model = ResEncUNet(1, 2, [16, 16, 16], [1.0, 1.0, 1.0], use_deep_supervision=True)
    model.train()
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert len(model.encoder.stages) == 3
    assert len(output["deep_supervision"]) == 1


# Encoder state dict and transfer tests.
def test_resenc_encoder_state_dict_keys(base_kwargs):
    """Encoder weights use nnU-Net's key names and exclude the decoder."""
    model = ResEncUNet(**base_kwargs)
    encoder_sd = model.get_encoder_state_dict()
    assert all(k.startswith("encoder.") for k in encoder_sd)
    assert "encoder.stem.convs.0.conv.weight" in encoder_sd
    assert "encoder.stem.convs.0.all_modules.0.weight" in encoder_sd
    assert "encoder.stages.1.blocks.0.skip.1.conv.weight" in encoder_sd
    n_decoder_keys = sum(k.startswith("decoder.") for k in model.state_dict())
    assert len(encoder_sd) + n_decoder_keys == len(model.state_dict())


def test_resenc_pretrained_encoder_transfers_across_channels(tmp_path, base_kwargs):
    """A single-channel encoder loads into a four-channel model by key."""
    source = ResEncUNet(**base_kwargs)
    weights_path = tmp_path / "source.pt"
    torch.save(source.state_dict(), weights_path)

    base_kwargs["in_channels"] = 4
    target = ResEncUNet(**base_kwargs)
    target, summary = load_pretrained_encoder(target, str(weights_path), "average")

    assert summary["skipped"] == []
    assert sorted(summary["channel_strategy_applied"]) == [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.stem.convs.0.conv.weight",
    ]
    key = "encoder.stages.3.blocks.5.conv2.conv.weight"
    assert torch.equal(target.state_dict()[key], source.state_dict()[key])
