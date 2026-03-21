"""Unit tests for model construction and loading utilities."""
from collections import OrderedDict
from unittest.mock import patch, MagicMock
import pytest
import torch

# MIST imports.
from mist.models.model_loader import (
    average_fold_weights,
    validate_encoder_compatibility,
    load_pretrained_encoder,
    validate_mist_config_for_model_loading,
    convert_legacy_model_config,
    get_model,
    load_model_from_config
)
from mist.models.nnunet.mist_nnunet import NNUNet

@pytest.fixture
def valid_mist_config():
    """Fixture for a valid MIST model configuration."""
    return {
        "model": {
            "architecture": "nnunet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
                "target_spacing": [1.0, 1.0, 1.0],
                "patch_size": [64, 64, 64],
                "use_residual_blocks": False,
                "use_deep_supervision": False,
                "use_pocket_model": False,
            }
        }
    }


@pytest.fixture
def legacy_mist_config():
    """Fixture for a legacy MIST model configuration."""
    return {
        "model_name": "nnunet",
        "n_channels": 1,
        "n_classes": 2,
        "deep_supervision": False,
        "pocket": False,
        "patch_size": [64, 64, 64],
        "target_spacing": [1.0, 1.0, 1.0],
        "use_res_block": False,
    }


# ---------------------------------------------------------------------------
# average_fold_weights tests
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_state_dict():
    """A minimal state dict with one weight tensor."""
    return OrderedDict({
        "encoder.weight": torch.ones(4, 1, 3, 3, 3),
        "encoder.bias": torch.zeros(4),
    })


def test_average_fold_weights_returns_ordered_dict(simple_state_dict, tmp_path):
    """Averaging two identical state dicts returns a valid OrderedDict."""
    p1 = tmp_path / "fold0.pt"
    p2 = tmp_path / "fold1.pt"
    torch.save(simple_state_dict, p1)
    torch.save(simple_state_dict, p2)

    avg = average_fold_weights([str(p1), str(p2)])
    assert isinstance(avg, OrderedDict)
    assert set(avg.keys()) == set(simple_state_dict.keys())


def test_average_fold_weights_correct_mean(tmp_path):
    """Averaging two state dicts produces the correct element-wise mean."""
    sd1 = OrderedDict({"weight": torch.ones(2, 2)})
    sd2 = OrderedDict({"weight": torch.full((2, 2), 3.0)})
    p1, p2 = tmp_path / "f0.pt", tmp_path / "f1.pt"
    torch.save(sd1, p1)
    torch.save(sd2, p2)

    avg = average_fold_weights([str(p1), str(p2)])
    expected = torch.full((2, 2), 2.0)
    assert torch.allclose(avg["weight"], expected)


def test_average_fold_weights_strips_ddp_prefix(tmp_path):
    """DDP module. prefix is stripped before averaging."""
    sd = OrderedDict({
        "module.encoder.weight": torch.ones(4, 1, 3, 3, 3),
        "module.encoder.bias": torch.zeros(4),
    })
    p = tmp_path / "ddp.pt"
    torch.save(sd, p)

    avg = average_fold_weights([str(p)])
    assert "encoder.weight" in avg
    assert "encoder.bias" in avg
    assert not any(k.startswith("module.") for k in avg)


def test_average_fold_weights_saves_to_output_path(simple_state_dict, tmp_path):
    """Averaged weights are saved when output_path is provided."""
    p = tmp_path / "fold.pt"
    out = tmp_path / "avg.pt"
    torch.save(simple_state_dict, p)

    average_fold_weights([str(p)], output_path=str(out))
    assert out.exists()

    loaded = torch.load(str(out), weights_only=True)
    assert set(loaded.keys()) == set(simple_state_dict.keys())


def test_average_fold_weights_mismatched_keys_raises(tmp_path):
    """Checkpoints with different keys raise ValueError."""
    sd1 = OrderedDict({"weight": torch.ones(2)})
    sd2 = OrderedDict({"different_weight": torch.ones(2)})
    p1, p2 = tmp_path / "f0.pt", tmp_path / "f1.pt"
    torch.save(sd1, p1)
    torch.save(sd2, p2)

    with pytest.raises(ValueError, match="different keys"):
        average_fold_weights([str(p1), str(p2)])


# ---------------------------------------------------------------------------
# validate_encoder_compatibility tests
# ---------------------------------------------------------------------------

@pytest.fixture
def nnunet_config():
    return {
        "model": {
            "architecture": "nnunet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
                "patch_size": [32, 32, 32],
                "target_spacing": [1.0, 1.0, 1.0],
                "use_residual_blocks": False,
                "use_deep_supervision": False,
                "use_pocket_model": True,
            }
        }
    }


@pytest.fixture
def mednext_config():
    return {
        "model": {
            "architecture": "mednext_small",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
                "use_residual_blocks": False,
                "use_deep_supervision": False,
                "use_pocket_model": False,
            }
        }
    }


def test_validate_encoder_compatibility_identical_configs_passes(nnunet_config):
    """Identical configs are compatible."""
    validate_encoder_compatibility(nnunet_config, nnunet_config)


def test_validate_encoder_compatibility_allows_in_out_channel_diff(nnunet_config):
    """Differing in_channels and out_channels are allowed."""
    target = {
        "model": {
            "architecture": "nnunet",
            "params": {
                **nnunet_config["model"]["params"],
                "in_channels": 4,
                "out_channels": 5,
            }
        }
    }
    validate_encoder_compatibility(nnunet_config, target)


def test_validate_encoder_compatibility_arch_mismatch_raises(
    nnunet_config, mednext_config
):
    """Different architectures raise ValueError."""
    with pytest.raises(ValueError, match="Architecture mismatch"):
        validate_encoder_compatibility(nnunet_config, mednext_config)


def test_validate_encoder_compatibility_arch_mismatch_force_warns(
    nnunet_config, mednext_config
):
    """force=True emits a warning instead of raising."""
    with pytest.warns(UserWarning, match="Architecture mismatch"):
        validate_encoder_compatibility(nnunet_config, mednext_config, force=True)


def test_validate_encoder_compatibility_patch_size_mismatch_raises(nnunet_config):
    """Differing patch_size raises for adaptive architectures."""
    target = {
        "model": {
            "architecture": "nnunet",
            "params": {**nnunet_config["model"]["params"], "patch_size": [64, 64, 64]},
        }
    }
    with pytest.raises(ValueError, match="patch_size mismatch"):
        validate_encoder_compatibility(nnunet_config, target)


def test_validate_encoder_compatibility_spacing_mismatch_raises(nnunet_config):
    """Differing target_spacing raises for adaptive architectures."""
    target = {
        "model": {
            "architecture": "nnunet",
            "params": {
                **nnunet_config["model"]["params"],
                "target_spacing": [0.5, 0.5, 0.5],
            },
        }
    }
    with pytest.raises(ValueError, match="target_spacing mismatch"):
        validate_encoder_compatibility(nnunet_config, target)


def test_validate_encoder_compatibility_non_adaptive_ignores_patch_size(
    mednext_config,
):
    """Non-adaptive architectures only require matching architecture name."""
    target = {
        "model": {
            "architecture": "mednext_small",
            "params": {**mednext_config["model"]["params"], "in_channels": 4},
        }
    }
    validate_encoder_compatibility(mednext_config, target)


# ---------------------------------------------------------------------------
# load_pretrained_encoder tests
# ---------------------------------------------------------------------------

@pytest.fixture
def nnunet_source():
    """Minimal source NNUNet with in_channels=1."""
    return NNUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )


@pytest.fixture
def source_checkpoint(nnunet_source, tmp_path):
    """Save source model checkpoint and return path."""
    path = str(tmp_path / "source.pt")
    torch.save(nnunet_source.state_dict(), path)
    return path


def test_load_pretrained_encoder_same_channels_returns_summary(
    nnunet_source, source_checkpoint
):
    """Loading into same-channel model populates the summary."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # Different out_channels — decoder should be fresh.
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    _, summary = load_pretrained_encoder(target, source_checkpoint)
    assert len(summary["loaded"]) > 0
    assert len(summary["channel_strategy_applied"]) == 0


def test_load_pretrained_encoder_encoder_weights_transferred(
    nnunet_source, source_checkpoint
):
    """Encoder weights in target should match source after loading."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    target, _ = load_pretrained_encoder(target, source_checkpoint)

    source_enc = nnunet_source.get_encoder_state_dict()
    target_enc = target.get_encoder_state_dict()
    for key in source_enc:
        assert torch.allclose(source_enc[key].float(), target_enc[key].float())


def test_load_pretrained_encoder_decoder_unchanged(
    nnunet_source, source_checkpoint
):
    """Decoder weights must not be overwritten by encoder loading."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    decoder_before = {
        k: v.clone() for k, v in target.state_dict().items()
        if not k.startswith(("unet.input_block.", "unet.encoder_layers.", "unet.bottleneck."))
    }
    target, _ = load_pretrained_encoder(target, source_checkpoint)
    decoder_after = {
        k: v for k, v in target.state_dict().items()
        if not k.startswith(("unet.input_block.", "unet.encoder_layers.", "unet.bottleneck."))
    }
    for key in decoder_before:
        assert torch.allclose(decoder_before[key].float(), decoder_after[key].float())


def test_load_pretrained_encoder_in_channel_average_strategy(
    source_checkpoint, tmp_path
):
    """'average' strategy produces a weight with the correct target shape."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=4,  # Different from source (1).
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    target, summary = load_pretrained_encoder(
        target, source_checkpoint, in_channel_strategy="average"
    )
    assert len(summary["channel_strategy_applied"]) > 0
    # Verify the affected weights have the target shape.
    target_enc = target.get_encoder_state_dict()
    for key in summary["channel_strategy_applied"]:
        assert target_enc[key].shape[1] == 4


def test_load_pretrained_encoder_in_channel_first_strategy(
    source_checkpoint,
):
    """'first' strategy applies without error and produces correct shape."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    target, summary = load_pretrained_encoder(
        target, source_checkpoint, in_channel_strategy="first"
    )
    assert len(summary["channel_strategy_applied"]) > 0
    target_enc = target.get_encoder_state_dict()
    for key in summary["channel_strategy_applied"]:
        assert target_enc[key].shape[1] == 2


def test_load_pretrained_encoder_in_channel_skip_strategy(
    source_checkpoint,
):
    """'skip' strategy skips the mismatched layer entirely."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    _, summary = load_pretrained_encoder(
        target, source_checkpoint, in_channel_strategy="skip"
    )
    assert len(summary["channel_strategy_applied"]) == 0


def test_load_pretrained_encoder_invalid_strategy_raises(source_checkpoint):
    """An invalid in_channel_strategy raises ValueError."""
    target = NNUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    with pytest.raises(ValueError, match="in_channel_strategy must be one of"):
        load_pretrained_encoder(target, source_checkpoint, in_channel_strategy="bad")


def test_load_pretrained_encoder_strips_ddp_prefix(nnunet_source, tmp_path):
    """Source checkpoints with DDP module. prefix are handled correctly."""
    ddp_sd = OrderedDict(
        {f"module.{k}": v for k, v in nnunet_source.state_dict().items()}
    )
    path = str(tmp_path / "ddp.pt")
    torch.save(ddp_sd, path)

    target = NNUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        patch_size=[32, 32, 32],
        target_spacing=[1.0, 1.0, 1.0],
        use_residual_blocks=False,
        use_deep_supervision=False,
        use_pocket_model=True,
    )
    _, summary = load_pretrained_encoder(target, path)
    assert len(summary["loaded"]) > 0


def test_load_pretrained_encoder_no_get_encoder_raises(source_checkpoint):
    """Models without get_encoder_state_dict raise AttributeError."""
    class BareModel(torch.nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(AttributeError, match="does not implement get_encoder_state_dict"):
        load_pretrained_encoder(BareModel(), source_checkpoint)


# ---------------------------------------------------------------------------
# Model construction and loading tests
# ---------------------------------------------------------------------------

def test_get_model_success(valid_mist_config):
    """Test model construction from valid configuration."""
    validate_mist_config_for_model_loading(valid_mist_config)
    model = get_model(
        valid_mist_config["model"]["architecture"],
        **valid_mist_config["model"]["params"],
    )
    assert isinstance(model, NNUNet)


def test_validate_missing_model_key(valid_mist_config):
    """Test ValueError is raised when 'model' key is missing."""
    valid_mist_config.pop("model")
    with pytest.raises(
        ValueError, match="Missing required key 'model' in configuration."
    ):
        validate_mist_config_for_model_loading(valid_mist_config)


def test_validate_missing_architecture_key(valid_mist_config):
    """Test ValueError is raised when 'architecture' key is missing."""
    valid_mist_config["model"].pop("architecture")
    with pytest.raises(
        ValueError,
        match="Missing required key 'architecture' in model section.",
    ):
        validate_mist_config_for_model_loading(valid_mist_config)


def test_validate_missing_required_params_key(valid_mist_config):
    """Test ValueError is raised when a required parameter is missing."""
    valid_mist_config["model"]["params"].pop("in_channels")
    with pytest.raises(
        ValueError,
        match="Missing required key 'in_channels' in model parameters.",
    ):
        validate_mist_config_for_model_loading(valid_mist_config)


@patch("torch.load")
def test_load_model_from_config_strips_ddp_prefix(
    mock_torch_load, valid_mist_config
):
    """DDP checkpoints (with 'module.' prefix) are stripped before loading."""
    dummy_model = MagicMock(spec=NNUNet)

    # Return a dummy model instance from registry constructor.
    with patch("mist.models.model_loader.get_model", return_value=dummy_model):
        # Fake DDP-wrapped weights.
        mock_torch_load.return_value = {
            "module.encoder.weight": torch.randn(4, 1, 3, 3, 3),
            "module.encoder.bias": torch.randn(4),
        }

        model = load_model_from_config("mock_weights.pth", valid_mist_config)

        # Verify keys were stripped.
        loaded_state_dict = dummy_model.load_state_dict.call_args[0][0]
        assert "encoder.weight" in loaded_state_dict
        assert "encoder.bias" in loaded_state_dict
        assert all(
            not k.startswith("module.") for k in loaded_state_dict.keys()
        )
        assert model is dummy_model


@patch("torch.load")
def test_load_model_from_config_keeps_non_ddp_keys(
    mock_torch_load, valid_mist_config
):
    """Non-DDP checkpoints are loaded without key modification."""
    dummy_model = MagicMock(spec=NNUNet)

    with patch("mist.models.model_loader.get_model", return_value=dummy_model):
        # Raw (non-DDP) state dict.
        mock_torch_load.return_value = {
            "encoder.weight": torch.randn(4, 1, 3, 3, 3),
            "encoder.bias": torch.randn(4),
        }

        model = load_model_from_config("mock_weights.pth", valid_mist_config)

        loaded_state_dict = dummy_model.load_state_dict.call_args[0][0]
        assert "encoder.weight" in loaded_state_dict
        assert "encoder.bias" in loaded_state_dict
        # Ensure nothing was stripped.
        assert all(
            k in ["encoder.weight", "encoder.bias"]
            for k in loaded_state_dict.keys()
        )
        assert model is dummy_model


def test_convert_legacy_model_config_success(legacy_mist_config):
    """Test conversion of legacy model config to new format."""
    new_config = convert_legacy_model_config(legacy_mist_config)
    assert new_config["model"]["architecture"] == "nnunet"
    assert new_config["model"]["params"]["in_channels"] == 1
    assert new_config["model"]["params"]["out_channels"] == 2
    assert new_config["model"]["params"]["patch_size"] == [64, 64, 64]
    assert new_config["model"]["params"]["target_spacing"] == [1.0, 1.0, 1.0]
    assert not new_config["model"]["params"]["use_residual_blocks"]
    assert not new_config["model"]["params"]["use_deep_supervision"]
    assert not new_config["model"]["params"]["use_pocket_model"]


def test_convert_legacy_model_config_missing_keys(legacy_mist_config):
    """Test conversion raises ValueError for missing keys."""
    legacy_mist_config.pop("model_name")
    with pytest.raises(
        ValueError, match="Missing required key 'model_name' in legacy model"
    ):
        convert_legacy_model_config(legacy_mist_config)
