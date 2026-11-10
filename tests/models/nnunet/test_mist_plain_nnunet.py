"""Tests for nnunet-plain, the nnunet alias, and nnU-Net's planning rules."""

import pytest
import torch

from mist.models import model_loader
from mist.models.model_registry import canonical_architecture, get_model_from_registry
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.nnunet.mist_plain_nnunet import PlainNNUNet
from mist.models.nnunet.nnunet_utils import get_plain_unet_params, get_unet_params


def _build(name="nnunet-plain", patch_size=(32, 32, 32), spacing=(1.0, 1.0, 1.0)):
    return get_model_from_registry(
        name,
        in_channels=2,
        out_channels=3,
        patch_size=list(patch_size),
        target_spacing=list(spacing),
    )


# Topology. Expected values below came from nnU-Net v2.8.1's get_pool_and_conv_props.


def test_plain_topology_isotropic_128():
    """128 is downsampled to a bottleneck of 4 along every axis."""
    kernels, strides, divisors = get_plain_unet_params([128, 128, 128], [1.0, 1.0, 1.0])
    assert strides == [[1, 1, 1]] + [[2, 2, 2]] * 5
    assert kernels == [[3, 3, 3]] * 6
    assert divisors == [32, 32, 32]


def test_plain_topology_has_no_depth_limit():
    """Unlike the other planners, a large patch gets a sixth downsampling."""
    _, strides, divisors = get_plain_unet_params([256, 256, 256], [1.0, 1.0, 1.0])
    assert len(strides) == 7
    assert divisors == [64, 64, 64]
    assert len(get_unet_params([256, 256, 256], [1.0, 1.0, 1.0])[1]) == 6


def test_plain_topology_anisotropic():
    """A coarse axis joins in only once the others have caught up to it."""
    kernels, strides, divisors = get_plain_unet_params([320, 320, 20], [0.625, 0.625, 3.6])
    assert strides == [
        [1, 1, 1],
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 1],
        [2, 2, 1],
    ]
    assert divisors == [64, 64, 4]
    assert kernels[0] == [3, 3, 1]
    assert kernels[-1] == [3, 3, 3]


def test_plain_topology_kernels_stay_three():
    """Once an axis's kernel is 3, it never goes back to 1."""
    kernels, _, _ = get_plain_unet_params([320, 320, 20], [0.625, 0.625, 3.6])
    for axis in range(3):
        values = [k[axis] for k in kernels]
        assert values == sorted(values)


def test_plain_topology_spacing_ratio_is_strictly_below_two():
    """A ratio of exactly 2 waits, where the other planners would downsample."""
    _, strides, _ = get_plain_unet_params([128, 128, 128], [1.0, 1.0, 2.0])
    assert strides[1] == [2, 2, 1]
    assert get_unet_params([128, 128, 128], [1.0, 1.0, 2.0])[1][1] == [2, 2, 2]


def test_plain_topology_rejects_mismatched_lengths():
    """Patch size and spacing must have the same length."""
    with pytest.raises(ValueError, match="same length"):
        get_plain_unet_params([128, 128], [1.0, 1.0, 1.0])


# The network.


def test_nnunet_plain_state_dict_uses_nnunet_names():
    """Keys match nnU-Net's PlainConvUNet, with two convolutions per stage."""
    keys = _build().state_dict().keys()
    assert "encoder.stages.0.0.convs.0.conv.weight" in keys
    assert "encoder.stages.0.0.convs.0.all_modules.0.weight" in keys
    assert "encoder.stages.0.0.convs.1.norm.weight" in keys
    assert "encoder.stages.0.0.convs.2.conv.weight" not in keys
    assert "decoder.transpconvs.0.weight" in keys
    assert "decoder.stages.0.convs.1.conv.weight" in keys
    assert "decoder.seg_layers.0.weight" in keys
    assert not any(k.startswith("decoder.encoder.") for k in keys)
    assert not any("conv3" in k or "lateral" in k for k in keys)


def test_nnunet_plain_filters_double_up_to_320():
    """Filters follow nnU-Net's 32 x 2^stage, capped at 320."""
    model = _build(patch_size=(256, 256, 256))
    assert model.encoder.output_channels == [32, 64, 128, 256, 320, 320, 320]


def test_nnunet_plain_eval_returns_prediction():
    """In eval mode, the output is the full-resolution prediction."""
    model = _build().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 2, 32, 32, 32))
    assert out.shape == (1, 3, 32, 32, 32)


def test_nnunet_plain_training_returns_deep_supervision():
    """In training, two lower-resolution heads come back at full resolution."""
    model = _build(patch_size=(64, 64, 64)).train()
    out = model(torch.zeros(2, 2, 64, 64, 64))
    assert out["prediction"].shape == (2, 3, 64, 64, 64)
    assert len(out["deep_supervision"]) == 2
    assert all(o.shape == (2, 3, 64, 64, 64) for o in out["deep_supervision"])


@pytest.mark.parametrize(
    "patch_size, spacing",
    [
        ((128, 128, 64), (1.0, 1.0, 1.0)),
        ((64, 64, 16), (0.6, 0.6, 3.6)),
        ((96, 64, 32), (1.0, 1.2, 0.8)),
    ],
)
def test_nnunet_plain_runs_on_compatible_patches(patch_size, spacing):
    """Skip connections match for planned topologies."""
    model = _build(patch_size=patch_size, spacing=spacing).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 2, *patch_size)).shape == (1, 3, *patch_size)


def test_nnunet_plain_rejects_incompatible_patch_with_clear_error():
    """A patch the decoder can't rebuild fails at construction, not mid-training."""
    with pytest.raises(ValueError, match="must be divisible by"):
        _build(patch_size=(288, 288, 288))


def test_nnunet_plain_encoder_state_dict_holds_only_encoder():
    """Transfer learning takes the encoder alone."""
    keys = _build().get_encoder_state_dict().keys()
    assert keys and all(k.startswith("encoder.") for k in keys)


# The alias.


def test_nnunet_and_nnunet_res_build_the_same_network():
    """Old configs and checkpoints under either name stay interchangeable."""
    alias, res = _build("nnunet"), _build("nnunet-res")
    assert isinstance(alias, NNUNet) and isinstance(res, NNUNet)
    assert alias.state_dict().keys() == res.state_dict().keys()
    res.load_state_dict(alias.state_dict(), strict=True)


def test_nnunet_keeps_its_residual_blocks():
    """The name nnunet keeps meaning the residual network from earlier versions."""
    assert any("conv3" in k for k in _build("nnunet").state_dict())
    assert isinstance(_build("nnunet-plain"), PlainNNUNet)


def test_canonical_architecture():
    """Only the alias changes."""
    assert canonical_architecture("nnunet") == "nnunet-res"
    assert canonical_architecture("nnunet-res") == "nnunet-res"
    assert canonical_architecture("nnunet-plain") == "nnunet-plain"
    assert canonical_architecture("resenc") == "resenc"


def _config(architecture, patch_size=(64, 64, 64)):
    return {
        "model": {"architecture": architecture},
        "spatial_config": {"patch_size": list(patch_size), "target_spacing": [1.0, 1.0, 1.0]},
    }


@pytest.mark.parametrize("source, target", [("nnunet", "nnunet-res"), ("nnunet-res", "nnunet")])
def test_encoder_compatibility_treats_alias_as_same(source, target):
    """Pretrained weights move freely between the alias and its target."""
    model_loader.validate_encoder_compatibility(_config(source), _config(target))


def test_encoder_compatibility_alias_still_checks_patch_size():
    """The alias gets the residual network's strict spatial check."""
    with pytest.raises(ValueError, match="patch_size mismatch"):
        model_loader.validate_encoder_compatibility(
            _config("nnunet"), _config("nnunet-res", patch_size=(32, 32, 32))
        )


def test_encoder_compatibility_plain_and_residual_differ():
    """The plain and residual encoders can't exchange weights."""
    with pytest.raises(ValueError, match="Architecture mismatch"):
        model_loader.validate_encoder_compatibility(_config("nnunet"), _config("nnunet-plain"))


def test_encoder_compatibility_plain_patch_mismatch_warns():
    """Plain shapes depend only on stage and kernel, so shared stages transfer."""
    with pytest.warns(UserWarning, match="matching stages still transfer"):
        model_loader.validate_encoder_compatibility(
            _config("nnunet-plain"), _config("nnunet-plain", patch_size=(32, 32, 32))
        )
