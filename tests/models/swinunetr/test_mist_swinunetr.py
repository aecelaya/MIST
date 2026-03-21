"""Tests for the MistSwinUNETR wrapper and SwinUNETR-V2 registry."""

import torch
import pytest

from mist.models.swinunetr.mist_swinunetr import MistSwinUNETR
from mist.models.swinunetr.swinunetr_registry import create_swinunetr
from mist.models.model_registry import get_model_from_registry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_kwargs():
    return {
        "in_channels": 1,
        "out_channels": 2,
        "patch_size": [64, 64, 64],
        "target_spacing": [1.0, 1.0, 1.0],
    }


@pytest.fixture
def small_input():
    """64³ input — divisible by 32 as required by SwinUNETR."""
    return torch.randn(1, 1, 64, 64, 64)


# ---------------------------------------------------------------------------
# MistSwinUNETR construction
# ---------------------------------------------------------------------------

class TestMistSwinUNETRInit:
    """Tests for MistSwinUNETR.__init__."""

    def test_constructs_with_defaults(self):
        model = MistSwinUNETR(in_channels=1, out_channels=2)
        assert isinstance(model, torch.nn.Module)

    def test_extra_kwargs_are_ignored(self):
        """Interface kwargs (patch_size, target_spacing, etc.) don't raise."""
        model = MistSwinUNETR(
            in_channels=1, out_channels=2,
            patch_size=[64, 64, 64],
            target_spacing=[1.0, 1.0, 1.0],
        )
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("feature_size", [24, 48, 96])
    def test_feature_sizes_construct(self, feature_size):
        """All three standard feature sizes construct without error."""
        model = MistSwinUNETR(
            in_channels=1, out_channels=2, feature_size=feature_size
        )
        assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# MistSwinUNETR forward
# ---------------------------------------------------------------------------

class TestMistSwinUNETRForward:
    """Tests for MistSwinUNETR.forward."""

    def test_eval_returns_tensor(self, small_input):
        model = MistSwinUNETR(in_channels=1, out_channels=2)
        model.eval()
        with torch.no_grad():
            output = model(small_input)
        assert isinstance(output, torch.Tensor)

    def test_eval_output_shape(self, small_input):
        """Output spatial dims match input; channels equal out_channels."""
        model = MistSwinUNETR(in_channels=1, out_channels=2)
        model.eval()
        with torch.no_grad():
            output = model(small_input)
        assert output.shape == (1, 2, 64, 64, 64)

    def test_train_returns_dict(self, small_input):
        model = MistSwinUNETR(in_channels=1, out_channels=2)
        model.train()
        with torch.no_grad():
            output = model(small_input)
        assert isinstance(output, dict)
        assert "prediction" in output
        assert "deep_supervision" in output
        assert output["deep_supervision"] is None

    def test_train_prediction_shape(self, small_input):
        model = MistSwinUNETR(in_channels=1, out_channels=2)
        model.train()
        with torch.no_grad():
            output = model(small_input)
        assert output["prediction"].shape == (1, 2, 64, 64, 64)

    def test_multi_channel_input(self):
        model = MistSwinUNETR(in_channels=4, out_channels=3)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 4, 64, 64, 64))
        assert output.shape == (1, 3, 64, 64, 64)


# ---------------------------------------------------------------------------
# create_swinunetr factory
# ---------------------------------------------------------------------------

class TestCreateSwinUNETR:
    """Tests for the create_swinunetr factory function."""

    def test_creates_small(self, base_kwargs):
        model = create_swinunetr("small", **base_kwargs)
        assert isinstance(model, MistSwinUNETR)

    def test_creates_base(self, base_kwargs):
        model = create_swinunetr("base", **base_kwargs)
        assert isinstance(model, MistSwinUNETR)

    def test_creates_large(self, base_kwargs):
        model = create_swinunetr("large", **base_kwargs)
        assert isinstance(model, MistSwinUNETR)

    def test_variant_is_case_insensitive(self, base_kwargs):
        model = create_swinunetr("SMALL", **base_kwargs)
        assert isinstance(model, MistSwinUNETR)

    def test_unknown_variant_raises(self, base_kwargs):
        with pytest.raises(ValueError, match="Unknown SwinUNETR variant"):
            create_swinunetr("xlarge", **base_kwargs)

    @pytest.mark.parametrize("missing_key", ["in_channels", "out_channels"])
    def test_missing_required_key_raises(self, base_kwargs, missing_key):
        del base_kwargs[missing_key]
        with pytest.raises(ValueError, match=f"Missing required key '{missing_key}'"):
            create_swinunetr("small", **base_kwargs)

    def test_small_has_smaller_feature_size_than_base(self, base_kwargs):
        """Variants differ in capacity: small < base < large."""
        small = create_swinunetr("small", **base_kwargs)
        base = create_swinunetr("base", **base_kwargs)
        large = create_swinunetr("large", **base_kwargs)

        def param_count(m):
            return sum(p.numel() for p in m.parameters())

        assert param_count(small) < param_count(base) < param_count(large)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegisteredModels:
    """Tests that all three variants are registered in the model registry."""

    @pytest.mark.parametrize("name", [
        "swinunetr-small", "swinunetr-base", "swinunetr-large"
    ])
    def test_variant_is_registered(self, base_kwargs, name):
        model = get_model_from_registry(name, **base_kwargs)
        assert isinstance(model, MistSwinUNETR)
