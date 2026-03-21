"""Tests for the MGNet model registry."""

import pytest

from mist.models.mgnets.mgnets_registry import create_mgnet
from mist.models.mgnets.mist_mgnets import MGNet
from mist.models.model_registry import get_model_from_registry


@pytest.fixture
def base_kwargs():
    return {
        "in_channels": 1,
        "out_channels": 2,
        "patch_size": [32, 32, 32],
        "target_spacing": [1.0, 1.0, 1.0],
        "use_residual_blocks": False,
        "use_deep_supervision": True,
    }


class TestCreateMgnet:
    """Tests for the create_mgnet factory function."""

    def test_creates_fmgnet(self, base_kwargs):
        model = create_mgnet("fmgnet", **base_kwargs)
        assert isinstance(model, MGNet)

    def test_creates_wnet(self, base_kwargs):
        model = create_mgnet("wnet", **base_kwargs)
        assert isinstance(model, MGNet)

    def test_variant_is_case_insensitive(self, base_kwargs):
        model = create_mgnet("FMGNet", **base_kwargs)
        assert isinstance(model, MGNet)

    def test_fmgnet_has_progressive_spike_schedule(self, base_kwargs):
        """FMGNet spike schedule is [1, 2, …, bottleneck_layer_idx]."""
        model = create_mgnet("fmgnet", **base_kwargs)
        expected = list(range(1, model.bottleneck_layer_idx + 1))
        assert model.spike_height_schedule == expected

    def test_unknown_variant_raises(self, base_kwargs):
        with pytest.raises(ValueError, match="Unknown MGNet variant"):
            create_mgnet("unknown_variant", **base_kwargs)

    @pytest.mark.parametrize("missing_key", [
        "in_channels", "out_channels", "patch_size", "target_spacing",
        "use_residual_blocks", "use_deep_supervision",
    ])
    def test_missing_required_key_raises(self, base_kwargs, missing_key):
        del base_kwargs[missing_key]
        with pytest.raises(ValueError, match=f"Missing required key '{missing_key}'"):
            create_mgnet("fmgnet", **base_kwargs)

    def test_optional_num_deep_supervision_heads_forwarded(self, base_kwargs):
        """num_deep_supervision_heads is optional and forwarded when provided."""
        base_kwargs["num_deep_supervision_heads"] = 2
        model = create_mgnet("fmgnet", **base_kwargs)
        assert model.num_aux_heads == 1  # total_heads=2 → num_aux=1


class TestRegisteredModels:
    """Tests that fmgnet and wnet are registered in the model registry."""

    def test_fmgnet_is_registered(self, base_kwargs):
        model = get_model_from_registry("fmgnet", **base_kwargs)
        assert isinstance(model, MGNet)

    def test_wnet_is_registered(self, base_kwargs):
        model = get_model_from_registry("wnet", **base_kwargs)
        assert isinstance(model, MGNet)
