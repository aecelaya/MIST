"""Unit tests for MIST-compatible MedNeXt implementation."""
import torch
import pytest

# MIST imports.
from mist.models.mednext.mist_mednext import MedNeXt


# ---------------------------------------------------------------------------
# patch_size divisibility guard
# ---------------------------------------------------------------------------

class TestMedNeXtPatchSizeGuard:
    """Tests for patch_size divisibility guard in MedNeXt.__init__."""

    def test_no_patch_size_constructs(self):
        """patch_size=None (default) never raises."""
        model = MedNeXt(in_channels=1, out_channels=2)
        assert isinstance(model, torch.nn.Module)

    def test_divisible_patch_size_passes(self):
        """All dimensions divisible by 16 (default 4-stage) should pass."""
        model = MedNeXt(in_channels=1, out_channels=2, patch_size=[64, 64, 64])
        assert isinstance(model, torch.nn.Module)

    def test_non_divisible_dimension_raises(self):
        """A dimension not divisible by 16 raises ValueError."""
        with pytest.raises(ValueError, match="divisible by 2 \\*\\* len"):
            MedNeXt(in_channels=1, out_channels=2, patch_size=[64, 64, 5])

    def test_custom_blocks_down_changes_divisor(self):
        """2-stage architecture only needs divisibility by 4."""
        # 12 is not divisible by 16 (default 4-stage), but IS by 4 (2-stage).
        model = MedNeXt(
            in_channels=1, out_channels=2,
            patch_size=[64, 64, 12],
            blocks_down=(2, 2), blocks_up=(2, 2),
        )
        assert isinstance(model, torch.nn.Module)

    def test_custom_blocks_down_still_enforces_divisibility(self):
        """2-stage architecture still rejects odd spatial sizes."""
        with pytest.raises(ValueError, match="divisible by 2 \\*\\* len"):
            MedNeXt(
                in_channels=1, out_channels=2,
                patch_size=[64, 64, 5],
                blocks_down=(2, 2), blocks_up=(2, 2),
            )

    def test_extra_kwargs_ignored(self):
        """Unknown kwargs (e.g. target_spacing) do not raise."""
        model = MedNeXt(
            in_channels=1, out_channels=2,
            patch_size=[32, 32, 32],
            target_spacing=[1.0, 1.0, 1.0],
        )
        assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_mednext_forward_eval_mode():
    """Test MedNeXt forward pass in eval mode (no deep supervision)."""
    model = MedNeXt(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=False,
    )
    model.eval()
    x = torch.randn(1, 1, 32, 64, 64)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 3


def test_mednext_forward_train_mode():
    """Test MedNeXt forward pass in train mode without deep supervision."""
    model = MedNeXt(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=False,
    )
    model.train()
    x = torch.randn(1, 1, 32, 64, 64)
    output = model(x)
    assert isinstance(output, dict)
    assert "prediction" in output
    assert output["prediction"].shape[0] == x.shape[0]
    assert output["prediction"].shape[1] == 3
    assert output["deep_supervision"] is None


def test_mednext_forward_with_deep_supervision():
    """Test MedNeXt forward pass in train mode with deep supervision."""
    model = MedNeXt(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=True,
        blocks_up=(1, 1),
        blocks_down=(1, 1),
        blocks_bottleneck=1,
    )
    model.train()
    x = torch.randn(1, 1, 32, 64, 64)
    output = model(x)

    assert isinstance(output, dict)
    assert "prediction" in output
    assert "deep_supervision" in output
    assert isinstance(output["deep_supervision"], list)
    assert all(
        ds.shape == output["prediction"].shape for
        ds in output["deep_supervision"]
    )
