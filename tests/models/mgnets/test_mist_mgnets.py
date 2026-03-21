"""Tests for the MGNet (FMGNet / W-Net) model implementation."""

import torch
import pytest

from mist.models.mgnets.mist_mgnets import MGNet
from mist.models.mgnets.mgnets_constants import MGNetConstants as mgnet_constants
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_kwargs():
    """Minimal valid kwargs for constructing an MGNet."""
    return {
        "in_channels": 1,
        "out_channels": 2,
        "patch_size": [32, 32, 32],
        "target_spacing": [1.0, 1.0, 1.0],
        "use_residual_blocks": False,
        "use_deep_supervision": True,
    }


@pytest.fixture
def small_input():
    """Small 3-D input tensor matching the 32³ patch size."""
    return torch.randn(1, 1, 32, 32, 32)


# ---------------------------------------------------------------------------
# _generate_sparse_w_sequence
# ---------------------------------------------------------------------------

class TestGenerateSparseWSequence:
    """Tests for MGNet._generate_sparse_w_sequence."""

    @pytest.fixture(autouse=True)
    def model(self, base_kwargs):
        self.model = MGNet(**base_kwargs)

    @pytest.mark.parametrize("max_height", [0, 1])
    def test_degenerate_returns_single_one(self, max_height):
        """Heights <= 1 return a single spike of height 1."""
        assert self.model._generate_sparse_w_sequence(max_height) == [1]

    def test_max_height_two(self):
        assert self.model._generate_sparse_w_sequence(2) == [1, 2, 1]

    def test_max_height_three(self):
        assert self.model._generate_sparse_w_sequence(3) == [1, 2, 1, 3, 1, 2, 1]

    def test_max_height_four(self):
        assert self.model._generate_sparse_w_sequence(4) == [
            1, 2, 1, 3, 1, 4, 1, 3, 1, 2, 1
        ]

    @pytest.mark.parametrize("max_height", range(1, 6))
    def test_sequence_is_palindrome(self, max_height):
        """The W-cycle schedule is symmetric (palindrome)."""
        seq = self.model._generate_sparse_w_sequence(max_height)
        assert seq == seq[::-1]

    @pytest.mark.parametrize("max_height", range(1, 6))
    def test_max_value_equals_max_height(self, max_height):
        """The tallest spike reaches exactly max_height."""
        seq = self.model._generate_sparse_w_sequence(max_height)
        assert max(seq) == max_height


# ---------------------------------------------------------------------------
# __init__ — construction
# ---------------------------------------------------------------------------

class TestMGNetInit:
    """Tests for MGNet.__init__ covering topology and attribute setup."""

    def test_fmgnet_constructs(self, base_kwargs):
        """FMGNet can be instantiated without error."""
        assert isinstance(MGNet(mg_net="fmgnet", **base_kwargs), torch.nn.Module)

    def test_wnet_constructs(self, base_kwargs):
        """W-Net can be instantiated without error."""
        assert isinstance(MGNet(mg_net="wnet", **base_kwargs), torch.nn.Module)

    def test_unet_fallback_constructs(self, base_kwargs):
        """Unknown mg_net string falls back to single-spike (U-Net) topology."""
        assert isinstance(MGNet(mg_net="unet", **base_kwargs), torch.nn.Module)

    def test_mg_net_string_is_case_insensitive(self, base_kwargs):
        assert isinstance(MGNet(mg_net="FMGNet", **base_kwargs), torch.nn.Module)

    def test_extra_kwargs_are_ignored(self, base_kwargs):
        """**kwargs absorbs unknown keys without raising."""
        assert isinstance(
            MGNet(**base_kwargs, some_unknown_arg="ignored"), torch.nn.Module
        )

    def test_non_3d_patch_size_raises(self, base_kwargs):
        """2D patch_size raises a clear 3D-only error."""
        base_kwargs["patch_size"] = [32, 32]
        base_kwargs["target_spacing"] = [1.0, 1.0]
        with pytest.raises(ValueError, match="3D patch_size"):
            MGNet(**base_kwargs)

    # --- Pocket paradigm ---

    def test_filters_per_layer_are_constant(self, base_kwargs):
        """All depths use the same filter count (pocket paradigm)."""
        model = MGNet(**base_kwargs)
        assert all(f == constants.INITIAL_FILTERS for f in model.filters_per_layer)

    def test_filters_do_not_depend_on_depth(self, base_kwargs):
        """Filter count is identical at the shallowest and deepest levels."""
        model = MGNet(**base_kwargs)
        assert model.filters_per_layer[0] == model.filters_per_layer[-1]

    # --- Spike schedules ---

    def test_fmgnet_spike_schedule_is_progressive(self, base_kwargs):
        """FMGNet schedule is [1, 2, …, bottleneck_layer_idx]."""
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        expected = list(range(1, model.bottleneck_layer_idx + 1))
        assert model.spike_height_schedule == expected

    def test_wnet_spike_schedule_max_equals_bottleneck(self, base_kwargs):
        """W-Net's tallest spike reaches the bottleneck layer."""
        model = MGNet(mg_net="wnet", **base_kwargs)
        assert max(model.spike_height_schedule) == model.bottleneck_layer_idx

    def test_wnet_spike_schedule_starts_with_one(self, base_kwargs):
        """W-Net schedule starts with a height-1 spike."""
        model = MGNet(mg_net="wnet", **base_kwargs)
        assert model.spike_height_schedule[0] == 1

    def test_unet_spike_schedule_is_single_max_spike(self, base_kwargs):
        """U-Net fallback uses exactly one spike at maximum height."""
        model = MGNet(mg_net="unet", **base_kwargs)
        assert model.spike_height_schedule == [model.bottleneck_layer_idx]

    # --- Block variant ---

    def test_residual_blocks(self, base_kwargs):
        """use_residual_blocks=True constructs without error."""
        base_kwargs["use_residual_blocks"] = True
        assert isinstance(MGNet(**base_kwargs), torch.nn.Module)

    # --- Deep supervision ---

    def test_deep_supervision_disabled_sets_zero_aux_heads(self, base_kwargs):
        base_kwargs["use_deep_supervision"] = False
        model = MGNet(**base_kwargs)
        assert model.num_aux_heads == 0
        assert len(model.deep_supervision_heads) == 0

    def test_deep_supervision_uses_two_aux_heads(self, base_kwargs):
        """Deep supervision always uses 2 auxiliary heads, matching nnUNet."""
        model = MGNet(**base_kwargs)
        assert model.num_aux_heads == 2


# ---------------------------------------------------------------------------
# _make_block
# ---------------------------------------------------------------------------

class TestMakeBlock:
    """Tests for MGNet._make_block."""

    @pytest.fixture(autouse=True)
    def model(self, base_kwargs):
        self.model = MGNet(**base_kwargs)

    def test_returns_module_for_normal_channels(self):
        """Small in_channels produces a plain block (no projection)."""
        block = self.model._make_block(
            in_channels=32, out_channels=32,
            kernel_size=[3, 3, 3], stride=[1, 1, 1],
        )
        assert isinstance(block, torch.nn.Module)
        assert not isinstance(block, torch.nn.Sequential)

    def test_returns_sequential_with_projection_for_large_in_channels(self):
        """in_channels > REDUCTION_THRESHOLD prepends a 1×1 projection conv."""
        block = self.model._make_block(
            in_channels=mgnet_constants.REDUCTION_THRESHOLD + 1,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
        )
        assert isinstance(block, torch.nn.Sequential)
        projection = block[0]
        assert isinstance(projection, torch.nn.Conv3d)
        assert projection.in_channels == mgnet_constants.REDUCTION_THRESHOLD + 1
        assert projection.out_channels == mgnet_constants.REDUCTION_THRESHOLD


    def test_no_projection_at_exact_threshold(self):
        """in_channels == REDUCTION_THRESHOLD does not trigger projection."""
        block = self.model._make_block(
            in_channels=mgnet_constants.REDUCTION_THRESHOLD,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
        )
        assert not isinstance(block, torch.nn.Sequential)

    def test_projected_block_uses_constant_for_intermediate_channels(self):
        """The inner block after projection uses mgnet_constants.REDUCTION_THRESHOLD."""
        block = self.model._make_block(
            in_channels=mgnet_constants.REDUCTION_THRESHOLD + 64,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
        )
        # block[0] is the projection; block[1] is the conv block
        inner_in = block[1].conv1.conv.in_channels
        assert inner_in == mgnet_constants.REDUCTION_THRESHOLD


# ---------------------------------------------------------------------------
# _make_upsample
# ---------------------------------------------------------------------------

class TestMakeUpsample:
    """Tests for MGNet._make_upsample."""

    @pytest.fixture(autouse=True)
    def model(self, base_kwargs):
        self.model = MGNet(**base_kwargs)

    def test_returns_conv_transpose_3d(self):
        up = self.model._make_upsample(in_channels=32, scale_factor=[2, 2, 2])
        assert isinstance(up, torch.nn.ConvTranspose3d)


    def test_kernel_size_equals_stride(self):
        """kernel_size == stride ensures artifact-free upsampling."""
        scale = [2, 2, 1]
        up = self.model._make_upsample(in_channels=32, scale_factor=scale)
        assert list(up.kernel_size) == scale
        assert list(up.stride) == scale

    def test_channels_are_preserved(self):
        """Transposed conv preserves the channel count (upsamples spatially)."""
        up = self.model._make_upsample(in_channels=64, scale_factor=[2, 2, 2])
        assert up.in_channels == 64
        assert up.out_channels == 64

    def test_anisotropic_scale_factor(self):
        """Anisotropic strides are handled correctly."""
        up = self.model._make_upsample(in_channels=32, scale_factor=[2, 2, 1])
        assert list(up.stride) == [2, 2, 1]


# ---------------------------------------------------------------------------
# _init_weights
# ---------------------------------------------------------------------------

class TestInitWeights:
    """Tests for MGNet._init_weights (verified via model construction)."""

    def test_conv3d_bias_initialized_to_constant(self, base_kwargs):
        """Conv3d biases are set to INITIAL_BIAS_VALUE after construction."""
        model = MGNet(**base_kwargs)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv3d) and m.bias is not None:
                assert torch.all(m.bias == constants.INITIAL_BIAS_VALUE)

    def test_instance_norm_weight_one_bias_zero(self, base_kwargs):
        """Affine InstanceNorm3d params are initialized to weight=1, bias=0."""
        model = MGNet(**base_kwargs)
        for m in model.modules():
            if isinstance(m, torch.nn.InstanceNorm3d) and m.weight is not None:
                assert torch.all(m.weight == 1.0)
                assert torch.all(m.bias == 0.0)

    def test_conv_transpose_3d_initialized(self, base_kwargs):
        """ConvTranspose3d weights are initialized (non-zero after Kaiming)."""
        model = MGNet(**base_kwargs)
        for m in model.modules():
            if isinstance(m, torch.nn.ConvTranspose3d):
                assert not torch.all(m.weight == 0)
                break


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

class TestMGNetForward:
    """Tests for MGNet.forward covering all return paths."""

    @pytest.mark.parametrize("mg_net", ["fmgnet", "wnet", "unet"])
    def test_eval_returns_tensor(self, base_kwargs, small_input, mg_net):
        """Eval mode always returns a plain tensor regardless of topology."""
        model = MGNet(mg_net=mg_net, **base_kwargs)
        model.eval()
        with torch.no_grad():
            output = model(small_input)
        assert isinstance(output, torch.Tensor)

    @pytest.mark.parametrize("mg_net", ["fmgnet", "wnet", "unet"])
    def test_eval_output_shape(self, base_kwargs, small_input, mg_net):
        """Output spatial dims match input; channel dim equals out_channels."""
        model = MGNet(mg_net=mg_net, **base_kwargs)
        model.eval()
        with torch.no_grad():
            output = model(small_input)
        assert output.shape == (1, base_kwargs["out_channels"], 32, 32, 32)

    def test_train_with_deep_supervision_returns_dict(self, base_kwargs, small_input):
        """Training mode with deep supervision returns a dict."""
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        model.train()
        with torch.no_grad():
            output = model(small_input)
        assert isinstance(output, dict)
        assert "prediction" in output
        assert "deep_supervision" in output

    def test_train_deep_supervision_outputs_are_list(self, base_kwargs, small_input):
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        model.train()
        with torch.no_grad():
            output = model(small_input)
        assert isinstance(output["deep_supervision"], list)
        assert len(output["deep_supervision"]) > 0

    def test_train_deep_supervision_shapes_match_prediction(
        self, base_kwargs, small_input
    ):
        """All deep supervision outputs are upsampled to match the prediction."""
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        model.train()
        with torch.no_grad():
            output = model(small_input)
        for ds in output["deep_supervision"]:
            assert ds.shape == output["prediction"].shape

    def test_train_without_deep_supervision_returns_tensor(
        self, base_kwargs, small_input
    ):
        """Training mode without deep supervision returns a plain tensor."""
        base_kwargs["use_deep_supervision"] = False
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        model.train()
        with torch.no_grad():
            output = model(small_input)
        assert isinstance(output, torch.Tensor)


    def test_multi_channel_input(self, base_kwargs):
        """Model handles multi-channel inputs correctly."""
        base_kwargs["in_channels"] = 4
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 4, 32, 32, 32))
        assert output.shape[1] == base_kwargs["out_channels"]

    def test_anisotropic_spacing(self, base_kwargs):
        """Transposed conv upsampling handles anisotropic strides correctly."""
        base_kwargs["patch_size"] = [64, 64, 32]
        base_kwargs["target_spacing"] = [1.0, 1.0, 3.0]
        model = MGNet(mg_net="fmgnet", **base_kwargs)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 64, 64, 32))
        assert output.shape[0] == 1
        assert output.shape[1] == base_kwargs["out_channels"]

