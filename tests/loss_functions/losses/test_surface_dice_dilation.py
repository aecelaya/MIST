"""Tests for Surface Dice Dilation losses: SurfaceDilationLogic, VolumetricSDDL, VesselSDDL."""

import math
import warnings
from typing import Tuple
from unittest import mock

import torch
import torch.nn.functional as F
import pytest

from mist.loss_functions.losses.surface_dice_dilation import (
    SurfaceDilationLogic,
    VolumetricSDDL,
    VesselSDDL,
)
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.losses.cl_dice import CLDice
from mist.loss_functions.loss_registry import get_loss


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ISOTROPIC = (1.0, 1.0, 1.0)
ANISOTROPIC = (1.0, 1.0, 3.0)  # coarser z-axis


def _make_mock_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random raw inputs: label ints (B, 1, H, W, D) and logits (B, C, H, W, D)."""
    y_true = torch.randint(0, n_classes, size=(batch_size, 1, size, size, size))
    y_pred = torch.randn((batch_size, n_classes, size, size, size))
    return y_true, y_pred


def _make_preprocessed_data(
    n_classes: int = 3,
    batch_size: int = 2,
    size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocessed inputs: one-hot (B, C, H, W, D) and softmax (B, C, H, W, D).

    SurfaceDilationLogic.forward receives already-preprocessed tensors from
    the parent loss, so tests that call it directly need this form.
    """
    y_true, y_pred = _make_mock_data(n_classes=n_classes, batch_size=batch_size, size=size)
    y_true_oh = F.one_hot(
        y_true.squeeze(1).long(), num_classes=n_classes
    ).permute(0, 4, 1, 2, 3).float()
    y_pred_prob = F.softmax(y_pred, dim=1)
    return y_true_oh, y_pred_prob


def _make_logic(spacing=ISOTROPIC, tau_mm="auto") -> SurfaceDilationLogic:
    return SurfaceDilationLogic(
        spacing_xyz=spacing,
        tau_mm=tau_mm,
        tau_safety_factor=1.25,
        boundary_ksize=3,
        eps=1e-6,
    )


# ---------------------------------------------------------------------------
# SurfaceDilationLogic — init / auto-tau / kernel sizes
# ---------------------------------------------------------------------------

class TestSurfaceDilationLogicInit:
    """Tests for SurfaceDilationLogic.__init__."""

    def test_auto_tau_equals_max_spacing_times_safety_factor(self):
        logic = _make_logic(spacing=ANISOTROPIC, tau_mm="auto")
        assert logic.tau_mm == pytest.approx(3.0 * 1.25)

    def test_auto_tau_case_insensitive(self):
        logic = SurfaceDilationLogic(
            spacing_xyz=ISOTROPIC, tau_mm="AUTO",
            tau_safety_factor=1.0, boundary_ksize=3, eps=1e-6,
        )
        assert logic.tau_mm == pytest.approx(1.0 * 1.0)

    def test_explicit_tau_stored_correctly(self):
        logic = _make_logic(spacing=ISOTROPIC, tau_mm=2.0)
        assert logic.tau_mm == 2.0

    def test_explicit_tau_below_max_spacing_warns(self):
        """tau_mm < max(spacing) should emit a UserWarning."""
        with pytest.warns(UserWarning, match="tau_mm"):
            _make_logic(spacing=ANISOTROPIC, tau_mm=0.5)  # max spacing = 3.0

    def test_explicit_tau_at_or_above_max_spacing_no_warning(self):
        """tau_mm >= max(spacing) must not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _make_logic(spacing=ANISOTROPIC, tau_mm=3.0)  # exactly at max, no warn

    def test_invalid_spacing_length_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            SurfaceDilationLogic(
                spacing_xyz=(1.0, 1.0),  # 2D, not 3D
                tau_mm="auto", tau_safety_factor=1.25, boundary_ksize=3, eps=1e-6,
            )


class TestSurfaceDilationLogicKernelSizes:
    """Tests for _get_kernel_sizes."""

    def test_isotropic_spacing_produces_equal_kernels(self):
        logic = _make_logic(spacing=ISOTROPIC, tau_mm=2.0)
        kx, ky, kz = logic.kxkykz
        assert kx == ky == kz

    def test_all_kernels_are_odd(self):
        """Kernels must always be odd (2*radius + 1)."""
        logic = _make_logic(spacing=(1.0, 2.0, 3.0), tau_mm=3.0)
        for k in logic.kxkykz:
            assert k % 2 == 1

    def test_coarser_axis_gets_smaller_kernel(self):
        """Larger voxel spacing → fewer voxels per mm → smaller kernel."""
        with pytest.warns(UserWarning, match="tau_mm"):
            logic = _make_logic(spacing=(1.0, 1.0, 4.0), tau_mm=2.0)
        kx, ky, kz = logic.kxkykz
        assert kz < kx  # z-axis is coarser (4 mm/vox vs 1 mm/vox)

    def test_kernel_value_matches_formula(self):
        """kx = 2 * ceil(tau / sx) + 1."""
        tau, sx = 2.0, 1.0
        logic = _make_logic(spacing=(sx, sx, sx), tau_mm=tau)
        expected = 2 * math.ceil(tau / sx) + 1
        assert logic.kxkykz[0] == expected

    def test_minimum_kernel_size_is_one(self):
        """Very small tau relative to spacing clamps radius to 0 → kernel = 1."""
        with pytest.warns(UserWarning, match="tau_mm"):
            logic = _make_logic(spacing=(100.0, 100.0, 100.0), tau_mm=1.0)
        for k in logic.kxkykz:
            assert k >= 1


# ---------------------------------------------------------------------------
# SurfaceDilationLogic — forward
# ---------------------------------------------------------------------------

class TestSurfaceDilationLogicForward:
    """Tests for SurfaceDilationLogic.forward."""

    @pytest.fixture
    def logic(self):
        return _make_logic(spacing=ISOTROPIC)

    def test_returns_scalar(self, logic):
        y_true_oh, y_pred_prob = _make_preprocessed_data()
        out = logic(y_true_oh, y_pred_prob, exclude_background=False)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_loss_in_valid_range(self, logic):
        """Surface Dice loss = 1 - Dice, so it should be in [0, 1]."""
        y_true_oh, y_pred_prob = _make_preprocessed_data()
        out = logic(y_true_oh, y_pred_prob, exclude_background=False)
        assert 0.0 <= out.item() <= 1.0 + 1e-5

    def test_no_nan(self, logic):
        y_true_oh, y_pred_prob = _make_preprocessed_data()
        out = logic(y_true_oh, y_pred_prob, exclude_background=False)
        assert torch.isfinite(out)

    def test_exclude_background_false_and_true_are_equivalent(self, logic):
        """Both code paths (strip inside vs. strip before) must agree."""
        y_true_oh, y_pred_prob = _make_preprocessed_data(n_classes=3)

        # Path A: pass all channels, let the logic strip channel 0.
        out_a = logic(y_true_oh, y_pred_prob, exclude_background=False)

        # Path B: pre-strip channel 0 and tell logic it was already done.
        out_b = logic(y_true_oh[:, 1:], y_pred_prob[:, 1:], exclude_background=True)

        assert torch.isclose(out_a, out_b, atol=1e-5)

    def test_perfect_prediction_gives_zero_loss(self, logic):
        """Identical prediction and ground truth → surface Dice = 1 → loss = 0."""
        # Build a foreground blob: 8×8×8 cube of 1s inside a 16³ volume.
        b, s = 1, 16
        y_true_oh = torch.zeros(b, 2, s, s, s)
        y_true_oh[:, 0] = 1.0            # background everywhere
        y_true_oh[:, 1, 4:12, 4:12, 4:12] = 1.0  # foreground blob
        y_true_oh[:, 0, 4:12, 4:12, 4:12] = 0.0  # clear bg inside blob

        # Perfect prediction: probabilities match ground truth exactly.
        y_pred_prob = y_true_oh.clone()

        # Pass only the foreground channel (exclude_background=True path).
        loss = logic(
            y_true_oh[:, 1:], y_pred_prob[:, 1:], exclude_background=True
        )
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# VolumetricSDDL
# ---------------------------------------------------------------------------

class TestVolumetricSDDL:
    """Tests for VolumetricSDDL (Dice+CE + Surface Dice Dilation)."""

    def test_forward_returns_scalar(self):
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        y_true, y_pred = _make_mock_data()
        loss = loss_fn(y_true, y_pred)
        assert loss.ndim == 0

    def test_loss_is_finite(self):
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_alpha_stability(self, alpha):
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred, alpha=alpha))

    def test_alpha_one_returns_dice_ce(self):
        """alpha=1.0 must return pure DiceCE without calling surface logic."""
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        y_true, y_pred = _make_mock_data()

        with mock.patch.object(
            loss_fn.surface_logic, "forward", wraps=loss_fn.surface_logic.forward
        ) as mock_surf:
            loss = loss_fn(y_true, y_pred, alpha=1.0)
            mock_surf.assert_not_called()

        # Value must match DiceCELoss on the same inputs.
        expected = DiceCELoss()(y_true, y_pred)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_alpha_zero_uses_only_surface_term(self):
        """alpha=0.0 must call the surface logic and return only that term."""
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        y_true, y_pred = _make_mock_data()

        with mock.patch.object(
            loss_fn.surface_logic, "forward", wraps=loss_fn.surface_logic.forward
        ) as mock_surf:
            loss_fn(y_true, y_pred, alpha=0.0)
            mock_surf.assert_called_once()

    def test_changing_alpha_changes_loss(self):
        """Different alpha values must produce different loss values."""
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        y_true, y_pred = _make_mock_data()
        loss_a = loss_fn(y_true, y_pred, alpha=0.0)
        loss_b = loss_fn(y_true, y_pred, alpha=1.0)
        assert not torch.isclose(loss_a, loss_b)

    def test_exclude_background(self):
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC, exclude_background=True)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    def test_anisotropic_spacing(self):
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ANISOTROPIC)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    def test_explicit_tau_mm(self):
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC, tau_mm=2.0)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    def test_invalid_spacing_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            VolumetricSDDL(sddl_spacing_xyz=(1.0, 1.0))  # 2-tuple, not 3

    def test_tau_below_max_spacing_warns(self):
        with pytest.warns(UserWarning, match="tau_mm"):
            VolumetricSDDL(sddl_spacing_xyz=ANISOTROPIC, tau_mm=0.5)

    def test_registered_in_loss_registry(self):
        assert get_loss("volumetric_sddl") is VolumetricSDDL

    def test_surface_logic_is_submodule(self):
        """surface_logic must be an nn.Module so its parameters are tracked."""
        loss_fn = VolumetricSDDL(sddl_spacing_xyz=ISOTROPIC)
        assert isinstance(loss_fn.surface_logic, torch.nn.Module)


# ---------------------------------------------------------------------------
# VesselSDDL
# ---------------------------------------------------------------------------

class TestVesselSDDL:
    """Tests for VesselSDDL (CLDice + Surface Dice Dilation)."""

    def test_forward_returns_scalar(self):
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()
        loss = loss_fn(y_true, y_pred)
        assert loss.ndim == 0

    def test_loss_is_finite(self):
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_alpha_stability(self, alpha):
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred, alpha=alpha))

    def test_alpha_one_returns_cldice(self):
        """alpha=1.0 must return pure CLDice without calling surface logic."""
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()

        with mock.patch.object(
            loss_fn.surface_logic, "forward", wraps=loss_fn.surface_logic.forward
        ) as mock_surf:
            loss = loss_fn(y_true, y_pred, alpha=1.0)
            mock_surf.assert_not_called()

        # Value must match CLDice on the same inputs.
        reference = CLDice(iterations=2)(y_true, y_pred, alpha=1.0)
        assert torch.isclose(loss, reference, atol=1e-5)

    def test_alpha_zero_uses_only_surface_term(self):
        """alpha=0.0 must call the surface logic and return only that term."""
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()

        with mock.patch.object(
            loss_fn.surface_logic, "forward", wraps=loss_fn.surface_logic.forward
        ) as mock_surf:
            loss_fn(y_true, y_pred, alpha=0.0)
            mock_surf.assert_called_once()

    def test_changing_alpha_changes_loss(self):
        """Different alpha values must produce different loss values."""
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()
        loss_a = loss_fn(y_true, y_pred, alpha=0.0)
        loss_b = loss_fn(y_true, y_pred, alpha=1.0)
        assert not torch.isclose(loss_a, loss_b)

    def test_exclude_background(self):
        loss_fn = VesselSDDL(
            sddl_spacing_xyz=ISOTROPIC, exclude_background=True, iterations=2
        )
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    def test_anisotropic_spacing(self):
        loss_fn = VesselSDDL(sddl_spacing_xyz=ANISOTROPIC, iterations=2)
        y_true, y_pred = _make_mock_data()
        assert torch.isfinite(loss_fn(y_true, y_pred))

    def test_invalid_spacing_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            VesselSDDL(sddl_spacing_xyz=(1.0, 1.0))

    def test_tau_below_max_spacing_warns(self):
        with pytest.warns(UserWarning, match="tau_mm"):
            VesselSDDL(sddl_spacing_xyz=ANISOTROPIC, tau_mm=0.5)

    def test_registered_in_loss_registry(self):
        assert get_loss("vessel_sddl") is VesselSDDL

    def test_surface_logic_is_submodule(self):
        """surface_logic must be an nn.Module so its parameters are tracked."""
        loss_fn = VesselSDDL(sddl_spacing_xyz=ISOTROPIC)
        assert isinstance(loss_fn.surface_logic, torch.nn.Module)
