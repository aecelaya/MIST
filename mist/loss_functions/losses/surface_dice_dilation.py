"""Surface Dice Dilation Losses (Volumetric & Vessel variants).

This module implements the Surface Dice Dilation Loss (SDDL) family.
These losses are designed to tolerate clinically insignificant spatial errors
by measuring overlap between *dilated* boundaries rather than exact pixels.

The module provides two variants:
1. VolumetricSDDL: For blob-like structures (tumors, organs).
   Combines Dice+CE with Surface Tolerance.
2. VesselSDDL: For tubular structures (vessels, ducts).
   Combines clDice (Topology) with Surface Tolerance.

Implementation Note:
    We use the "Composition over Inheritance" pattern for the surface logic
    to avoid Method Resolution Order (MRO) conflicts common when mixing
    complex loss hierarchies (like SegmentationLoss -> Dice -> DiceCELoss).
"""

from typing import Sequence, Tuple, Union
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.losses.cl_dice import CLDice


class SurfaceDilationLogic(nn.Module):
    """Logic module for Surface Dice Dilation.

    This module implements the "Dice of Dilated Boundaries" surrogate:
        S_hat = 2 * |D(∂P) ∩ D(∂T)| / (|D(∂P)| + |D(∂T)|)

    Where:
        - ∂P, ∂T are soft boundaries (Prediction, Target).
        - D(.) is an anisotropic dilation based on physical tolerance `tau_mm`.

    Key Responsibilities:
    1. Calculating anisotropic kernels based on voxel spacing.
    2. Handling coordinate mapping between spacing (W, H, D) and Tensors
        (H, W, D).
    3. Performing differentiable morphological operations.

    Attributes:
        boundary_ksize: The kernel size used for soft boundary extraction
            (gradient computation via erosion).
        eps: Small constant for numerical stability to avoid division by zero.
        spacing_xyz: The physical voxel spacing in millimeters, stored as
            (sx, sy, sz).
        tau_mm: The final resolved physical surface tolerance in mm. If
            initialized with "auto", this holds the calculated value.
        kxkykz: The pre-calculated kernel sizes for anisotropic dilation,
            corresponding to dimensions (x, y, z). Note that during forward
            pass, these are remapped to PyTorch's (Depth, Height, Width)
            convention.
    """

    def __init__(
        self,
        spacing_xyz: Sequence[float],
        tau_mm: Union[float, str],
        tau_safety_factor: float,
        boundary_ksize: int,
        eps: float,
    ):
        """Initialize the Surface Dilation logic.

        Args:
            spacing_xyz: Tuple of (sx, sy, sz) voxel spacing in mm.
                Must correspond to the (Width, Height, Depth) dimensions.
            tau_mm: Physical surface tolerance in mm.
                - If float: Uses that specific tolerance.
                - If "auto": Calculates tolerance as max(spacing) * safety_factor.
            tau_safety_factor: Multiplier for auto-tolerance calculation.
                Defaults to 1.25 (125% of the coarsest voxel dimension) to ensure
                the kernel never collapses to 1 pixel on coarse axes.
            boundary_ksize: Odd kernel size for soft boundary extraction (gradient).
            eps: Numerical stability constant.
        """
        super().__init__()
        self.boundary_ksize = int(boundary_ksize)
        self.eps = float(eps)

        if len(spacing_xyz) != 3:
            raise ValueError(
                "sddl_spacing_xyz must be a 3-tuple (sx, sy, sz), got "
                f"{spacing_xyz}"
            )
        self.spacing_xyz = tuple(float(s) for s in spacing_xyz)

        # --- Auto Tolerance Logic ---
        if isinstance(tau_mm, str) and tau_mm.lower() == "auto":
            # Heuristic: We cannot be more precise than our coarsest axis.
            base_tau = max(self.spacing_xyz)
            self.tau_mm = base_tau * float(tau_safety_factor)
        else:
            self.tau_mm = float(tau_mm)
            max_spacing = max(self.spacing_xyz)
            if self.tau_mm < max_spacing:
                warnings.warn(
                    f"[SDDL] tau_mm ({self.tau_mm:.2f}) < "
                    f"max spacing ({max_spacing:.2f}). Dilation may be "
                    "suboptimal (clamped to 1 voxel in coarse axis).",
                    UserWarning,
                    stacklevel=2,
                )

        # Pre-calculate kernels immediately.
        self.kxkykz = self._get_kernel_sizes(self.tau_mm, self.spacing_xyz)

    def _get_kernel_sizes(
        self, tau: float, spacing: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Calculate odd kernel sizes (kx, ky, kz) based on physical spacing.

        Calculates radius = ceil(tau / spacing) and kernel = 2 * radius + 1.
        """
        sx, sy, sz = spacing
        # Avoid division by zero
        sx, sy, sz = max(1e-12, sx), max(1e-12, sy), max(1e-12, sz)

        rx = math.ceil(tau / sx)
        ry = math.ceil(tau / sy)
        rz = math.ceil(tau / sz)

        # Kernel size is 2 * radius + 1
        return (
            2 * max(0, int(rx)) + 1,
            2 * max(0, int(ry)) + 1,
            2 * max(0, int(rz)) + 1,
        )

    def _soft_boundary(self, p: torch.Tensor) -> torch.Tensor:
        """Compute soft morphological boundary: p - Erode(p).

        Implemented as p - (-MaxPool(-p)) to remain differentiable.
        """
        if self.boundary_ksize <= 0:
            return torch.zeros_like(p)

        padding = self.boundary_ksize // 2
        # Erode(p) = -MaxPool3d(-p)
        eroded = -F.max_pool3d(
            -p, kernel_size=self.boundary_ksize, stride=1, padding=padding
        )
        return p - eroded

    def _soft_dilation_xyz(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anisotropic soft dilation using pre-calculated xyz kernels.

        CRITICAL COORDINATE MAPPING:
        - Input Tensor shape: (Batch, Class, Height, Width, Depth)
        - Input Spacing order: (sx, sy, sz) -> (Width, Height, Depth)
        - Kernel sizes (kxkykz) were calculated based on spacing (W, H, D).

        PyTorch MaxPool3d kernel/padding argument order is (Depth, Height, Width).
        Therefore, we must map our calculated kernels (kx, ky, kz) as follows:
            - Dim 2 (Height) <- ky
            - Dim 3 (Width)  <- kx
            - Dim 4 (Depth)  <- kz
        """
        kx, ky, kz = self.kxkykz

        return F.max_pool3d(
            x,
            kernel_size=(ky, kx, kz),
            stride=1,
            padding=(ky // 2, kx // 2, kz // 2),
        )

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        exclude_background: bool,
    ) -> torch.Tensor:
        """Compute the Surface Dice Dilation Loss component.

        Args:
            y_true: Ground Truth One-Hot tensor (B, C, H, W, D).
            y_pred: Prediction Softmax Probabilities (B, C, H, W, D).
            exclude_background: Whether to drop channel 0 before calculation.

        Returns:
            Scalar loss: 1.0 - mean(SurfaceDice).
        """
        # Strip background if preprocess() hasn't already removed it.
        # When exclude_background=False, preprocess() kept all channels, so
        # we remove channel 0 here. When exclude_background=True, preprocess()
        # already stripped it, so no further slicing is needed.
        if not exclude_background:
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        # 1. Soft boundaries.
        partial_p = self._soft_boundary(y_pred)
        partial_t = self._soft_boundary(y_true)

        # 2. Anisotropic dilation.
        dilated_p = self._soft_dilation_xyz(partial_p)
        dilated_t = self._soft_dilation_xyz(partial_t)

        # 3. Dice score (sum over spatial dims D, H, W).
        spatial_dims = (2, 3, 4)
        overlap = (dilated_p * dilated_t).sum(dim=spatial_dims)
        union = dilated_p.sum(dim=spatial_dims) + \
            dilated_t.sum(dim=spatial_dims)

        s_hat = (2.0 * overlap) / (union + self.eps)
        return 1.0 - torch.mean(s_hat)


@register_loss(name="volumetric_sddl")
class VolumetricSDDL(DiceCELoss):
    """Combined Volumetric (Dice+CE) + Surface Dice Dilation Loss.

    Designed for solid organs and tumors (Liver, Kidney, HCC).

    Formula:
        Loss = alpha * (Dice + CE) + (1 - alpha) * SurfaceDice

    Attributes:
        surface_logic (SurfaceDilationLogic): The helper module that handles
            the anisotropic dilation and surface overlap calculation.
    """

    def __init__(
        self,
        sddl_spacing_xyz: Sequence[float],
        tau_mm: Union[float, str] = "auto",
        tau_safety_factor: float = 1.25,
        boundary_ksize: int = 3,
        eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize Volumetric SDDL.

        Args:
            sddl_spacing_xyz: (sx, sy, sz) voxel spacing in mm.
                Named specifically to catch injection from the Trainer.
            tau_mm: Tolerance in mm, or "auto" (default).
            tau_safety_factor: Multiplier if tau_mm="auto".
            boundary_ksize: Kernel size for boundary extraction.
            eps: Numerical stability constant.
            **kwargs: Arguments passed to DiceCELoss (e.g., exclude_background,
                lambda_ce, lambda_dice).
        """
        # 1. Initialize Parent (DiceCELoss) normally
        super().__init__(**kwargs)

        # 2. Instantiate Surface Logic via Composition (Submodule)
        self.surface_logic = SurfaceDilationLogic(
            spacing_xyz=sddl_spacing_xyz,
            tau_mm=tau_mm,
            tau_safety_factor=tau_safety_factor,
            boundary_ksize=boundary_ksize,
            eps=eps,
        )

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            y_true: Ground Truth (B, 1, H, W, D).
            y_pred: Logits (B, C, H, W, D).
            **kwargs: Must contain 'alpha' (float) to control weighting.
                      Defaults to 0.5 if not present.

        Returns:
            Weighted loss scalar.
        """
        alpha = kwargs.get("alpha", 0.5)

        # Volumetric Loss (Dice + CE)
        loss_vol = super().forward(y_true, y_pred, **kwargs)

        # Optimization: Skip surface calculation if alpha is 1.0
        if alpha >= 1.0:
            return loss_vol

        # Surface Loss
        # Preprocess: y_true -> OneHot, y_pred -> Softmax
        y_true_oh, y_pred_prob = self.preprocess(y_true, y_pred)

        # Call the submodule
        loss_surf = self.surface_logic(
            y_true_oh, y_pred_prob, self.exclude_background
        )

        return alpha * loss_vol + (1.0 - alpha) * loss_surf


@register_loss(name="vessel_sddl")
class VesselSDDL(CLDice):
    """Combined Vessel Topology (CLDice) + Surface Dice Dilation Loss.

    Designed for thin, tubular structures (Vessels, Ducts, Nerves).

    Formula:
        Loss = alpha * (CLDice_Composite) + (1 - alpha) * SurfaceDice

    Where CLDice_Composite is usually:
        0.5 * (Dice+CE) + 0.5 * clDice (Topology)

    Attributes:
        surface_logic (SurfaceDilationLogic): The helper module that handles
            the anisotropic dilation and surface overlap calculation.
    """

    def __init__(
        self,
        sddl_spacing_xyz: Sequence[float],
        tau_mm: Union[float, str] = "auto",
        tau_safety_factor: float = 1.25,
        boundary_ksize: int = 3,
        eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize Vessel SDDL.

        Args:
            sddl_spacing_xyz: (sx, sy, sz) voxel spacing in mm.
            tau_mm: Tolerance in mm, or "auto".
            tau_safety_factor: Multiplier if tau_mm="auto".
            boundary_ksize: Kernel size for boundary extraction.
            eps: Numerical stability constant.
            **kwargs: Arguments passed to CLDice (e.g., iterations, smooth).
        """
        # 1. Initialize Parent (CLDice) normally
        super().__init__(**kwargs)

        # 2. Instantiate Surface Logic via Composition
        self.surface_logic = SurfaceDilationLogic(
            spacing_xyz=sddl_spacing_xyz,
            tau_mm=tau_mm,
            tau_safety_factor=tau_safety_factor,
            boundary_ksize=boundary_ksize,
            eps=eps,
        )

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            y_true: Ground Truth (B, 1, H, W, D).
            y_pred: Logits (B, C, H, W, D).
            **kwargs: Must contain 'alpha' (float) to control weighting.
                      Defaults to 0.5 if not present.

        Returns:
            Weighted loss scalar.
        """
        alpha = kwargs.get("alpha", 0.5)

        # Vessel Loss (CLDice + Dice + CE)
        loss_cl = super().forward(y_true, y_pred, **kwargs)

        if alpha >= 1.0:
            return loss_cl

        # Surface Loss
        y_true_oh, y_pred_prob = self.preprocess(y_true, y_pred)

        # Call the submodule
        loss_surf = self.surface_logic(
            y_true_oh, y_pred_prob, self.exclude_background
        )

        return alpha * loss_cl + (1.0 - alpha) * loss_surf
