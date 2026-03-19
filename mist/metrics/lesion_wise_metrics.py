"""Lesion-wise metrics for segmentation evaluation."""
from typing import Any, Dict, List, Set, Tuple, Union
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure

# MIST imports.
from mist.metrics.metrics_constants import LesionWiseMetricsConstants
from mist.metrics.segmentation_metrics import (
    compute_dice_coefficient,
    compute_surface_distances,
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
)


def _consolidate_gt_lesions(
    labeled_gt: np.ndarray,
    struct: np.ndarray,
    consolidation_iters: int,
) -> np.ndarray:
    """Merge GT lesions whose dilated footprints overlap into single lesions.

    Dilates the binary GT mask, re-labels the dilated result, then projects
    the new labels back onto the original GT voxels. Any two lesions whose
    dilated footprints connect share a label and are treated as a single
    lesion in subsequent analysis.

    Args:
        labeled_gt: Integer-labeled GT mask (from scipy.ndimage.label).
        struct: Binary structure element for dilation.
        consolidation_iters: Number of dilation iterations.

    Returns:
        Relabeled GT mask where nearby lesions share a label. Background
        voxels remain zero.
    """
    gt_dilated = binary_dilation(
        labeled_gt > 0, structure=struct, iterations=consolidation_iters
    )
    labeled_dilated, _ = label(gt_dilated, struct)
    # Project original GT voxels onto the dilated component labels.
    # Lesions whose dilated footprints merge receive the same new label and
    # are treated as one lesion going forward.
    return labeled_dilated * (labeled_gt > 0)


def compute_lesion_wise_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float],
    metrics: List[str],
    min_lesion_volume: float = LesionWiseMetricsConstants.MIN_LESION_VOLUME,
    surface_dice_tolerance_mm: float = LesionWiseMetricsConstants.SURFACE_DICE_TOLERANCE_MM,
    dilation_iters: int = LesionWiseMetricsConstants.DILATION_ITERS,
    gt_consolidation_iters: int = LesionWiseMetricsConstants.GT_CONSOLIDATION_ITERS,
    reduction: str = "mean",
) -> Union[List[Dict[str, Any]], Dict[str, float]]:
    """Compute lesion-wise metrics following the BraTS evaluation protocol.

    Each GT lesion above the volume threshold is matched to overlapping
    prediction components via dilation. Unmatched predicted components are
    counted as false positives (FP). The aggregate score penalizes both
    undetected GT lesions (FN) and FPs:

        score = sum(per_lesion_scores) / (num_gt_above_thresh + num_fp)

    FN lesions contribute 0 (Dice / surface Dice) or the image diagonal
    (HD95) to the numerator. Each FP contributes 0 (Dice / surface Dice)
    or the image diagonal (HD95) to the numerator, and 1 to the denominator.

    Args:
        pred: Binary prediction mask.
        gt: Binary ground truth mask.
        spacing: Voxel spacing in mm (dx, dy, dz).
        metrics: Metrics to compute. Supported values: 'dice', 'haus95',
            'surface_dice'.
        min_lesion_volume: Minimum GT lesion volume in mm³ to include.
            Lesions below this threshold are excluded from analysis.
        surface_dice_tolerance_mm: Tolerance in mm for surface Dice.
        dilation_iters: Dilation iterations used when searching for
            predicted components that overlap a GT lesion.
        gt_consolidation_iters: Dilation iterations for GT lesion
            consolidation before analysis. GT lesions whose dilated
            footprints connect are merged into a single lesion. Set to 0
            (default) to skip consolidation. Set equal to dilation_iters
            to replicate BraTS-style consolidation.
        reduction: Aggregation mode.
            - 'mean': BraTS-style sum / (num_gt_above_thresh + num_fp).
            - 'none': Return raw per-lesion result dicts (for debugging).

    Returns:
        If reduction == 'none': List of per-lesion result dicts, one per
            GT lesion above the volume threshold.
        If reduction == 'mean': Dict mapping metric name to aggregated
            score. Returns an empty dict when the denominator is zero (no
            GT lesions above threshold and no predicted lesions).

    Raises:
        ValueError: If an unsupported reduction mode is specified.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(
            f"Unsupported reduction: '{reduction}'. Use 'mean' or 'none'."
        )

    struct = generate_binary_structure(3, 2)

    # Label GT connected components, then optionally consolidate nearby ones.
    labeled_gt, _ = label(gt, struct)
    if gt_consolidation_iters > 0:
        labeled_gt = _consolidate_gt_lesions(
            labeled_gt, struct, gt_consolidation_iters
        )

    # Label prediction connected components.
    labeled_pred, num_pred = label(pred, struct)

    # Worst-case distance for HD95 FP / FN penalty: image diagonal in mm.
    diagonal_mm = float(
        np.linalg.norm(np.array(gt.shape) * np.array(spacing))
    )

    # Unique non-zero GT lesion labels (after optional consolidation some
    # original labels may have merged, so use np.unique rather than a range).
    gt_lesion_ids = np.unique(labeled_gt)
    gt_lesion_ids = gt_lesion_ids[gt_lesion_ids > 0]

    matched_pred_labels: Set[int] = set()
    num_gt_above_thresh = 0
    results: List[Dict[str, Any]] = []
    needs_surface = bool({"haus95", "surface_dice"}.intersection(metrics))

    for lesion_id in gt_lesion_ids:
        gt_lesion = labeled_gt == lesion_id
        lesion_vol = float(np.sum(gt_lesion) * np.prod(spacing))
        if lesion_vol < min_lesion_volume:
            continue

        num_gt_above_thresh += 1

        # Find prediction components that overlap the dilated GT lesion.
        dilated = binary_dilation(
            gt_lesion, structure=struct, iterations=dilation_iters
        )
        pred_overlap_labels = np.unique(labeled_pred[dilated])
        pred_overlap_labels = pred_overlap_labels[pred_overlap_labels > 0]
        matched_pred_labels.update(pred_overlap_labels.tolist())

        detected = len(pred_overlap_labels) > 0
        pred_lesion = np.isin(labeled_pred, pred_overlap_labels)

        lesion_result: Dict[str, Any] = {
            "gt_lesion_id": int(lesion_id),
            "gt_volume_mm3": lesion_vol,
            "num_matched_pred_labels": len(pred_overlap_labels),
            "detected": detected,
        }

        if "dice" in metrics:
            lesion_result["lesion_wise_dice"] = (
                float(compute_dice_coefficient(gt_lesion, pred_lesion))
                if detected else 0.0
            )

        # Compute surface distances once, shared by HD95 and surface Dice.
        surface_dist = None
        if detected and needs_surface:
            try:
                surface_dist = compute_surface_distances(
                    gt_lesion, pred_lesion, spacing
                )
            except ValueError:
                pass  # Treat as undetected for surface-based metrics.

        if "haus95" in metrics:
            lesion_result["lesion_wise_haus95"] = (
                float(compute_robust_hausdorff(surface_dist, percent=95.0))
                if surface_dist is not None else diagonal_mm
            )

        if "surface_dice" in metrics:
            lesion_result["lesion_wise_surf_dice"] = (
                float(compute_surface_dice_at_tolerance(
                    surface_dist, surface_dice_tolerance_mm
                ))
                if surface_dist is not None else 0.0
            )

        results.append(lesion_result)

    # False positives: predicted components not matched to any GT lesion.
    num_fp = len(set(range(1, num_pred + 1)) - matched_pred_labels)

    # Return raw per-lesion list before any aggregation.
    if reduction == "none":
        return results

    denominator = num_gt_above_thresh + num_fp
    if denominator == 0:
        return {}

    aggregate: Dict[str, float] = {}

    if "dice" in metrics:
        aggregate["lesion_wise_dice"] = (
            sum(r["lesion_wise_dice"] for r in results) / denominator
        )

    if "haus95" in metrics:
        haus_sum = sum(r["lesion_wise_haus95"] for r in results)
        # Each FP is penalized with the image diagonal distance.
        aggregate["lesion_wise_haus95"] = (
            (haus_sum + num_fp * diagonal_mm) / denominator
        )

    if "surface_dice" in metrics:
        aggregate["lesion_wise_surf_dice"] = (
            sum(r["lesion_wise_surf_dice"] for r in results) / denominator
        )

    return aggregate
