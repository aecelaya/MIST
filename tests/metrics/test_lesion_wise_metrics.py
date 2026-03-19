"""Unit tests for lesion-wise metric computation."""
import numpy as np
import pytest
from unittest.mock import patch

# MIST imports.
from mist.metrics import lesion_wise_metrics
from mist.metrics.lesion_wise_metrics import (
    _consolidate_gt_lesions,
    compute_lesion_wise_metrics,
)
from scipy.ndimage import label, generate_binary_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPACING = (1.0, 1.0, 1.0)


def make_vol(*regions, shape=(20, 20, 20)):
    """Return a bool volume with True voxels in each (slice, slice, slice)."""
    vol = np.zeros(shape, dtype=bool)
    for r in regions:
        vol[r] = True
    return vol


# Two well-separated lesion positions used across many tests.
LESION_A = (slice(1, 4), slice(1, 4), slice(1, 4))   # 27 voxels, 27 mm³
LESION_B = (slice(15, 18), slice(15, 18), slice(15, 18))  # 27 voxels, 27 mm³

# Two lesions with a 1-voxel gap (for consolidation tests).
NEAR_A = (slice(1, 4), slice(1, 4), slice(1, 4))
NEAR_B = (slice(5, 8), slice(1, 4), slice(1, 4))   # gap at x=4


# ---------------------------------------------------------------------------
# _consolidate_gt_lesions
# ---------------------------------------------------------------------------

def test_consolidate_merges_nearby_lesions():
    """Lesions 1 voxel apart should merge with consolidation_iters=1."""
    gt = make_vol(NEAR_A, NEAR_B)
    struct = generate_binary_structure(3, 2)
    labeled_gt, _ = label(gt, struct)

    consolidated = _consolidate_gt_lesions(labeled_gt, struct, consolidation_iters=1)

    # Both lesion regions should share a single label.
    labels_in_a = np.unique(consolidated[NEAR_A])
    labels_in_b = np.unique(consolidated[NEAR_B])
    assert labels_in_a[labels_in_a > 0].tolist() == labels_in_b[labels_in_b > 0].tolist()


def test_consolidate_does_not_merge_distant_lesions():
    """Well-separated lesions should remain distinct with consolidation_iters=1."""
    gt = make_vol(LESION_A, LESION_B)
    struct = generate_binary_structure(3, 2)
    labeled_gt, _ = label(gt, struct)

    consolidated = _consolidate_gt_lesions(labeled_gt, struct, consolidation_iters=1)

    ids_a = set(np.unique(consolidated[LESION_A])) - {0}
    ids_b = set(np.unique(consolidated[LESION_B])) - {0}
    assert ids_a.isdisjoint(ids_b), "Distant lesions should not share a label."


def test_consolidate_preserves_background():
    """Background voxels must remain zero after consolidation."""
    gt = make_vol(NEAR_A, NEAR_B)
    struct = generate_binary_structure(3, 2)
    labeled_gt, _ = label(gt, struct)

    consolidated = _consolidate_gt_lesions(labeled_gt, struct, consolidation_iters=1)

    assert consolidated[~gt].sum() == 0


# ---------------------------------------------------------------------------
# Aggregation formula: Dice
# ---------------------------------------------------------------------------

def test_perfect_prediction_dice_is_one():
    """Exact GT/pred overlap with no FP gives lesion_wise_dice == 1.0."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A, LESION_B)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    assert result["lesion_wise_dice"] == pytest.approx(1.0)


def test_fn_reduces_dice():
    """Undetected GT lesion (FN) pulls dice toward 0."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A)  # Misses LESION_B.
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    # num_gt=2, num_fp=0 → denominator=2; scores=[1.0, 0.0]
    assert result["lesion_wise_dice"] == pytest.approx(0.5)


def test_fp_reduces_dice():
    """Spurious predicted lesion (FP) pulls dice toward 0."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_A, LESION_B)  # LESION_B is a FP.
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    # num_gt=1, num_fp=1 → denominator=2; scores=[1.0]
    assert result["lesion_wise_dice"] == pytest.approx(0.5)


def test_fn_and_fp_both_penalize_dice():
    """FN and FP penalties are additive in the denominator."""
    gt = make_vol(LESION_A, LESION_B)
    # Pred: detects LESION_A, misses LESION_B, adds FP at a third location.
    LESION_C = (slice(8, 11), slice(8, 11), slice(8, 11))
    pred = make_vol(LESION_A, LESION_C)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    # num_gt=2, num_fp=1 → denominator=3; scores=[1.0, 0.0]
    assert result["lesion_wise_dice"] == pytest.approx(1.0 / 3.0)


def test_all_gt_below_volume_threshold_no_pred_returns_empty():
    """Returns {} when all GT lesions are below the volume threshold."""
    gt = make_vol(LESION_A)
    pred = make_vol()  # Empty prediction.
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=1000.0,  # All lesions filtered.
    )
    assert result == {}


def test_fp_only_all_gt_below_threshold():
    """FP predictions with all GT below threshold give dice == 0."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_B)  # Unrelated prediction — FP.
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=1000.0, dilation_iters=1,
    )
    # num_gt_above_thresh=0, num_fp=1 → denominator=1; sum=0
    assert result["lesion_wise_dice"] == pytest.approx(0.0)


def test_volume_threshold_excludes_small_gt_lesions():
    """GT lesions below min_lesion_volume are excluded from analysis."""
    gt = make_vol(LESION_A)   # 27 mm³ — below threshold.
    pred = make_vol()          # Empty pred → no FP either.
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=50.0, dilation_iters=1,
    )
    # GT filtered, no pred → denominator=0 → {}.
    assert result == {}


# ---------------------------------------------------------------------------
# Aggregation formula: HD95
# ---------------------------------------------------------------------------

def test_perfect_prediction_haus95_is_zero():
    """Exact GT/pred overlap gives lesion_wise_haus95 == 0.0."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A, LESION_B)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["haus95"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    assert result["lesion_wise_haus95"] == pytest.approx(0.0)


def test_fn_haus95_penalized_with_diagonal():
    """Undetected GT lesion (FN) contributes image diagonal to HD95."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A)  # Misses LESION_B.
    diagonal = float(np.linalg.norm(np.array((20, 20, 20)) * np.array(SPACING)))
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["haus95"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    # num_gt=2, num_fp=0 → denominator=2; haus95=[0.0, diagonal]
    expected = (0.0 + diagonal) / 2
    assert result["lesion_wise_haus95"] == pytest.approx(expected)


def test_fp_haus95_penalized_with_diagonal():
    """Each FP predicted lesion adds image diagonal to HD95 numerator."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_A, LESION_B)  # LESION_B is a FP.
    diagonal = float(np.linalg.norm(np.array((20, 20, 20)) * np.array(SPACING)))
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["haus95"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    # num_gt=1, num_fp=1 → denominator=2; haus95=[0.0] + 1*diagonal
    expected = (0.0 + 1 * diagonal) / 2
    assert result["lesion_wise_haus95"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Aggregation formula: surface Dice
# ---------------------------------------------------------------------------

def test_perfect_prediction_surface_dice_is_one():
    """Exact GT/pred overlap gives lesion_wise_surf_dice == 1.0."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A, LESION_B)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["surface_dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    assert result["lesion_wise_surf_dice"] == pytest.approx(1.0)


def test_fn_surface_dice_penalized():
    """Undetected GT lesion (FN) contributes 0 to surface Dice numerator."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A)  # Misses LESION_B.
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["surface_dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    # num_gt=2, num_fp=0 → denominator=2; surf=[1.0, 0.0]
    assert result["lesion_wise_surf_dice"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GT consolidation effect on aggregation
# ---------------------------------------------------------------------------

def test_gt_consolidation_merges_nearby_lesions_into_one():
    """Two nearby GT lesions treated as one lesion with consolidation."""
    gt = make_vol(NEAR_A, NEAR_B)  # 1-voxel gap → merge with iters=1.
    pred = make_vol(NEAR_A, NEAR_B)
    result_no_consol = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1, gt_consolidation_iters=0,
    )
    result_consol = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1, gt_consolidation_iters=1,
    )
    # Both should give dice=1.0 with a perfect prediction, but the
    # consolidated version processes one lesion instead of two.
    assert result_no_consol["lesion_wise_dice"] == pytest.approx(1.0)
    assert result_consol["lesion_wise_dice"] == pytest.approx(1.0)


def test_gt_consolidation_changes_denominator():
    """Consolidation reduces the effective GT lesion count."""
    # Two nearby GT lesions, pred only covers first.
    gt = make_vol(NEAR_A, NEAR_B)
    pred = make_vol(NEAR_A)
    result_no_consol = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1, gt_consolidation_iters=0,
    )
    result_consol = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=0.0, dilation_iters=1, gt_consolidation_iters=1,
    )
    # Without consolidation: 2 GT lesions, 1 FN → dice=(1.0+0.0)/2=0.5.
    assert result_no_consol["lesion_wise_dice"] == pytest.approx(0.5)
    # With consolidation: 1 merged lesion partially covered → dice > 0 and != 0.5.
    assert result_consol["lesion_wise_dice"] != pytest.approx(0.5)
    assert result_consol["lesion_wise_dice"] > 0.0


# ---------------------------------------------------------------------------
# reduction="none" and invalid reduction
# ---------------------------------------------------------------------------

def test_reduction_none_returns_list_of_per_lesion_dicts():
    """reduction='none' returns one dict per GT lesion above threshold."""
    gt = make_vol(LESION_A, LESION_B)
    pred = make_vol(LESION_A, LESION_B)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice", "haus95", "surface_dice"],
        min_lesion_volume=0.0, dilation_iters=1, reduction="none",
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for lesion in result:
        assert "lesion_wise_dice" in lesion
        assert "lesion_wise_haus95" in lesion
        assert "lesion_wise_surf_dice" in lesion
        assert "gt_volume_mm3" in lesion
        assert "detected" in lesion


def test_reduction_none_all_below_threshold_returns_empty_list():
    """reduction='none' with all GT filtered returns an empty list."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_A)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING, metrics=["dice"],
        min_lesion_volume=1000.0, reduction="none",
    )
    assert result == []


def test_invalid_reduction_raises_value_error():
    """Unsupported reduction mode raises ValueError."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_A)
    with pytest.raises(ValueError, match="Unsupported reduction"):
        compute_lesion_wise_metrics(
            pred, gt, SPACING, metrics=["dice"],
            min_lesion_volume=0.0, reduction="median",
        )


def test_invalid_reduction_raises_before_loop():
    """ValueError is raised immediately, even when all GT is below threshold."""
    gt = make_vol(LESION_A)
    pred = make_vol()
    with pytest.raises(ValueError, match="Unsupported reduction"):
        compute_lesion_wise_metrics(
            pred, gt, SPACING, metrics=["dice"],
            min_lesion_volume=1000.0, reduction="median",
        )


# ---------------------------------------------------------------------------
# Multiple metrics computed together
# ---------------------------------------------------------------------------

def test_all_three_metrics_in_aggregate():
    """All three metrics are present when requested together."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_A)
    result = compute_lesion_wise_metrics(
        pred, gt, SPACING,
        metrics=["dice", "haus95", "surface_dice"],
        min_lesion_volume=0.0, dilation_iters=1,
    )
    assert "lesion_wise_dice" in result
    assert "lesion_wise_haus95" in result
    assert "lesion_wise_surf_dice" in result


def test_surface_distance_value_error_falls_back_to_worst_case():
    """If compute_surface_distances raises ValueError, surface metrics use
    the worst-case fallback (diagonal for HD95, 0.0 for surface Dice)."""
    gt = make_vol(LESION_A)
    pred = make_vol(LESION_A)
    diagonal = float(np.linalg.norm(np.array((20, 20, 20)) * np.array(SPACING)))

    with patch(
        "mist.metrics.lesion_wise_metrics.compute_surface_distances",
        side_effect=ValueError("forced error"),
    ):
        result = compute_lesion_wise_metrics(
            pred, gt, SPACING,
            metrics=["haus95", "surface_dice"],
            min_lesion_volume=0.0, dilation_iters=1,
        )

    # HD95: 1 FN-equivalent (surface_dist=None) → diagonal / 1
    assert result["lesion_wise_haus95"] == pytest.approx(diagonal)
    # Surface Dice: 0.0 fallback / 1
    assert result["lesion_wise_surf_dice"] == pytest.approx(0.0)
