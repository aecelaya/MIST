"""Unit tests for metrics_registry module in MIST."""
import numpy as np
import pytest

# MIST imports.
from mist.metrics.metrics_registry import (
    get_metric,
    list_registered_metrics,
    Metric,
    METRIC_REGISTRY,
    DiceCoefficient,
)

# pylint: disable=redefined-outer-name
@pytest.fixture
def synthetic_masks():
    """Fixture returning two 3D masks and spacing."""
    mask_gt = np.zeros((10, 10, 10), dtype=bool)
    mask_pred = np.zeros((10, 10, 10), dtype=bool)
    mask_gt[2:5, 2:5, 2:5] = True
    mask_pred[3:6, 3:6, 3:6] = True
    spacing = (1.0, 1.0, 1.0)
    return mask_gt, mask_pred, spacing


def test_metric_subclass_missing_required_attr_raises():
    """Subclass missing name, best, or worst raises TypeError at definition."""
    with pytest.raises(TypeError, match="must define class attribute 'name'"):
        class BadMetric(Metric):  # pylint: disable=unused-variable
            best = 1.0
            worst = 0.0
            def __call__(self, truth, pred, spacing, **kwargs):
                pass  # pragma: no cover


def test_registry_contains_all_metrics():
    """Should register all metric classes by name."""
    expected_names = {
        "dice", "haus95", "surf_dice", "avg_surf",
        "lesion_wise_dice", "lesion_wise_haus95", "lesion_wise_surf_dice",
    }
    assert expected_names.issubset(set(METRIC_REGISTRY.keys()))


def test_get_metric_valid():
    """Should retrieve a metric instance by name."""
    dice = get_metric("dice")
    assert isinstance(dice, DiceCoefficient)


def test_get_metric_invalid():
    """Should raise ValueError if metric is not registered."""
    with pytest.raises(ValueError, match="not registered"):
        get_metric("non_existent_metric")


def test_list_registered_metrics():
    """Should return a sorted list of metric names."""
    registered = list_registered_metrics()
    assert isinstance(registered, list)
    assert "dice" in registered
    assert sorted(registered) == registered


def test_dice_coefficient_metric(synthetic_masks):
    """Should compute valid dice value between 0 and 1."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("dice")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_hausdorff95_metric(synthetic_masks):
    """Should compute a positive Hausdorff distance."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("haus95")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert result > 0.0


def test_surface_dice_metric(synthetic_masks):
    """Should compute surface dice in [0, 1] with default tolerance."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("surf_dice")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_average_surface_distance_metric(synthetic_masks):
    """Should compute a positive average surface distance."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("avg_surf")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert result > 0.0


# ---------------------------------------------------------------------------
# Lesion-wise metrics: kwargs forwarding
# ---------------------------------------------------------------------------

@pytest.fixture
def lesion_masks():
    """Two well-separated 3D lesions with perfect overlap."""
    gt = np.zeros((20, 20, 20), dtype=bool)
    pred = np.zeros((20, 20, 20), dtype=bool)
    gt[1:4, 1:4, 1:4] = True
    gt[15:18, 15:18, 15:18] = True
    pred[1:4, 1:4, 1:4] = True
    pred[15:18, 15:18, 15:18] = True
    spacing = (1.0, 1.0, 1.0)
    return gt, pred, spacing


def test_lesion_wise_dice_registered():
    """lesion_wise_dice should be in the metric registry."""
    assert "lesion_wise_dice" in METRIC_REGISTRY


def test_lesion_wise_haus95_registered():
    """lesion_wise_haus95 should be in the metric registry."""
    assert "lesion_wise_haus95" in METRIC_REGISTRY


def test_lesion_wise_surf_dice_registered():
    """lesion_wise_surf_dice should be in the metric registry."""
    assert "lesion_wise_surf_dice" in METRIC_REGISTRY


def test_lesion_wise_dice_perfect_overlap(lesion_masks):
    """lesion_wise_dice returns 1.0 for exact GT/pred overlap."""
    gt, pred, spacing = lesion_masks
    metric = get_metric("lesion_wise_dice")
    result = metric(gt, pred, spacing, min_lesion_volume=0.0, dilation_iters=1)
    assert result == pytest.approx(1.0)


def test_lesion_wise_haus95_perfect_overlap(lesion_masks):
    """lesion_wise_haus95 returns 0.0 for exact GT/pred overlap."""
    gt, pred, spacing = lesion_masks
    metric = get_metric("lesion_wise_haus95")
    result = metric(gt, pred, spacing, min_lesion_volume=0.0, dilation_iters=1)
    assert result == pytest.approx(0.0)


def test_lesion_wise_surf_dice_perfect_overlap(lesion_masks):
    """lesion_wise_surf_dice returns 1.0 for exact GT/pred overlap."""
    gt, pred, spacing = lesion_masks
    metric = get_metric("lesion_wise_surf_dice")
    result = metric(gt, pred, spacing, min_lesion_volume=0.0, dilation_iters=1)
    assert result == pytest.approx(1.0)


def test_lesion_wise_dice_min_volume_kwarg_forwarded(lesion_masks):
    """min_lesion_volume kwarg is forwarded: high threshold with empty pred → best case."""
    gt, pred, spacing = lesion_masks
    metric = get_metric("lesion_wise_dice")
    # All GT lesions filtered AND prediction is empty → denominator=0 → best case.
    empty_pred = np.zeros_like(pred)
    result = metric(gt, empty_pred, spacing, min_lesion_volume=10000.0, dilation_iters=1)
    assert result == metric.best


def test_lesion_wise_dice_all_gt_filtered_with_fp_returns_zero(lesion_masks):
    """All GT filtered but pred non-empty → FP penalty → dice=0.0 (not best)."""
    gt, pred, spacing = lesion_masks
    metric = get_metric("lesion_wise_dice")
    # GT filtered, pred non-empty → num_fp>0, denominator>0 → dice=0.
    result = metric(gt, pred, spacing, min_lesion_volume=10000.0, dilation_iters=1)
    assert result == pytest.approx(0.0)


def test_lesion_wise_haus95_fp_penalized(lesion_masks):
    """FP prediction increases lesion_wise_haus95 above 0."""
    gt, pred, spacing = lesion_masks
    # Add an extra FP lesion to pred.
    pred_with_fp = pred.copy()
    pred_with_fp[8:11, 8:11, 8:11] = True
    metric = get_metric("lesion_wise_haus95")
    result = metric(
        gt, pred_with_fp, spacing,
        min_lesion_volume=0.0, dilation_iters=1,
    )
    assert result > 0.0


def test_lesion_wise_surf_dice_tolerance_kwarg_forwarded(lesion_masks):
    """tolerance kwarg is forwarded as surface_dice_tolerance_mm."""
    gt, pred, spacing = lesion_masks
    metric = get_metric("lesion_wise_surf_dice")
    result_tight = metric(
        gt, pred, spacing, min_lesion_volume=0.0, dilation_iters=1, tolerance=0.1
    )
    result_loose = metric(
        gt, pred, spacing, min_lesion_volume=0.0, dilation_iters=1, tolerance=10.0
    )
    # Both should be 1.0 for perfect overlap; this at least verifies no crash.
    assert isinstance(result_tight, float)
    assert isinstance(result_loose, float)
