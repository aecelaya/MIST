"""Unit tests for the _welford_merge and _percentile_from_histogram helpers."""
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pytest

from mist.analyze_data.analyzer import _welford_merge, _percentile_from_histogram
from mist.analyze_data.analyzer_constants import AnalyzeConstants as constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(arr: ArrayLike) -> tuple[int, float, float]:
    """Return (n, mean, M2) for a numpy array."""
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    mean = float(np.mean(a))
    M2 = float(np.sum((a - mean) ** 2))
    return n, mean, M2


def _make_hist(
    data: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Build a CT-range histogram for *data* using the standard bin edges."""
    bin_edges = np.linspace(
        constants.CT_HU_HIST_MIN,
        constants.CT_HU_HIST_MAX,
        constants.CT_HU_HIST_BINS + 1,
    )
    hist, _ = np.histogram(data, bins=bin_edges)
    return hist.astype(np.int64), bin_edges


# ---------------------------------------------------------------------------
# _welford_merge
# ---------------------------------------------------------------------------

class TestWelfordMerge:
    """Tests for _welford_merge."""

    def test_single_group_matches_numpy(self):
        """A single group reproduces np.mean and np.std exactly."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, got_mean, got_std = _welford_merge([_make_stats(data)])
        assert got_mean == pytest.approx(np.mean(data))
        assert got_std == pytest.approx(np.std(data))

    def test_two_groups_matches_numpy(self):
        """Two groups merged produce the same result as concatenating them."""
        rng = np.random.default_rng(0)
        a = rng.normal(100.0, 50.0, 200)
        b = rng.normal(-200.0, 100.0, 300)
        _, got_mean, got_std = _welford_merge([_make_stats(a), _make_stats(b)])
        combined = np.concatenate([a, b])
        assert got_mean == pytest.approx(np.mean(combined), rel=1e-9)
        assert got_std == pytest.approx(np.std(combined), rel=1e-9)

    def test_many_groups_matches_numpy(self):
        """Many groups produce the same result as concatenating all of them."""
        rng = np.random.default_rng(42)
        groups = [rng.normal(i * 100.0, 30.0, 150) for i in range(10)]
        stats = [_make_stats(g) for g in groups]
        _, got_mean, got_std = _welford_merge(stats)
        combined = np.concatenate(groups)
        assert got_mean == pytest.approx(np.mean(combined), rel=1e-9)
        assert got_std == pytest.approx(np.std(combined), rel=1e-9)

    def test_zero_count_groups_are_skipped(self):
        """Groups with n == 0 do not affect the result."""
        data = np.array([10.0, 20.0, 30.0])
        empty = (0, 0.0, 0.0)
        _, got_mean, got_std = _welford_merge(
            [empty, _make_stats(data), empty]
        )
        assert got_mean == pytest.approx(np.mean(data))
        assert got_std == pytest.approx(np.std(data))

    def test_all_zero_count_returns_zeros(self):
        """An input of only zero-count groups returns (0, 0.0, 0.0)."""
        total_n, mean, std = _welford_merge([(0, 0.0, 0.0), (0, 0.0, 0.0)])
        assert total_n == 0
        assert mean == 0.0
        assert std == 0.0

    def test_empty_input_returns_zeros(self):
        """An empty stats list returns (0, 0.0, 0.0)."""
        total_n, mean, std = _welford_merge([])
        assert total_n == 0
        assert mean == 0.0
        assert std == 0.0

    def test_constant_values_give_zero_std(self):
        """A group where all values are the same produces std == 0."""
        data = np.full(100, 42.0)
        _, got_mean, got_std = _welford_merge([_make_stats(data)])
        assert got_mean == pytest.approx(42.0)
        assert got_std == pytest.approx(0.0, abs=1e-12)

    def test_total_n_is_correct(self):
        """total_n equals the sum of the individual group sizes."""
        groups = [np.ones(n) for n in [10, 20, 30]]
        total_n, _, _ = _welford_merge([_make_stats(g) for g in groups])
        assert total_n == 60


# ---------------------------------------------------------------------------
# _percentile_from_histogram
# ---------------------------------------------------------------------------

class TestPercentileFromHistogram:
    """Tests for _percentile_from_histogram."""

    def test_median_of_uniform_hu_range(self):
        """50th percentile of uniform HU data matches np.percentile within 1 HU."""
        data = np.arange(-200, 200, dtype=np.float64)
        hist, bin_edges = _make_hist(data)
        result = _percentile_from_histogram(hist, bin_edges, 50.0)
        assert abs(result - np.percentile(data, 50.0)) <= 1.0

    def test_low_percentile_accuracy(self):
        """0.5th percentile estimate is within 2 HU of the exact value."""
        data = np.arange(-500, 500, dtype=np.float64)
        hist, bin_edges = _make_hist(data)
        result = _percentile_from_histogram(
            hist, bin_edges, constants.CT_GLOBAL_CLIP_MIN_PERCENTILE
        )
        assert abs(result - np.percentile(data, constants.CT_GLOBAL_CLIP_MIN_PERCENTILE)) <= 2.0

    def test_high_percentile_accuracy(self):
        """99.5th percentile estimate is within 2 HU of the exact value."""
        data = np.arange(-500, 500, dtype=np.float64)
        hist, bin_edges = _make_hist(data)
        result = _percentile_from_histogram(
            hist, bin_edges, constants.CT_GLOBAL_CLIP_MAX_PERCENTILE
        )
        assert abs(result - np.percentile(data, constants.CT_GLOBAL_CLIP_MAX_PERCENTILE)) <= 2.0

    def test_empty_histogram_returns_zero(self):
        """An all-zero histogram returns 0.0."""
        _, bin_edges = _make_hist([])
        hist = np.zeros(constants.CT_HU_HIST_BINS, dtype=np.int64)
        assert _percentile_from_histogram(hist, bin_edges, 50.0) == 0.0

    def test_single_bin_value(self):
        """All values in one bin → all percentiles return that bin's edge."""
        data = np.full(100, 0.0)  # all at 0 HU
        hist, bin_edges = _make_hist(data)
        for pct in [0.5, 50.0, 99.5]:
            result = _percentile_from_histogram(hist, bin_edges, pct)
            assert abs(result - 0.0) <= 1.0

    def test_aggregated_histograms_match_pooled_data(self):
        """Summing per-patient histograms gives the same result as pooling data."""
        rng = np.random.default_rng(7)
        patients = [rng.uniform(-300, 300, 500) for _ in range(5)]
        all_data = np.concatenate(patients)
        hist_all, bin_edges = _make_hist(all_data)

        # Sum individual histograms.
        combined = np.zeros(constants.CT_HU_HIST_BINS, dtype=np.int64)
        for p in patients:
            h, _ = _make_hist(p)
            combined += h

        for pct in [0.5, 50.0, 99.5]:
            from_all = _percentile_from_histogram(hist_all, bin_edges, pct)
            from_combined = _percentile_from_histogram(combined, bin_edges, pct)
            assert from_all == from_combined
