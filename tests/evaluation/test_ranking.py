"""Tests for mist.evaluation.ranking."""
import numpy as np
import pandas as pd
import pytest

from mist.evaluation import ranking
from mist.evaluation.ranking import (
    SUMMARY_ROW_IDS,
    _direction_for_column,
    _strip_summary_rows,
    _suffix_match_metric,
    rank_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(ids, **cols):
    """Build a small results DataFrame for tests."""
    return pd.DataFrame({"id": ids, **cols})


# ---------------------------------------------------------------------------
# _strip_summary_rows
# ---------------------------------------------------------------------------

class TestStripSummaryRows:
    """Tests for ranking._strip_summary_rows."""

    def test_drops_known_summary_ids(self):
        """Rows whose id is a known summary label are dropped."""
        df = _df(
            ids=["p1", "p2", "Mean", "Std", "Median"],
            WT_dice=[0.9, 0.8, 0.85, 0.05, 0.85],
        )
        out = _strip_summary_rows(df, "id")
        assert list(out["id"]) == ["p1", "p2"]

    def test_keeps_all_when_no_summary_rows(self):
        """All rows survive when no summary labels are present."""
        df = _df(ids=["p1", "p2"], WT_dice=[0.9, 0.8])
        out = _strip_summary_rows(df, "id")
        assert list(out["id"]) == ["p1", "p2"]

    def test_summary_row_ids_constant(self):
        """SUMMARY_ROW_IDS contains the exact labels emitted by evaluator."""
        assert SUMMARY_ROW_IDS == frozenset({
            "Mean", "Std", "25th Percentile", "Median", "75th Percentile"
        })

    def test_missing_id_column_raises(self):
        """A DataFrame without the id column raises ValueError."""
        df = pd.DataFrame({"WT_dice": [0.9]})
        with pytest.raises(ValueError, match="missing the required id column"):
            _strip_summary_rows(df, "id")

    def test_alternate_id_column(self):
        """A custom id column name is honored."""
        df = pd.DataFrame({
            "patient": ["p1", "Mean"],
            "WT_dice": [0.9, 0.8],
        })
        out = _strip_summary_rows(df, "patient")
        assert list(out["patient"]) == ["p1"]

    def test_does_not_mutate_input(self):
        """Input DataFrame is not modified by stripping."""
        df = _df(ids=["p1", "Mean"], WT_dice=[0.9, 0.8])
        _ = _strip_summary_rows(df, "id")
        assert list(df["id"]) == ["p1", "Mean"]


# ---------------------------------------------------------------------------
# _suffix_match_metric
# ---------------------------------------------------------------------------

class TestSuffixMatchMetric:
    """Tests for ranking._suffix_match_metric."""

    def test_matches_simple_metric(self):
        """A column ending in _dice resolves to 'dice'."""
        assert _suffix_match_metric("WT_dice", ["dice", "haus95"]) == "dice"

    def test_longest_match_wins(self):
        """A multi-word metric wins over a shorter substring."""
        keys = ["dice", "lesion_wise_dice", "surf_dice"]
        assert _suffix_match_metric(
            "WT_lesion_wise_dice", keys
        ) == "lesion_wise_dice"
        assert _suffix_match_metric(
            "ET_surf_dice", keys
        ) == "surf_dice"

    def test_no_match_returns_none(self):
        """Columns with no matching suffix return None."""
        assert _suffix_match_metric("WT_custom", ["dice", "haus95"]) is None

    def test_column_equal_to_metric_does_not_match(self):
        """A bare metric column with no class prefix does not match."""
        # "_dice" suffix requires content before the underscore.
        assert _suffix_match_metric("dice", ["dice"]) is None

    def test_empty_metric_keys_returns_none(self):
        """An empty registry returns None."""
        assert _suffix_match_metric("WT_dice", []) is None


# ---------------------------------------------------------------------------
# _direction_for_column
# ---------------------------------------------------------------------------

class TestDirectionForColumn:
    """Tests for ranking._direction_for_column."""

    def test_higher_is_better_metric(self):
        """Dice (best=1.0, worst=0.0) resolves to 'higher'."""
        assert _direction_for_column("WT_dice", None) == "higher"

    def test_lower_is_better_metric(self):
        """Hausdorff95 (best=0.0, worst=inf) resolves to 'lower'."""
        assert _direction_for_column("WT_haus95", None) == "lower"

    def test_lesion_wise_metric_resolves(self):
        """Multi-word lesion-wise metrics resolve correctly."""
        assert _direction_for_column(
            "WT_lesion_wise_dice", None
        ) == "higher"
        assert _direction_for_column(
            "WT_lesion_wise_haus95", None
        ) == "lower"

    def test_override_takes_precedence(self):
        """An override beats the registry default."""
        # Even though dice would normally be 'higher', an explicit override
        # of 'lower' wins.
        assert _direction_for_column(
            "WT_dice", {"WT_dice": "lower"}
        ) == "lower"

    def test_invalid_override_raises(self):
        """Override values must be 'higher' or 'lower'."""
        with pytest.raises(ValueError, match="must be 'higher' or 'lower'"):
            _direction_for_column("custom_col", {"custom_col": "ascending"})

    def test_unknown_column_without_override_raises(self):
        """Columns not matching the registry require an override."""
        with pytest.raises(ValueError, match="Cannot determine ranking"):
            _direction_for_column("WT_custom_metric", None)

    def test_unknown_column_with_override_resolves(self):
        """An override allows ranking of unregistered metrics."""
        assert _direction_for_column(
            "WT_custom_metric",
            {"WT_custom_metric": "higher"},
        ) == "higher"


# ---------------------------------------------------------------------------
# rank_results — happy paths
# ---------------------------------------------------------------------------

class TestRankResults:
    """Tests for ranking.rank_results."""

    def test_two_strategies_higher_better(self):
        """Higher-is-better metric: larger values get rank 1."""
        df_a = _df(ids=["p1", "p2"], WT_dice=[0.9, 0.8])
        df_b = _df(ids=["p1", "p2"], WT_dice=[0.5, 0.4])

        summary, detailed = rank_results([df_a, df_b], names=["a", "b"])

        # a beats b on every patient → average rank 1.0; b → 2.0.
        a_row = summary.loc[summary["strategy"] == "a"].iloc[0]
        b_row = summary.loc[summary["strategy"] == "b"].iloc[0]
        assert a_row["average_rank"] == pytest.approx(1.0)
        assert b_row["average_rank"] == pytest.approx(2.0)
        # Summary is sorted ascending by average_rank.
        assert summary.iloc[0]["strategy"] == "a"

        # Detailed has one column per metric.
        assert "WT_dice" in detailed.columns
        a_detail = detailed.loc[detailed["strategy"] == "a"].iloc[0]
        assert a_detail["WT_dice"] == pytest.approx(1.0)

    def test_two_strategies_lower_better(self):
        """Lower-is-better metric: smaller values get rank 1."""
        df_a = _df(ids=["p1", "p2"], WT_haus95=[1.0, 2.0])
        df_b = _df(ids=["p1", "p2"], WT_haus95=[5.0, 6.0])

        summary, _ = rank_results([df_a, df_b], names=["a", "b"])

        a_row = summary.loc[summary["strategy"] == "a"].iloc[0]
        assert a_row["average_rank"] == pytest.approx(1.0)

    def test_ties_use_average_ranks(self):
        """Tied values share the average of their positions."""
        df_a = _df(ids=["p1"], WT_dice=[0.9])
        df_b = _df(ids=["p1"], WT_dice=[0.9])
        df_c = _df(ids=["p1"], WT_dice=[0.5])

        summary, _ = rank_results(
            [df_a, df_b, df_c], names=["a", "b", "c"]
        )

        # a and b tie for 1st (average rank (1+2)/2 = 1.5); c is 3rd.
        a_row = summary.loc[summary["strategy"] == "a"].iloc[0]
        b_row = summary.loc[summary["strategy"] == "b"].iloc[0]
        c_row = summary.loc[summary["strategy"] == "c"].iloc[0]
        assert a_row["average_rank"] == pytest.approx(1.5)
        assert b_row["average_rank"] == pytest.approx(1.5)
        assert c_row["average_rank"] == pytest.approx(3.0)

    def test_mixed_metrics_aggregate_correctly(self):
        """Higher-better and lower-better metrics combine in the average."""
        # On WT_dice (higher better) a wins. On WT_haus95 (lower better)
        # a also wins. Average rank should be 1.0 for a, 2.0 for b.
        df_a = _df(
            ids=["p1"], WT_dice=[0.9], WT_haus95=[1.0]
        )
        df_b = _df(
            ids=["p1"], WT_dice=[0.5], WT_haus95=[5.0]
        )

        summary, detailed = rank_results([df_a, df_b], names=["a", "b"])
        a_row = summary.loc[summary["strategy"] == "a"].iloc[0]
        assert a_row["average_rank"] == pytest.approx(1.0)
        # Detailed has one column per metric.
        assert {"WT_dice", "WT_haus95"} <= set(detailed.columns)

    def test_summary_rows_are_stripped(self):
        """Aggregate summary rows are excluded before ranking."""
        df_a = _df(
            ids=["p1", "p2", "Mean"],
            WT_dice=[0.9, 0.8, 0.85],
        )
        df_b = _df(
            ids=["p1", "p2", "Mean"],
            WT_dice=[0.5, 0.4, 0.45],
        )

        summary, _ = rank_results([df_a, df_b], names=["a", "b"])
        # If Mean had not been stripped, ranking would still hold here, but
        # the test guards against future regressions where Mean might tie or
        # invert the result. We simply assert no crash and clear winner.
        a_row = summary.loc[summary["strategy"] == "a"].iloc[0]
        assert a_row["average_rank"] == pytest.approx(1.0)

    def test_default_strategy_names(self):
        """Default names follow strategy_<index> when names is None."""
        df_a = _df(ids=["p1"], WT_dice=[0.9])
        df_b = _df(ids=["p1"], WT_dice=[0.5])

        summary, _ = rank_results([df_a, df_b])
        assert set(summary["strategy"]) == {"strategy_0", "strategy_1"}

    def test_rows_aligned_by_id(self):
        """DataFrames in different row order still align by id."""
        df_a = _df(
            ids=["p1", "p2"],
            WT_dice=[0.9, 0.5],  # a wins p1, loses p2
        )
        df_b = _df(
            ids=["p2", "p1"],   # reversed order
            WT_dice=[0.9, 0.5],  # b wins p2, loses p1
        )

        summary, _ = rank_results([df_a, df_b], names=["a", "b"])
        # a and b each win one patient → tie at average rank (1+2)/2 = 1.5.
        a_row = summary.loc[summary["strategy"] == "a"].iloc[0]
        b_row = summary.loc[summary["strategy"] == "b"].iloc[0]
        assert a_row["average_rank"] == pytest.approx(1.5)
        assert b_row["average_rank"] == pytest.approx(1.5)

    def test_three_strategies_distinct_ranks(self):
        """Three strategies produce per-metric ranks 1/2/3."""
        df_a = _df(ids=["p1"], WT_dice=[0.9])
        df_b = _df(ids=["p1"], WT_dice=[0.7])
        df_c = _df(ids=["p1"], WT_dice=[0.5])

        summary, _ = rank_results(
            [df_a, df_b, df_c], names=["a", "b", "c"]
        )
        ranks_by_name = dict(zip(summary["strategy"], summary["average_rank"]))
        assert ranks_by_name["a"] == pytest.approx(1.0)
        assert ranks_by_name["b"] == pytest.approx(2.0)
        assert ranks_by_name["c"] == pytest.approx(3.0)

    def test_custom_id_column(self):
        """A non-default id column is honored."""
        df_a = pd.DataFrame({"patient": ["p1"], "WT_dice": [0.9]})
        df_b = pd.DataFrame({"patient": ["p1"], "WT_dice": [0.5]})
        summary, _ = rank_results(
            [df_a, df_b], names=["a", "b"], id_column="patient"
        )
        winner = summary.iloc[0]["strategy"]
        assert winner == "a"

    def test_direction_overrides_for_unregistered_metric(self):
        """A non-MIST metric column ranks via overrides."""
        df_a = pd.DataFrame({"id": ["p1"], "WT_custom": [10.0]})
        df_b = pd.DataFrame({"id": ["p1"], "WT_custom": [5.0]})
        summary, _ = rank_results(
            [df_a, df_b],
            names=["a", "b"],
            direction_overrides={"WT_custom": "higher"},
        )
        # Higher better → a wins.
        assert summary.iloc[0]["strategy"] == "a"

    def test_detailed_per_metric_means(self):
        """Detailed output reports mean rank per (strategy, metric)."""
        # Two metrics, two patients, two strategies. Per-metric mean ranks
        # should match per-patient ranks averaged across patients.
        df_a = _df(
            ids=["p1", "p2"],
            WT_dice=[0.9, 0.5],  # a is better on p1, worse on p2.
            WT_haus95=[1.0, 5.0],  # a is better on p1, worse on p2.
        )
        df_b = _df(
            ids=["p1", "p2"],
            WT_dice=[0.5, 0.9],
            WT_haus95=[5.0, 1.0],
        )
        _, detailed = rank_results([df_a, df_b], names=["a", "b"])
        # Each metric averages rank 1 and rank 2 → 1.5 for each strategy.
        a_row = detailed.loc[detailed["strategy"] == "a"].iloc[0]
        b_row = detailed.loc[detailed["strategy"] == "b"].iloc[0]
        assert a_row["WT_dice"] == pytest.approx(1.5)
        assert a_row["WT_haus95"] == pytest.approx(1.5)
        assert b_row["WT_dice"] == pytest.approx(1.5)
        assert b_row["WT_haus95"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# rank_results — error paths
# ---------------------------------------------------------------------------

class TestRankResultsErrors:
    """Error handling in ranking.rank_results."""

    def test_fewer_than_two_results_raises(self):
        """At least two DataFrames are required."""
        df = _df(ids=["p1"], WT_dice=[0.9])
        with pytest.raises(ValueError, match="at least 2 DataFrames"):
            rank_results([df])

    def test_names_length_mismatch_raises(self):
        """names length must equal len(results)."""
        df_a = _df(ids=["p1"], WT_dice=[0.9])
        df_b = _df(ids=["p1"], WT_dice=[0.5])
        with pytest.raises(ValueError, match="names has length"):
            rank_results([df_a, df_b], names=["only_one"])

    def test_duplicate_names_raises(self):
        """names must be unique."""
        df_a = _df(ids=["p1"], WT_dice=[0.9])
        df_b = _df(ids=["p1"], WT_dice=[0.5])
        with pytest.raises(ValueError, match="names must be unique"):
            rank_results([df_a, df_b], names=["same", "same"])

    def test_mismatched_columns_raises(self):
        """All DataFrames must share the same columns."""
        df_a = _df(ids=["p1"], WT_dice=[0.9])
        df_b = _df(ids=["p1"], TC_dice=[0.5])
        with pytest.raises(ValueError, match="different columns"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_mismatched_patient_ids_raises(self):
        """All DataFrames must share the same patient ID set."""
        df_a = _df(ids=["p1", "p2"], WT_dice=[0.9, 0.8])
        df_b = _df(ids=["p1", "p3"], WT_dice=[0.5, 0.4])
        with pytest.raises(ValueError, match="different patient IDs"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_duplicate_patient_ids_raises(self):
        """Duplicate IDs within a DataFrame are rejected."""
        df_a = _df(ids=["p1", "p1"], WT_dice=[0.9, 0.8])
        df_b = _df(ids=["p1", "p2"], WT_dice=[0.5, 0.4])
        with pytest.raises(ValueError, match="duplicate patient IDs"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_empty_after_stripping_raises(self):
        """A DataFrame with only summary rows raises ValueError."""
        df_a = _df(
            ids=["Mean", "Std"],
            WT_dice=[0.85, 0.05],
        )
        df_b = _df(ids=["p1"], WT_dice=[0.9])
        with pytest.raises(ValueError, match="no rows after removing summary"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_no_metric_columns_raises(self):
        """A DataFrame with only the id column raises ValueError."""
        df_a = pd.DataFrame({"id": ["p1"]})
        df_b = pd.DataFrame({"id": ["p1"]})
        with pytest.raises(ValueError, match="No metric columns"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_nan_values_raise(self):
        """NaN values in metric columns raise ValueError."""
        df_a = _df(ids=["p1", "p2"], WT_dice=[0.9, np.nan])
        df_b = _df(ids=["p1", "p2"], WT_dice=[0.5, 0.4])
        with pytest.raises(ValueError, match="NaN values in column"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_unknown_column_without_override_raises(self):
        """A non-MIST metric without an override raises ValueError."""
        df_a = pd.DataFrame({"id": ["p1"], "WT_custom": [1.0]})
        df_b = pd.DataFrame({"id": ["p1"], "WT_custom": [0.5]})
        with pytest.raises(ValueError, match="Cannot determine ranking"):
            rank_results([df_a, df_b], names=["a", "b"])

    def test_missing_id_column_raises(self):
        """A DataFrame without the id column raises ValueError."""
        df_a = pd.DataFrame({"WT_dice": [0.9]})
        df_b = pd.DataFrame({"WT_dice": [0.5]})
        with pytest.raises(ValueError, match="missing the required id column"):
            rank_results([df_a, df_b], names=["a", "b"])
