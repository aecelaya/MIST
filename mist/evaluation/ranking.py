"""BraTS-style ranking of evaluation result DataFrames.

Provides a generic, framework-neutral ranking utility: given N result
DataFrames (e.g., the outputs of mist_evaluate run on N different models or
postprocessing strategies), produce a per-strategy summary rank and a
per-metric breakdown.

The ranking scheme follows the BraTS challenge convention:
    1. For each (patient, metric) cell, rank strategies from best (1) to
       worst, with tied values receiving the average of their positions.
    2. Aggregate per-strategy by averaging across all (patient, metric)
       cells.

Direction (whether higher or lower values are better) is auto-detected for
any column whose suffix matches a metric registered in
mist.metrics.metrics_registry. Columns from external metrics can be handled
via a direction_overrides mapping.
"""
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mist.metrics.metrics_registry import METRIC_REGISTRY


# Summary-row IDs added by mist.evaluation.evaluation_utils.compute_results_stats.
# These are stripped before ranking so aggregate rows do not contaminate the
# per-patient ranking.
SUMMARY_ROW_IDS: frozenset[str] = frozenset({
    "Mean", "Std", "25th Percentile", "Median", "75th Percentile",
})


def _strip_summary_rows(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Drop rows whose id is a known aggregate label (Mean, Std, ...)."""
    if id_column not in df.columns:
        raise ValueError(
            f"DataFrame is missing the required id column '{id_column}'."
        )
    return df.loc[~df[id_column].astype(str).isin(SUMMARY_ROW_IDS)].copy()


def _suffix_match_metric(
    column: str, metric_keys: list[str]
) -> str | None:
    """Return the registered metric name matching `column`'s suffix.

    Matches the longest registered metric name first so that, for example,
    "WT_lesion_wise_dice" resolves to "lesion_wise_dice" rather than "dice".
    """
    for metric in sorted(metric_keys, key=len, reverse=True):
        suffix = f"_{metric}"
        if column.endswith(suffix) and len(column) > len(suffix):
            return metric
    return None


def _direction_for_column(
    column: str,
    overrides: Mapping[str, str] | None,
) -> Literal["higher", "lower"]:
    """Resolve the ranking direction for a single metric column.

    Resolution order:
        1. If `column` is in `overrides`, use that.
        2. Else, suffix-match against MIST's metric registry and infer
           direction from the matched metric's best/worst attributes.
        3. Else, raise ValueError.

    Raises:
        ValueError: If overrides supplies an invalid direction or if the
            column cannot be matched to a registered metric and has no
            override.
    """
    if overrides and column in overrides:
        direction = overrides[column]
        if direction not in ("higher", "lower"):
            raise ValueError(
                f"Direction override for column '{column}' must be 'higher' "
                f"or 'lower', got {direction!r}."
            )
        return direction

    metric_name = _suffix_match_metric(column, list(METRIC_REGISTRY.keys()))
    if metric_name is None:
        raise ValueError(
            f"Cannot determine ranking direction for column '{column}'. "
            "Either rename it to end with a registered MIST metric "
            f"(one of: {sorted(METRIC_REGISTRY.keys())}), or pass a "
            "direction for this column via direction_overrides."
        )
    metric = METRIC_REGISTRY[metric_name]
    return "higher" if metric.best > metric.worst else "lower"


def rank_results(
    results: list[pd.DataFrame],
    names: list[str] | None = None,
    direction_overrides: Mapping[str, str] | None = None,
    id_column: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank N result DataFrames BraTS-style.

    For each (patient, metric) cell, strategies are ranked from best (1) to
    worst with average tie handling. The summary rank for a strategy is the
    mean of its ranks across all (patient, metric) cells.

    Args:
        results: List of result DataFrames (>= 2). Each must share the same
            id column and the same metric columns.
        names: Optional friendly labels, one per DataFrame. Defaults to
            "strategy_0", "strategy_1", ...
        direction_overrides: Optional mapping of column name to "higher" or
            "lower". Required only for metric columns whose suffix does not
            match a registered MIST metric.
        id_column: Name of the column identifying each patient. Defaults to
            "id".

    Returns:
        Tuple of (summary_df, detailed_df).
            summary_df has columns ["strategy", "average_rank"], sorted
                ascending by average_rank.
            detailed_df has columns ["strategy", *metric_columns], where each
                value is the mean rank for that strategy on that metric
                across all patients.

    Raises:
        ValueError: If there are fewer than 2 DataFrames, name length or
            uniqueness checks fail, the id column is missing, columns or
            patient IDs differ across DataFrames, patient IDs duplicate
            within a DataFrame, metric values contain NaNs, or a metric
            column has no resolvable ranking direction.
    """
    if len(results) < 2:
        raise ValueError(
            f"rank_results requires at least 2 DataFrames, got {len(results)}."
        )

    if names is None:
        names = [f"strategy_{i}" for i in range(len(results))]
    elif len(names) != len(results):
        raise ValueError(
            f"names has length {len(names)} but results has length "
            f"{len(results)}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"names must be unique, got {names}.")

    # Strip summary rows and validate per-DataFrame.
    cleaned: list[pd.DataFrame] = []
    for i, df in enumerate(results):
        df = _strip_summary_rows(df, id_column)
        if df[id_column].duplicated().any():
            raise ValueError(
                f"DataFrame at index {i} has duplicate patient IDs in "
                f"column '{id_column}'."
            )
        if df.empty:
            raise ValueError(
                f"DataFrame at index {i} has no rows after removing summary "
                "rows. Cannot rank empty results."
            )
        cleaned.append(df)

    # Validate columns are identical across all DataFrames.
    reference_cols = list(cleaned[0].columns)
    for i, df in enumerate(cleaned[1:], start=1):
        if list(df.columns) != reference_cols:
            raise ValueError(
                f"DataFrame at index {i} has different columns than "
                "DataFrame at index 0. All result DataFrames must share "
                "the same columns."
            )

    # Validate patient IDs are identical across all DataFrames.
    reference_ids = sorted(cleaned[0][id_column].astype(str).tolist())
    for i, df in enumerate(cleaned[1:], start=1):
        df_ids = sorted(df[id_column].astype(str).tolist())
        if df_ids != reference_ids:
            raise ValueError(
                f"DataFrame at index {i} has different patient IDs than "
                "DataFrame at index 0. All result DataFrames must share "
                "the same patient set."
            )

    # Sort each DataFrame by id (as string) so rows align across strategies.
    aligned = [
        df.assign(_sort_key=df[id_column].astype(str))
          .sort_values("_sort_key")
          .drop(columns="_sort_key")
          .reset_index(drop=True)
        for df in cleaned
    ]

    metric_columns = [c for c in reference_cols if c != id_column]
    if not metric_columns:
        raise ValueError(
            "No metric columns found in result DataFrames "
            f"(all columns: {reference_cols})."
        )

    # Validate metric values are finite.
    for i, df in enumerate(aligned):
        for col in metric_columns:
            if df[col].isna().any():
                raise ValueError(
                    f"DataFrame at index {i} has NaN values in column "
                    f"'{col}'. Replace or drop NaN values before ranking."
                )

    # Resolve direction for each metric column.
    directions = {
        col: _direction_for_column(col, direction_overrides)
        for col in metric_columns
    }

    n_strategies = len(aligned)
    n_patients = len(aligned[0])
    n_metrics = len(metric_columns)

    # ranks[m, p, s] = rank of strategy s for metric m on patient p.
    ranks = np.empty((n_metrics, n_patients, n_strategies), dtype=float)
    for m_idx, col in enumerate(metric_columns):
        # Stack values across strategies: shape (n_patients, n_strategies).
        values = np.stack(
            [df[col].to_numpy(dtype=float) for df in aligned], axis=1
        )
        # rankdata(method="average") ranks ascending (smallest -> 1).
        # For "higher is better" metrics, negate so the largest -> 1.
        if directions[col] == "higher":
            values = -values
        for p_idx in range(n_patients):
            ranks[m_idx, p_idx, :] = rankdata(
                values[p_idx, :], method="average"
            )

    # Summary: mean per strategy across all (metric, patient).
    avg_ranks = ranks.mean(axis=(0, 1))
    summary_df = pd.DataFrame({
        "strategy": names,
        "average_rank": avg_ranks,
    }).sort_values("average_rank", ignore_index=True)

    # Detailed: mean per (strategy, metric) across patients.
    per_metric_means = ranks.mean(axis=1)  # (n_metrics, n_strategies)
    detailed_df = pd.DataFrame({"strategy": names})
    for m_idx, col in enumerate(metric_columns):
        detailed_df[col] = per_metric_means[m_idx, :]

    return summary_df, detailed_df
