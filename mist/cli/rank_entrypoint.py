"""Command line tool to rank multiple evaluation result CSVs BraTS-style."""
import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd

from mist.cli.args import ArgParser
from mist.evaluation.ranking import rank_results
from mist.utils import io


def _parse_rank_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ranking."""
    parser = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Rank N evaluation result CSVs BraTS-style. For each (patient, "
            "metric) cell, strategies are ranked from best (1) to worst with "
            "average tie handling, then aggregated by mean rank per strategy."
        ),
    )
    parser.arg(
        "--results", type=str, nargs="+", required=True,
        help=(
            "Paths to two or more evaluation result CSVs (e.g., outputs of "
            "mist_evaluate). Each CSV must share the same id column and "
            "metric columns."
        ),
    )
    parser.arg(
        "--names", type=str, nargs="+", default=None,
        help=(
            "Optional friendly labels, one per --results CSV in the same "
            "order. Defaults to the file stem of each results CSV."
        ),
    )
    parser.arg(
        "--output-csv", type=str, required=True,
        help=(
            "Path where the summary ranking CSV will be written. Columns: "
            "'strategy', 'average_rank'."
        ),
    )
    parser.arg(
        "--output-detailed-csv", type=str, default=None,
        help=(
            "Optional path for a per-metric breakdown CSV containing mean "
            "ranks per strategy per metric column."
        ),
    )
    parser.arg(
        "--metric-direction-overrides", type=str, default=None,
        help=(
            "Optional path to a JSON file mapping metric column name to "
            "'higher' or 'lower'. Required only for columns whose suffix "
            "does not match a registered MIST metric."
        ),
    )
    parser.arg(
        "--id-column", type=str, default="id",
        help="Name of the column identifying each patient.",
    )

    ns = parser.parse_args(argv)
    return ns


def _ensure_output_dir(output_csv: Path) -> None:
    """Create the parent directory for the output CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)


def run_rank(ns: argparse.Namespace) -> None:
    """Load CSVs, run ranking, and write output(s)."""
    results_paths = [Path(p).expanduser().resolve() for p in ns.results]
    if len(results_paths) < 2:
        raise ValueError(
            "mist_rank requires at least two --results CSVs to rank."
        )

    if ns.names is not None:
        names = list(ns.names)
        if len(names) != len(results_paths):
            raise ValueError(
                f"--names has {len(names)} entries but --results has "
                f"{len(results_paths)}."
            )
    else:
        names = [p.stem for p in results_paths]

    direction_overrides = None
    if ns.metric_direction_overrides is not None:
        overrides_path = (
            Path(ns.metric_direction_overrides).expanduser().resolve()
        )
        direction_overrides = io.read_json_file(overrides_path)
        if not isinstance(direction_overrides, dict):
            raise ValueError(
                f"Direction overrides file at {overrides_path} must contain "
                "a JSON object mapping column names to 'higher' or 'lower'."
            )

    results = [pd.read_csv(p) for p in results_paths]

    summary_df, detailed_df = rank_results(
        results=results,
        names=names,
        direction_overrides=direction_overrides,
        id_column=ns.id_column,
    )

    output_csv = Path(ns.output_csv).expanduser().resolve()
    _ensure_output_dir(output_csv)
    summary_df.to_csv(output_csv, index=False)

    if ns.output_detailed_csv is not None:
        detailed_csv = Path(ns.output_detailed_csv).expanduser().resolve()
        _ensure_output_dir(detailed_csv)
        detailed_df.to_csv(detailed_csv, index=False)


def rank_entry(argv: list[str] | None = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_rank_args(argv)
    run_rank(ns)


if __name__ == "__main__":
    rank_entry()  # pragma: no cover
