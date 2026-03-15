"""Command line tool to evaluate predictions from MIST output."""

import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mist.cli.args import ArgParser
from mist.evaluation.evaluator import Evaluator
from mist.utils import io


def _parse_eval_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI for evaluation."""
    parser = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Evaluate predictions produced by MIST.",
    )
    parser.arg(
        "--config", type=str, required=True,
        help="Path to config.json from a MIST run (must contain evaluation.final_classes)."
    )
    parser.arg(
        "--paths-csv", type=str, required=True,
        help="CSV with paths to predictions/masks."
    )
    parser.arg(
        "--output-csv", type=str, required=True,
        help="Where to write the evaluation results CSV."
    )
    parser.arg(
        "--num-workers", type=int, required=False,
        help="Number of parallel workers to use for evaluation."
    )

    ns = parser.parse_args(argv)
    return ns


def _ensure_output_dir(output_csv: Path) -> None:
    """Create the parent directory for the output CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)


def run_evaluation(ns: argparse.Namespace) -> None:
    """Load inputs, construct Evaluator, and run."""
    config_path = Path(ns.config).expanduser().resolve()
    paths_csv = Path(ns.paths_csv).expanduser().resolve()
    output_csv = Path(ns.output_csv).expanduser().resolve()

    _ensure_output_dir(output_csv)

    # Load inputs.
    df = pd.read_csv(paths_csv)
    full_config = io.read_json_file(config_path)

    # Extract just the evaluation portion (fallback to the full config if it's
    # already scoped).
    eval_config = full_config.get("evaluation", full_config)

    # Initialize and run.
    evaluator = Evaluator(
        filepaths_dataframe=df,
        evaluation_config=eval_config,
        output_csv_path=str(output_csv),
    )
    evaluator.run(max_workers=ns.num_workers)


def evaluation_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_eval_args(argv)
    run_evaluation(ns)


if __name__ == "__main__":
    evaluation_entry()  # pragma: no cover
