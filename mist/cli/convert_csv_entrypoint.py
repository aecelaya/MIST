"""Command line tool for converting CSV-format datasets to MIST format."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
import argparse

from mist.cli.args import ArgParser
from mist.conversion_tools.csv import convert_csv


def _parse_convert_csv_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for CSV dataset conversion."""
    parser = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Convert a CSV-format dataset to MIST format.",
    )
    parser.arg(
        "--train-csv", type=str, required=True,
        help=(
            "Path to training CSV with columns: id, mask, image1 [, image2, ...]."
        ),
    )
    parser.arg(
        "--output", type=str, required=True,
        help="Directory to save the converted MIST-format dataset.",
    )
    parser.arg(
        "--test-csv", type=str, required=False,
        help=(
            "Path to optional test CSV with columns: id, image1 [, image2, ...]."
        ),
    )
    parser.arg(
        "--num-workers-conversion", type=int, default=1,
        help="Number of parallel threads for file copying.",
    )
    return parser.parse_args(argv)


def run_convert_csv(ns: argparse.Namespace) -> None:
    """Resolve paths and call the CSV converter."""
    train_csv = Path(ns.train_csv).expanduser().resolve()
    output = Path(ns.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_csv = (
        Path(ns.test_csv).expanduser().resolve() if ns.test_csv else None
    )
    convert_csv(train_csv, output, test_csv, max_workers=ns.num_workers_conversion)


def convert_csv_entry(argv: list[str] | None = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_convert_csv_args(argv)
    run_convert_csv(ns)


if __name__ == "__main__":
    convert_csv_entry()  # pragma: no cover
