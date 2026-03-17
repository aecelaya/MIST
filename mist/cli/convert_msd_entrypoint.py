"""Command line tool for converting MSD datasets to MIST format."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional
import argparse

from mist.cli.args import ArgParser
from mist.conversion_tools.msd import convert_msd


def _parse_convert_msd_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Parse CLI arguments for MSD dataset conversion."""
    parser = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Convert a Medical Segmentation Decathlon dataset to MIST format.",
    )
    parser.arg(
        "--source", type=str, required=True,
        help="Path to the MSD dataset directory (must contain dataset.json).",
    )
    parser.arg(
        "--output", type=str, required=True,
        help="Directory to save the converted MIST-format dataset.",
    )
    parser.arg(
        "--num-workers", type=int, required=False,
        help=(
            "Number of parallel threads for file copying. Defaults to the "
            "ThreadPoolExecutor default. Reduce if you encounter I/O errors."
        ),
    )
    return parser.parse_args(argv)


def run_convert_msd(ns: argparse.Namespace) -> None:
    """Resolve paths and call the MSD converter."""
    source = Path(ns.source).expanduser().resolve()
    output = Path(ns.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    convert_msd(source, output, max_workers=ns.num_workers)


def convert_msd_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_convert_msd_args(argv)
    run_convert_msd(ns)


if __name__ == "__main__":
    convert_msd_entry()  # pragma: no cover
