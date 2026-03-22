"""Entrypoint for averaging MIST fold checkpoint weights."""
from typing import List, Optional
import argparse

from mist.cli.args import ArgParser
from mist.models.model_loader import average_fold_weights


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = ArgParser(
        description=(
            "Average weights from multiple MIST fold checkpoints. "
            "Produces a single initialization checkpoint by element-wise "
            "averaging across all provided folds. The averaged weights "
            "generalize better than any single fold model and are the "
            "recommended input for --pretrained-weights."
        )
    )
    parser.arg(
        "--weights",
        nargs="+",
        required=True,
        metavar="CHECKPOINT",
        help="Paths to fold checkpoint files (.pt or .pth). Provide all folds "
             "from a cross-validation run, e.g. fold_0.pt fold_1.pt ...",
    )
    parser.arg(
        "--output",
        type=str,
        required=True,
        help="Output path for the averaged weights file (e.g. pretrained_init.pt).",
    )
    return parser.parse_args(argv)


def average_weights_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for the mist_average_weights command."""
    ns = _parse_args(argv)
    average_fold_weights(ns.weights, output_path=ns.output)
    print(f"Averaged weights from {len(ns.weights)} checkpoints saved to {ns.output}")


if __name__ == "__main__":
    average_weights_entry()  # pragma: no cover
