"""Run analyze, preprocess, and train in one command."""
import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional

# MIST imports.
from mist.cli import args as argmod
from mist.cli.analyze_entrypoint import analyze_entry
from mist.cli.preprocess_entrypoint import preprocess_entry
from mist.cli.train_entrypoint import train_entry


def _ns_to_argv(ns: argparse.Namespace, keys: List[str]) -> List[str]:
    """Convert a subset of Namespace fields into an argv list."""
    argv: List[str] = []
    for k in keys:
        if not hasattr(ns, k):
            continue
        v = getattr(ns, k)
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv.extend([flag, str(v)])
    return argv


def _parse_run_all_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build and parse args for the run-all entrypoint."""
    parser = argmod.ArgParser(
        prog="mist_run_all",
        description="Run Analyzer → Preprocess → Train in one go.",
        conflict_handler="resolve",  # critical: resolve duplicates
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Add all stage args directly into this parser (no parents).
    argmod.add_analyzer_args(parser)
    argmod.add_preprocess_args(parser)
    argmod.add_train_args(parser)

    ns = parser.parse_args(argv)

    # Provide defaults expected downstream.
    if not getattr(ns, "results", None):
        ns.results = str(Path("./results").expanduser().resolve())
    if not getattr(ns, "numpy", None):
        ns.numpy = str(Path("./numpy").expanduser().resolve())

    # Minimal top-level validation.
    if not getattr(ns, "data", None):
        parser.error("Missing required argument for analyze: --data")
    return ns


def run_all_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for running analyze, preprocess, and train sequentially."""
    ns = _parse_run_all_args(argv)

    analyzer_keys = [
        "data", "results", "nfolds", "num_workers_analyze", "overwrite"
    ]
    preprocess_keys = [
        "results", "numpy", "num_workers_preprocess",
        "no_preprocess", "compute_dtms", "overwrite"
    ]
    train_keys = [
        "results", "numpy",
        "gpus",
        "model", "patch_size",
        "loss", "use_dtms", "composite_loss_weighting",
        "epochs", "batch_size_per_gpu", "learning_rate",
        "lr_scheduler", "optimizer", "l2_penalty",
        "folds", "num_workers_evaluate", "overwrite",
    ]

    # Run each stage with the appropriate subset of args.
    analyze_entry(_ns_to_argv(ns, analyzer_keys))
    preprocess_entry(_ns_to_argv(ns, preprocess_keys))
    train_entry(_ns_to_argv(ns, train_keys))


if __name__ == "__main__":
    run_all_entry()  # pragma: no cover
