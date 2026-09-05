"""Entrypoint for finalizing a MIST training run.

Aggregates out-of-fold predictions into results.csv and, if a held-out test
set is configured, runs ensembled test-set inference -- the same steps
train_entry() runs automatically when a single invocation trains every
configured fold.

When folds are instead trained across separate per-node jobs sharing a
--results directory (one or a few folds per node, e.g. on an HPC cluster),
each node's mist_train invocation skips this tail on its own (see
train_entrypoint.py) rather than racing every other node to write the same
shared results.csv/evaluation_paths.csv/test predictions. This command is
what ties everything together: run it once, after every per-fold job has
finished, e.g. as a Slurm job with --dependency=afterok:<fold-job-ids>.
"""

import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd
import torch

# MIST imports.
from mist.cli import args as argmod
from mist.evaluation import evaluation_utils
from mist.evaluation.evaluator import Evaluator
from mist.inference import inference_utils
from mist.inference.inference_runners import infer_from_dataframe
from mist.utils import io
from mist.utils.console import (
    print_error,
    print_info,
    print_section_header,
    print_warning,
)


def _parse_finalize_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI for the finalize command.

    Falls back to ./results if not provided, matching the other entrypoints.
    """
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Finalize a MIST training run: aggregate out-of-fold predictions "
            "into results.csv and run held-out test-set inference, if "
            "configured. Safe to re-run -- each run recomputes both from "
            "whatever is currently in the results directory. Run this once, "
            "after every per-fold mist_train job sharing --results has "
            "finished (in particular, the one-fold-per-node pattern, where "
            "each node runs `mist_train --folds i`)."
        ),
    )
    parser.arg("--results", type=str, help="Path to output of the MIST pipeline.")
    parser.arg(
        "--num-workers-evaluate",
        type=argmod.positive_int,
        default=1,
        help="Number of parallel workers for evaluation.",
    )
    parser.arg(
        "--device",
        type=str,
        default="cuda",
        help=(
            "Device for held-out test-set inference (ignored if no test set "
            "is configured): 'cpu', 'cuda', or a CUDA index like '0'."
        ),
    )
    ns = parser.parse_args(argv)

    if not ns.results:
        ns.results = str(Path("./results").expanduser().resolve())
    return ns


def _check_folds_complete(results_dir: Path, config: dict) -> None:
    """Warn if any configured fold's model checkpoint is missing.

    Running mist_finalize before every per-node fold job has finished would
    silently produce a partial/wrong results.csv (and, worse, a wrong
    ensembled test-set prediction) -- exactly the failure mode this command
    exists to avoid. Warn rather than raise: a partial run may still be a
    deliberate, informed choice (e.g. checking in on progress).

    Args:
        results_dir: The MIST results directory.
        config: The run's config.json, already loaded.
    """
    nfolds = config["training"]["nfolds"]
    models_dir = results_dir / "models"
    missing = [fold for fold in range(nfolds) if not (models_dir / f"fold_{fold}.pt").is_file()]
    if missing:
        print_warning(
            f"Fold(s) {missing} have no saved model in {models_dir} -- "
            "results.csv (and test-set predictions, if configured) will "
            "only reflect the folds that have finished so far."
        )


def run_finalize(ns: argparse.Namespace) -> None:
    """Aggregate out-of-fold predictions and, optionally, run test inference.

    Args:
        ns: Namespace with `results`, `num_workers_evaluate`, and `device`
            (matching _parse_finalize_args' output, or an equivalent
            Namespace built by train_entry() for a single-invocation run).
    """
    results_dir = Path(ns.results).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    config_path = results_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = io.read_json_file(str(config_path))

    train_paths_csv = results_dir / "train_paths.csv"
    if not train_paths_csv.is_file():
        raise FileNotFoundError(f"Training paths file not found: {train_paths_csv}")

    print_section_header("Finalizing MIST run")
    _check_folds_complete(results_dir, config)

    # Evaluate out-of-fold predictions.
    filepaths_df, eval_warnings = evaluation_utils.build_evaluation_dataframe(
        train_paths_csv=str(train_paths_csv),
        prediction_folder=str(results_dir / "predictions" / "train" / "raw"),
    )
    if eval_warnings:
        print_warning(eval_warnings)

    if filepaths_df.empty:
        print_error("No valid prediction-mask pairs. Skipping evaluation.")
    else:
        evaluation_csv = results_dir / "evaluation_paths.csv"
        filepaths_df.to_csv(evaluation_csv, index=False)

        results_csv = results_dir / "results.csv"
        evaluator = Evaluator(
            filepaths_dataframe=filepaths_df,
            evaluation_config=config["evaluation"],
            output_csv_path=results_csv,
        )
        evaluator.run(max_workers=ns.num_workers_evaluate)
        print_info(f"Wrote {results_csv}")

    # Optional held-out test-set inference.
    test_paths_csv = results_dir / "test_paths.csv"
    if test_paths_csv.is_file():
        device = inference_utils.resolve_device(ns.device)
        test_df = pd.read_csv(test_paths_csv)
        test_pred_dir = results_dir / "predictions" / "test"
        with torch.no_grad():
            infer_from_dataframe(
                paths_dataframe=test_df,
                output_directory=str(test_pred_dir),
                mist_configuration=config,
                models_directory=str(results_dir / "models"),
                postprocessing_strategy_filepath=None,
                device=device,
            )
        print_info(f"Wrote test-set predictions to {test_pred_dir}")


def finalize_entry(argv: list[str] | None = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_finalize_args(argv)
    run_finalize(ns)


if __name__ == "__main__":
    finalize_entry()  # pragma: no cover
