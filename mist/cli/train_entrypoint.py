"""Entrypoint for running the MIST training pipeline."""

import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

# MIST imports.
from mist.cli import args as argmod
from mist.cli.finalize_entrypoint import run_finalize
from mist.inference.inference_runners import test_on_fold
from mist.training.trainers.patch_3d_trainer import Patch3DTrainer
from mist.utils import io
from mist.utils.console import print_info


def _parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI for the training pipeline.

    Falls back to ./results and ./numpy if not provided, then downstream
    functions validate that the expected artifacts exist.
    """
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="MIST training pipeline.",
    )
    # Add training-specific args.
    argmod.add_train_args(parser)
    ns = parser.parse_args(argv)

    # Fallbacks for convenience/consistency with other entrypoints.
    if not ns.results:
        ns.results = str(Path("./results").expanduser().resolve())
    if not ns.numpy:
        ns.numpy = str(Path("./numpy").expanduser().resolve())
    return ns


def _ensure_required_artifacts(ns: argparse.Namespace) -> tuple[Path, bool]:
    """Verify results & numpy folders contain the expected structure.

    Returns:
        Tuple[Path, bool]: (results_dir, has_test_paths)
    """
    results_dir = Path(ns.results).expanduser().resolve()
    numpy_dir = Path(ns.numpy).expanduser().resolve()

    # Results folder must already exist and contain required files.
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    required_files = ["config.json", "train_paths.csv", "fg_bboxes.csv"]
    missing = [f for f in required_files if not (results_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in results directory: "
            + ", ".join(missing)
            + f" (in {results_dir})"
        )

    has_test_paths = (results_dir / "test_paths.csv").is_file()

    # NumPy directory must exist with required subfolders.
    if not numpy_dir.is_dir():
        raise FileNotFoundError(f"NumPy directory does not exist: {numpy_dir}")

    required_np_subdirs = ["images", "labels"]  # 'dtms' is optional.
    missing_np = [d for d in required_np_subdirs if not (numpy_dir / d).is_dir()]
    if missing_np:
        raise FileNotFoundError(
            "Missing required subfolder(s) in NumPy directory: "
            + ", ".join(missing_np)
            + f" (in {numpy_dir})"
        )
    return results_dir, has_test_paths


def _create_train_dirs(results_dir: Path, has_test_paths: bool) -> None:
    """Create output directories inside results for logs/models/predictions."""
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)

    train_pred_dir = results_dir / "predictions" / "train" / "raw"
    train_pred_dir.mkdir(parents=True, exist_ok=True)
    if has_test_paths:
        test_pred_dir = results_dir / "predictions" / "test"
        test_pred_dir.mkdir(parents=True, exist_ok=True)


def train_entry(argv: list[str] | None = None) -> None:
    """Entrypoint for the training command."""
    ns = _parse_train_args(argv)

    # Validate artifacts from analyze + preprocess
    results_dir, has_test_paths = _ensure_required_artifacts(ns)

    # --resume and --overwrite are mutually exclusive.
    if getattr(ns, "resume", False) and getattr(ns, "overwrite", False):
        raise ValueError(
            "--resume and --overwrite are mutually exclusive. "
            "Use --resume to continue an interrupted run, or "
            "--overwrite to start a fresh run."
        )

    # Avoid accidental overwrite of an existing results.csv.
    results_csv = results_dir / "results.csv"
    if results_csv.exists() and not getattr(ns, "overwrite", False):
        raise FileExistsError(
            f"Found existing results at {results_csv}. Use --overwrite to replace them."
        )

    _create_train_dirs(results_dir, has_test_paths)

    # Determine which folds THIS invocation is responsible for before
    # training starts, and before any other invocation sharing this
    # --results directory (e.g. one-fold-per-node runs on an HPC cluster)
    # can mutate config.json's own "folds" key: BaseTrainer persists a
    # --folds override there, so reading it back *after* training would
    # race against every other node doing the same to the same shared file.
    # "nfolds" itself is never touched by mist_train, so it's safe to trust
    # even after training.
    initial_config = io.read_json_file(str(results_dir / "config.json"))
    nfolds = initial_config["training"]["nfolds"]
    own_folds = (
        [int(fold) for fold in ns.folds]
        if getattr(ns, "folds", None) is not None
        else list(initial_config["training"]["folds"])
    )
    covers_all_folds = set(own_folds) == set(range(nfolds))

    # Train
    trainer = Patch3DTrainer(ns)
    trainer.fit()

    # Post-training: generate out-of-fold predictions for the folds this
    # invocation trained. Each fold's predictions land at disjoint filenames
    # (train_paths.csv assigns each case to exactly one fold), so this is
    # safe to run concurrently with other invocations sharing --results.
    for fold in own_folds:
        test_on_fold(ns, fold)

    if not covers_all_folds:
        # This invocation only trained a subset of the configured folds (the
        # one-fold-per-node pattern). Evaluating results.csv/test-set
        # inference now would mean reading whatever the *other* folds'
        # predictions/models happen to look like at this exact moment --
        # possibly still missing, mid-write, or not started at all. Defer to
        # mist_finalize, run once after every per-fold job has finished.
        print_info(
            f"Trained fold(s) {own_folds} of {nfolds} total. Once every "
            f"fold has finished, run `mist_finalize --results {results_dir}` "
            "to produce the final results.csv"
            + (" and test-set predictions." if has_test_paths else ".")
        )
        return

    # This invocation trained every configured fold itself: finalize now.
    run_finalize(
        argparse.Namespace(
            results=str(results_dir),
            num_workers_evaluate=ns.num_workers_evaluate,
            device="cuda",
        )
    )


if __name__ == "__main__":
    train_entry()  # pragma: no cover
