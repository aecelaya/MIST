"""Command line tool for postprocessing predictions from MIST output."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Optional, List
import argparse
import shutil

import pandas as pd

# MIST imports.
from mist.postprocessing.postprocessor import Postprocessor
from mist.evaluation.evaluator import Evaluator
from mist.cli.args import ArgParser, positive_int
from mist.utils import io


def _parse_postprocess_args(
    argv: Optional[List[str]]=None
) -> argparse.Namespace:
    """Parse command line arguments for postprocessing predictions."""
    p = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Apply a postprocessing strategy to MIST predictions.",
    )

    # Required arguments.
    p.arg(
        "--base-predictions", type=str, required=True,
        help="Directory containing base predictions (e.g., NIfTI files)."
    )
    p.arg(
        "--output", type=str, required=True,
        help=(
            "Root output directory. Postprocessed masks are written to "
            "output/predictions/, the strategy is copied to "
            "output/strategy.json, and evaluation results (if requested) are "
            "written to output/postprocess_results.csv."
        ),
    )
    p.arg(
        "--postprocess-strategy", type=str, required=True,
        help="Path to JSON file specifying the postprocessing strategy."
    )
    p.arg(
        "--num-workers-postprocess", type=positive_int, default=1,
        help="Number of parallel workers for postprocessing.",
    )

    # Optional evaluation arguments.
    p.arg(
        "--paths-csv", type=str, required=False, default=None,
        help=(
            "CSV with columns 'id' and 'mask' containing patient IDs and "
            "paths to ground truth masks. When provided alongside "
            "--eval-config, evaluation is automatically run on the "
            "postprocessed predictions and results are saved to "
            "output/postprocess_results.csv."
        ),
    )
    p.arg(
        "--eval-config", type=str, required=False, default=None,
        help=(
            "Path to evaluation config JSON. Required when --paths-csv is "
            "provided. Accepts a full MIST config.json (the 'evaluation' key "
            "is extracted automatically) or a standalone evaluation config."
        ),
    )
    return p.parse_args(argv)


def _prepare_io(
    ns: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    """Validate inputs and create the output directory structure.

    Returns (base_dir, output_dir, predictions_dir, strategy_path).
    """
    base_dir = Path(ns.base_predictions).expanduser().resolve()
    output_dir = Path(ns.output).expanduser().resolve()
    strategy_path = Path(ns.postprocess_strategy).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base predictions directory not found: {base_dir}"
        )
    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Postprocess strategy file not found: {strategy_path}"
        )

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, output_dir, predictions_dir, strategy_path


def _validate_eval_args(ns: argparse.Namespace) -> None:
    """Raise if only one of --paths-csv / --eval-config is provided."""
    has_paths = ns.paths_csv is not None
    has_config = ns.eval_config is not None
    if has_paths != has_config:
        raise ValueError(
            "--paths-csv and --eval-config must both be provided together."
        )


def _build_eval_filepaths_df(
    paths_csv: Path,
    predictions_dir: Path,
) -> pd.DataFrame:
    """Build the filepaths DataFrame for evaluation.

    Reads a CSV with 'id' and 'mask' columns, then constructs the
    'prediction' column from the postprocessed files in predictions_dir.

    Args:
        paths_csv: CSV with 'id' and 'mask' columns.
        predictions_dir: Directory containing postprocessed predictions.

    Returns:
        DataFrame with 'id', 'mask', and 'prediction' columns.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    df = pd.read_csv(paths_csv)
    missing = [c for c in ("id", "mask") if c not in df.columns]
    if missing:
        raise ValueError(
            f"paths-csv is missing required column(s): {', '.join(missing)}"
        )
    df = df[["id", "mask"]].copy()
    df["prediction"] = df["id"].apply(
        lambda pid: str(predictions_dir / f"{pid}.nii.gz")
    )
    return df


def _run_evaluation_after_postprocess(
    ns: argparse.Namespace,
    output_dir: Path,
    predictions_dir: Path,
) -> None:
    """Run evaluation on the postprocessed predictions.

    Args:
        ns: Parsed argument namespace.
        output_dir: Root output directory (results CSV is written here).
        predictions_dir: Directory containing postprocessed predictions.
    """
    paths_csv = Path(ns.paths_csv).expanduser().resolve()
    eval_config_path = Path(ns.eval_config).expanduser().resolve()

    if not paths_csv.exists():
        raise FileNotFoundError(f"Paths CSV not found: {paths_csv}")
    if not eval_config_path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found: {eval_config_path}"
        )

    filepaths_df = _build_eval_filepaths_df(paths_csv, predictions_dir)

    full_config = io.read_json_file(str(eval_config_path))
    eval_config = full_config.get("evaluation", full_config)

    evaluator = Evaluator(
        filepaths_dataframe=filepaths_df,
        evaluation_config=eval_config,
        output_csv_path=output_dir / "postprocess_results.csv",
    )
    evaluator.run()


def run_postprocess(ns: argparse.Namespace) -> None:
    """Main runner for postprocessing."""
    _validate_eval_args(ns)
    base_dir, output_dir, predictions_dir, strategy_path = _prepare_io(ns)

    shutil.copy(strategy_path, output_dir / "strategy.json")

    postprocessor = Postprocessor(strategy_path=str(strategy_path))
    postprocessor.run(
        base_dir=base_dir,
        output_dir=predictions_dir,
        num_workers=ns.num_workers_postprocess,
    )

    if ns.paths_csv is not None:
        _run_evaluation_after_postprocess(ns, output_dir, predictions_dir)


def postprocess_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_postprocess_args(argv)
    run_postprocess(ns)


if __name__ == "__main__":
    postprocess_entry()  # pragma: no cover
