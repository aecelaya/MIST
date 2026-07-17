"""Command line tool MIST inference on a given dataset."""

import argparse
import warnings
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd
import torch

# MIST imports
from mist.cli.args import ArgParser
from mist.inference import inference_runners, inference_utils
from mist.utils import io


def _parse_inference_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for MIST inference."""
    p = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Run MIST inference using trained models.",
    )

    # Required.
    p.arg(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing saved models (e.g., results/models/).",
    )
    p.arg(
        "--config",
        type=str,
        required=True,
        help="Path to config.json from a MIST training run.",
    )
    p.arg(
        "--paths-csv",
        type=str,
        required=True,
        help=(
            "CSV with an 'id' column and one column per image type matching "
            "the dataset's image keys (e.g., 't1', 't2'). See docs for the "
            "required format."
        ),
    )
    p.arg(
        "--output",
        type=str,
        required=True,
        help="Directory to write predictions (NIfTI).",
    )

    # Optional.
    p.arg(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on: 'cpu', 'cuda', or a CUDA index like '0'.",
    )
    p.arg(
        "--postprocess-strategy",
        type=str,
        default=None,
        help="Path to postprocessing strategy JSON file.",
    )
    p.flag(
        "--output-probs",
        help=(
            "Also write each prediction's final (post fold/TTA-ensemble, "
            "pre-argmax) probability volume to '<output>/probabilities/'. "
            "When set, the discrete prediction is written to "
            "'<output>/discrete/' instead of directly under '<output>/'. "
            "Use this to later combine probabilities across multiple models "
            "with 'mist_ensemble --input-type probabilities'."
        ),
    )
    return p.parse_args(argv)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string (e.g. 'cpu', 'cuda', '0') to a torch.device."""
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Numeric (CUDA index) case.
    try:
        idx = int(device_str)
    except ValueError as e:
        raise ValueError(f"Invalid device specification: {device_str}") from e

    if torch.cuda.is_available() and idx < torch.cuda.device_count():
        return torch.device(f"cuda:{idx}")

    warnings.warn(f"CUDA device {idx} not available; falling back to CPU.", stacklevel=2)
    return torch.device("cpu")


def _prepare_io(ns: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Validate inputs and create output directory.

    Returns (models_dir, config, paths_csv, output_dir).
    """
    models_dir = Path(ns.models_dir).expanduser().resolve()
    config = Path(ns.config).expanduser().resolve()
    paths_csv = Path(ns.paths_csv).expanduser().resolve()
    output_dir = Path(ns.output).expanduser().resolve()

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")
    if not paths_csv.exists():
        raise FileNotFoundError(f"Paths CSV not found: {paths_csv}")

    # Optional postprocess strategy
    if ns.postprocess_strategy is not None:
        pps = Path(ns.postprocess_strategy).expanduser().resolve()
        if not pps.exists():
            raise FileNotFoundError(f"Postprocess strategy file not found: {pps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, config, paths_csv, output_dir


def run_inference(ns: argparse.Namespace) -> None:
    """Main runner for MIST inference."""
    device = _resolve_device(ns.device)
    models_dir, config_path, paths_csv, output_dir = _prepare_io(ns)

    # Load & validate inputs
    df = pd.read_csv(paths_csv)
    inference_utils.validate_paths_dataframe(df)  # Raises if invalid.
    mist_cfg = io.read_json_file(str(config_path))

    # Execute inference
    inference_runners.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=str(output_dir),
        mist_configuration=mist_cfg,
        models_directory=str(models_dir),
        postprocessing_strategy_filepath=(
            str(Path(ns.postprocess_strategy).expanduser().resolve())
            if ns.postprocess_strategy is not None
            else None
        ),
        device=device,
        output_probs=ns.output_probs,
    )


def inference_entry(argv: list[str] | None = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_inference_args(argv)
    run_inference(ns)


if __name__ == "__main__":
    inference_entry()  # pragma: no cover
