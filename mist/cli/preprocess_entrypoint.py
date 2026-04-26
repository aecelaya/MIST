"""Entrypoint for the preprocessing pipeline."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
import argparse

# MIST imports.
from mist.cli import args as argmod
from mist.preprocessing import preprocess
from mist.utils.console import print_warning


def _parse_preprocess_args(
    argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse CLI for the preprocessing pipeline."""
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Preprocess a dataset into NumPy tensors for MIST.",
    )
    # Add preprocessing-specific args.
    argmod.add_preprocess_args(parser)

    ns = parser.parse_args(argv)

    # Enforce/derive required paths for this pipeline.
    if not ns.results:
        parser.error(
            "--results is required (directory containing config.json, "
            "train_paths.csv, fg_bboxes.csv)."
        )

    # Default NumPy output to ./numpy (under the current working directory)
    # if not provided.
    if not ns.numpy:
        ns.numpy = str(Path("./numpy").expanduser().resolve())
    return ns


def _ensure_analyze_artifacts(ns: argparse.Namespace) -> Path:
    """Verify the results dir exists and has analyzer outputs."""
    results_dir = Path(ns.results).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    required = ["config.json", "train_paths.csv", "fg_bboxes.csv"]
    missing = [f for f in required if not (results_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) produced by analyze: "
            + ", ".join(missing)
            + f" (in {results_dir})"
        )
    return results_dir


def _prepare_preprocess_dirs(ns: argparse.Namespace) -> None:
    """Validate and create the numpy directory for saving preprocessed data."""
    numpy_dir = Path(ns.numpy).expanduser().resolve()

    # If the target exists and is non-empty, require --overwrite.
    if numpy_dir.exists():
        # Consider “non-empty” to be having any file/dir inside.
        non_empty = any(numpy_dir.iterdir())
        if non_empty and not getattr(ns, "overwrite", False):
            raise FileExistsError(
                f"Destination {numpy_dir} already contains files. "
                "Use --overwrite or choose a different --numpy directory."
            )
        if non_empty and getattr(ns, "overwrite", False):
            print_warning(
                f"Overwriting existing preprocessed data in {numpy_dir}"
            )

    numpy_dir.mkdir(parents=True, exist_ok=True)


def preprocess_entry(argv: list[str] | None = None) -> None:
    """Entrypoint for the preprocess command."""
    ns = _parse_preprocess_args(argv)
    _ensure_analyze_artifacts(ns)
    _prepare_preprocess_dirs(ns)
    preprocess.preprocess_dataset(ns)


if __name__ == "__main__":
    preprocess_entry()  # pragma: no cover
