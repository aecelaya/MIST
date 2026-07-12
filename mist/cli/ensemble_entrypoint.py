"""Command line tool to ensemble predictions from multiple MIST models."""

import argparse
import concurrent.futures
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import SimpleITK as sitk

from mist.cli.args import ArgParser, positive_int
from mist.inference.label_ensemblers.label_ensembler_registry import (
    get_label_ensembler,
    list_label_ensemblers,
)
from mist.utils import progress_bar
from mist.utils.console import print_error, print_success


def _parse_ensemble_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for mist_ensemble."""
    parser = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Combine predictions from multiple MIST models into a single "
            "consensus segmentation."
        ),
    )
    parser.arg(
        "--predictions",
        nargs="+",
        required=True,
        help=(
            "Two or more directories, each containing NIfTI predictions "
            "(one file per patient, named <patient_id>.nii.gz). All "
            "directories must contain the same set of patient files."
        ),
    )
    parser.arg(
        "--output",
        type=str,
        required=True,
        help="Directory where the consensus predictions will be written.",
    )
    parser.arg(
        "--ensemble-backend",
        type=str,
        choices=list_label_ensemblers(),
        default="staple",
        help="Algorithm used to combine label maps.",
    )
    parser.arg(
        "--num-workers-ensemble",
        type=positive_int,
        default=1,
        help="Number of parallel workers for ensembling.",
    )
    return parser.parse_args(argv)


def _validate_prediction_dirs(prediction_dirs: list[str]) -> list[Path]:
    """Resolve and validate that all prediction directories exist.

    Args:
        prediction_dirs: List of directory path strings.

    Returns:
        List of resolved Path objects.

    Raises:
        FileNotFoundError: If any directory does not exist.
        ValueError: If fewer than two directories are provided.
    """
    if len(prediction_dirs) < 2:
        raise ValueError("mist_ensemble requires at least two prediction directories.")
    resolved = []
    for d in prediction_dirs:
        path = Path(d).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {path}")
        resolved.append(path)
    return resolved


def _get_patient_ids(dirs: list[Path]) -> list[str]:
    """Extract and validate patient IDs across all prediction directories.

    Patient IDs are inferred from the NIfTI filenames in the first directory.
    All subsequent directories must contain exactly the same set of files.

    Args:
        dirs: List of resolved prediction directory paths.

    Returns:
        Sorted list of patient ID strings (filename stems without .nii.gz).

    Raises:
        ValueError: If patient IDs do not match across directories.
    """
    reference_ids = {p.name.replace(".nii.gz", "") for p in dirs[0].glob("*.nii.gz")}
    for d in dirs[1:]:
        ids = {p.name.replace(".nii.gz", "") for p in d.glob("*.nii.gz")}
        if ids != reference_ids:
            missing = reference_ids - ids
            extra = ids - reference_ids
            msg_parts = []
            if missing:
                msg_parts.append(f"missing from {d}: {sorted(missing)}")
            if extra:
                msg_parts.append(f"extra in {d}: {sorted(extra)}")
            raise ValueError(
                "Patient IDs do not match across prediction directories. "
                + "; ".join(msg_parts)
            )
    return sorted(reference_ids)


def _ensemble_single_patient(
    patient_id: str,
    dirs: list[Path],
    ensemble_backend: str,
    output_dir: Path,
) -> str | None:
    """Ensemble one patient's label maps and write the consensus to disk.

    Rebuilds the ensembler from its registry name rather than accepting an
    instance, so this function can be sent to a worker process.

    Args:
        patient_id: Patient identifier shared across all prediction dirs.
        dirs: Resolved prediction directories.
        ensemble_backend: Name of the registered label ensembler to use.
        output_dir: Directory where the consensus prediction is written.

    Returns:
        An error message string on failure, or None on success.
    """
    try:
        ensembler = get_label_ensembler(ensemble_backend)
        label_maps = [
            sitk.ReadImage(str(d / f"{patient_id}.nii.gz")) for d in dirs
        ]
        consensus = ensembler(label_maps)
        sitk.WriteImage(
            consensus,
            str(output_dir / f"{patient_id}.nii.gz"),
        )
        return None
    except Exception as e:  # pylint: disable=broad-except
        return f"Ensemble failed for {patient_id}: {str(e)}"


def run_ensemble(ns: argparse.Namespace) -> None:
    """Load inputs, run the ensemble, and write output predictions.

    Args:
        ns: Parsed argument namespace from _parse_ensemble_args.
    """
    dirs = _validate_prediction_dirs(ns.predictions)
    patient_ids = _get_patient_ids(dirs)

    output_dir = Path(ns.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    error_messages = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=ns.num_workers_ensemble
    ) as executor:
        future_to_patient = {
            executor.submit(
                _ensemble_single_patient,
                patient_id,
                dirs,
                ns.ensemble_backend,
                output_dir,
            ): patient_id
            for patient_id in patient_ids
        }

        with progress_bar.get_progress_bar("Ensembling predictions") as pb:
            for future in pb.track(
                concurrent.futures.as_completed(future_to_patient),
                total=len(patient_ids),
            ):
                error_message = future.result()
                if error_message:
                    error_messages.append(error_message)

    if error_messages:
        for message in error_messages:
            print_error(message)
    else:
        print_success("Ensemble completed successfully.")


def ensemble_entry(argv: list[str] | None = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_ensemble_args(argv)
    run_ensemble(ns)


if __name__ == "__main__":
    ensemble_entry()  # pragma: no cover
