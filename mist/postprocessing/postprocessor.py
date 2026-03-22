"""Postprocessor class for applying transforms to prediction masks."""
import concurrent.futures
import shutil
from pathlib import Path
from typing import Dict, List, Union, cast
import ants
import numpy as np
from rich.table import Table

# MIST imports.
from mist.utils import io, progress_bar
from mist.utils.console import (
    console,
    print_section_header,
    print_info,
    print_warning,
    print_success,
)
from mist.postprocessing.transform_registry import (
    get_transform,
    POSTPROCESSING_TRANSFORMS,
)
from mist.postprocessing.postprocessing_utils import StrategyStep


def _postprocess_single_file(
    input_path: Path,
    output_path: Path,
    transforms: List[str],
    apply_to_labels: List[List[int]],
    per_label: List[bool],
    transform_kwargs: List[Dict],
) -> List[str]:
    """Copy a mask to the output directory then apply all transforms.

    Copying first ensures the original is preserved in the output directory
    even if a transform fails partway through.

    Args:
        input_path: Path to the input mask file.
        output_path: Destination path for the postprocessed mask.
        transforms: List of transform names to apply.
        apply_to_labels: Label groups for each transform.
        per_label: Per-label application flags for each transform.
        transform_kwargs: Keyword arguments for each transform.

    Returns:
        A list of error/warning message strings (empty on full success).
    """
    # Copy first so the original is always present in output_dir.
    shutil.copy(input_path, output_path)

    patient_id = input_path.name.removesuffix(".nii.gz")
    mask = ants.image_read(str(input_path))
    messages = []

    for transform_name, per_label_flag, label_group, kwargs in zip(
        transforms, apply_to_labels, per_label, transform_kwargs
    ):
        try:
            transform_fn = get_transform(transform_name)
            updated_npy = transform_fn(
                mask.numpy().astype(np.uint8),
                labels_list=label_group,
                per_label=per_label_flag,
                **kwargs
            ).astype(np.uint8)
            mask = mask.new_image_like(updated_npy)  # type: ignore
        except ValueError as e:
            messages.append(
                f"[red]Error applying {transform_name} to {patient_id}: "
                f"{e}[/red]"
            )

    ants.image_write(mask, str(output_path))
    return messages


class Postprocessor:
    """Postprocessor class for applying postprocessing transforms to masks.

    This class is responsible for applying one or more user-defined transforms
    (e.g., morphological operations) to a set of prediction masks. The masks
    are discovered automatically in a prediction directory. The transformed
    masks are saved to an output directory.

    Attributes:
        transforms: List of transform names to apply.
        apply_to_labels: List of label groups to which transforms are applied.
        per_label: List of bools indicating per-label application.
        transform_kwargs: Keyword arguments to be passed to each transform.
    """

    def __init__(
        self,
        strategy_path: str,
    ):
        """Initialize a Postprocessor.

        Args:
            strategy_path: Path to a JSON file specifying the transform
                strategy.
        """
        # Load strategy file and extract transform details.
        strategy = self._load_strategy(strategy_path)
        self.transforms = [step["transform"] for step in strategy]
        self.apply_to_labels = [step["apply_to_labels"] for step in strategy]
        self.per_label = [step["per_label"] for step in strategy]
        self.transform_kwargs = [step.get("kwargs", {}) for step in strategy]

    def _load_strategy(self, strategy_path: str) -> List[StrategyStep]:
        """Load and validate the transform strategy from a JSON file.

        Args:
            strategy_path: Path to the strategy JSON file.

        Returns:
            Validated strategy as a list of StrategyStep.

        Raises:
            ValueError: If the strategy is not a list of valid steps.
        """
        # Load the raw strategy from the JSON file.
        raw_strategy = io.read_json_file(strategy_path)

        # Ensure the loaded data is a list of dicts.
        if not isinstance(raw_strategy, list):
            raise ValueError(
                "Strategy file must contain a list of strategy steps."
            )

        # Cast the raw strategy to a list of StrategyStep.
        # This is a type hinting step and does not perform any validation.
        strategy = cast(List[StrategyStep], raw_strategy)

        # Validate each step has required fields.
        for i, step in enumerate(strategy):
            if "transform" not in step:
                raise ValueError(f"Missing 'transform' key in step {i}.")
            if "apply_to_labels" not in step:
                raise ValueError(f"Missing 'apply_to_labels' key in step {i}.")
            if "per_label" not in step:
                raise ValueError(
                    f"Missing 'apply_sequentially' key in step {i}."
                )
            if (
                not isinstance(step["apply_to_labels"], list) or
                not all(isinstance(lbl, int) for lbl in step["apply_to_labels"])
            ):
                raise ValueError(
                    f"'apply_to_labels' in step {i} must be a list of integers."
                )
            if not isinstance(step["per_label"], bool):
                raise ValueError(
                    f"'apply_sequentially' in step {i} must be a boolean."
                )
            if step["transform"] not in POSTPROCESSING_TRANSFORMS:
                raise ValueError(
                    f"Unknown transform '{step['transform']}' in step {i}. "
                    f"Available transforms: "
                    f"{sorted(POSTPROCESSING_TRANSFORMS.keys())}."
                )
            if (
                step["transform"] == "replace_small_objects_with_label"
                and not step["per_label"]
            ):
                raise ValueError(
                    f"'replace_small_objects_with_label' in step {i} requires "
                    "'apply_sequentially': true."
                )
        return strategy

    def _gather_base_filepaths(self, base_dir: Path) -> List[Path]:
        """Gather all .nii.gz files from the base directory.

        We only want valid files so we iterate through all candidates and check
        if they exist/are files. If not, we skip them. This function also warns
        the user if any .nii.gz files were skipped.

        Args:
            base_dir: Directory to search for .nii.gz files.

        Returns:
            List of valid .nii.gz file paths.
        """
        all_candidates = [
            p for p in base_dir.iterdir() if p.name.endswith(".nii.gz")
        ]

        valid_files = []
        skipped_files = []
        for p in all_candidates:
            if p.is_file():
                valid_files.append(p)
            else:
                skipped_files.append(p.name)

        if skipped_files:
            print_warning(
                f"Skipped {len(skipped_files)} .nii.gz file(s) that are not "
                "valid files (e.g., directories or broken symlinks):"
            )
            for fname in skipped_files:
                print_info(f" - {fname}")
        return valid_files

    def _print_strategy(self) -> None:
        """Print the transform strategy in a formatted table."""
        print_section_header("Postprocessing predictions")

        table = Table(title="Strategy Summary", show_lines=True)
        table.add_column("Transform", style="bold")
        table.add_column("Per Label", justify="center")
        table.add_column("Target Labels", justify="center")

        for name, per_label_flag, label_group in zip(
            self.transforms, self.per_label, self.apply_to_labels
        ):
            labels = ', '.join(map(str, label_group))
            table.add_row(name, str(per_label_flag), labels)
        console.print(table)

    def apply_strategy_to_single_example(
            self,
            patient_id: str,
            mask: ants.core.ants_image.ANTsImage
    ) -> tuple[ants.core.ants_image.ANTsImage, List[str]]:
        """Apply all transforms in the strategy to a single ANTsImage mask.

        Args:
            patient_id: Unique identifier for the patient or example.
            mask: ANTsImage object to which transforms will be applied.

        Returns:
            A tuple of the transformed ANTsImage and a list of messages.
        """
        messages: List[str] = []

        for transform_name, per_label_flag, label_group, kwargs in zip(
            self.transforms,
            self.per_label,
            self.apply_to_labels,
            self.transform_kwargs
        ):
            try:
                transform_fn = get_transform(transform_name)
                updated_npy = transform_fn(
                    mask.numpy().astype(np.uint8),
                    labels_list=label_group,
                    per_label=per_label_flag,
                    **kwargs
                ).astype(np.uint8)
            except ValueError as e:
                messages.append(
                    f"[red]Error applying {transform_name} to {patient_id}: "
                    f"{e}[/red]"
                )
            else:
                mask = mask.new_image_like(updated_npy)  # type: ignore
        return mask, messages

    def run(
        self,
        base_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_workers: int = 1,
    ) -> None:
        """Apply strategy to all prediction masks in a base directory.

        Args:
            base_dir: Directory containing the prediction masks.
            output_dir: Directory where the postprocessed masks will be saved.
            num_workers: Number of parallel workers. Defaults to 1.
        """
        base_dir = Path(base_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_filepaths = self._gather_base_filepaths(base_dir)
        if not base_filepaths:
            print_warning(
                "No .nii.gz files found in base directory. "
                "Nothing to postprocess."
            )
            return
        self._print_strategy()

        messages = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_path = {
                executor.submit(
                    _postprocess_single_file,
                    fp,
                    output_dir / fp.name,
                    self.transforms,
                    self.apply_to_labels,
                    self.per_label,
                    self.transform_kwargs,
                ): fp
                for fp in base_filepaths
            }
            with progress_bar.get_progress_bar(
                "Applying strategy to examples"
            ) as pb:
                for future in pb.track(
                    concurrent.futures.as_completed(future_to_path),
                    total=len(base_filepaths),
                ):
                    try:
                        messages.extend(future.result())
                    except Exception as e:  # pylint: disable=broad-except
                        fp = future_to_path[future]
                        messages.append(
                            f"[red]Unexpected error processing "
                            f"{fp.name}: {e}[/red]"
                        )

        if messages:
            print_section_header(
                "Postprocessing completed with the following messages:"
            )
            for message in messages:
                print_info(message)
        else:
            print_success("Postprocessing completed successfully.")
