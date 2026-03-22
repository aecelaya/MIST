"""Utility functions for MIST trainers."""
from pathlib import Path
from typing import List, Sequence, Union


class RunningMean:
    """Simple moving average for loss tracking.

    This class tracks the mean of a series of values (e.g., loss values) over
    time. It is reset after each epoch.

    Attributes:
        count: Number of values added.
        total: Sum of values added.
    """
    def __init__(self):
        self.count = 0
        self.total = 0

    def __call__(self, loss: float) -> float:
        """Update the mean with a new loss value."""
        self.total += loss
        self.count += 1
        return self.result()

    def result(self) -> float:
        """Return the current mean."""
        return self.total / self.count if self.count != 0 else 0.0

    def reset_states(self):
        """Reset the mean tracker."""
        self.count = 0
        self.total = 0


def sanity_check_fold_data(
    fold: int,
    train_images: Sequence[Union[str, Path]],
    train_labels: Sequence[Union[str, Path]],
    val_images: Sequence[Union[str, Path]],
    val_labels: Sequence[Union[str, Path]],
    train_dtms: Sequence[Union[str, Path]],
) -> None:
    """Sanity check the fold data for training and validation.

    Ensures:
      - Non-empty train/val sets.
      - 1:1 pairing (counts) within train and within val.
      - No duplicates within each set.
      - No train/val leakage (images and labels).
      - If train_dtms is provided, they exist, match train count, have no
        duplicates, and their file stems align with train_images.
      - Filenames (stems) align across modalities (img/label and optionally DTM).

    Args:
        fold: The fold number being checked.
        train_images: List of training image paths.
        train_labels: List of training label paths.
        val_images: List of validation image paths.
        val_labels: List of validation label paths.
        train_dtms: List of training DTM paths, or None if not using DTMs.

    Raises:
        ValueError: If any of the checks fail, providing details on the issue.
    """
    def _normalize(paths: Sequence[Union[str, Path]]) -> List[str]:
        return [str(Path(p).expanduser().resolve()) for p in paths]

    # Normalize everything to absolute string paths
    tr_img = _normalize(train_images)
    tr_lbl = _normalize(train_labels)
    va_img = _normalize(val_images)
    va_lbl = _normalize(val_labels)
    tr_dtm = _normalize(train_dtms) if train_dtms is not None else None

    n_tr_img, n_tr_lbl = len(tr_img), len(tr_lbl)
    n_va_img, n_va_lbl = len(va_img), len(va_lbl)

    # Non-empty sets.
    if min(n_tr_img, n_tr_lbl, n_va_img, n_va_lbl) == 0:
        raise ValueError(
            f"Fold {fold} has empty data after split. "
            f"counts: train_images={n_tr_img}, train_labels={n_tr_lbl}, "
            f"val_images={n_va_img}, val_labels={n_va_lbl}"
        )

    # 1:1 pairing within train and within val.
    if n_tr_img != n_tr_lbl:
        raise ValueError(
            f"Fold {fold} mismatch: train_images ({n_tr_img}) "
            f"!= train_labels ({n_tr_lbl})"
        )
    if n_va_img != n_va_lbl:
        raise ValueError(
            f"Fold {fold} mismatch: val_images ({n_va_img}) "
            f"!= val_labels ({n_va_lbl})"
        )

    # No duplicates inside each split.
    if len(set(tr_img)) != n_tr_img:
        raise ValueError(f"Fold {fold} has duplicate entries in train_images.")
    if len(set(tr_lbl)) != n_tr_lbl:
        raise ValueError(f"Fold {fold} has duplicate entries in train_labels.")
    if len(set(va_img)) != n_va_img:
        raise ValueError(f"Fold {fold} has duplicate entries in val_images.")
    if len(set(va_lbl)) != n_va_lbl:
        raise ValueError(f"Fold {fold} has duplicate entries in val_labels.")

    # No train/val overlap (data leakage) — check both images and labels.
    overlap_img = set(tr_img) & set(va_img)
    overlap_lbl = set(tr_lbl) & set(va_lbl)
    if overlap_img:
        example = next(iter(overlap_img))
        raise ValueError(
            f"Fold {fold} train/val overlap in images ({len(overlap_img)} "
            f"files), e.g.: {example}"
        )
    if overlap_lbl:
        example = next(iter(overlap_lbl))
        raise ValueError(
            f"Fold {fold} train/val overlap in labels ({len(overlap_lbl)} "
            f"files), e.g.: {example}"
        )

    # Optional: ensure file stems (patient ids) align within each split.
    def _stems(paths: Sequence[str]) -> List[str]:
        return [Path(p).stem for p in paths]

    if _stems(tr_img) != _stems(tr_lbl):
        raise ValueError(
            f"Fold {fold} image/label stem mismatch in training set."
        )
    if _stems(va_img) != _stems(va_lbl):
        raise ValueError(
            f"Fold {fold} image/label stem mismatch in validation set."
        )

    # If DTMs are provided, ensure they align with training images.
    if tr_dtm is not None:
        if len(tr_dtm) != n_tr_img:
            raise ValueError(
                f"Fold {fold} mismatch: train_dtms ({len(tr_dtm)}) "
                f"!= train_images ({n_tr_img})"
            )
        if len(set(tr_dtm)) != len(tr_dtm):
            raise ValueError(f"Fold {fold} has duplicate entries in DTMs.")
        if _stems(tr_img) != _stems(tr_dtm):
            raise ValueError(
                f"Fold {fold} image/DTM stem mismatch in training set."
            )


def get_npy_paths(
    data_dir: Union[str, Path],
    patient_ids: Sequence[Union[str, Path]],
    *,
    suffix: str = ".npy",
    must_exist: bool = True,
) -> List[str]:
    """Get paths to .npy files for given patient IDs.

    Args:
        data_dir: Directory containing the .npy files.
        patient_ids: List of patient IDs to construct file paths.
        suffix: File suffix to append to each patient ID (default: ".npy").
        must_exist: If True, raise FileNotFoundError if any expected file is
            missing.

    Returns:
        List of absolute paths to the .npy files for the given patient IDs.

    Raises:
        FileNotFoundError: If must_exist is True and any expected file does not
            exist.
    """
    base = Path(data_dir).expanduser().resolve()
    paths = [str((base / f"{pid}{suffix}").resolve()) for pid in patient_ids]
    if must_exist:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            more = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"Missing {len(missing)} expected files under {base}: "
                f"{preview}{more}"
            )
    return paths
