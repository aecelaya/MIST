"""Where each foreground class is in a case, for foreground-oversampled patches.

mist_preprocess finds these once per case and saves them next to the labels, in
a `foreground` folder, as nnU-Net does. The generic loader reads them instead of
scanning every full label in every worker, which is slow for large images such
as abdominal CT. Data preprocessed before this existed has no such folder, so
the loader falls back to scanning the label.

This module imports nothing from DALI, so the generic loader can use it on
machines without DALI.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Most voxel locations kept per class per case, as in nnU-Net. A uniform subset
# of a class's voxels still gives uniform picks, and keeps each case's locations
# to about 60 KB per class.
MAX_LOCATIONS = 10000

# Folder next to the preprocessed "labels" folder that holds the locations.
FOLDER = "foreground"

# Key for the fallback locations of any nonzero voxel, used when no configured
# class is present.
_ANY_KEY = "any"


def find(
    label: np.ndarray,
    labels: Sequence[int],
    rng: np.random.Generator,
) -> dict[int | None, np.ndarray]:
    """Find where each foreground class is in a case's (D, H, W, C) label.

    Args:
        label: The case's label array.
        labels: The configured foreground classes.
        rng: Random generator for subsampling large classes.

    Returns:
        Up to MAX_LOCATIONS (d, h, w) voxel coordinates per class present. If no
        configured class is present, the same for any nonzero voxel, under the
        key None. Empty if there is no foreground at all.
    """

    def subset(mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask)
        if coords.shape[0] > MAX_LOCATIONS:
            keep = rng.choice(coords.shape[0], MAX_LOCATIONS, replace=False)
            coords = coords[keep]
        return coords.astype(np.uint16)

    label_channel = np.asarray(label[..., 0])
    locations = {}
    for class_id in labels:
        coords = subset(label_channel == class_id)
        if coords.size:
            locations[class_id] = coords
    if not locations:
        coords = subset(label_channel != 0)
        if coords.size:
            locations[None] = coords
    return locations


def path_for(label_path: str | Path) -> Path:
    """Return where a preprocessed label's foreground locations are saved."""
    label_path = Path(label_path)
    return label_path.parent.parent / FOLDER / f"{label_path.stem}.npz"


def save(
    path: str | Path,
    locations: dict[int | None, np.ndarray],
    labels: Sequence[int],
) -> None:
    """Save a case's foreground locations, with the classes they were found for.

    Args:
        path: Where to save them, usually path_for(the label's path).
        locations: The output of find.
        labels: The classes find was given.
    """
    arrays = {(_ANY_KEY if key is None else str(key)): value for key, value in locations.items()}
    np.savez(path, labels=np.asarray(list(labels), dtype=np.int64), **arrays)


def load(path: str | Path, labels: Sequence[int]) -> dict[int | None, np.ndarray] | None:
    """Load a case's saved foreground locations.

    Args:
        path: Where they were saved.
        labels: The classes training samples from.

    Returns:
        The locations, as find returns them, or None if there is no file, it
        can't be read, or it was saved for different classes. Callers then scan
        the label instead.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path) as saved:
            if saved["labels"].tolist() != [int(label) for label in labels]:
                return None
            return {
                (None if key == _ANY_KEY else int(key)): saved[key]
                for key in saved.files
                if key != "labels"
            }
    except (OSError, ValueError, KeyError):
        return None
