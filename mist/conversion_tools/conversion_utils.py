"""Utilities for converting datasets to MIST format."""
import shutil
from pathlib import Path


def copy_image_from_source_to_dest(
    image_source: str | Path,
    image_destination: str | Path,
) -> None:
    """Copy image from source to destination.

    Args:
        image_source: Path to the source image.
        image_destination: Path to the destination file or directory.

    Returns:
        None
    """
    src = Path(image_source)
    dst = Path(image_destination)

    # Ensure destination directory exists.
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)
