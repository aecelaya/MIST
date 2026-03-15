"""Shared test helpers for tests/analyze_data."""
import numpy as np
import ants


class FakePB:
    """Minimal progress-bar context manager that yields items unchanged."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def track(self, it):
        """Yield items from the given iterable."""
        return it


def fake_get_progress_bar(_text: str) -> FakePB:
    """Return a FakePB instance regardless of the text argument."""
    return FakePB()


def make_ants_image(
    shape=(10, 10, 10),
    spacing=(1.0, 1.0, 1.0),
    fill=1.0,
) -> ants.ANTsImage:
    """Create an ANTs image filled with *fill* at the given *shape* and
    *spacing*."""
    arr = np.full(shape, fill, dtype=np.float32)
    img = ants.from_numpy(arr)
    img.set_spacing(spacing)
    return img
