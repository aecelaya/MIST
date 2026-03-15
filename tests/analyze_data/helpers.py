"""Shared test helpers for tests/analyze_data."""
import types
from typing import Any, Self

import numpy as np
import ants


class FakePB:
    """Minimal progress-bar context manager that yields items unchanged."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        return False

    def track(self, it: Any, total: int | None = None) -> Any:
        """Yield items from the given iterable."""
        return it


def fake_get_progress_bar(_text: str) -> FakePB:
    """Return a FakePB instance regardless of the text argument."""
    return FakePB()


def make_ants_image(
    shape: tuple[int, int, int] = (10, 10, 10),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill: float = 1.0,
) -> ants.ANTsImage:
    """Create an ANTs image filled with *fill* at the given *shape* and
    *spacing*."""
    arr = np.full(shape, fill, dtype=np.float32)
    img = ants.from_numpy(arr)
    img.set_spacing(spacing)
    return img
