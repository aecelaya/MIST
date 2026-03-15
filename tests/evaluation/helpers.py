"""Shared test helpers for tests/evaluation."""
import concurrent.futures

import ants
import numpy as np


class FakePB:
    """Minimal progress-bar context manager that yields items unchanged."""

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, *_):
        """Exit the context manager."""
        return False

    def track(self, it, **kwargs):
        """Yield items from the given iterable."""
        return it


def fake_get_progress_bar(_text: str) -> FakePB:
    """Return a FakePB regardless of the text argument."""
    return FakePB()


class FakeExecutor:
    """Synchronous executor that runs submitted tasks in the calling process.

    Replaces ProcessPoolExecutor in tests so that monkeypatching applied in
    the main process is visible inside the 'worker'.
    """

    def __init__(self, max_workers=None):
        """Accept and ignore max_workers to match the real API."""

    def __enter__(self):
        """Return self as the context manager."""
        return self

    def __exit__(self, *_):
        """Exit without suppressing exceptions."""
        return False

    def submit(self, fn, *args, **kwargs):
        """Run *fn* immediately and return a completed Future."""
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pylint: disable=broad-except
            future.set_exception(exc)
        return future


def make_ants_image(
    shape=(10, 10, 10),
    spacing=(1.0, 1.0, 1.0),
    fill=1.0,
) -> ants.ANTsImage:
    """Create an ANTs image filled with *fill* at the given shape and spacing."""
    arr = np.full(shape, fill, dtype=np.float32)
    img = ants.from_numpy(arr)
    img.set_spacing(spacing)
    return img


def make_eval_config(classes=None) -> dict:
    """Return a minimal valid evaluation_config in the new nested format.

    Args:
        classes: Mapping of class name → label list. Defaults to
            ``{"tumor": [1]}``.

    Returns:
        A dict suitable for passing to ``Evaluator`` or
        ``initialize_results_dataframe``.
    """
    if classes is None:
        classes = {"tumor": [1]}
    return {
        name: {"labels": labels, "metrics": {"dice": {}}}
        for name, labels in classes.items()
    }
