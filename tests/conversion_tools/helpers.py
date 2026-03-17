"""Shared test helpers for tests/conversion_tools."""


class DummyProgressBar:
    """A dummy progress bar that passes items through unchanged."""

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, *args):
        """Exit the context manager."""

    def track(self, iterable, total=None):
        """Yield items from the iterable without any progress tracking."""
        return iterable
