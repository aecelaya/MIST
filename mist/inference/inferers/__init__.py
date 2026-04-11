"""Initialize and register all available inferers."""
# Explicitly import all inferer modules to trigger decorator-based registration.
from .sliding_window import SlidingWindowInferer  # noqa: F401
