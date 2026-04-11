"""Initialize and register all available ensemblers."""
# Import ensembler implementations to trigger registration decorators.
from .mean import MeanEnsembler  # noqa: F401
