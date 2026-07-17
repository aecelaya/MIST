"""Initialize and register all available probability ensemblers."""

# Import probability ensembler implementations to trigger registration
# decorators.
from .mean import MeanProbabilityEnsembler  # noqa: F401
