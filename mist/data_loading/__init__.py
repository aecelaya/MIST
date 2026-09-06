"""Initialize and register all MIST data loading backends."""

from mist.data_loading import data_loader_registry, generic_loader

# Always available: pure CPU/NumPy, no NVIDIA dependency.
data_loader_registry.register_data_loader("generic", generic_loader)

# Only available when nvidia-dali-cuda120 is installed (CUDA hardware only).
# Guarded so importing mist.data_loading itself never requires DALI -- see
# cpu_rocm_support_plan.md Stage 2.
try:
    from mist.data_loading import dali_loader
except ImportError:
    pass
else:
    data_loader_registry.register_data_loader("dali", dali_loader)
