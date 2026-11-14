"""Initialize and register MIST's data loading backend."""

from mist.data_loading import data_loader_registry, generic_loader

# Pure CPU/NumPy patch reads; runs on CPU, NVIDIA, and AMD ROCm machines.
data_loader_registry.register_data_loader("generic", generic_loader)
