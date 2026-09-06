"""Data loader registry for managing training data loading backends in MIST.

Mirrors mist/models/model_registry.py's pattern, with one difference: a data
loading backend is a *module* exposing three functions with matching
signatures -- get_training_dataset, get_validation_dataset, get_test_dataset
(see dali_loader.py and generic_loader.py) -- rather than a single builder
function, so the whole module is registered as one unit instead of wrapping
each function separately with a decorator.
"""

from types import ModuleType

# Functions every registered data loader module must expose.
_REQUIRED_ATTRS = ("get_training_dataset", "get_validation_dataset", "get_test_dataset")

# Dictionary mapping data loader names to their backend module.
DATA_LOADER_REGISTRY: dict[str, ModuleType] = {}


def register_data_loader(name: str, module: ModuleType) -> None:
    """Register a data loading backend module under a unique name.

    Args:
        name: A unique string identifier for the loader (e.g., "dali",
            "generic").
        module: The loader module. Must expose get_training_dataset,
            get_validation_dataset, and get_test_dataset.

    Raises:
        ValueError: If name is already registered, or module is missing one
            of the three required functions.
    """
    if name in DATA_LOADER_REGISTRY:
        raise ValueError(f"Data loader '{name}' is already registered.")

    missing = [attr for attr in _REQUIRED_ATTRS if not callable(getattr(module, attr, None))]
    if missing:
        raise ValueError(
            f"Data loader module for '{name}' is missing required function(s): {missing}."
        )

    DATA_LOADER_REGISTRY[name] = module


def get_data_loader_from_registry(name: str) -> ModuleType:
    """Retrieve a registered data loader module by name.

    Args:
        name: Registered name of the data loader.

    Returns:
        The loader module, exposing get_training_dataset,
        get_validation_dataset, and get_test_dataset.

    Raises:
        ValueError: If the loader name is not registered.
    """
    if name not in DATA_LOADER_REGISTRY:
        raise ValueError(
            f"Data loader '{name}' is not registered.\n"
            f"Available data loaders: {sorted(DATA_LOADER_REGISTRY.keys())}"
        )
    return DATA_LOADER_REGISTRY[name]


def list_registered_data_loaders() -> list[str]:
    """List all available registered data loader names.

    Returns:
        A sorted list of registered data loader names.
    """
    return sorted(DATA_LOADER_REGISTRY.keys())
