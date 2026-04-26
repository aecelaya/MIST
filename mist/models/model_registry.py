"""Model registry for managing architecture creation in MIST."""
from collections.abc import Callable

# Dictionary mapping model names to builder functions.
MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model-building function.

    Args:
        name: A unique string identifier for the model (e.g., "nnunet").

    Returns:
        The original function, unmodified.
    """
    def decorator(fn: Callable) -> Callable:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model_from_registry(model_name: str, **kwargs):
    """
    Construct a model from the registry.

    Args:
        model_name: Registered name of the model.
        **kwargs: Keyword arguments passed to the model-building function.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If the model name is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered.\n"
            f"Available models: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


def list_registered_models() -> list[str]:
    """
    List all available registered models.

    Returns:
        A sorted list of registered model names.
    """
    return sorted(MODEL_REGISTRY.keys())
