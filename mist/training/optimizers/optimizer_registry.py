"""Registry for optimizers used in training."""
from typing import Callable, Dict, Iterable, List
import torch
from torch.optim import Optimizer

# MIST imports.
from mist.training.optimizers.optimizer_constants import OptimizerConstants


def _adam_optimizer(
    params: Iterable,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> Optimizer:
    """Internal Adam optimizer."""
    return torch.optim.Adam(
        params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps
    )


def _adamw_optimizer(
    params: Iterable,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> Optimizer:
    """Internal AdamW optimizer."""
    return torch.optim.AdamW(
        params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps
    )


def _sgd_optimizer(
    params: Iterable,
    learning_rate: float,
    weight_decay: float,
    eps: float,  # pylint:disable=unused-argument
) -> Optimizer:
    """SGD optimizer."""
    return torch.optim.SGD(
        params=params,
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=OptimizerConstants.SGD_MOMENTUM,
        nesterov=OptimizerConstants.SGD_NESTEROV
    )


OPTIMIZER_REGISTRY: Dict[str, Callable[..., Optimizer]] = {
    "adam": _adam_optimizer,
    "adamw": _adamw_optimizer,
    "sgd": _sgd_optimizer,
}


def get_optimizer(name: str, params: Iterable, **kwargs) -> Optimizer:
    """Factory function for optimizers.

    Args:
        name: Optimizer name (adam, adamw, sgd).
        params: Model parameters to optimize.
        **kwargs: Optimizer-specific keyword arguments.

    Returns:
        torch.optim.Optimizer

    Raises:
        ValueError: If optimizer name is invalid.
    """
    name = name.lower()
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {list_optimizers()}"
        )
    return OPTIMIZER_REGISTRY[name](params, **kwargs)


def list_optimizers() -> List[str]:
    """Return the list of available optimizer names."""
    return sorted(OPTIMIZER_REGISTRY.keys())
