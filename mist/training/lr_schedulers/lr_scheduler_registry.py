"""Registry for learning rate schedulers used in training."""
from typing import Callable, Dict, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# MIST imports.
from mist.training.lr_schedulers.lr_schedulers_constants import (
    LRSchedulerConstants
)


def _cosine_scheduler(optimizer: Optimizer, epochs: int) -> LRScheduler:
    """Cosine annealing LR schedule."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _polynomial_scheduler(optimizer: Optimizer, epochs: int) -> LRScheduler:
    """Polynomial decay LR schedule."""
    return torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=epochs,
        power=LRSchedulerConstants.POLYNOMIAL_DECAY,
    )


def _constant_scheduler(optimizer: Optimizer, epochs: int) -> LRScheduler:
    """Constant learning rate schedule."""
    return torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=LRSchedulerConstants.CONSTANT_LR_FACTOR
    )


LR_SCHEDULER_REGISTRY: Dict[str, Callable[..., LRScheduler]] = {
    "cosine": _cosine_scheduler,
    "polynomial": _polynomial_scheduler,
    "constant": _constant_scheduler,
}


def get_lr_scheduler(
    name: str,
    optimizer: Optimizer,
    epochs: int,
    warmup_epochs: int = 0,
) -> LRScheduler:
    """Factory function for learning rate schedulers.

    When warmup_epochs > 0, a linear warmup phase is prepended to the
    requested schedule. The warmup ramps the LR from
    LRSchedulerConstants.WARMUP_START_FACTOR × base_lr up to base_lr over
    warmup_epochs steps. The main scheduler then runs for the remaining
    epochs - warmup_epochs steps so the full decay budget is preserved.

    Args:
        name: Scheduler name (cosine, polynomial, constant).
        optimizer: The optimizer whose LR will be scheduled.
        epochs: Total number of training epochs.
        warmup_epochs: Number of linear warmup epochs (default: 0).

    Returns:
        LRScheduler — a plain scheduler when warmup_epochs == 0, or a
        SequentialLR that chains warmup → main schedule otherwise.

    Raises:
        ValueError: If name is not in the registry or warmup_epochs >= epochs.
    """
    name = name.lower()
    if name not in LR_SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Available: {list_lr_schedulers()}"
        )

    if warmup_epochs < 0:
        raise ValueError(
            f"warmup_epochs must be >= 0, got {warmup_epochs}."
        )

    if warmup_epochs == 0:
        return LR_SCHEDULER_REGISTRY[name](optimizer, epochs)

    if warmup_epochs >= epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be less than "
            f"epochs ({epochs})."
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=LRSchedulerConstants.WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    main = LR_SCHEDULER_REGISTRY[name](optimizer, epochs - warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_epochs],
    )


def list_lr_schedulers() -> List[str]:
    """Return the list of available scheduler names."""
    return sorted(LR_SCHEDULER_REGISTRY.keys())
