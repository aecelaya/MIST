"""Alpha schedulers for dynamic composite loss weighting.

This module provides scheduler classes to dynamically adjust the weighting
(alpha) of loss components during training.
"""

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type


class AlphaScheduler(ABC):
    """Abstract base class for alpha schedulers."""

    @abstractmethod
    def __call__(self, epoch: int) -> float:
        """Calculates the alpha value for the given epoch.

        Args:
            epoch: The current training epoch (0-indexed).

        Returns:
            The calculated alpha value as a float.
        """


class ConstantScheduler(AlphaScheduler):
    """Returns a constant alpha value throughout training.

    Attributes:
        value: The constant alpha value to be returned.
    """

    def __init__(self, value: float = 0.5):
        """Initializes the ConstantScheduler.

        Args:
            value: The constant alpha value. Defaults to 0.5.
        """
        self.value = float(value)

    def __call__(self, epoch: int) -> float:
        """Returns the constant alpha value.

        Args:
            epoch: The current training epoch (ignored).

        Returns:
            The constant alpha value.
        """
        return self.value


class LinearScheduler(AlphaScheduler):
    """Linearly interpolates alpha from a start value to an end value.

    The scheduler maintains the start value for a specified number of initial
    epochs (init_pause), then linearly interpolates to the end value over the
    remaining epochs.

    Attributes:
        start_val: The initial alpha value.
        end_val: The final alpha value.
        init_pause: The number of epochs to wait before decaying.
        decay_duration: The number of epochs over which the decay occurs.
    """

    def __init__(
        self,
        num_epochs: int,
        init_pause: int = 5,
        start_val: float = 1.0,
        end_val: float = 0.0,
    ):
        """Initializes the LinearScheduler.

        Args:
            num_epochs: The total number of training epochs.
            init_pause: The number of epochs to hold the start value.
                Defaults to 5.
            start_val: The starting alpha value. Defaults to 1.0.
            end_val: The target alpha value after decay. Defaults to 0.0.
        """
        self.num_epochs = num_epochs
        self.start_val = start_val
        self.end_val = end_val
        self.init_pause = init_pause
        decay_steps = max(1, num_epochs - 1)
        self.decay_duration = max(1, decay_steps - self.init_pause)

    def __call__(self, epoch: int) -> float:
        """Calculates the alpha value for the linear schedule.

        Args:
            epoch: The current training epoch.

        Returns:
            The interpolated alpha value.
        """
        if epoch <= self.init_pause:
            return self.start_val

        steps_past_pause = epoch - self.init_pause
        progress = min(1.0, steps_past_pause / self.decay_duration)
        return self.start_val + progress * (self.end_val - self.start_val)


class CosineScheduler(AlphaScheduler):
    """Decays alpha following a cosine curve (half-period).

    This scheduler creates a smooth transition from start_val to end_val using
    a cosine function, which often results in better stability than linear
    decay.

    Attributes:
        start_val: The initial alpha value.
        end_val: The final alpha value.
        init_pause: The number of epochs to wait before decaying.
        decay_duration: The number of epochs over which the decay occurs.
    """

    def __init__(
        self,
        num_epochs: int,
        init_pause: int = 5,
        start_val: float = 1.0,
        end_val: float = 0.0,
    ):
        """Initializes the CosineScheduler.

        Args:
            num_epochs: The total number of training epochs.
            init_pause: The number of epochs to hold the start value.
                Defaults to 5.
            start_val: The starting alpha value. Defaults to 1.0.
            end_val: The target alpha value after decay. Defaults to 0.0.
        """
        self.num_epochs = num_epochs
        self.start_val = start_val
        self.end_val = end_val
        self.init_pause = init_pause
        decay_steps = max(1, num_epochs - 1)
        self.decay_duration = max(1, decay_steps - self.init_pause)

    def __call__(self, epoch: int) -> float:
        """Calculates the alpha value for the cosine schedule.

        Args:
            epoch: The current training epoch.

        Returns:
            The cosine-interpolated alpha value.
        """
        if epoch <= self.init_pause:
            return self.start_val

        steps_past_pause = epoch - self.init_pause
        progress = min(1.0, steps_past_pause / self.decay_duration)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end_val + (self.start_val - self.end_val) * cosine_factor


ALPHA_SCHEDULER_REGISTRY: Dict[str, Type[AlphaScheduler]] = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
}


def get_default_scheduler_config(name: str) -> Dict[str, Any]:
    """Build a ``{name, params}`` config dict with constructor defaults.

    Extracts default parameter values directly from the scheduler's
    ``__init__`` signature, excluding ``num_epochs`` (which is runtime
    context supplied by the trainer, not a user config value).

    Args:
        name: Registered scheduler name (e.g. ``"linear"``).

    Returns:
        A dict of the form ``{"name": name, "params": {...}}``.

    Raises:
        ValueError: If ``name`` is not registered.

    Example::

        >>> get_default_scheduler_config("linear")
        {"name": "linear", "params": {"init_pause": 5, "start_val": 1.0, "end_val": 0.0}}
    """
    if name not in ALPHA_SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Available: {list(ALPHA_SCHEDULER_REGISTRY.keys())}"
        )
    cls = ALPHA_SCHEDULER_REGISTRY[name]
    params = {
        k: v.default
        for k, v in inspect.signature(cls.__init__).parameters.items()
        if k not in ("self", "num_epochs")
        and v.default is not inspect.Parameter.empty
    }
    return {"name": name, "params": params}


def get_alpha_scheduler(name: str, num_epochs: int, **params) -> AlphaScheduler:
    """Factory function to retrieve an alpha scheduler by name.

    Args:
        name: The name of the scheduler (e.g., 'linear', 'cosine', 'constant').
        num_epochs: Total training epochs. Passed to schedulers that declare
            a ``num_epochs`` parameter; ignored for those that don't (e.g.,
            ConstantScheduler).
        **params: Scheduler-specific keyword arguments (e.g., ``init_pause``,
            ``start_val``, ``end_val``, ``value``).

    Returns:
        An instance of the requested AlphaScheduler.

    Raises:
        ValueError: If the scheduler name is not found in the registry.
    """
    if name not in ALPHA_SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Available: {list(ALPHA_SCHEDULER_REGISTRY.keys())}"
        )
    cls = ALPHA_SCHEDULER_REGISTRY[name]
    if "num_epochs" in inspect.signature(cls.__init__).parameters:
        return cls(num_epochs=num_epochs, **params)
    return cls(**params)


def list_alpha_schedulers() -> List[str]:
    """Lists all registered alpha scheduler names.

    Returns:
        A sorted list of available scheduler names.
    """
    return sorted(list(ALPHA_SCHEDULER_REGISTRY.keys()))
