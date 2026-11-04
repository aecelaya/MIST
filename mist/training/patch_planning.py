"""Choose a patch size and batch size from a network's estimated GPU memory.

The search follows nnU-Net's planner: start from a large patch shaped by the target
spacing, clipped to the median image size, and shrink the axis that takes up the
largest share of the image until the network fits in memory at the minimum batch
size. Spare memory goes to the patch first; the batch size only
grows once the patch has reached the image size, and is capped so that one step
sees a small share of the dataset.

Memory comes from counting a network's activations (see estimate_network), which
works for any registered architecture and doesn't need a GPU. Planning happens at
the start of training, when the real GPU, network, and precision are known, and
the result is saved in config.json for later runs.
"""

import dataclasses
import functools
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

import mist.models  # noqa: F401  # Registers every architecture.
from mist.models.model_registry import get_model_from_registry
from mist.models.nnunet.nnunet_utils import get_plain_unet_params, get_unet_params
from mist.training.planning_constants import PlanningConstants as constants
from mist.utils.console import print_warning

_GIB = 1024**3

# Patch and batch sizes with this value are planned the first time training runs.
AUTO = "auto"

# Layers whose outputs stand in for a network's memory use.
_ACTIVATION_LAYERS = (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)

# What every patch axis must be divisible by, for networks with a fixed design.
_FIXED_DIVISORS = {"mednext": 16, "swinunetr": 32, "primus": 8}

# Patch used to measure networks that can't run on PyTorch's meta device. Their
# fixed design makes activations proportional to patch volume.
_MEASURING_PATCH = (64, 64, 64)


@dataclasses.dataclass(frozen=True)
class NetworkEstimate:
    """A network's size at some patch.

    Attributes:
        activations: Activation values per sample.
        parameters: Number of parameters.
    """

    activations: float
    parameters: int


@dataclasses.dataclass(frozen=True)
class Plan:
    """A planned patch and batch size.

    Attributes:
        patch_size: The patch size.
        batch_size_per_gpu: The batch size per GPU.
        estimated_bytes: Estimated peak memory per GPU.
    """

    patch_size: list[int]
    batch_size_per_gpu: int
    estimated_bytes: float


def _count_activations(
    architecture: str,
    in_channels: int,
    out_channels: int,
    patch_size: Sequence[int],
    target_spacing: Sequence[float],
    device: str,
) -> NetworkEstimate:
    """Run one forward pass and add up the activation layers' output sizes."""
    if device == "meta":
        with torch.device("meta"):
            model = get_model_from_registry(
                architecture,
                in_channels=in_channels,
                out_channels=out_channels,
                patch_size=list(patch_size),
                target_spacing=list(target_spacing),
            )
    else:
        model = get_model_from_registry(
            architecture,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=list(patch_size),
            target_spacing=list(target_spacing),
        )
    model.train()

    total = [0]

    def count(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        total[0] += output.numel()

    hooks = [
        module.register_forward_hook(count)
        for module in model.modules()
        if isinstance(module, _ACTIVATION_LAYERS)
    ]
    with torch.no_grad():
        model(torch.empty(1, in_channels, *patch_size, device=device))
    for hook in hooks:
        hook.remove()
    return NetworkEstimate(float(total[0]), sum(p.numel() for p in model.parameters()))


@functools.cache
def _measure_on_cpu(
    architecture: str,
    in_channels: int,
    out_channels: int,
    target_spacing: tuple[float, ...],
) -> NetworkEstimate:
    """Measure a network once on the CPU at a small patch."""
    return _count_activations(
        architecture, in_channels, out_channels, _MEASURING_PATCH, target_spacing, "cpu"
    )


def estimate_network(
    architecture: str,
    in_channels: int,
    out_channels: int,
    patch_size: Sequence[int],
    target_spacing: Sequence[float],
) -> NetworkEstimate:
    """Estimate a network's size at a patch size, without using a GPU.

    The network runs on PyTorch's meta device, which tracks shapes without
    allocating memory. A few networks can't run there (e.g., Primus and SwinUNETR);
    they're measured once on the CPU at a small patch, then scaled by volume.

    Args:
        architecture: Registered model name.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        patch_size: Patch size. Must be valid for the network.
        target_spacing: Target spacing.

    Returns:
        The network's activations per sample and parameter count.

    Raises:
        ValueError: If the patch size isn't valid for the network.
    """
    try:
        return _count_activations(
            architecture, in_channels, out_channels, patch_size, target_spacing, "meta"
        )
    except (NotImplementedError, RuntimeError):
        measured = _measure_on_cpu(architecture, in_channels, out_channels, tuple(target_spacing))
        scale = float(np.prod(patch_size)) / float(np.prod(_MEASURING_PATCH))
        return NetworkEstimate(measured.activations * scale, measured.parameters)


def estimated_bytes(estimate: NetworkEstimate, batch_size: int, amp: bool) -> float:
    """Estimate peak GPU memory per GPU for a network and batch size.

    Args:
        estimate: The network's size.
        batch_size: Batch size per GPU.
        amp: Whether training uses BF16 mixed precision.

    Returns:
        Estimated peak memory in bytes.
    """
    per_activation = constants.BYTES_PER_ACTIVATION
    if amp:
        per_activation *= constants.BF16_ACTIVATION_FACTOR
    return (
        constants.OVERHEAD_BYTES
        + constants.BYTES_PER_PARAMETER * estimate.parameters
        + per_activation * estimate.activations * batch_size
    )


def axis_divisors(
    architecture: str,
    patch_size: Sequence[int],
    target_spacing: Sequence[float],
) -> list[int]:
    """Return what each patch axis must be divisible by for a network.

    Args:
        architecture: Registered model name.
        patch_size: Patch size, which adaptive networks plan their depth from.
        target_spacing: Target spacing.

    Returns:
        One divisor per axis.
    """
    for prefix, divisor in _FIXED_DIVISORS.items():
        if architecture.startswith(prefix):
            return [divisor] * len(patch_size)
    if architecture == "nnunet-plain":
        return get_plain_unet_params(patch_size, target_spacing)[2]
    _, strides, _ = get_unet_params(patch_size, target_spacing)
    return [int(np.prod([s[axis] for s in strides])) for axis in range(len(patch_size))]


def _snap_down(
    architecture: str,
    patch_size: Sequence[int],
    target_spacing: Sequence[float],
) -> list[int]:
    """Reduce a patch until the network fits it; depth can change, so repeat."""
    patch = [int(s) for s in patch_size]
    while True:
        divisors = axis_divisors(architecture, patch, target_spacing)
        snapped = [max((s // d) * d, d) for s, d in zip(patch, divisors, strict=True)]
        if snapped == patch:
            return patch
        patch = snapped


def _initial_patch(
    median_size: Sequence[int],
    target_spacing: Sequence[float],
) -> list[int]:
    """Return a patch shaped by the spacing, larger than needed, clipped to the image."""
    inverse = 1.0 / np.asarray(target_spacing, dtype=float)
    scale = (constants.INITIAL_EDGE**3 / np.prod(inverse)) ** (1.0 / 3.0)
    return [min(round(float(i * scale)), int(m)) for i, m in zip(inverse, median_size, strict=True)]


def _shrink_axis(
    architecture: str,
    patch: list[int],
    median_size: Sequence[int],
    target_spacing: Sequence[float],
) -> list[int] | None:
    """Reduce the axis taking up the most of the image, or None if none can shrink."""
    divisors = axis_divisors(architecture, patch, target_spacing)
    order = np.argsort([p / m for p, m in zip(patch, median_size, strict=True)])[::-1]
    for axis in order:
        axis = int(axis)
        smaller = list(patch)
        # Taking off the current divisor could skip a valid size, since a smaller
        # patch can have a smaller divisor: 256 needs 64, but 224 is valid too.
        smaller[axis] -= divisors[axis]
        if smaller[axis] < constants.MIN_AXIS_SIZE:
            continue
        step = axis_divisors(architecture, smaller, target_spacing)[axis]
        smaller = list(patch)
        smaller[axis] -= step
        smaller = _snap_down(architecture, smaller, target_spacing)
        # A patch too small to downsample would leave the network with no decoder.
        downsampled = max(axis_divisors(architecture, smaller, target_spacing)) > 1
        if smaller != patch and smaller[axis] >= constants.MIN_AXIS_SIZE and downsampled:
            return smaller
    return None


def plan_patch_and_batch(
    architecture: str,
    in_channels: int,
    out_channels: int,
    median_size: Sequence[int],
    target_spacing: Sequence[float],
    gpu_memory_bytes: float,
    amp: bool,
    num_cases: int,
    num_gpus: int = 1,
    batch_size: int | None = None,
) -> Plan:
    """Choose a patch size and batch size for a network and a GPU.

    Args:
        architecture: Registered model name.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        median_size: Median resampled image size.
        target_spacing: Target spacing.
        gpu_memory_bytes: Memory of the GPU training will run on.
        amp: Whether training will use BF16 mixed precision.
        num_cases: Number of training cases, for capping the batch size.
        num_gpus: Number of GPUs sharing the capped global batch.
        batch_size: A batch size set by hand, to plan the patch around.

    Returns:
        The plan. If even the smallest patch doesn't fit, the plan holds that
        patch, and its estimate exceeds the memory available.
    """
    budget = gpu_memory_bytes * constants.MEMORY_FRACTION
    minimum = batch_size or constants.MIN_BATCH_SIZE

    def estimate_for(patch_size: list[int]) -> NetworkEstimate | None:
        """Return the estimate, or None if the network can't use this patch."""
        try:
            return estimate_network(
                architecture, in_channels, out_channels, patch_size, target_spacing
            )
        except ValueError:
            return None

    patch = _snap_down(architecture, _initial_patch(median_size, target_spacing), target_spacing)
    estimate = estimate_for(patch)
    # Some networks need a minimum depth, which a small image may not allow. As
    # with any patch larger than an image, training pads to fit.
    for _ in range(constants.MAX_GROW_STEPS):
        if estimate is not None:
            break
        patch = _snap_down(architecture, [size * 2 for size in patch], target_spacing)
        estimate = estimate_for(patch)
    if estimate is None:
        raise ValueError(f"Could not find a patch size that {architecture} can use.")

    shrunk = False
    while estimated_bytes(estimate, minimum, amp) > budget:
        smaller = _shrink_axis(architecture, patch, median_size, target_spacing)
        smaller_estimate = estimate_for(smaller) if smaller is not None else None
        if smaller_estimate is None:
            break
        patch, estimate = smaller, smaller_estimate
        shrunk = True

    planned_batch = minimum
    if batch_size is None and not shrunk:
        unit = estimated_bytes(estimate, 1, amp) - estimated_bytes(estimate, 0, amp)
        fits = int((budget - estimated_bytes(estimate, 0, amp)) // unit)
        # Keep one global batch to a small share of the dataset's voxels.
        dataset_voxels = float(num_cases) * float(np.prod(median_size))
        capped = dataset_voxels * constants.MAX_DATASET_FRACTION / float(np.prod(patch))
        planned_batch = max(min(fits, math.floor(capped / max(1, num_gpus))), minimum)

    return Plan(patch, planned_batch, estimated_bytes(estimate, planned_batch, amp))


def detect_gpu_memory_bytes() -> tuple[float | None, int]:
    """Return the smallest visible GPU's memory, or None, and the GPU count."""
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return None, 0
    count = torch.cuda.device_count()
    return (
        float(min(torch.cuda.get_device_properties(i).total_memory for i in range(count))),
        count,
    )


def is_auto(value: Any) -> bool:
    """Return whether a patch or batch size is still to be planned."""
    return value is None or value == AUTO


def plan_for_config(config: dict[str, Any], num_cases: int, num_gpus: int) -> Plan:
    """Plan whatever a configuration leaves as "auto", for this machine's GPU.

    A patch or batch size set by hand is kept and the other is chosen around it.
    With a patch set by hand, the batch size is the minimum.

    Args:
        config: MIST configuration, with its AMP setting already resolved.
        num_cases: Number of training cases.
        num_gpus: Number of GPUs that will train.

    Returns:
        The plan.
    """
    architecture = config["model"]["architecture"]
    params = config["model"]["params"]
    spatial = config["spatial_config"]
    training = config["training"]
    batch_size = None if is_auto(training["batch_size_per_gpu"]) else training["batch_size_per_gpu"]
    amp = bool(training["amp"])
    memory = detect_gpu_memory_bytes()[0] or constants.DEFAULT_GPU_MEMORY_GB * _GIB

    if not is_auto(spatial["patch_size"]):
        patch = [int(size) for size in spatial["patch_size"]]
        batch = batch_size or constants.MIN_BATCH_SIZE
        estimate = estimate_network(
            architecture,
            params["in_channels"],
            params["out_channels"],
            patch,
            spatial["target_spacing"],
        )
        return Plan(patch, batch, estimated_bytes(estimate, batch, amp))

    return plan_patch_and_batch(
        architecture=architecture,
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        median_size=config["preprocessing"]["median_resampled_image_size"],
        target_spacing=spatial["target_spacing"],
        gpu_memory_bytes=memory,
        amp=amp,
        num_cases=num_cases,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )


def check_plan_fits(config: dict[str, Any], patch_was_set: bool = False) -> None:
    """Warn if the training GPU doesn't suit a saved plan.

    A plan is saved the first time training runs, and later runs reuse it. This
    warns rather than replanning, since a patch that silently changed with the
    machine would make runs hard to reproduce and compare. Configs without a saved
    plan aren't checked, and neither are machines with no GPU.

    Args:
        config: The resolved MIST configuration.
        patch_was_set: Whether the patch size was set by hand for this run.
    """
    planning = config.get("planning")
    if not planning:
        return
    memory, _ = detect_gpu_memory_bytes()
    if memory is None:
        return

    architecture = config["model"]["architecture"]
    spatial = config["spatial_config"]
    training = config["training"]
    try:
        estimate = estimate_network(
            architecture,
            config["model"]["params"]["in_channels"],
            config["model"]["params"]["out_channels"],
            spatial["patch_size"],
            spatial["target_spacing"],
        )
    except Exception as error:  # pylint: disable=broad-except
        warnings.warn(f"Could not estimate GPU memory for this run: {error}", stacklevel=2)
        return

    needed = estimated_bytes(estimate, training["batch_size_per_gpu"], training["amp"])
    if needed > memory:
        print_warning(
            f"{architecture} with patch {spatial['patch_size']} and batch size "
            f"{training['batch_size_per_gpu']} may need about {needed / _GIB:.1f} GiB per GPU, "
            f"but this GPU has {memory / _GIB:.1f} GiB. If training runs out of memory, use a "
            'smaller --patch-size, or set patch_size and batch_size_per_gpu to "auto" in '
            "config.json to plan again."
        )
    elif (
        not patch_was_set and memory >= constants.UNDERUSED_RATIO * planning["gpu_memory_gb"] * _GIB
    ):
        print_warning(
            f"The patch size was planned for {planning['gpu_memory_gb']:.0f} GiB per GPU, but "
            f"this GPU has {memory / _GIB:.0f} GiB. A larger patch may fit: set patch_size and "
            'batch_size_per_gpu to "auto" in config.json to plan again.'
        )


def inference_batch_size(config: dict[str, Any], device: torch.device | str) -> int:
    """Choose how many patches to predict at once on a device.

    Predictions don't depend on this, so it's chosen for each machine rather than
    saved. It's capped because larger batches add memory risk for little speed.
    This is only a starting point, since memory also depends on the image: running
    out of memory lowers it during inference (see sliding_window.predict_with_fallback).

    Args:
        config: MIST configuration with a planned patch size.
        device: Device that will run inference.

    Returns:
        The batch size, from 1 to MAX_INFERENCE_BATCH_SIZE.
    """
    cap = constants.MAX_INFERENCE_BATCH_SIZE
    device = torch.device(device)
    if device.type != "cuda":
        return cap

    try:
        spatial = config["spatial_config"]
        estimate = estimate_network(
            config["model"]["architecture"],
            config["model"]["params"]["in_channels"],
            config["model"]["params"]["out_channels"],
            spatial["patch_size"],
            spatial["target_spacing"],
        )
        total = torch.cuda.get_device_properties(device).total_memory
        available = total * constants.MEMORY_FRACTION - torch.cuda.memory_allocated(device)
    except Exception as error:  # pylint: disable=broad-except
        warnings.warn(f"Could not size the inference batch: {error}", stacklevel=2)
        return cap

    per_activation = constants.INFERENCE_BYTES_PER_ACTIVATION
    if config["training"].get("amp"):
        per_activation *= constants.BF16_ACTIVATION_FACTOR
    fits = int(available // (per_activation * estimate.activations))
    return max(1, min(cap, fits))
