"""Hardware capability and accelerator-detection helpers for MIST.

MIST's AMP uses BF16 autocast, which is only hardware-accelerated on NVIDIA
Ampere or newer GPUs (A100, RTX 30xx, H100). On pre-Ampere cards (T4, V100,
RTX 20xx) BF16 autocast runs without tensor-core support and is slower than
plain FP32 — and on CPU it is unavailable entirely. These helpers resolve a
requested AMP setting against the current hardware so callers can transparently
fall back to FP32 instead of silently running a slow or unsupported path.

This module also detects which accelerator (CUDA, AMD ROCm, or CPU-only) MIST
is running on, so training can pick a working `torch.distributed` backend and
data loader without the user having to know these details themselves — see
``cpu_rocm_support_plan.md`` Stage 1.
"""

import contextlib
import warnings
from typing import Literal

import torch

AcceleratorType = Literal["cuda", "rocm", "cpu"]


def get_accelerator_type() -> AcceleratorType:
    """Return the accelerator MIST is currently running on.

    PyTorch's ROCm build reuses the ``torch.cuda`` namespace and the
    ``"cuda"`` device string as a compatibility shim, so
    ``torch.cuda.is_available()``, ``get_device_properties()``,
    ``device_count()``, and ``get_device_name()`` all already work correctly
    on AMD GPUs without any change. ``torch.version.hip`` is the documented
    way to tell the two apart at runtime: it's a version string on ROCm
    builds and ``None`` on CUDA/CPU-only builds.
    """
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None) is not None:
        return "rocm"
    return "cuda"


def resolve_communication_backend(requested: str) -> str:
    """Resolve a requested ``torch.distributed`` backend against hardware.

    Mirrors ``resolve_amp``'s and the data loader's resolve-and-persist
    pattern: an explicit, non-``"auto"`` value always passes through
    unchanged, so a user who sets ``training.hardware.communication_backend``
    directly in ``config.json`` is never second-guessed. Only the ``"auto"``
    sentinel gets resolved here: CUDA and ROCm both use ``"nccl"`` as the
    ``torch.distributed`` backend name — on ROCm it transparently routes to
    RCCL, AMD's NCCL-compatible collective library, with no code change
    required — while CPU-only hardware has no NCCL/RCCL and uses ``"gloo"``.

    Args:
        requested: The configured backend, or ``"auto"`` to detect one.

    Returns:
        The resolved backend name to pass to ``dist.init_process_group``.
    """
    if requested != "auto":
        return requested
    return "gloo" if get_accelerator_type() == "cpu" else "nccl"


def resolve_data_loader(requested: str) -> str:
    """Resolve a requested data loader implementation against hardware.

    Mirrors ``resolve_communication_backend``'s "auto" sentinel contract.
    DALI requires an NVIDIA GPU (``nvidia-dali-cuda120``), so it's only
    selected automatically on CUDA; ROCm and CPU-only hardware fall back to
    MIST's generic, accelerator-agnostic loader (see Stage 2-4 of
    ``cpu_rocm_support_plan.md``). An explicit, non-``"auto"`` value always
    passes through unchanged.

    Args:
        requested: The configured loader, or ``"auto"`` to detect one.

    Returns:
        The resolved data loader name, as registered in
        ``mist.data_loading.data_loader_registry``.
    """
    if requested != "auto":
        return requested
    return "dali" if get_accelerator_type() == "cuda" else "generic"


def bf16_supported() -> bool:
    """Return True if the current accelerator *natively* supports BF16.

    On CUDA, checks the compute capability (Ampere / SM 8.0 or newer) rather
    than ``torch.cuda.is_bf16_supported()``, which by default returns True on
    pre-Ampere GPUs (T4, V100) via slow software emulation — which would defeat
    the FP32 fallback this module exists to provide. ROCm has no documented
    equivalent emulation quirk, so ``torch.cuda.is_bf16_supported()`` (which
    already dispatches to the current ROCm device via the same compatibility
    shim as the rest of ``torch.cuda``) is trusted directly there.
    """
    accelerator = get_accelerator_type()
    if accelerator == "cpu":
        return False
    if accelerator == "rocm":
        return torch.cuda.is_bf16_supported()
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_amp(requested: bool, *, warn: bool = True) -> bool:
    """Resolve a requested AMP setting against the current hardware.

    Returns ``True`` only when AMP was requested *and* BF16 is hardware
    supported. When a request is downgraded — because there is no CUDA device
    or the GPU is pre-Ampere — a warning is emitted (per the ``warnings.warn``
    convention documented in ``mist.utils.console``) so callers know FP32 will
    be used instead.

    Args:
        requested: The AMP setting requested via config (``training.amp``).
        warn: Whether to warn when the request is downgraded to FP32.

    Returns:
        The effective AMP setting for the current hardware.
    """
    if not requested:
        return False
    if bf16_supported():
        return True
    if warn:
        device = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no CUDA device)"
        )
        warnings.warn(
            f"AMP was requested but {device} has no hardware BF16 support; "
            "falling back to FP32. Set training.amp = false to silence this.",
            stacklevel=2,
        )
    return False


def autocast_context(enabled: bool):
    """Return a BF16 autocast context for the current accelerator, or null.

    ``resolve_amp``/``bf16_supported`` already guarantee ``enabled`` is only
    ``True`` when the current accelerator has native BF16 support, so this
    just needs to point ``torch.autocast`` at the right device type: ``"cuda"``
    covers both real CUDA and ROCm (the same compatibility shim used
    everywhere else in this module), and CPU never reaches the ``True``
    branch since ``bf16_supported()`` is always ``False`` there.
    """
    if enabled:
        device_type = "cpu" if get_accelerator_type() == "cpu" else "cuda"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
