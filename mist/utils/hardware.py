"""Hardware capability helpers for MIST automatic mixed precision (AMP).

MIST's AMP uses BF16 autocast, which is only hardware-accelerated on NVIDIA
Ampere or newer GPUs (A100, RTX 30xx, H100). On pre-Ampere cards (T4, V100,
RTX 20xx) BF16 autocast runs without tensor-core support and is slower than
plain FP32 — and on CPU it is unavailable entirely. These helpers resolve a
requested AMP setting against the current hardware so callers can transparently
fall back to FP32 instead of silently running a slow or unsupported path.
"""

import contextlib
import warnings

import torch


def bf16_supported() -> bool:
    """Return True if the current CUDA device *natively* supports BF16.

    Checks the compute capability (Ampere / SM 8.0 or newer) rather than
    ``torch.cuda.is_bf16_supported()``, which by default returns True on
    pre-Ampere GPUs (T4, V100) via slow software emulation — which would defeat
    the FP32 fallback this module exists to provide.
    """
    if not torch.cuda.is_available():
        return False
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
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU (no CUDA device)"
        )
        warnings.warn(
            f"AMP was requested but {device} has no hardware BF16 support; "
            "falling back to FP32. Set training.amp = false to silence this.",
            stacklevel=2,
        )
    return False


def autocast_context(enabled: bool):
    """Return a BF16 CUDA autocast context when enabled, else a null context."""
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
