"""Tests for mist.utils.hardware."""

import contextlib
import warnings

import pytest
import torch

# MIST imports.
from mist.utils import hardware


def test_get_accelerator_type_cpu_without_cuda(monkeypatch) -> None:
    """get_accelerator_type is "cpu" when no CUDA/ROCm device is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hardware.get_accelerator_type() == "cpu"


def test_get_accelerator_type_cuda_when_hip_unset(monkeypatch) -> None:
    """get_accelerator_type is "cuda" when a device is visible and not ROCm."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert hardware.get_accelerator_type() == "cuda"


def test_get_accelerator_type_rocm_when_hip_set(monkeypatch) -> None:
    """get_accelerator_type is "rocm" when torch.version.hip is set.

    torch.version.hip is a version string on ROCm builds of PyTorch and None
    otherwise -- the documented way to distinguish ROCm from real CUDA, since
    both report through the same torch.cuda compatibility shim.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    assert hardware.get_accelerator_type() == "rocm"


@pytest.mark.parametrize(
    ("requested", "accelerator", "expected"),
    [
        ("auto", "cpu", "gloo"),
        ("auto", "cuda", "nccl"),
        ("auto", "rocm", "nccl"),
        ("mpi", "cuda", "mpi"),
        ("gloo", "cpu", "gloo"),
    ],
)
def test_resolve_communication_backend(monkeypatch, requested, accelerator, expected) -> None:
    """ "auto" resolves per accelerator; any other value passes through."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    assert hardware.resolve_communication_backend(requested) == expected


@pytest.mark.parametrize(
    ("requested", "accelerator", "expected"),
    [
        ("auto", "cpu", "generic"),
        ("auto", "rocm", "generic"),
        ("generic", "cuda", "generic"),
        ("dali", "cpu", "dali"),  # An explicit value always passes through untouched.
    ],
)
def test_resolve_data_loader(monkeypatch, requested, accelerator, expected) -> None:
    """ "auto" resolves per accelerator; any other value passes through.

    Cases that would actually reach the CUDA/"dali"-registered branch are
    covered by the two dedicated tests below instead, since that branch
    depends on data_loader_registry state, not just the accelerator.
    """
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    assert hardware.resolve_data_loader(requested) == expected


def test_resolve_data_loader_cuda_with_dali_registered(monkeypatch) -> None:
    """ "auto" resolves to "dali" on CUDA when it's actually installed."""
    from mist.data_loading import data_loader_registry

    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cuda")
    monkeypatch.setattr(
        data_loader_registry, "list_registered_data_loaders", lambda: ["dali", "generic"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_data_loader("auto") == "dali"


def test_resolve_data_loader_cuda_without_dali_warns_and_falls_back(monkeypatch) -> None:
    """ "auto" on CUDA without DALI installed warns and falls back to "generic".

    Regression guard: a CUDA machine that skipped `pip install
    "mist-medical[train-cuda]"` used to hit a ValueError deep inside
    build_dataloaders() ("Data loader 'dali' is not registered") instead of
    training on the generic loader with a clear warning.
    """
    from mist.data_loading import data_loader_registry

    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cuda")
    monkeypatch.setattr(data_loader_registry, "list_registered_data_loaders", lambda: ["generic"])
    with pytest.warns(UserWarning, match="train-cuda"):
        assert hardware.resolve_data_loader("auto") == "generic"


@pytest.mark.parametrize(
    ("accelerator", "expected"),
    [
        ("cpu", torch.device("cpu")),
        ("cuda", torch.device("cuda", 1)),
        ("rocm", torch.device("cuda", 1)),
    ],
)
def test_get_device_for_rank(monkeypatch, accelerator, expected) -> None:
    """CPU has no per-rank device; CUDA/ROCm both target "cuda:<rank>"."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    assert hardware.get_device_for_rank(rank=1) == expected


def test_bf16_supported_false_without_cuda(monkeypatch) -> None:
    """bf16_supported is False when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hardware.bf16_supported() is False


def test_bf16_supported_true_on_ampere(monkeypatch) -> None:
    """bf16_supported is True on Ampere or newer (compute capability >= 8)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert hardware.bf16_supported() is True


def test_bf16_supported_false_on_pre_ampere(monkeypatch) -> None:
    """bf16_supported is False on pre-Ampere GPUs (e.g. T4, SM 7.5).

    Regression guard: torch.cuda.is_bf16_supported() reports True on a T4 via
    software emulation, so the capability check must be used instead.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    assert hardware.bf16_supported() is False


def test_bf16_supported_on_rocm_trusts_is_bf16_supported(monkeypatch) -> None:
    """On ROCm, bf16_supported defers to torch.cuda.is_bf16_supported().

    Unlike NVIDIA pre-Ampere GPUs, there's no documented ROCm emulation quirk
    that would make this report a false positive, so no SM-style capability
    check is applied here.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert hardware.bf16_supported() is True

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert hardware.bf16_supported() is False


def test_resolve_amp_not_requested_returns_false() -> None:
    """A False request always resolves to False and never warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp(False) is False


def test_resolve_amp_supported_returns_true(monkeypatch) -> None:
    """A True request on Ampere+ hardware resolves to True."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp(True) is True


def test_resolve_amp_no_cuda_falls_back_with_warning(monkeypatch) -> None:
    """A True request with no CUDA device downgrades to False and warns."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="BF16"):
        assert hardware.resolve_amp(True) is False


def test_resolve_amp_pre_ampere_falls_back_with_device_name(monkeypatch) -> None:
    """A True request on a pre-Ampere GPU downgrades and names the device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "Tesla T4")
    with pytest.warns(UserWarning, match="Tesla T4"):
        assert hardware.resolve_amp(True) is False


def test_resolve_amp_warn_false_is_silent(monkeypatch) -> None:
    """warn=False suppresses the downgrade warning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "Tesla T4")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp(True, warn=False) is False


def test_autocast_context_disabled_is_nullcontext() -> None:
    """A disabled context is a no-op null context."""
    assert isinstance(hardware.autocast_context(False), contextlib.nullcontext)


def test_autocast_context_enabled_is_autocast() -> None:
    """An enabled context is a torch autocast (not a null context)."""
    # Constructing a CUDA autocast on a CPU-only host warns; that is unrelated
    # to what we assert here (the returned type), so suppress it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = hardware.autocast_context(True)
    assert not isinstance(ctx, contextlib.nullcontext)
    assert isinstance(ctx, torch.autocast)


def test_autocast_context_enabled_uses_cuda_device_type_on_rocm(monkeypatch) -> None:
    """ROCm reuses the "cuda" autocast device type, same as real CUDA."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "rocm")
    # Constructing a CUDA autocast on this CPU-only dev host warns; unrelated
    # to what we assert here (the device type it was constructed with).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = hardware.autocast_context(True)
    assert ctx.device == "cuda"


def test_autocast_context_enabled_uses_cpu_device_type_on_cpu(monkeypatch) -> None:
    """A CPU accelerator gets a "cpu" autocast device type, not "cuda".

    In practice bf16_supported() is always False on CPU, so resolve_amp()
    never lets enabled=True reach here on real CPU-only hardware -- this
    covers the defensive branch directly in case a caller bypasses that.
    """
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cpu")
    ctx = hardware.autocast_context(True)
    assert ctx.device == "cpu"
