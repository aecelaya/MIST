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
        ("auto", "cuda", "generic"),
        ("generic", "cuda", "generic"),
        ("dali", "cuda", "dali"),  # An explicit value always passes through untouched.
        ("dali", "cpu", "dali"),
    ],
)
def test_resolve_data_loader(monkeypatch, requested, accelerator, expected) -> None:
    """ "auto" is the generic loader on every machine, even with DALI installed."""
    from mist.data_loading import data_loader_registry

    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    monkeypatch.setattr(
        data_loader_registry, "list_registered_data_loaders", lambda: ["dali", "generic"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_data_loader(requested) == expected


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


class _FakeDeviceProperties:
    """Minimal stand-in for torch.cuda.get_device_properties()'s return."""

    def __init__(self, gcn_arch_name: str) -> None:
        self.gcnArchName = gcn_arch_name


def test_bf16_supported_true_on_cdna(monkeypatch) -> None:
    """bf16_supported is True on CDNA (MFMA matrix hardware), e.g. MI210."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx=0: _FakeDeviceProperties("gfx90a:sramecc+:xnack-"),
    )
    assert hardware.bf16_supported() is True


def test_bf16_supported_true_on_rdna3(monkeypatch) -> None:
    """bf16_supported is True on RDNA3+ (WMMA matrix hardware), e.g. RX 7900."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx=0: _FakeDeviceProperties("gfx1100"),
    )
    assert hardware.bf16_supported() is True


def test_bf16_supported_false_on_rdna2(monkeypatch) -> None:
    """bf16_supported is False on RDNA1/2 (no matrix hardware), e.g. RX 6800.

    Regression guard: torch.cuda.is_bf16_supported() reports True on RDNA2
    via software emulation on plain shader ALUs -- confirmed empirically to
    regress training speed vs. FP32 on a real gfx1030 card -- so the
    gcnArchName allow-list must be used instead of trusting it.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.3.42134", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx=0: _FakeDeviceProperties("gfx1030"),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert hardware.bf16_supported() is False


@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "rocm"])
def test_resolve_compile_not_requested_returns_false(monkeypatch, accelerator) -> None:
    """Without a request, compile stays off everywhere and nothing warns."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_compile(False) is False


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
def test_resolve_compile_on_a_gpu(monkeypatch, accelerator) -> None:
    """A request on a GPU is met."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    monkeypatch.setattr(hardware.os, "name", "posix")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_compile(True) is True


@pytest.mark.parametrize("accelerator, os_name", [("cpu", "posix"), ("cuda", "nt")])
def test_resolve_compile_unsupported_is_quiet(monkeypatch, accelerator, os_name) -> None:
    """Where compile isn't supported, it stays off without a warning, since it's the default."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    monkeypatch.setattr(hardware.os, "name", os_name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_compile(True) is False


def test_resolve_amp_not_requested_returns_false() -> None:
    """A False request always resolves to False and never warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp(False) is False


@pytest.mark.parametrize("supported", [True, False])
def test_resolve_amp_auto_follows_the_hardware_without_warning(monkeypatch, supported) -> None:
    """ "auto" didn't ask for BF16, so getting FP32 isn't worth a warning."""
    monkeypatch.setattr(hardware, "bf16_supported", lambda: supported)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp("auto") is supported


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


def test_resolve_amp_keeps_fp16_on_a_gpu(monkeypatch) -> None:
    """FP16 runs on any GPU, so it's kept quietly, even on a pre-Ampere card."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp("fp16") == hardware.FP16


def test_resolve_amp_fp16_falls_back_to_fp32_without_a_gpu(monkeypatch) -> None:
    """FP16 on a CPU-only machine becomes FP32, with a warning."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cpu")
    with pytest.warns(UserWarning, match="FP16"):
        assert hardware.resolve_amp("fp16") is False


def test_autocast_context_uses_fp16_or_bf16(monkeypatch) -> None:
    """ "fp16" autocasts to FP16, and True to BF16."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cuda")
    created = []
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: created.append(kwargs))
    hardware.autocast_context("fp16")
    hardware.autocast_context(True)
    assert [kwargs["dtype"] for kwargs in created] == [torch.float16, torch.bfloat16]
