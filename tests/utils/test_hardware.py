"""Tests for mist.utils.hardware."""

import contextlib
import warnings

import pytest
import torch

# MIST imports.
from mist.utils import hardware


def test_bf16_supported_false_without_cuda(monkeypatch) -> None:
    """bf16_supported is False when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hardware.bf16_supported() is False


def test_bf16_supported_true_when_cuda_and_bf16(monkeypatch) -> None:
    """bf16_supported is True when CUDA is available and BF16 is supported."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert hardware.bf16_supported() is True


def test_resolve_amp_not_requested_returns_false() -> None:
    """A False request always resolves to False and never warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hardware.resolve_amp(False) is False


def test_resolve_amp_supported_returns_true(monkeypatch) -> None:
    """A True request on BF16-capable hardware resolves to True."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
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
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "Tesla T4")
    with pytest.warns(UserWarning, match="Tesla T4"):
        assert hardware.resolve_amp(True) is False


def test_resolve_amp_warn_false_is_silent(monkeypatch) -> None:
    """warn=False suppresses the downgrade warning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
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
