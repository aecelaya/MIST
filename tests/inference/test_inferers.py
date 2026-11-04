"""Tests for MIST inferers."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

# MIST imports.
from mist.inference.inferers.base import AbstractInferer
from mist.inference.inferers.inferer_registry import (
    get_inferer,
    list_inferers,
    register_inferer,
)
from mist.inference.inferers.sliding_window import SlidingWindowInferer, predict_with_fallback


class DummyInferer(AbstractInferer):
    """Dummy Inferer for base class testing."""

    def infer(
        self, image: torch.Tensor, model: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        return model(image)


def test_dummy_inferer_call_and_repr():
    """Test __call__, __eq__, and __repr__ for AbstractInferer."""

    def model(x):
        return x + 1

    image = torch.rand(1, 1, 8, 8, 8)
    dummy = DummyInferer()
    result = dummy(image, model)
    assert torch.allclose(result, image + 1)
    assert dummy == DummyInferer()
    assert repr(dummy) == "DummyInferer(name='dummyinferer')"


def test_inferer_hash_and_eq():
    """Test __hash__ and __eq__ for AbstractInferer."""
    a = DummyInferer()
    b = DummyInferer()
    c = object()  # Not an AbstractInferer

    # Hashes should match for same name.
    assert hash(a) == hash(b)

    # __eq__ should return False for non-inferer comparison.
    assert a != c


# SlidingWindowInferer tests.
@patch("mist.inference.inferers.sliding_window.monai.inferers.sliding_window_inference")
def test_sliding_window_inferer_basic(mock_sw_inference):
    """Test basic functionality of SlidingWindowInferer."""
    image = torch.rand(1, 1, 32, 32, 32)
    model = MagicMock(return_value=image)
    mock_sw_inference.return_value = image

    inferer = SlidingWindowInferer(patch_size=(16, 16, 16))
    result = inferer(image, model)

    assert result.shape == image.shape
    mock_sw_inference.assert_called_once()


@patch("mist.inference.inferers.sliding_window.monai.inferers.sliding_window_inference")
def test_sliding_window_inferer_sw_batch_size_passed_to_monai(mock_sw_inference):
    """sw_batch_size is forwarded to MONAI's sliding_window_inference."""
    image = torch.rand(1, 1, 32, 32, 32)
    mock_sw_inference.return_value = image

    inferer = SlidingWindowInferer(patch_size=(16, 16, 16), sw_batch_size=4)
    inferer(image, MagicMock(return_value=image))

    assert mock_sw_inference.call_args.kwargs["sw_batch_size"] == 4


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"patch_size": (16, 16)}, "patch_size must be a tuple of length 3"),
        ({"patch_size": (16, -1, 16)}, "must be positive integers"),
        (
            {"patch_size": (16, 16, 16), "sw_batch_size": 0},
            "sw_batch_size must be a positive integer",
        ),
        (
            {"patch_size": (16, 16, 16), "patch_overlap": 1.5},
            "patch_overlap must be in the range",
        ),
        (
            {"patch_size": (16, 16, 16), "patch_blend_mode": "linear"},
            "Unsupported blend mode",
        ),
    ],
)
def test_sliding_window_inferer_invalid_params(kwargs, match):
    """Test that invalid constructor arguments raise ValueError."""
    with pytest.raises(ValueError, match=match):
        SlidingWindowInferer(**kwargs)


# Registry tests.
def test_registry_get_inferer_sliding_window():
    """Test get_inferer returns the SlidingWindowInferer class."""
    cls = get_inferer("sliding_window")
    assert cls is SlidingWindowInferer


def test_registry_list_inferers_includes_sliding_window():
    """Test that 'sliding_window' is in the inferer registry."""
    assert "sliding_window" in list_inferers()


def test_registry_get_inferer_invalid_name():
    """Test that get_inferer raises KeyError for unknown name."""
    with pytest.raises(KeyError, match="Inferer 'invalid' is not registered"):
        get_inferer("invalid")


def test_registry_rejects_invalid_class():
    """Test that registering a non-Inferer class raises TypeError."""

    class NotAnInferer:
        pass

    with pytest.raises(TypeError, match="must inherit from AbstractInferer"):

        @register_inferer("invalid_class")
        class Invalid(NotAnInferer):
            pass


def test_registry_rejects_duplicate_name():
    """Test that duplicate registration raises KeyError."""
    with pytest.raises(KeyError, match="already registered"):

        @register_inferer("sliding_window")
        class DuplicateInferer(AbstractInferer):
            def infer(self, image, model):
                return model(image)


# Recovering from running out of memory.


def _oom_until(max_batch=None, stitch_on_cpu=False):
    """Fake MONAI inference that runs out of memory above some limit."""
    calls = []

    def fake(inputs, sw_batch_size, device, **_):
        calls.append((sw_batch_size, torch.device(device).type))
        too_large = max_batch is not None and sw_batch_size > max_batch
        needs_cpu = stitch_on_cpu and torch.device(device).type != "cpu"
        if too_large or needs_cpu:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return torch.zeros_like(inputs)

    return fake, calls


@pytest.fixture
def fake_empty_cache(monkeypatch):
    """Count cache clears; a CPU-only build has no real CUDA cache."""
    cleared = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(1))
    return cleared


@patch("monai.inferers.sliding_window_inference")
def test_fallback_is_silent_when_memory_suffices(mock_swi, fake_empty_cache):
    """Normal runs make one call and clear nothing."""
    mock_swi.side_effect, calls = _oom_until()
    predict_with_fallback(torch.zeros(1, 1, 4, 4, 4), 4, "cpu", roi_size=(2, 2, 2))
    assert calls == [(4, "cpu")]
    assert not fake_empty_cache


@patch("monai.inferers.sliding_window_inference")
def test_fallback_halves_the_batch_until_it_fits(mock_swi, fake_empty_cache):
    """Running out of memory retries at half the batch, clearing the cache first."""
    mock_swi.side_effect, calls = _oom_until(max_batch=1)
    result = predict_with_fallback(torch.zeros(1, 1, 4, 4, 4), 4, "cpu", roi_size=(2, 2, 2))
    assert [c[0] for c in calls] == [4, 2, 1]
    assert len(fake_empty_cache) == 2
    assert result.shape == (1, 1, 4, 4, 4)


@patch("monai.inferers.sliding_window_inference")
def test_fallback_stitches_on_cpu_as_a_last_resort(mock_swi, fake_empty_cache):
    """At a batch of 1, the output moves off the GPU."""
    mock_swi.side_effect, calls = _oom_until(stitch_on_cpu=True)
    predict_with_fallback(torch.zeros(1, 1, 4, 4, 4), 2, "meta", roi_size=(2, 2, 2))
    assert calls == [(2, "meta"), (1, "meta"), (1, "cpu")]


@patch("monai.inferers.sliding_window_inference")
def test_fallback_gives_up_when_nothing_is_left_to_try(mock_swi, fake_empty_cache):
    """With nothing left to try, the error reaches the caller."""
    mock_swi.side_effect = torch.cuda.OutOfMemoryError("out of memory")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        predict_with_fallback(torch.zeros(1, 1, 4, 4, 4), 1, "cpu", roi_size=(2, 2, 2))


def test_batch_size_does_not_change_predictions():
    """Falling back to a smaller batch is safe because results stay identical."""
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 2, 3, padding=1), torch.nn.InstanceNorm3d(2)
    ).eval()
    image = torch.randn(1, 1, 24, 24, 24)
    with torch.no_grad():
        results = [
            predict_with_fallback(
                image,
                batch,
                "cpu",
                roi_size=(12, 12, 12),
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
            for batch in (1, 2, 4)
        ]
    assert torch.equal(results[0], results[1]) and torch.equal(results[0], results[2])
