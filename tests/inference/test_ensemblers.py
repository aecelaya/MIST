"""Tests for MIST ensemblers."""
import torch
import pytest

# MIST imports.
from mist.inference.ensemblers.base import AbstractEnsembler
from mist.inference.ensemblers.mean import MeanEnsembler
from mist.inference.ensemblers.ensembler_registry import (
    get_ensembler,
    list_ensemblers,
    register_ensembler
)


class DummyEnsembler(AbstractEnsembler):
    """Dummy Ensembler for base class testing."""

    def combine(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        return predictions[0]


def test_dummy_ensembler_call_and_repr():
    """Test __call__, __eq__, and __repr__ for AbstractEnsembler."""
    x = torch.rand(1, 2, 3, 4, 5)
    dummy = DummyEnsembler()
    assert torch.equal(dummy([x]), x)
    assert dummy == DummyEnsembler()
    assert repr(dummy) == "DummyEnsembler(name='dummyensembler')"


# MeanEnsembler tests.
def test_mean_ensembler_average():
    """Test averaging behavior of MeanEnsembler."""
    x1 = torch.ones(1, 2, 3, 4, 4)
    x2 = torch.zeros(1, 2, 3, 4, 4)
    expected = 0.5 * (x1 + x2)

    ensembler = MeanEnsembler()
    result = ensembler([x1, x2])
    assert torch.allclose(result, expected)
    assert result.shape == expected.shape  # Should be (1, 2, 3, 4, 4).


def test_mean_ensembler_empty_input():
    """Test that empty input raises ValueError."""
    ensembler = MeanEnsembler()
    with pytest.raises(ValueError, match="requires at least one prediction"):
        ensembler([])


def test_mean_ensembler_repr():
    """Test __repr__ for MeanEnsembler."""
    ens = MeanEnsembler()
    assert repr(ens) == "MeanEnsembler(name='meanensembler')"


# Registry tests.
def test_registry_get_ensembler_mean():
    """Test get_ensembler returns the MeanEnsembler instance."""
    ens = get_ensembler("mean")
    assert isinstance(ens, MeanEnsembler)


def test_registry_list_ensemblers_includes_mean():
    """Test that 'mean' is in the ensembler registry."""
    assert "mean" in list_ensemblers()


def test_registry_get_ensembler_invalid_name():
    """Test that get_ensembler raises KeyError for unknown name."""
    with pytest.raises(KeyError, match="Ensembler 'invalid' is not registered"):
        get_ensembler("invalid")


def test_registry_rejects_invalid_class():
    """Test that registering a non-Ensembler class raises TypeError."""
    class NotAnEnsembler:
        pass

    with pytest.raises(TypeError, match="must inherit from AbstractEnsembler"):
        @register_ensembler("invalid_class")
        class Invalid(NotAnEnsembler):
            pass


def test_registry_rejects_duplicate_name():
    """Test that duplicate registration raises KeyError."""
    with pytest.raises(KeyError, match="already registered"):
        @register_ensembler("mean")
        class DuplicateMean(AbstractEnsembler):
            def combine(self, predictions):
                return predictions[0]


def test_ensembler_hash_and_eq():
    """Test __hash__ and __eq__ for AbstractEnsembler."""
    a = DummyEnsembler()
    b = DummyEnsembler()
    c = object()  # Not an AbstractEnsembler

    # Hashes should match for same name.
    assert hash(a) == hash(b)

    # __eq__ should return False for non-ensembler comparison.
    assert a != c
