"""Tests for MIST probability-space ensemblers."""

import numpy as np
import pytest

from mist.inference.probability_ensemblers.base import AbstractProbabilityEnsembler
from mist.inference.probability_ensemblers.mean import MeanProbabilityEnsembler
from mist.inference.probability_ensemblers.probability_ensembler_registry import (
    get_probability_ensembler,
    list_probability_ensemblers,
    register_probability_ensembler,
)


class DummyProbabilityEnsembler(AbstractProbabilityEnsembler):
    """Minimal concrete implementation for base class tests."""

    def combine(self, probabilities: list[np.ndarray]) -> np.ndarray:
        return probabilities[0]


def test_dummy_probability_ensembler_call_and_repr():
    """Test __call__, __eq__, and __repr__ for AbstractProbabilityEnsembler."""
    x = np.random.rand(4, 5, 6, 3)
    dummy = DummyProbabilityEnsembler()
    assert np.array_equal(dummy([x]), x)
    assert dummy == DummyProbabilityEnsembler()
    assert repr(dummy) == "DummyProbabilityEnsembler(name='dummyprobabilityensembler')"


def test_probability_ensembler_hash_and_eq():
    """Test __hash__ and __eq__ for AbstractProbabilityEnsembler."""
    a = DummyProbabilityEnsembler()
    b = DummyProbabilityEnsembler()
    c = object()

    assert hash(a) == hash(b)
    assert a != c


# ---------------------------------------------------------------------------
# MeanProbabilityEnsembler tests
# ---------------------------------------------------------------------------


def test_mean_probability_ensembler_average():
    """Test averaging behavior of MeanProbabilityEnsembler."""
    x1 = np.ones((4, 5, 6, 3), dtype=np.float32)
    x2 = np.zeros((4, 5, 6, 3), dtype=np.float32)
    expected = 0.5 * (x1 + x2)

    ensembler = MeanProbabilityEnsembler()
    result = ensembler([x1, x2])
    assert np.allclose(result, expected)
    assert result.shape == expected.shape


def test_mean_probability_ensembler_empty_input():
    """Test that empty input raises ValueError."""
    ensembler = MeanProbabilityEnsembler()
    with pytest.raises(ValueError, match="requires at least one probability volume"):
        ensembler([])


def test_mean_probability_ensembler_mismatched_shapes_raises():
    """Test that mismatched shapes across volumes raises ValueError."""
    x1 = np.ones((4, 5, 6, 3), dtype=np.float32)
    x2 = np.ones((4, 5, 6, 2), dtype=np.float32)

    ensembler = MeanProbabilityEnsembler()
    with pytest.raises(ValueError, match="must have the same shape"):
        ensembler([x1, x2])


def test_mean_probability_ensembler_repr():
    """Test __repr__ for MeanProbabilityEnsembler."""
    ens = MeanProbabilityEnsembler()
    assert repr(ens) == "MeanProbabilityEnsembler(name='meanprobabilityensembler')"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_get_probability_ensembler_mean():
    """Test get_probability_ensembler returns a MeanProbabilityEnsembler instance."""
    ens = get_probability_ensembler("mean")
    assert isinstance(ens, MeanProbabilityEnsembler)


def test_registry_list_probability_ensemblers_includes_mean():
    """Test that 'mean' is in the probability ensembler registry."""
    assert "mean" in list_probability_ensemblers()


def test_registry_get_probability_ensembler_invalid_name():
    """Test that get_probability_ensembler raises KeyError for unknown name."""
    with pytest.raises(KeyError, match="not registered"):
        get_probability_ensembler("invalid")


def test_registry_rejects_invalid_class():
    """Test that registering a non-ensembler class raises TypeError."""

    class NotAnEnsembler:
        pass

    with pytest.raises(
        TypeError, match="must inherit from AbstractProbabilityEnsembler"
    ):

        @register_probability_ensembler("invalid_class")
        class Invalid(NotAnEnsembler):
            pass


def test_registry_rejects_duplicate_name():
    """Test that duplicate registration raises KeyError."""
    with pytest.raises(KeyError, match="already registered"):

        @register_probability_ensembler("mean")
        class DuplicateMean(AbstractProbabilityEnsembler):
            def combine(self, probabilities):
                return probabilities[0]
