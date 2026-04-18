"""Tests for MIST TTA transforms and strategies."""
import torch
import pytest

# MIST imports.
from mist.inference.tta.transforms import (
    AbstractTransform,
    get_transform,
    register_transform,
)
from mist.inference.tta.strategies import (
    TTAStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)


class DummyTransform(AbstractTransform):
    """Dummy transform for base class testing."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image + 1

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction - 1


def test_abstract_transform_call_and_repr():
    """Test __call__, __eq__, and __repr__ for AbstractTransform."""
    x = torch.tensor([1.0])
    t = DummyTransform()
    assert torch.equal(t(x), x + 1)
    assert t == DummyTransform()
    assert repr(t) == "DummyTransform(name='dummytransform')"


def test_transform_hash_and_eq():
    """Test __hash__ and __eq__ for AbstractTransform."""
    a = DummyTransform()
    b = DummyTransform()
    c = object()  # Not a transform

    assert hash(a) == hash(b)
    assert a != c


# Parameterized tests for TTA flips.
@pytest.mark.parametrize("name, dims", [
    ("identity", []),
    ("flip_x", [2]),
    ("flip_y", [3]),
    ("flip_z", [4]),
    ("flip_xy", [2, 3]),
    ("flip_xz", [2, 4]),
    ("flip_yz", [3, 4]),
    ("flip_xyz", [2, 3, 4]),
])
def test_transform_forward_inverse_symmetry(name, dims):
    """Test that forward + inverse = identity for registered transforms."""
    transform = get_transform(name)
    input_tensor = torch.rand(1, 1, 4, 4, 4)
    output = transform.inverse(transform.forward(input_tensor))
    assert torch.allclose(output, input_tensor, atol=1e-6)


# TTAStrategy tests.
class DummyStrategy(TTAStrategy):
    def get_transforms(self):
        return [get_transform("identity")]


def test_tta_strategy_call_and_repr():
    """Test __call__, __eq__, and __repr__ for TTAStrategy."""
    strategy = DummyStrategy()
    transforms = strategy()
    assert len(transforms) == 1
    assert strategy == DummyStrategy()
    assert repr(strategy) == "DummyStrategy(name='dummystrategy')"


def test_tta_strategy_hash_and_eq():
    """Test __hash__ and __eq__ for TTAStrategy."""
    a = DummyStrategy()
    b = DummyStrategy()
    c = object()  # Not a strategy

    assert hash(a) == hash(b)
    assert a != c


def test_get_strategy_valid():
    """Test that get_strategy returns a registered strategy."""
    s = get_strategy("none")
    assert isinstance(s, TTAStrategy)


def test_list_strategies_contains_registered():
    """Test that all registered strategies are in the list."""
    assert "none" in list_strategies()
    assert "all_flips" in list_strategies()


def test_get_strategy_invalid_name():
    """Test that requesting an unknown strategy raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        get_strategy("not_a_strategy")


def test_register_strategy_invalid_class():
    """Test that registering a non-strategy raises TypeError."""
    class NotAStrategy:
        pass

    with pytest.raises(TypeError, match="must inherit from TTAStrategy"):
        @register_strategy("bad")
        class Bad(NotAStrategy):
            pass


def test_register_strategy_duplicate_name():
    """Test that registering a duplicate strategy name raises KeyError."""
    with pytest.raises(KeyError, match="already registered"):
        @register_strategy("none")
        class Duplicate(TTAStrategy):
            def get_transforms(self):
                return []


def test_no_tta_strategy_returns_identity():
    """Test that NoTTAStrategy returns only the identity transform."""
    strategy = get_strategy("none")
    transforms = strategy.get_transforms()
    assert len(transforms) == 1
    assert transforms[0].name == "identitytransform"


def test_all_flips_strategy_returns_all_flip_transforms():
    """Test that AllFlipsStrategy returns all expected flip transforms."""
    expected_names = {
        "identitytransform",
        "flipxtransform",
        "flipytransform",
        "flipztransform",
        "flipxytransform",
        "flipxztransform",
        "flipyztransform",
        "flipxyztransform",
    }
    strategy = get_strategy("all_flips")
    transforms = strategy.get_transforms()
    returned_names = {t.name for t in transforms}
    assert returned_names == expected_names


# Transform registry error handling.
def test_get_transform_invalid_name():
    """Test that get_transform raises KeyError for unknown name."""
    with pytest.raises(KeyError, match="not registered"):
        get_transform("bad_transform")


def test_register_transform_invalid_class():
    """Test that registering a non-transform raises TypeError."""
    class NotATransform:
        pass

    with pytest.raises(TypeError, match="must subclass AbstractTransform"):
        @register_transform("bad")
        class Bad(NotATransform):
            pass


def test_register_transform_duplicate_name():
    """Test that duplicate transform registration raises KeyError."""
    with pytest.raises(KeyError, match="already registered"):
        @register_transform("identity")
        class DuplicateTransform(AbstractTransform):
            def forward(self, image): return image
            def inverse(self, prediction): return prediction
