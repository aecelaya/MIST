"""Unit tests for the data loader registry mechanism in MIST."""

from types import SimpleNamespace

import pytest

# MIST imports.
from mist.data_loading.data_loader_registry import (
    DATA_LOADER_REGISTRY,
    get_data_loader_from_registry,
    list_registered_data_loaders,
    register_data_loader,
)


def _fake_loader_module(**overrides):
    """Build a stand-in module exposing the three required functions."""
    defaults = {
        "get_training_dataset": lambda **kwargs: "train",
        "get_validation_dataset": lambda **kwargs: "val",
        "get_test_dataset": lambda **kwargs: "test",
    }
    return SimpleNamespace(**{**defaults, **overrides})


@pytest.fixture(autouse=True)
def clear_registry():
    """Ensure the data loader registry is clean before and after each test."""
    DATA_LOADER_REGISTRY.clear()
    yield
    DATA_LOADER_REGISTRY.clear()


def test_register_data_loader_success():
    """A module exposing all three required functions registers successfully."""
    module = _fake_loader_module()
    register_data_loader("dummy", module)
    assert "dummy" in DATA_LOADER_REGISTRY
    assert get_data_loader_from_registry("dummy") is module


def test_register_data_loader_duplicate_name():
    """Registering a loader with a duplicate name raises an error."""
    register_data_loader("dup", _fake_loader_module())
    with pytest.raises(ValueError, match="Data loader 'dup' is already registered"):
        register_data_loader("dup", _fake_loader_module())


@pytest.mark.parametrize(
    "missing_fn",
    ["get_training_dataset", "get_validation_dataset", "get_test_dataset"],
)
def test_register_data_loader_missing_function_raises(missing_fn):
    """A module missing one of the three required functions is rejected."""
    module = _fake_loader_module(**{missing_fn: None})
    with pytest.raises(ValueError, match=f"missing required function.*{missing_fn}"):
        register_data_loader("incomplete", module)
    assert "incomplete" not in DATA_LOADER_REGISTRY


def test_get_data_loader_from_registry_not_found():
    """Requesting an unregistered loader raises an error."""
    with pytest.raises(ValueError, match="Data loader 'missing' is not registered"):
        get_data_loader_from_registry("missing")


def test_list_registered_data_loaders_returns_sorted():
    """list_registered_data_loaders returns sorted loader names."""
    register_data_loader("z_loader", _fake_loader_module())
    register_data_loader("a_loader", _fake_loader_module())
    assert list_registered_data_loaders() == ["a_loader", "z_loader"]
