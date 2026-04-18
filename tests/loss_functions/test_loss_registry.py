"""Unit tests for the loss function registry in MIST."""
import pytest
from torch import nn

# MIST imports.
from mist.loss_functions import loss_registry
from mist.loss_functions.base import SegmentationLoss


class DummyLoss(SegmentationLoss):
    """A dummy loss function for testing the registry."""

    def forward(self, y_true, y_pred, *args, **kwargs):
        return y_pred.sum()


def test_register_and_get_loss():
    """Test registering and retrieving a loss function."""
    name = "dummy_loss"
    # Register
    decorator = loss_registry.register_loss(name)
    decorated_class = decorator(DummyLoss)
    assert decorated_class is DummyLoss

    # Retrieve
    retrieved = loss_registry.get_loss(name)
    assert retrieved is DummyLoss

    # Instantiate and check behavior
    loss_instance = retrieved()
    assert isinstance(loss_instance, nn.Module)
    assert isinstance(loss_instance, SegmentationLoss)


def test_register_duplicate_loss_raises():
    """Test that registering the same loss name twice raises an error."""
    name = "duplicate_loss"
    loss_registry.register_loss(name)(DummyLoss)
    with pytest.raises(
        ValueError, match=f"Loss '{name}' is already registered."
    ):
        loss_registry.register_loss(name)(DummyLoss)


def test_get_unregistered_loss_raises():
    """Test that requesting an unknown loss raises ValueError."""
    with pytest.raises(ValueError, match="Loss 'not_found' is not registered"):
        loss_registry.get_loss("not_found")


def test_list_registered_losses_contains_expected():
    """Test that listing registered losses works."""
    name = "another_dummy"
    loss_registry.register_loss(name)(DummyLoss)
    registered = loss_registry.list_registered_losses()
    assert name in registered
