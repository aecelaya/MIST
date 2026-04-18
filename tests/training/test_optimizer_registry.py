"""Tests for the optimizer registry in MIST."""
import pytest
import torch
from torch import nn

# MIST imports.
from mist.training.optimizers import optimizer_registry as reg
from mist.training.optimizers.optimizer_constants import OptimizerConstants


@pytest.fixture
def model():
    """Simple model to supply parameter iterables to the optimizers."""
    return nn.Linear(4, 2)


def test_list_optimizers_sorted():
    """Should be sorted & contain all registered names."""
    assert reg.list_optimizers() == ["adam", "adamw", "sgd"]


def test_get_optimizer_adam_hyperparams(model):
    """Test getting Adam optimizer with specific hyperparameters."""
    opt = reg.get_optimizer(
        "adam",
        params=model.parameters(),
        learning_rate=0.005,
        weight_decay=0.01,
        eps=1e-7,
    )
    assert isinstance(opt, torch.optim.Adam)
    pg = opt.param_groups[0]
    assert pg["lr"] == pytest.approx(0.005)
    assert pg["weight_decay"] == pytest.approx(0.01)
    # eps is stored in defaults (and usually mirrored in param_groups too).
    assert opt.defaults["eps"] == pytest.approx(1e-7)


def test_get_optimizer_adamw_case_insensitive_and_hyperparams(model):
    """Test getting AdamW optimizer with specific hyperparameters."""
    opt = reg.get_optimizer(
        "ADAMW",  # case-insensitive.
        params=model.parameters(),
        learning_rate=0.002,
        weight_decay=0.1,
        eps=1e-6,
    )
    assert isinstance(opt, torch.optim.AdamW)
    pg = opt.param_groups[0]
    assert pg["lr"] == pytest.approx(0.002)
    assert pg["weight_decay"] == pytest.approx(0.1)
    assert opt.defaults["eps"] == pytest.approx(1e-6)


def test_get_optimizer_sgd_uses_constants_and_ignores_eps(model):
    """Test getting SGD optimizer with constants and ignoring eps."""
    opt = reg.get_optimizer(
        "sgd",
        params=model.parameters(),
        learning_rate=0.1,
        weight_decay=0.05,
        eps=123.456,  # Intentionally meaningless for SGD.
    )
    assert isinstance(opt, torch.optim.SGD)
    pg = opt.param_groups[0]
    assert pg["lr"] == pytest.approx(0.1)
    assert pg["weight_decay"] == pytest.approx(0.05)
    # Momentum/nesterov should come from OptimizerConstants.
    assert pg["momentum"] == pytest.approx(OptimizerConstants.SGD_MOMENTUM)
    assert pg["nesterov"] == OptimizerConstants.SGD_NESTEROV


def test_get_optimizer_unknown_raises_with_list(model):
    """Test unknown optimizer raises an error with available names."""
    with pytest.raises(ValueError) as exc:
        reg.get_optimizer(
            "rmsprop",
            params=model.parameters(),
            learning_rate=0.01,
            weight_decay=0.0,
            eps=1e-8,
        )
    msg = str(exc.value)
    assert "Unknown optimizer 'rmsprop'" in msg
    # Should include available names in the error message.
    for name in reg.list_optimizers():
        assert name in msg
