"""Tests for the learning rate scheduler registry in MIST."""
import pytest
import torch

# MIST imports.
from mist.training.lr_schedulers import lr_scheduler_registry as reg
from mist.training.lr_schedulers.lr_schedulers_constants import (
    LRSchedulerConstants,
)


def _make_optimizer(lr: float=0.1):
    """Minimal param to satisfy torch optimizer."""
    param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    return torch.optim.SGD([param], lr=lr)


def test_list_lr_schedulers_sorted_and_complete():
    """Should be sorted & contain all registered names."""
    names = reg.list_lr_schedulers()
    assert names == sorted(names)
    # Sanity: expected set.
    assert set(names) >= {"cosine", "polynomial", "constant"}


def test_get_lr_scheduler_cosine_type_and_tmax():
    """Test getting CosineAnnealingLR with specific T_max."""
    opt = _make_optimizer(lr=0.1)
    epochs = 5
    sched = reg.get_lr_scheduler("cosine", optimizer=opt, epochs=epochs)

    # Type/attrs without stepping (no warning triggered).
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == epochs

    # If you want to exercise a step, do it after optimizer.step().
    opt.step()
    sched.step()


def test_get_lr_scheduler_polynomial_type_and_power():
    """Test getting PolynomialLR with specific power."""
    opt = _make_optimizer(lr=0.1)
    epochs = 7
    sched = reg.get_lr_scheduler("polynomial", optimizer=opt, epochs=epochs)

    assert isinstance(sched, torch.optim.lr_scheduler.PolynomialLR)
    # Power matches our constants.
    assert sched.power == pytest.approx(LRSchedulerConstants.POLYNOMIAL_DECAY)


def test_get_lr_scheduler_constant_type_and_factor():
    """Test getting ConstantLR with specific factor."""
    base_lr = 0.2
    opt = _make_optimizer(lr=base_lr)
    sched = reg.get_lr_scheduler("constant", optimizer=opt, epochs=10)

    assert isinstance(sched, torch.optim.lr_scheduler.ConstantLR)

    factor = LRSchedulerConstants.CONSTANT_LR_FACTOR
    opt.step()
    sched.step()

    current_lr = opt.param_groups[0]["lr"]
    assert current_lr == pytest.approx(base_lr * factor)


def test_get_lr_scheduler_invalid_raises():
    """Test unknown scheduler raises an error with available names."""
    opt = _make_optimizer()
    with pytest.raises(ValueError) as ei:
        reg.get_lr_scheduler("nope", optimizer=opt, epochs=3)
    msg = str(ei.value)
    assert "Unknown scheduler" in msg and "Available:" in msg


def test_get_lr_scheduler_warmup_returns_sequential():
    """warmup_epochs > 0 returns a SequentialLR chaining warmup + main."""
    opt = _make_optimizer(lr=0.1)
    sched = reg.get_lr_scheduler("cosine", optimizer=opt, epochs=10, warmup_epochs=3)
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


def test_get_lr_scheduler_warmup_ramps_lr():
    """LR increases during warmup then follows main schedule."""
    base_lr = 0.1
    opt = _make_optimizer(lr=base_lr)
    sched = reg.get_lr_scheduler("cosine", optimizer=opt, epochs=10, warmup_epochs=3)

    lrs = []
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # LR should increase during warmup (steps 0-2) then decrease.
    assert lrs[0] < lrs[1] < lrs[2]
    # After warmup the cosine decay should bring LR below the peak.
    assert lrs[-1] < lrs[2]


def test_get_lr_scheduler_warmup_zero_equals_no_warmup():
    """warmup_epochs=0 returns the plain scheduler, not a SequentialLR."""
    opt = _make_optimizer(lr=0.1)
    sched = reg.get_lr_scheduler("cosine", optimizer=opt, epochs=10, warmup_epochs=0)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_get_lr_scheduler_warmup_negative_raises():
    """Negative warmup_epochs raises ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_epochs must be >= 0"):
        reg.get_lr_scheduler("cosine", optimizer=opt, epochs=10, warmup_epochs=-1)


def test_get_lr_scheduler_warmup_gte_epochs_raises():
    """warmup_epochs >= epochs raises ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_epochs"):
        reg.get_lr_scheduler("cosine", optimizer=opt, epochs=5, warmup_epochs=5)
