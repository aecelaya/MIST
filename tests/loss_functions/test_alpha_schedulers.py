"""Unit tests for alpha schedulers."""

import pytest

from mist.loss_functions.alpha_schedulers import (
    ConstantScheduler,
    LinearScheduler,
    CosineScheduler,
    get_alpha_scheduler,
    get_default_scheduler_config,
    list_alpha_schedulers,
)

class TestConstantScheduler:
    """Tests for the ConstantScheduler."""

    def test_initialization_defaults(self):
        """Test default value initialization."""
        scheduler = ConstantScheduler()
        assert scheduler(0) == 0.5
        assert scheduler(9) == 0.5

    def test_custom_value(self):
        """Test with a specific alpha value."""
        scheduler = ConstantScheduler(value=0.8)
        assert scheduler(0) == 0.8
        assert scheduler(50) == 0.8
        assert scheduler(99) == 0.8

    def test_value_only(self):
        """ConstantScheduler accepts only value — no num_epochs needed."""
        scheduler = ConstantScheduler(value=0.3)
        assert scheduler(0) == 0.3


class TestLinearScheduler:
    """Tests for the LinearScheduler."""

    def test_pause_phase(self):
        """Test that alpha stays at start_val during the pause phase."""
        pause = 5
        scheduler = LinearScheduler(
            num_epochs=20,
            init_pause=pause,
            start_val=1.0,
            end_val=0.0
        )
        for i in range(pause + 1):
            assert scheduler(i) == 1.0

    def test_linear_decay_math(self):
        """Test the linear interpolation math."""
        scheduler = LinearScheduler(
            num_epochs=11,
            init_pause=0,
            start_val=1.0,
            end_val=0.0
        )
        assert scheduler(0) == 1.0
        assert scheduler(5) == pytest.approx(0.5)
        assert scheduler(10) == pytest.approx(0.0)

    def test_clamping_at_end(self):
        """Test that alpha clamps to end_val after decay duration."""
        scheduler = LinearScheduler(
            num_epochs=10, 
            init_pause=2,
            start_val=1.0,
            end_val=0.2
        )
        assert scheduler(100) == pytest.approx(0.2)

    def test_single_epoch_edge_case(self):
        """Test behavior with num_epochs=1."""
        scheduler = LinearScheduler(num_epochs=1, start_val=1.0, end_val=0.0)
        assert scheduler(0) == pytest.approx(1.0)


class TestCosineScheduler:
    """Tests for the CosineScheduler."""

    def test_cosine_decay_shape(self):
        """Test specific points on the cosine curve."""
        # Setup: 11 epochs (0-10). Pause 0.
        # Makes the math clean: progress goes from 0.0 to 1.0 over 10 steps.
        scheduler = CosineScheduler(
            num_epochs=11,
            init_pause=0,
            start_val=1.0,
            end_val=0.0
        )
        assert scheduler(0) == pytest.approx(1.0)
        assert scheduler(10) == pytest.approx(0.0)
        assert scheduler(5) == pytest.approx(0.5)
        assert scheduler(3) == pytest.approx(0.79389, rel=1e-3)

    def test_cosine_with_pause(self):
        """Test that cosine decay respects the init_pause."""
        scheduler = CosineScheduler(
            num_epochs=20,
            init_pause=5,
            start_val=1.0,
            end_val=0.0
        )
        assert scheduler(3) == 1.0
        assert scheduler(6) < 1.0


class TestSchedulerRegistry:
    """Tests for the factory and registry functions."""

    def test_get_linear_scheduler(self):
        """Test factory creation of LinearScheduler."""
        scheduler = get_alpha_scheduler("linear", num_epochs=10, start_val=0.5)
        assert isinstance(scheduler, LinearScheduler)
        assert scheduler.start_val == 0.5

    def test_get_cosine_scheduler(self):
        """Test factory creation of CosineScheduler."""
        scheduler = get_alpha_scheduler("cosine", num_epochs=10)
        assert isinstance(scheduler, CosineScheduler)

    def test_get_constant_scheduler(self):
        """Test factory creation of ConstantScheduler — num_epochs not passed."""
        scheduler = get_alpha_scheduler("constant", num_epochs=10, value=0.9)
        assert isinstance(scheduler, ConstantScheduler)
        assert scheduler.value == 0.9

    def test_invalid_scheduler_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_alpha_scheduler("non_existent_scheduler", num_epochs=10)

    def test_list_schedulers(self):
        """Test listing available schedulers."""
        schedulers = list_alpha_schedulers()
        assert "linear" in schedulers
        assert "cosine" in schedulers
        assert "constant" in schedulers
        assert schedulers == sorted(schedulers)


class TestGetDefaultSchedulerConfig:
    """Tests for the get_default_scheduler_config utility."""

    def test_linear_defaults(self):
        """Linear scheduler config contains expected default params."""
        cfg = get_default_scheduler_config("linear")
        assert cfg["name"] == "linear"
        assert cfg["params"]["init_pause"] == 5
        assert cfg["params"]["start_val"] == 1.0
        assert cfg["params"]["end_val"] == 0.0

    def test_cosine_defaults(self):
        """Cosine scheduler config contains expected default params."""
        cfg = get_default_scheduler_config("cosine")
        assert cfg["name"] == "cosine"
        assert cfg["params"]["init_pause"] == 5
        assert cfg["params"]["start_val"] == 1.0
        assert cfg["params"]["end_val"] == 0.0

    def test_constant_defaults(self):
        """Constant scheduler config contains value but not num_epochs."""
        cfg = get_default_scheduler_config("constant")
        assert cfg["name"] == "constant"
        assert cfg["params"]["value"] == 0.5
        assert "num_epochs" not in cfg["params"]

    def test_no_num_epochs_in_any_params(self):
        """num_epochs is runtime context and must never appear in params."""
        for name in list_alpha_schedulers():
            cfg = get_default_scheduler_config(name)
            assert "num_epochs" not in cfg["params"], (
                f"num_epochs leaked into params for '{name}'"
            )

    def test_config_roundtrips_through_get_alpha_scheduler(self):
        """Default config can be fed directly to get_alpha_scheduler."""
        for name in list_alpha_schedulers():
            cfg = get_default_scheduler_config(name)
            scheduler = get_alpha_scheduler(
                cfg["name"], num_epochs=100, **cfg["params"]
            )
            assert callable(scheduler)

    def test_invalid_name_raises(self):
        """Unknown scheduler name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_default_scheduler_config("nonexistent")
