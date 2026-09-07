"""Unit tests for the model registry mechanism in MIST."""

import pytest

# MIST imports.
from mist.models.model_registry import (
    MODEL_REGISTRY,
    get_model_from_registry,
    list_registered_models,
    register_model,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Isolate the model registry for each test in this file.

    Snapshots and restores the *real* registrations afterward, rather than
    just clearing to empty -- mist.models registers every real model
    (nnunet, mednext, mgnets, swinunetr variants) exactly once, at that
    package's own import time, so other tests rely on those still being
    there once this file's tests are done. Clearing to empty permanently
    (the bug this replaces -- see the identical fix and rationale in
    tests/data_loading/test_data_loader_registry.py's clear_registry) left
    list_registered_models() empty for any test running afterward in the
    same session, which is exactly the trap
    tests/regression/cpu_rocm/test_stage4_cpu_end_to_end.py fell into: its
    real `mist_train --model nnunet-pocket` CLI parse failed with
    "invalid choice: 'nnunet-pocket' (choose from )" whenever this file's
    tests happened to run first.
    """
    saved = dict(MODEL_REGISTRY)
    MODEL_REGISTRY.clear()
    yield
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(saved)


def test_register_model_success():
    """Test that a model can be successfully registered and retrieved."""

    @register_model("dummy")
    def build_dummy():
        return "dummy_model"

    assert "dummy" in MODEL_REGISTRY
    assert get_model_from_registry("dummy") == "dummy_model"


def test_register_model_duplicate_name():
    """Test that registering a model with a duplicate name raises an error."""

    @register_model("dup")
    def model_one():
        return "one"

    with pytest.raises(ValueError, match="Model 'dup' is already registered"):

        @register_model("dup")
        def model_two():
            return "two"


def test_get_model_from_registry_not_found():
    """Test that requesting an unregistered model raises an error."""
    with pytest.raises(ValueError, match="Model 'missing' is not registered"):
        get_model_from_registry("missing")


def test_list_registered_models_returns_sorted():
    """Test that list_registered_models returns sorted model names."""

    @register_model("z_model")
    def z():
        return None

    @register_model("a_model")
    def a():
        return None

    registered = list_registered_models()
    assert registered == ["a_model", "z_model"]
