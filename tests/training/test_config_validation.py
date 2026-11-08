"""Tests for the warnings about mistakes in config.json."""

import copy

import pytest

from mist.analyze_data.analyzer_utils import build_base_config
from mist.training.config_validation import find_config_problems


@pytest.fixture(name="config")
def fixture_config() -> dict:
    """A config with the values that mist_analyze would fill in."""
    config = build_base_config()
    config["evaluation"] = {"tumor": {"labels": [1], "metrics": {"dice": {}}}}
    config["planning"] = {"amp_auto": True}
    return config


def test_a_new_config_has_no_problems(config):
    """Nothing MIST writes itself is reported."""
    assert find_config_problems(config) == []


def test_a_config_from_mist_analyze_has_no_problems(config):
    """Values that mist_analyze and mist_train fill in are accepted."""
    config["training"]["amp"] = True
    config["training"]["batch_size_per_gpu"] = 2
    config["training"]["folds"] = [0, 1]
    config["training"]["hardware"]["num_gpus"] = 2
    config["spatial_config"]["patch_size"] = [64, 64, 64]
    assert find_config_problems(config) == []


def test_a_missing_key_is_not_a_problem(config):
    """Older configs don't have the newest settings."""
    del config["training"]["compile"]
    del config["training"]["augmentation"]
    assert find_config_problems(config) == []


def test_a_misspelled_key_suggests_the_right_one(config):
    """The warning names the key and its likely spelling."""
    config["training"]["compil"] = True
    assert find_config_problems(config) == [
        "MIST doesn't use 'training.compil', so it has no effect. Did you mean 'training.compile'?"
    ]


def test_a_nested_misspelled_key_is_found(config):
    """Sections inside sections are checked."""
    config["training"]["augmentation"]["transforms"]["flip"] = True
    problems = find_config_problems(config)
    assert len(problems) == 1
    assert "'training.augmentation.transforms.flip'" in problems[0]
    assert "'training.augmentation.transforms.flips'" in problems[0]


def test_the_older_zoom_key_is_accepted(config):
    """Configs from before 2.5.0 have it, and it's read as scaling."""
    config["training"]["augmentation"]["transforms"]["zoom"] = True
    assert find_config_problems(config) == []


def test_an_unrelated_key_gets_no_suggestion(config):
    """A suggestion only appears when something is close."""
    config["zzz"] = 1
    assert find_config_problems(config) == ["MIST doesn't use 'zzz', so it has no effect."]


@pytest.mark.parametrize(
    "section",
    [
        ("model", "params"),
        ("evaluation",),
        ("planning",),
        ("inference", "inferer", "params"),
    ],
)
def test_free_form_sections_can_hold_any_key(config, section):
    """What's in these sections depends on the chosen model, metric, or inferer."""
    target = config
    for key in section:
        target = target[key]
    target["anything"] = 1
    assert find_config_problems(config) == []


def test_composite_loss_weighting_can_hold_any_key(config):
    """Each alpha scheduler has its own parameters."""
    config["training"]["loss"]["composite_loss_weighting"] = {"name": "linear", "init_pause": 5}
    assert find_config_problems(config) == []


@pytest.mark.parametrize(
    ("section", "key", "value", "expected"),
    [
        ("training", "compile", "false", "should be true or false"),
        ("training", "amp", "yes", 'should be true, false, or "auto"'),
        ("training", "epochs", "1000", "should be a whole number"),
        ("training", "epochs", 10.5, "should be a whole number"),
        ("training", "epochs", True, "should be a whole number"),
        ("training", "learning_rate", "1e-3", "should be a number"),
        ("training", "val_percent", 20, "should be a number from 0 to 1"),
        ("training", "dali_foreground_prob", -0.1, "should be a number from 0 to 1"),
        ("training", "augmentation", True, "should be a section"),
        ("training", "gpu_memory_budget_gb", "8", "should be a number of GiB above 0"),
        ("training", "gpu_memory_budget_gb", 0, "should be a number of GiB above 0"),
    ],
)
def test_invalid_values_are_found(config, section, key, value, expected):
    """Values that would be misread or would fail later are reported."""
    config[section][key] = value
    problems = find_config_problems(config)
    assert len(problems) == 1
    assert f"'{section}.{key}'" in problems[0]
    assert expected in problems[0]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("amp", "auto"),
        ("amp", False),
        ("epochs", 1000.0),
        ("learning_rate", 1),
        ("val_percent", 0),
        ("dali_foreground_prob", 1),
        ("gpu_memory_budget_gb", 24),
        ("gpu_memory_budget_gb", None),
    ],
)
def test_valid_values_are_accepted(config, key, value):
    """Each kind of check lets a good value through."""
    config["training"][key] = value
    assert find_config_problems(config) == []


def test_every_listed_key_is_in_the_base_config():
    """A renamed setting can't leave its check behind unnoticed."""
    from mist.training import config_validation as cv

    def get(path):
        target = build_base_config()
        for key in path.split("."):
            target = target[key]
        return target

    for path in cv._BOOLEANS | cv._INTEGERS | cv._NUMBERS | cv._FRACTIONS | cv._FREE_FORM_SECTIONS:
        if path not in cv._EXTRA_SECTIONS:
            get(path)


def test_the_config_is_not_changed(config):
    """Validation only reads."""
    config["training"]["compil"] = True
    before = copy.deepcopy(config)
    find_config_problems(config)
    assert config == before
