"""Warnings for mistakes in config.json, which people often edit by hand.

MIST reads many settings with a default, so a misspelled key would otherwise be
ignored without a word. These checks only warn, so a config that trains today
keeps training. Keys missing from a config aren't reported, since older configs
don't have the newest settings.
"""

import difflib
from typing import Any

from mist.analyze_data.analyzer_utils import build_base_config

# Top-level sections that aren't in the base config: mist_analyze adds the first,
# and the first mist_train adds the second.
_EXTRA_SECTIONS = frozenset({"evaluation", "planning"})

# Sections whose contents depend on the chosen model, scheduler, or inferer.
_FREE_FORM_SECTIONS = frozenset(
    {
        "evaluation",
        "model.params",
        "planning",
        "training.loss.composite_loss_weighting",
        "inference.inferer.params",
    }
)

# Older names that MIST still reads.
_LEGACY_KEYS = frozenset({"training.augmentation.transforms.zoom"})

_TRANSFORMS = (
    "flips",
    "rotation",
    "scaling",
    "noise",
    "blur",
    "brightness",
    "contrast",
    "low_resolution",
    "gamma",
)

# A string here would count as true, even "false".
_BOOLEANS = frozenset(
    {
        "preprocessing.skip",
        "preprocessing.compute_dtms",
        "preprocessing.normalize_dtms",
        "training.compile",
        "training.augmentation.enabled",
        "inference.tta.enabled",
        *(f"training.augmentation.transforms.{name}" for name in _TRANSFORMS),
    }
)

_INTEGERS = frozenset(
    {
        "training.seed",
        "training.nfolds",
        "training.validate_every",
        "training.epochs",
        "training.min_steps_per_epoch",
        "training.warmup_epochs",
        "training.hardware.num_cpu_workers",
    }
)

_NUMBERS = frozenset(
    {
        "training.learning_rate",
        "training.l2_penalty",
        "training.grad_clip_norm",
    }
)

_FRACTIONS = frozenset(
    {
        "training.val_percent",
        "training.dali_foreground_prob",
    }
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_problem(path: str, value: Any) -> str | None:
    """Describe what's wrong with a setting's value, if anything."""
    if path in _BOOLEANS and not isinstance(value, bool):
        return f"'{path}' should be true or false, but is {value!r}."
    if path == "training.amp" and not (isinstance(value, bool) or value in ("auto", "fp16")):
        return f'\'{path}\' should be true, false, "auto", or "fp16", but is {value!r}.'
    if path in _INTEGERS and not (_is_number(value) and float(value).is_integer()):
        return f"'{path}' should be a whole number, but is {value!r}."
    if path in _NUMBERS and not _is_number(value):
        return f"'{path}' should be a number, but is {value!r}."
    if path in _FRACTIONS and not (_is_number(value) and 0 <= value <= 1):
        return f"'{path}' should be a number from 0 to 1, but is {value!r}."
    if path == "training.gpu_memory_budget_gb" and not (
        value is None or (_is_number(value) and value > 0)
    ):
        return f"'{path}' should be a number of GiB above 0, or null, but is {value!r}."
    return None


def _check_section(
    section: dict[str, Any], schema: dict[str, Any], prefix: str, problems: list[str]
) -> None:
    """Add the problems in one section of the config, and in those inside it."""
    for key, value in section.items():
        path = f"{prefix}{key}"
        if path in _FREE_FORM_SECTIONS or path in _LEGACY_KEYS:
            continue

        if key not in schema:
            problem = f"MIST doesn't use '{path}', so it has no effect."
            close = difflib.get_close_matches(key, schema, n=1)
            if close:
                problem += f" Did you mean '{prefix}{close[0]}'?"
            problems.append(problem)
        elif isinstance(schema[key], dict):
            if isinstance(value, dict):
                _check_section(value, schema[key], f"{path}.", problems)
            else:
                problems.append(f"'{path}' should be a section, but is {value!r}.")
        else:
            problem = _value_problem(path, value)
            if problem:
                problems.append(problem)


def find_config_problems(config: dict[str, Any]) -> list[str]:
    """Find unknown keys and invalid values in a MIST config.

    Args:
        config: The contents of config.json.

    Returns:
        A description of each problem. Empty if there are none.
    """
    schema = build_base_config()
    for section in _EXTRA_SECTIONS:
        schema[section] = {}

    problems: list[str] = []
    _check_section(config, schema, "", problems)
    return problems
