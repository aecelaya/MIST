"""Tests for GPU-memory-based patch and batch size planning."""

import argparse
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from mist.models.model_registry import get_model_from_registry
from mist.training import patch_planning as pp
from mist.training.planning_constants import PlanningConstants as constants

GIB = 1024**3
ONE_OF_EACH_FAMILY = [
    "nnunet-plain",
    "nnunet-res",
    "nnunet-pocket",
    "resenc",
    "fmgnet",
    "mednext-small",
    "swinunetr-small",
    "primus-medium",
]


def _plan(architecture="nnunet-plain", memory_gb=16, amp=True, median=(400, 400, 300), **kwargs):
    defaults = {
        "in_channels": 1,
        "out_channels": 3,
        "target_spacing": [1.0, 1.0, 1.0],
        "num_cases": 500,
    }
    return pp.plan_patch_and_batch(
        architecture=architecture,
        median_size=list(median),
        gpu_memory_bytes=memory_gb * GIB,
        amp=amp,
        **{**defaults, **kwargs},
    )


# The estimate.


def test_meta_estimate_matches_a_real_forward_pass():
    """Counting on the meta device gives what real tensors give."""
    args = ("nnunet-plain", 2, 3, [32, 32, 32], [1.0, 1.0, 1.0])
    meta = pp.estimate_network(*args)
    real = pp._count_activations(*args, "cpu")
    assert meta == real


def test_estimate_grows_with_patch_and_output_is_per_sample():
    """A larger patch means more activations, and parameters don't double."""
    small = pp.estimate_network("nnunet-plain", 1, 3, [64, 64, 64], [1.0, 1.0, 1.0])
    large = pp.estimate_network("nnunet-plain", 1, 3, [128, 128, 128], [1.0, 1.0, 1.0])
    assert large.activations > 7 * small.activations
    assert large.parameters < 2 * small.parameters


@pytest.mark.parametrize("name", ["primus-medium", "swinunetr-small"])
def test_estimate_falls_back_to_cpu_and_scales_by_volume(name):
    """Networks that can't run on the meta device are measured and scaled."""
    base = pp.estimate_network(name, 1, 3, [64, 64, 64], [1.0, 1.0, 1.0])
    double = pp.estimate_network(name, 1, 3, [128, 64, 64], [1.0, 1.0, 1.0])
    assert base.activations > 0
    assert double.activations == pytest.approx(2 * base.activations)


def test_invalid_patch_raises_value_error():
    """A patch the network can't use isn't quietly estimated."""
    with pytest.raises(ValueError):
        pp.estimate_network("nnunet-plain", 1, 3, [288, 288, 288], [1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "architecture, patch, batch, amp, steady_gib",
    [
        ("nnunet-plain", [128, 128, 128], 4, False, 18.1),
        ("swinunetr-small", [96, 96, 96], 4, False, 13.5),
    ],
)
def test_estimate_covers_measured_steady_memory(architecture, patch, batch, amp, steady_gib):
    """The estimate stays above steady memory measured on a GPU, with some margin."""
    estimate = pp.estimate_network(architecture, 4, 5, patch, [1.0, 1.0, 1.0])
    estimated_gib = pp.estimated_bytes(estimate, batch, amp) / GIB
    assert steady_gib / estimated_gib == pytest.approx(0.8, abs=0.1)


@pytest.mark.parametrize(
    "architecture, expected",
    [
        ("nnunet-plain", (12.0, 0.55)),
        ("swinunetr-small", (17.0, 0.9)),
        ("primus-medium", (12.0, 0.7)),
        ("primusv2-small", (12.0, 0.55)),
    ],
)
def test_network_family_sets_memory_per_activation(architecture, expected):
    """Each family gets its own values, and primusv2 isn't taken for primus."""
    estimate = pp.estimate_network(architecture, 1, 5, [64, 64, 64], [1.0, 1.0, 1.0])
    assert (estimate.bytes_per_activation, estimate.bf16_factor) == expected


def test_bf16_lowers_the_estimate_and_batch_raises_it():
    """Mixed precision takes less memory, and each sample adds the same."""
    estimate = pp.NetworkEstimate(1e8, 10**7)
    assert pp.estimated_bytes(estimate, 2, True) < pp.estimated_bytes(estimate, 2, False)
    one, two, three = (pp.estimated_bytes(estimate, b, False) for b in (1, 2, 3))
    assert two - one == pytest.approx(three - two)


# Patch rules.


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mednext-base", [16, 16, 16]),
        ("swinunetr-base", [32, 32, 32]),
        ("primus-large", [8, 8, 8]),
        ("nnunet-plain", [64, 64, 64]),
        ("nnunet-res", [32, 32, 32]),
    ],
)
def test_axis_divisors(name, expected):
    """Each family has its own rule; plain has no depth limit."""
    assert pp.axis_divisors(name, [256, 256, 256], [1.0, 1.0, 1.0]) == expected


def test_snap_down_handles_depth_changing_with_size():
    """288 needs a multiple of 64 under plain, but not under the residual network."""
    assert pp._snap_down("nnunet-plain", [288, 288, 288], [1.0, 1.0, 1.0]) == [256, 256, 256]
    assert pp._snap_down("nnunet-res", [288, 288, 288], [1.0, 1.0, 1.0]) == [288, 288, 288]


def test_shrink_keeps_valid_sizes_a_large_divisor_would_skip():
    """From 256, plain's next size is 224, even though 256 needs a multiple of 64."""
    smaller = pp._shrink_axis("nnunet-plain", [256, 256, 256], [256, 256, 256], [1.0, 1.0, 1.0])
    assert sorted(smaller) == [224, 256, 256]


# Plans.


@pytest.mark.parametrize("name", ONE_OF_EACH_FAMILY)
@pytest.mark.parametrize(
    "median, spacing",
    [((200, 200, 160), [1.0, 1.0, 1.0]), ((320, 320, 20), [0.625, 0.625, 3.6])],
)
def test_every_family_gets_a_patch_it_can_build(name, median, spacing):
    """Whatever's planned builds, stays in the image, and fits the memory."""
    plan = _plan(name, memory_gb=12, median=median, target_spacing=spacing)
    get_model_from_registry(
        name,
        in_channels=1,
        out_channels=3,
        patch_size=plan.patch_size,
        target_spacing=spacing,
    )
    assert all(p <= m for p, m in zip(plan.patch_size, median, strict=True)) or name.startswith(
        ("primus", "swinunetr", "mednext")
    )
    assert plan.estimated_bytes <= 12 * GIB * constants.MEMORY_FRACTION + 1


def test_more_memory_gives_a_larger_patch():
    """The patch is what grows when the image is large."""
    sizes = [np.prod(_plan(memory_gb=gb).patch_size) for gb in (4, 8, 16, 32)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_heavier_networks_get_smaller_patches():
    """Planning for the network is the point of knowing the architecture."""
    light = np.prod(_plan("nnunet-plain").patch_size)
    heavy = np.prod(_plan("mednext-large").patch_size)
    assert heavy < light


def test_fp32_gets_a_smaller_patch_than_bf16():
    """A card without BF16 fits less."""
    assert np.prod(_plan(amp=False).patch_size) < np.prod(_plan(amp=True).patch_size)


def test_patch_shrinks_the_axis_taking_up_the_most_of_the_image():
    """Patches end up in proportion to the image, as in nnU-Net."""
    plan = _plan(median=(512, 512, 128), target_spacing=[1.0, 1.0, 2.5])
    assert plan.patch_size[0] == plan.patch_size[1] > plan.patch_size[2]
    ratios = [p / m for p, m in zip(plan.patch_size, (512, 512, 128), strict=True)]
    assert max(ratios) - min(ratios) < 0.25


def test_batch_stays_at_the_minimum_when_the_patch_had_to_shrink():
    """Spare memory goes to the patch first."""
    assert _plan(memory_gb=16).batch_size_per_gpu == constants.MIN_BATCH_SIZE


def test_batch_grows_once_the_patch_reaches_the_image():
    """With a small image and plenty of memory, the batch takes up the slack."""
    plan = _plan(memory_gb=48, median=(96, 96, 96), num_cases=1000)
    assert plan.patch_size == [96, 96, 96]
    assert plan.batch_size_per_gpu > constants.MIN_BATCH_SIZE
    assert plan.estimated_bytes <= 48 * GIB * constants.MEMORY_FRACTION


def test_batch_is_capped_by_the_dataset_and_shared_across_gpus():
    """One global batch sees at most 5% of the data, but never below the minimum."""
    many = _plan(memory_gb=48, median=(96, 96, 96), num_cases=200)
    assert many.batch_size_per_gpu == 10  # 200 cases * 5%.
    shared = _plan(memory_gb=48, median=(96, 96, 96), num_cases=200, num_gpus=2)
    assert shared.batch_size_per_gpu == 5
    few = _plan(memory_gb=48, median=(96, 96, 96), num_cases=10)
    assert few.batch_size_per_gpu == constants.MIN_BATCH_SIZE


@pytest.mark.parametrize("name", ["nnunet-res", "nnunet-pocket", "nnunet-plain", "resenc"])
def test_small_images_still_get_a_patch_the_network_can_build(name):
    """A network needing a minimum depth gets it, padding the small image."""
    plan = _plan(name, median=(20, 20, 16))
    get_model_from_registry(
        name,
        in_channels=1,
        out_channels=3,
        patch_size=plan.patch_size,
        target_spacing=[1.0, 1.0, 1.0],
    )


def test_impossibly_small_gpu_returns_the_smallest_patch():
    """The search stops rather than loops, and the estimate shows it doesn't fit."""
    plan = _plan(memory_gb=0.1)
    assert plan.estimated_bytes > 0.1 * GIB
    assert plan.batch_size_per_gpu == constants.MIN_BATCH_SIZE


# What to plan for.


def test_detect_gpu_memory_uses_the_smallest_gpu(monkeypatch):
    """Mixed GPUs are limited by the smallest."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    sizes = {0: 24 * GIB, 1: 12 * GIB}
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: argparse.Namespace(total_memory=sizes[i]),
    )
    assert pp.detect_gpu_memory_bytes() == (12.0 * GIB, 2)


# The check at training time.


def _config(patch=(128, 128, 128), batch=2, planned_gb=16.0, amp=True):
    return {
        "planning": {"gpu_memory_gb": planned_gb},
        "model": {"architecture": "nnunet-plain", "params": {"in_channels": 1, "out_channels": 3}},
        "spatial_config": {"patch_size": list(patch), "target_spacing": [1.0, 1.0, 1.0]},
        "training": {"batch_size_per_gpu": batch, "amp": amp},
    }


@pytest.fixture
def warned(monkeypatch):
    """Capture printed warnings."""
    messages = []
    monkeypatch.setattr(pp, "print_warning", messages.append)
    return messages


def _with_gpu(monkeypatch, gb):
    monkeypatch.setattr(pp, "detect_gpu_memory_bytes", lambda: (gb * GIB, 1))


def test_check_warns_when_the_plan_needs_more_than_the_gpu_has(monkeypatch, warned):
    """Planned on a big card, training on a small one."""
    _with_gpu(monkeypatch, 4)
    pp.check_plan_fits(_config())
    assert len(warned) == 1 and "--patch-size" in warned[0] and '"auto"' in warned[0]


def test_check_warns_when_the_gpu_is_much_bigger_than_planned(monkeypatch, warned):
    """Planned on a small card or a CPU, training on a big one."""
    _with_gpu(monkeypatch, 48)
    pp.check_plan_fits(_config(planned_gb=16))
    assert len(warned) == 1 and "48 GiB" in warned[0] and '"auto"' in warned[0]


def test_check_doesnt_suggest_a_larger_patch_for_a_budgeted_plan(monkeypatch, warned):
    """A plan made for a memory budget is meant to stay small on a big GPU."""
    _with_gpu(monkeypatch, 48)
    config = _config(planned_gb=16)
    config["planning"]["memory_budget_gb"] = 8.0
    pp.check_plan_fits(config)
    assert warned == []


def test_check_is_quiet_when_the_plan_suits_the_gpu(monkeypatch, warned):
    """A matching GPU says nothing."""
    _with_gpu(monkeypatch, 16)
    pp.check_plan_fits(_config(patch=(96, 96, 96)))
    assert warned == []


def test_check_leaves_a_hand_set_patch_alone_unless_it_may_not_fit(monkeypatch, warned):
    """A deliberate smaller patch isn't a mistake, but a memory risk still warns."""
    _with_gpu(monkeypatch, 48)
    pp.check_plan_fits(_config(planned_gb=16), patch_was_set=True)
    assert warned == []
    _with_gpu(monkeypatch, 2)
    pp.check_plan_fits(_config(planned_gb=16), patch_was_set=True)
    assert len(warned) == 1


def test_check_skips_old_configs_and_machines_without_gpus(monkeypatch, warned):
    """Nothing to compare against means nothing to say."""
    _with_gpu(monkeypatch, 2)
    config = _config()
    del config["planning"]
    pp.check_plan_fits(config)
    monkeypatch.setattr(pp, "detect_gpu_memory_bytes", lambda: (None, 0))
    pp.check_plan_fits(_config())
    assert warned == []


def test_check_follows_a_changed_architecture(monkeypatch, warned):
    """A heavier network than was planned for is estimated as itself."""
    _with_gpu(monkeypatch, 8)
    config = _config(patch=(96, 96, 96), planned_gb=8)
    pp.check_plan_fits(config)
    assert warned == []
    config["model"]["architecture"] = "mednext-large"
    pp.check_plan_fits(config)
    assert len(warned) == 1 and "mednext-large" in warned[0]


def test_check_survives_an_estimate_failure(monkeypatch, warned):
    """A check must never stop training."""
    _with_gpu(monkeypatch, 8)
    config = _config(patch=(288, 288, 288))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp.check_plan_fits(config)
    assert any("Could not estimate" in str(w.message) for w in caught)


def test_activation_layers_exclude_norms_and_activations():
    """The fitted constant is per convolution or linear output only."""
    assert nn.InstanceNorm3d not in pp._ACTIVATION_LAYERS


# Planning from a config.


def _auto_config(patch="auto", batch="auto", amp=True, architecture="nnunet-plain", budget=None):
    return {
        "model": {"architecture": architecture, "params": {"in_channels": 1, "out_channels": 3}},
        "preprocessing": {"median_resampled_image_size": [400, 400, 300]},
        "spatial_config": {"patch_size": patch, "target_spacing": [1.0, 1.0, 1.0]},
        "training": {"batch_size_per_gpu": batch, "amp": amp, "gpu_memory_budget_gb": budget},
    }


@pytest.mark.parametrize("value, expected", [("auto", True), (None, True), ([64, 64, 64], False)])
def test_is_auto(value, expected):
    """Both unset forms count as still to be planned."""
    assert pp.is_auto(value) is expected


def test_plan_for_config_uses_this_gpu(monkeypatch):
    """More memory on this machine means a larger planned patch."""
    _with_gpu(monkeypatch, 8)
    small = pp.plan_for_config(_auto_config(), 500, 1)
    _with_gpu(monkeypatch, 32)
    large = pp.plan_for_config(_auto_config(), 500, 1)
    assert np.prod(large.patch_size) > np.prod(small.patch_size)


def test_plan_for_config_stays_within_the_memory_budget(monkeypatch):
    """With a budget, a larger GPU plans the same patch as the budget allows."""
    _with_gpu(monkeypatch, 80)
    budgeted = pp.plan_for_config(_auto_config(budget=8), 500, 1)
    _with_gpu(monkeypatch, 24)
    assert pp.plan_for_config(_auto_config(budget=8), 500, 1) == budgeted
    assert budgeted.estimated_bytes <= 8 * GIB
    whole_gpu = pp.plan_for_config(_auto_config(budget=None), 500, 1)
    assert np.prod(whole_gpu.patch_size) > np.prod(budgeted.patch_size)


def test_a_budget_matches_nnunets_rule(monkeypatch):
    """With a budget, one sample may have nnU-Net's 560M activations per 8 GiB."""
    _with_gpu(monkeypatch, 80)
    for budget in (8, 24):
        plan = pp.plan_for_config(_auto_config(budget=budget), 500, 1)
        estimate = pp.estimate_network("nnunet-plain", 1, 3, plan.patch_size, [1.0, 1.0, 1.0])
        assert estimate.activations <= 560e6 * budget / 8


def test_a_budget_plans_the_same_patch_in_bf16_and_fp32(monkeypatch):
    """nnU-Net's rule doesn't depend on the precision, if the GPU is large enough."""
    _with_gpu(monkeypatch, 40)
    bf16 = pp.plan_for_config(_auto_config(budget=8, amp=True), 500, 1)
    fp32 = pp.plan_for_config(_auto_config(budget=8, amp=False), 500, 1)
    assert bf16.patch_size == fp32.patch_size
    assert fp32.estimated_bytes > bf16.estimated_bytes


def test_heavier_networks_get_a_lower_activation_limit():
    """Swin's activations take more memory, so fewer of them fit the budget."""
    unet = pp.estimate_network("nnunet-plain", 1, 3, [64, 64, 64], [1.0, 1.0, 1.0])
    swin = pp.estimate_network("swinunetr-small", 1, 3, [64, 64, 64], [1.0, 1.0, 1.0])
    assert pp.activation_limit(unet, 8) == 1120e6
    assert pp.activation_limit(swin, 8) < pp.activation_limit(unet, 8)


def test_a_budget_above_the_gpu_plans_for_the_gpu(monkeypatch):
    """The budget can't make a plan larger than the GPU allows."""
    _with_gpu(monkeypatch, 8)
    assert pp.plan_for_config(_auto_config(budget=64), 500, 1) == pp.plan_for_config(
        _auto_config(budget=None), 500, 1
    )


def test_configs_without_a_budget_get_the_default():
    """Older configs plan for the default budget, and null means the whole GPU."""
    config = _auto_config()
    del config["training"]["gpu_memory_budget_gb"]
    assert pp.memory_budget_gb(config) == 8.0
    assert pp.memory_budget_gb(_auto_config(budget=None)) is None


def test_plan_for_config_keeps_a_hand_set_patch(monkeypatch):
    """A hand-set patch stays, and the batch is the minimum."""
    _with_gpu(monkeypatch, 48)
    plan = pp.plan_for_config(_auto_config(patch=[64, 64, 64]), 500, 1)
    assert plan.patch_size == [64, 64, 64]
    assert plan.batch_size_per_gpu == constants.MIN_BATCH_SIZE


def test_plan_for_config_fits_the_patch_around_a_hand_set_batch(monkeypatch):
    """A larger hand-set batch leaves room for a smaller patch."""
    _with_gpu(monkeypatch, 16)
    default = pp.plan_for_config(_auto_config(), 500, 1)
    plan = pp.plan_for_config(_auto_config(batch=8), 500, 1)
    assert plan.batch_size_per_gpu == 8
    assert np.prod(plan.patch_size) < np.prod(default.patch_size)


def test_plan_for_config_without_a_gpu_uses_the_default_memory(monkeypatch):
    """CPU training still gets a plan."""
    _with_gpu(monkeypatch, constants.DEFAULT_GPU_MEMORY_GB)
    expected = pp.plan_for_config(_auto_config(), 500, 1)
    monkeypatch.setattr(pp, "detect_gpu_memory_bytes", lambda: (None, 0))
    assert pp.plan_for_config(_auto_config(), 500, 1) == expected


def test_plan_for_config_caps_the_batch_by_real_gpu_count(monkeypatch):
    """The dataset cap divides across the GPUs that will actually train."""
    _with_gpu(monkeypatch, 48)
    config = _auto_config()
    config["preprocessing"]["median_resampled_image_size"] = [96, 96, 96]
    assert pp.plan_for_config(config, 200, 1).batch_size_per_gpu == 10
    assert pp.plan_for_config(config, 200, 2).batch_size_per_gpu == 5


# Inference batch size.


def _planned_config(patch=(96, 96, 96)):
    return _auto_config(patch=list(patch), batch=2)


def test_inference_batch_is_capped_on_cpu():
    """There's no GPU limit to work out on a CPU."""
    assert pp.inference_batch_size(_planned_config(), "cpu") == constants.MAX_INFERENCE_BATCH_SIZE


@pytest.mark.parametrize("memory_gb, expected", [(100, 4), (0.3, 1)])
def test_inference_batch_follows_gpu_memory_within_limits(monkeypatch, memory_gb, expected):
    """A large GPU hits the cap of 4, however large; a tiny one still predicts."""
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _d: argparse.Namespace(total_memory=memory_gb * GIB),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d: 0)
    assert pp.inference_batch_size(_planned_config(), "cuda:0") == expected


def test_inference_batch_accounts_for_memory_already_in_use(monkeypatch):
    """During validation, the training state leaves less room."""
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _d: argparse.Namespace(total_memory=2 * GIB),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d: 0)
    idle = pp.inference_batch_size(_planned_config(), "cuda:0")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d: 1.5 * GIB)
    busy = pp.inference_batch_size(_planned_config(), "cuda:0")
    assert busy < idle


def test_inference_batch_survives_an_estimate_failure():
    """Sizing the batch must never stop inference."""
    config = _planned_config(patch=(288, 288, 288))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = pp.inference_batch_size(config, "cuda:0")
    assert result == constants.MAX_INFERENCE_BATCH_SIZE
