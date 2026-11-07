"""Tests for mist.training.precision_test."""

import json
import subprocess

import pytest

from mist.training import precision_test as pt
from mist.training.patch_planning import Plan
from mist.training.planning_constants import PlanningConstants as constants

GIB = 1024**3
MEMORY = 80.0 * GIB
MODEL_KWARGS = {
    "in_channels": 1,
    "out_channels": 2,
    "patch_size": "auto",
    "target_spacing": [1.0, 1.0, 1.0],
}


def _plan(patch=(64, 64, 64), batch=4, gib=10.0):
    return Plan(list(patch), batch, gib * GIB)


def _fake_speeds(monkeypatch, bf16, fp32):
    """Make run_measurement return the given measurements, and record its calls."""
    calls = []

    def run(architecture, model_kwargs, batch_size, amp, compiled):
        calls.append((architecture, model_kwargs["patch_size"], batch_size, amp, compiled))
        return bf16 if amp else fp32

    monkeypatch.setattr(pt, "run_measurement", run)
    return calls


def _choose(bf16_plan=None, fp32_plan=None, compiled=False):
    return pt.choose_precision(
        "nnunet-plain", MODEL_KWARGS, bf16_plan or _plan(), fp32_plan or _plan(), MEMORY, compiled
    )


def test_samples_per_second_accounts_for_batch_size():
    """A smaller batch at the same step time trains on fewer samples."""
    assert pt.Measurement(0.5, 4, 0.0).samples_per_second == 8.0
    assert pt.Measurement(0.5, 2, 0.0).samples_per_second == 4.0


def test_fp32_wins_when_clearly_faster(monkeypatch):
    """FP32 is chosen at the speedup, and the reason shows both speeds."""
    calls = _fake_speeds(
        monkeypatch, pt.Measurement(1.0, 4, 5.0 * GIB), pt.Measurement(0.5, 4, 9.0 * GIB)
    )
    choice = _choose()
    assert choice.amp is False
    assert "BF16 4.00 and FP32 8.00 samples per second" in choice.reason
    assert "100% faster" in choice.reason
    assert calls == [
        ("nnunet-plain", [64, 64, 64], 4, True, False),
        ("nnunet-plain", [64, 64, 64], 4, False, False),
    ]


def test_both_precisions_are_timed_compiled_if_training_will_compile(monkeypatch):
    """The test measures the setup that will train."""
    calls = _fake_speeds(
        monkeypatch, pt.Measurement(1.0, 4, 5.0 * GIB), pt.Measurement(1.0, 4, 9.0 * GIB)
    )
    _choose(compiled=True)
    assert [call[-1] for call in calls] == [True, True]


def test_bf16_stays_when_fp32_is_only_slightly_faster(monkeypatch):
    """A gain under the margin could be noise, so BF16 stays."""
    speedup = constants.FP32_MIN_SPEEDUP - 0.01
    _fake_speeds(
        monkeypatch,
        pt.Measurement(speedup, 4, 5.0 * GIB),
        pt.Measurement(1.0, 4, 9.0 * GIB),
    )
    choice = _choose()
    assert choice.amp is True
    assert "isn't clearly faster" in choice.reason


def test_comparison_is_per_sample_when_fp32_needs_a_smaller_batch(monkeypatch):
    """Faster steps at half the batch aren't faster training."""
    _fake_speeds(monkeypatch, pt.Measurement(1.0, 8, 5.0 * GIB), pt.Measurement(0.6, 4, 9.0 * GIB))
    assert _choose(_plan(batch=8), _plan(batch=4)).amp is True


def test_smaller_fp32_batch_can_still_win(monkeypatch):
    """FP32 at a smaller batch wins if it trains on more samples per second."""
    calls = _fake_speeds(
        monkeypatch, pt.Measurement(1.0, 8, 5.0 * GIB), pt.Measurement(0.25, 4, 9.0 * GIB)
    )
    assert _choose(_plan(batch=8), _plan(batch=4)).amp is False
    assert [call[2] for call in calls] == [8, 4]


def test_a_smaller_fp32_patch_keeps_bf16_without_running(monkeypatch):
    """Speed isn't worth a smaller patch, so nothing needs timing."""
    calls = _fake_speeds(monkeypatch, None, None)
    choice = _choose(_plan(patch=(128, 128, 128)), _plan(patch=(96, 96, 96)))
    assert choice.amp is True
    assert "smaller patch" in choice.reason
    assert calls == []


def test_fp32_estimated_over_budget_keeps_bf16_without_running(monkeypatch):
    """A patch and batch set by hand can leave FP32 too big to try."""
    calls = _fake_speeds(monkeypatch, None, None)
    choice = _choose(_plan(gib=10.0), _plan(gib=79.0))
    assert choice.amp is True
    assert "may not fit" in choice.reason
    assert calls == []


def test_fp32_measured_over_budget_keeps_bf16(monkeypatch):
    """Real memory overrides the estimate, however fast FP32 was."""
    _fake_speeds(monkeypatch, pt.Measurement(1.0, 4, 5.0 * GIB), pt.Measurement(0.1, 4, 79.0 * GIB))
    choice = _choose()
    assert choice.amp is True
    assert "too much memory" in choice.reason


@pytest.mark.parametrize(
    ("bf16", "fp32"),
    [
        (None, pt.Measurement(1.0, 4, 0.0)),
        (pt.Measurement(1.0, 4, 0.0), None),
    ],
)
def test_a_failed_measurement_keeps_bf16(monkeypatch, bf16, fp32):
    """Running out of memory in the test mustn't stop training."""
    _fake_speeds(monkeypatch, bf16, fp32)
    choice = _choose()
    assert choice.amp is True
    assert "couldn't finish" in choice.reason


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, "python"),
        subprocess.TimeoutExpired("python", 1),
        OSError("no python"),
    ],
)
def test_run_measurement_returns_none_when_the_process_fails(monkeypatch, error):
    """A crash, out-of-memory error, or hang counts as no result."""

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(pt.subprocess, "run", fail)
    assert pt.run_measurement("nnunet-plain", MODEL_KWARGS, 2, False) is None


@pytest.mark.parametrize("stdout", ["", "not json\n", '{"unexpected": 1}\n'])
def test_run_measurement_returns_none_on_unreadable_output(monkeypatch, stdout):
    """Output that isn't a measurement counts as no result."""
    monkeypatch.setattr(
        pt.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout, ""),
    )
    assert pt.run_measurement("nnunet-plain", MODEL_KWARGS, 2, False) is None


def test_run_measurement_reads_the_last_line(monkeypatch):
    """A network printing while it builds doesn't hide the result."""
    stdout = 'building...\n{"seconds_per_step": 0.5, "batch_size": 2, "peak_bytes": 3.0}\n'
    monkeypatch.setattr(
        pt.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout, ""),
    )
    assert pt.run_measurement("nnunet-plain", MODEL_KWARGS, 2, False) == pt.Measurement(0.5, 2, 3.0)


@pytest.mark.parametrize("compiled", [False, True])
def test_measure_steps_compiles_only_if_asked(monkeypatch, compiled):
    """The steps run on a compiled model only when training will compile."""
    compiles = []
    monkeypatch.setattr(pt.torch.nn.Module, "compile", lambda self: compiles.append(self))
    kwargs = {**MODEL_KWARGS, "patch_size": [16, 16, 16]}
    monkeypatch.setattr(constants, "SPEED_TEST_WARMUP_STEPS", 1, raising=False)
    monkeypatch.setattr(constants, "SPEED_TEST_STEPS", 1, raising=False)
    pt.measure_steps("nnunet-plain", kwargs, 1, False, compiled)
    assert len(compiles) == (1 if compiled else 0)


def test_run_measurement_passes_compiled_to_the_child(monkeypatch):
    """The child process receives the compile setting."""
    seen = []

    def run(command, **_kwargs):
        seen.append(json.loads(command[-1]))
        stdout = '{"seconds_per_step": 0.5, "batch_size": 2, "peak_bytes": 3.0}\n'
        return subprocess.CompletedProcess([], 0, stdout, "")

    monkeypatch.setattr(pt.subprocess, "run", run)
    pt.run_measurement("nnunet-plain", MODEL_KWARGS, 2, False, True)
    assert seen[0]["compiled"] is True


def test_measure_steps_stops_early_on_a_slow_network(monkeypatch):
    """A slow network stops at the time limit, once it has the minimum steps."""
    monkeypatch.setattr(constants, "SPEED_TEST_MAX_SECONDS", 0, raising=False)
    timed = []
    real_median = pt.statistics.median

    def median(times):
        timed.extend(times)
        return real_median(times)

    monkeypatch.setattr(pt.statistics, "median", median)
    kwargs = {**MODEL_KWARGS, "patch_size": [16, 16, 16]}
    pt.measure_steps("nnunet-plain", kwargs, 1, False)
    assert len(timed) == constants.SPEED_TEST_MIN_STEPS


@pytest.mark.parametrize("amp", [False, True])
def test_real_measurement_in_its_own_process(amp):
    """The whole path runs for real: a child process, a network, and its result."""
    kwargs = {**MODEL_KWARGS, "patch_size": [16, 16, 16]}
    measurement = pt.run_measurement("nnunet-plain", kwargs, 1, amp)
    assert measurement is not None
    assert measurement.seconds_per_step > 0
    assert measurement.batch_size == 1


def test_a_failing_real_process_returns_none():
    """An unknown network fails in the child and doesn't raise here."""
    assert pt.run_measurement("not-a-network", MODEL_KWARGS, 1, False) is None
