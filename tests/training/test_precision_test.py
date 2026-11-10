"""Tests for mist.training.precision_test."""

import json
import subprocess

import pytest

from mist.training import precision_test as pt
from mist.training.patch_planning import Plan
from mist.training.planning_constants import PlanningConstants as constants
from mist.utils.hardware import FP16

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


def _fake_speeds(monkeypatch, speeds):
    """Make run_measurement return speeds[amp], and record its calls."""
    calls = []

    def run(architecture, model_kwargs, batch_size, amp, compiled):
        calls.append((architecture, model_kwargs["patch_size"], batch_size, amp, compiled))
        return speeds.get(amp)

    monkeypatch.setattr(pt, "run_measurement", run)
    return calls


def _speed(seconds, batch=4, gib=5.0):
    return pt.Measurement(seconds, batch, gib * GIB)


def _choose(plans=None, compiled=False, bf16_hardware=True):
    if plans is None:
        plans = {False: _plan(), FP16: _plan()}
        if bf16_hardware:
            plans[True] = _plan()
    return pt.choose_precision("nnunet-plain", MODEL_KWARGS, plans, MEMORY, compiled)


def _timed(calls):
    return [call[3] for call in calls]


SLOW = 1.0
NOISE = 1.0 / (constants.PRECISION_MIN_SPEEDUP - 0.01)  # Faster, but within the margin.


def test_samples_per_second_accounts_for_batch_size():
    """A smaller batch at the same step time trains on fewer samples."""
    assert pt.Measurement(0.5, 4, 0.0).samples_per_second == 8.0
    assert pt.Measurement(0.5, 2, 0.0).samples_per_second == 4.0


# With BF16 hardware.


def test_bf16_clearly_faster_than_fp32_skips_fp16(monkeypatch):
    """FP16 isn't timed when BF16 already wins, since it would add nothing."""
    calls = _fake_speeds(monkeypatch, {True: _speed(0.5), False: _speed(1.0)})
    choice = _choose()
    assert choice.amp is True
    assert _timed(calls) == [True, False]
    assert "BF16 8.00, FP32 4.00 samples per second" in choice.reason
    assert "clearly faster than FP32" in choice.reason


def test_fp32_wins_when_clearly_faster_than_bf16_and_fp16(monkeypatch):
    """A GPU with slow BF16 trains in FP32, after FP16 is timed too."""
    calls = _fake_speeds(
        monkeypatch, {True: _speed(1.0), False: _speed(0.5), FP16: _speed(0.5 * NOISE)}
    )
    choice = _choose()
    assert choice.amp is False
    assert _timed(calls) == [True, False, FP16]
    assert "Using FP32, which is 100% faster than BF16" in choice.reason
    assert "FP16 isn't clearly faster than FP32" in choice.reason


def test_fp16_wins_when_bf16_is_slow_and_fp16_is_clearly_faster(monkeypatch):
    """FP16 beats the faster of BF16 and FP32 by the margin, so it's chosen."""
    calls = _fake_speeds(monkeypatch, {True: _speed(1.0), False: _speed(0.5), FP16: _speed(0.25)})
    choice = _choose()
    assert choice.amp == FP16
    assert _timed(calls) == [True, False, FP16]
    assert "Using FP16, which is 100% faster than FP32" in choice.reason


def test_bf16_stays_when_nothing_is_clearly_faster(monkeypatch):
    """Gains under the margin could be noise, so the default stays."""
    calls = _fake_speeds(
        monkeypatch, {True: _speed(SLOW), False: _speed(NOISE), FP16: _speed(NOISE)}
    )
    choice = _choose()
    assert choice.amp is True
    assert _timed(calls) == [True, False, FP16]
    assert "isn't clearly faster" in choice.reason


def test_every_precision_is_timed_compiled_if_training_will_compile(monkeypatch):
    """The test measures the setup that will train."""
    calls = _fake_speeds(monkeypatch, {True: _speed(1.0), False: _speed(1.0), FP16: _speed(1.0)})
    _choose(compiled=True)
    assert [call[-1] for call in calls] == [True, True, True]


def test_comparison_is_per_sample_when_fp32_needs_a_smaller_batch(monkeypatch):
    """Faster steps at half the batch aren't faster training."""
    _fake_speeds(
        monkeypatch,
        {True: _speed(1.0, batch=8), False: _speed(0.6, batch=4), FP16: _speed(1.0, batch=8)},
    )
    plans = {True: _plan(batch=8), False: _plan(batch=4), FP16: _plan(batch=8)}
    assert _choose(plans).amp is True


def test_smaller_fp32_batch_can_still_win(monkeypatch):
    """FP32 at a smaller batch wins if it trains on more samples per second."""
    calls = _fake_speeds(
        monkeypatch,
        {True: _speed(1.0, batch=8), False: _speed(0.25, batch=4), FP16: _speed(1.0, batch=8)},
    )
    plans = {True: _plan(batch=8), False: _plan(batch=4), FP16: _plan(batch=8)}
    assert _choose(plans).amp is False
    assert [call[2] for call in calls] == [8, 4, 8]


def test_a_smaller_fp32_patch_is_not_timed_but_fp16_is(monkeypatch):
    """Speed isn't worth a smaller patch; BF16 wasn't shown faster, so FP16 is timed."""
    calls = _fake_speeds(monkeypatch, {True: _speed(1.0), FP16: _speed(1.0)})
    plans = {
        True: _plan(patch=(128, 128, 128)),
        False: _plan(patch=(96, 96, 96)),
        FP16: _plan(patch=(128, 128, 128)),
    }
    choice = _choose(plans)
    assert choice.amp is True
    assert _timed(calls) == [True, FP16]
    assert "FP32 would need a smaller patch" in choice.reason


def test_fp32_estimated_over_budget_is_not_timed(monkeypatch):
    """A patch and batch set by hand can leave FP32 too big to try."""
    calls = _fake_speeds(monkeypatch, {True: _speed(1.0), FP16: _speed(1.0)})
    choice = _choose({True: _plan(gib=10.0), False: _plan(gib=79.0), FP16: _plan(gib=12.0)})
    assert choice.amp is True
    assert False not in _timed(calls)
    assert "may not fit" in choice.reason


def test_fp32_measured_over_budget_keeps_bf16(monkeypatch):
    """Real memory overrides the estimate, however fast FP32 was."""
    _fake_speeds(
        monkeypatch,
        {True: _speed(1.0), False: _speed(0.1, gib=79.0), FP16: _speed(1.0)},
    )
    choice = _choose()
    assert choice.amp is True
    assert "too much memory" in choice.reason
    assert "isn't clearly faster" in choice.reason  # Said of FP16 only.
    assert "FP32 or" not in choice.reason


def test_a_failed_bf16_measurement_keeps_bf16_and_stops(monkeypatch):
    """Running out of memory in the test mustn't stop training."""
    calls = _fake_speeds(monkeypatch, {False: _speed(1.0), FP16: _speed(1.0)})
    choice = _choose()
    assert choice.amp is True
    assert "couldn't finish" in choice.reason
    assert _timed(calls) == [True]


def test_a_failed_fp32_measurement_still_times_fp16(monkeypatch):
    """One failed timing leaves the others to compare."""
    calls = _fake_speeds(monkeypatch, {True: _speed(1.0), FP16: _speed(0.5)})
    choice = _choose()
    assert choice.amp == FP16
    assert _timed(calls) == [True, False, FP16]
    assert "FP32's timing couldn't finish" in choice.reason


# Without BF16 hardware.


def test_without_bf16_fp16_wins_when_clearly_faster(monkeypatch):
    """On a T4, FP16 trained about 4 times faster than FP32."""
    calls = _fake_speeds(monkeypatch, {False: _speed(2.5), FP16: _speed(0.63)})
    choice = _choose(bf16_hardware=False)
    assert choice.amp == FP16
    assert _timed(calls) == [False, FP16]
    assert "Using FP16, which is" in choice.reason
    assert "faster than FP32" in choice.reason


def test_without_bf16_fp32_stays_when_fp16_is_not_clearly_faster(monkeypatch):
    """Consumer Pascal GPUs run FP16 slowly, and FP32 stays."""
    _fake_speeds(monkeypatch, {False: _speed(SLOW), FP16: _speed(NOISE)})
    choice = _choose(bf16_hardware=False)
    assert choice.amp is False
    assert "FP16 isn't clearly faster" in choice.reason


def test_without_bf16_a_failed_fp32_measurement_keeps_fp32(monkeypatch):
    """Without a baseline there's nothing to compare."""
    calls = _fake_speeds(monkeypatch, {FP16: _speed(0.5)})
    choice = _choose(bf16_hardware=False)
    assert choice.amp is False
    assert "couldn't finish" in choice.reason
    assert _timed(calls) == [False]


def test_without_bf16_fp16_over_budget_keeps_fp32(monkeypatch):
    """FP16's own memory counts against it."""
    _fake_speeds(monkeypatch, {False: _speed(1.0), FP16: _speed(0.25, gib=79.0)})
    choice = _choose(bf16_hardware=False)
    assert choice.amp is False
    assert "FP16 used too much memory" in choice.reason


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


@pytest.mark.parametrize("amp", [False, True, "fp16"])
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
