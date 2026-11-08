"""Time training precisions on this GPU and choose which one to train with.

hardware.bf16_supported says a GPU runs BF16 in hardware, not that BF16 is faster
there: TF32 convolutions can make FP32 faster, and some older A100s run BF16
slowly. FP16 has tensor cores on GPUs without BF16 hardware (V100, T4, Quadro
RTX 6000). So the first mist_train runs a few real training steps in each
candidate and saves the choice, in tiers:

- With BF16 hardware, BF16 is timed against FP32. If BF16 is clearly faster, it's
  chosen and FP16 isn't timed, since FP16 adds gradient scaling for no gain there.
  Otherwise FP16 is timed too.
- Without BF16 hardware, FP32 is timed against FP16.

The default (BF16 with hardware for it, else FP32) is kept unless another
precision is clearly faster without shrinking the patch, since a larger patch can
help accuracy. A precision may use a smaller batch, so they're compared in samples
per second. The steps use random data, so they leave out data loading. They compile
the model if training will, since compiling speeds up some precisions more than
others, and FP16 steps scale gradients and compute the loss in FP32, as training
does.

Each measurement runs in its own process: using the GPU in mist_train's main process
would hold memory on it for the whole run, and running out of memory here shouldn't
stop training.
"""

import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

import mist.models  # noqa: F401  # Registers every architecture.
from mist.models.model_registry import get_model_from_registry
from mist.training.patch_planning import Plan
from mist.training.planning_constants import PlanningConstants as constants
from mist.utils import hardware


@dataclasses.dataclass(frozen=True)
class Measurement:
    """Speed and memory of a few training steps.

    Attributes:
        seconds_per_step: Median seconds per training step.
        batch_size: Batch size per GPU.
        peak_bytes: Peak reserved GPU memory.
    """

    seconds_per_step: float
    batch_size: int
    peak_bytes: float

    @property
    def samples_per_second(self) -> float:
        """Return how many patches this trains on per second."""
        return self.batch_size / self.seconds_per_step


@dataclasses.dataclass(frozen=True)
class Choice:
    """The precision to train with.

    Attributes:
        amp: True for BF16, False for FP32, or "fp16".
        reason: A sentence explaining the choice, to print.
    """

    amp: bool | str
    reason: str


def measure_steps(
    architecture: str,
    model_kwargs: dict[str, Any],
    batch_size: int,
    amp: bool | str,
    compiled: bool = False,
) -> Measurement:
    """Run real training steps on random data, set up as training sets up the GPU.

    Args:
        architecture: Registered model name.
        model_kwargs: Arguments for the model, including in_channels and patch_size.
        batch_size: Batch size per GPU.
        amp: True for BF16, False for FP32, or "fp16".
        compiled: Whether to compile the model, as training does.

    Returns:
        The measurement.
    """
    hardware.tune_gpu_math()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model_from_registry(architecture, **model_kwargs).to(device).train()
    if compiled:
        model.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shape = (batch_size, model_kwargs["in_channels"], *model_kwargs["patch_size"])
    # As in training, FP16 scales gradients and computes the loss in FP32.
    fp16 = amp == hardware.FP16
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)

    def mean(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().mean() if fp16 else tensor.mean()

    def step() -> None:
        optimizer.zero_grad()
        with hardware.autocast_context(amp):
            output = model(torch.randn(shape, device=device))
        with hardware.autocast_context(False if fp16 else amp):
            if isinstance(output, dict):
                extras = output.get("deep_supervision") or []
                loss = mean(output["prediction"]) + sum(mean(extra) for extra in extras)
            else:
                loss = mean(output)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if cuda:
            torch.cuda.synchronize()

    # The first steps compile the model and set up cuDNN and the optimizer's state.
    for _ in range(constants.SPEED_TEST_WARMUP_STEPS):
        step()

    times: list[float] = []
    started = time.perf_counter()
    for _ in range(constants.SPEED_TEST_STEPS):
        before = time.perf_counter()
        step()
        times.append(time.perf_counter() - before)
        out_of_time = time.perf_counter() - started > constants.SPEED_TEST_MAX_SECONDS
        if out_of_time and len(times) >= constants.SPEED_TEST_MIN_STEPS:
            break

    peak = float(torch.cuda.max_memory_reserved()) if cuda else 0.0
    return Measurement(statistics.median(times), batch_size, peak)


def run_measurement(
    architecture: str,
    model_kwargs: dict[str, Any],
    batch_size: int,
    amp: bool | str,
    compiled: bool = False,
) -> Measurement | None:
    """Run measure_steps in its own process.

    Args:
        architecture: Registered model name.
        model_kwargs: Arguments for the model, including in_channels and patch_size.
        batch_size: Batch size per GPU.
        amp: True for BF16, False for FP32, or "fp16".
        compiled: Whether to compile the model, as training does.

    Returns:
        The measurement, or None if it ran out of memory or otherwise failed.
    """
    request = json.dumps(
        {
            "architecture": architecture,
            "model_kwargs": model_kwargs,
            "batch_size": batch_size,
            "amp": amp,
            "compiled": compiled,
        }
    )
    try:
        finished = subprocess.run(
            [sys.executable, "-m", "mist.training.precision_test", request],
            capture_output=True,
            text=True,
            check=True,
            timeout=constants.SPEED_TEST_TIMEOUT_SECONDS,
        )
        # Networks may print while they build, so the result is the last line.
        return Measurement(**json.loads(finished.stdout.strip().splitlines()[-1]))
    except (subprocess.SubprocessError, OSError, ValueError, IndexError, TypeError):
        return None


def _name(amp: bool | str) -> str:
    """Name a training.amp value."""
    if amp == hardware.FP16:
        return "FP16"
    return "BF16" if amp else "FP32"


def choose_precision(
    architecture: str,
    model_kwargs: dict[str, Any],
    plans: dict[bool | str, Plan],
    gpu_memory_bytes: float,
    compiled: bool,
) -> Choice:
    """Choose a precision, keeping the default unless another is clearly faster.

    The default is BF16 if plans has a BF16 plan (the GPU has BF16 hardware), else
    FP32. See the module docstring for the tiers.

    Args:
        architecture: Registered model name.
        model_kwargs: Arguments for the model, without the patch size.
        plans: The plan for each precision to consider, keyed by training.amp value:
            True (BF16), False (FP32), and "fp16".
        gpu_memory_bytes: Memory of the GPU training will run on.
        compiled: Whether training will compile the model.

    Returns:
        The choice. Train with plans[choice.amp].
    """
    default = True if True in plans else False
    base = plans[default]
    budget = gpu_memory_bytes * constants.MEMORY_FRACTION
    kwargs = {**model_kwargs, "patch_size": base.patch_size}
    measured: dict[bool | str, Measurement] = {}
    too_big: set[bool | str] = set()
    notes: list[str] = []

    def can_try(amp: bool | str) -> bool:
        """Whether a precision fits the default's patch, by the estimate."""
        if amp not in plans:
            return False
        if plans[amp].patch_size != base.patch_size:
            notes.append(f"{_name(amp)} would need a smaller patch")
            return False
        if plans[amp].estimated_bytes > budget:
            notes.append(f"{_name(amp)} may not fit in memory at this patch")
            return False
        return True

    def measure(amp: bool | str) -> Measurement | None:
        """Time a precision; None if it failed or used too much memory."""
        result = run_measurement(architecture, kwargs, plans[amp].batch_size_per_gpu, amp, compiled)
        if result is None:
            notes.append(f"{_name(amp)}'s timing couldn't finish")
            return None
        measured[amp] = result
        if amp != default and result.peak_bytes > budget:
            notes.append(f"{_name(amp)} used too much memory")
            too_big.add(amp)
            return None
        return result

    def clearly_faster(amp: bool | str, than: bool | str) -> bool:
        ratio = measured[amp].samples_per_second / measured[than].samples_per_second
        return ratio >= constants.PRECISION_MIN_SPEEDUP

    if measure(default) is None:
        return Choice(default, f"Using {_name(default)}, since the speed test couldn't finish.")

    # replaced is the precision the winner was chosen over, for the message.
    best, replaced = default, None
    if default is True:
        fp32 = measure(False) if can_try(False) else None
        bf16_clearly_faster = fp32 is not None and clearly_faster(True, False)
        if fp32 is not None and clearly_faster(False, True):
            best, replaced = False, True
        if not bf16_clearly_faster and can_try(hardware.FP16):
            if measure(hardware.FP16) is not None and clearly_faster(hardware.FP16, best):
                best, replaced = hardware.FP16, best
    elif can_try(hardware.FP16):
        if measure(hardware.FP16) is not None and clearly_faster(hardware.FP16, False):
            best, replaced = hardware.FP16, False

    speeds = ", ".join(
        f"{_name(amp)} {result.samples_per_second:.2f}" for amp, result in measured.items()
    )
    parts = [f"{speeds} samples per second."] if len(measured) > 1 else []
    if notes:
        parts.append(notes[0][0].upper() + "; ".join(notes)[1:] + ".")
    others = [amp for amp in measured if amp != best and amp not in too_big]
    if replaced is not None:
        gain = measured[best].samples_per_second / measured[replaced].samples_per_second - 1
        parts.append(f"Using {_name(best)}, which is {gain:.0%} faster than {_name(replaced)}.")
        close = [amp for amp in others if amp != replaced and not clearly_faster(best, amp)]
        if close:
            names = " and ".join(_name(amp) for amp in close)
            parts.append(f"{names} isn't clearly faster than {_name(best)}.")
    elif not others:
        parts.append(f"Using {_name(best)}.")
    elif all(clearly_faster(best, amp) for amp in others):
        slowest = " and ".join(_name(amp) for amp in others)
        parts.append(f"Using {_name(best)}, which is clearly faster than {slowest}.")
    else:
        rivals = " or ".join(_name(amp) for amp in others)
        parts.append(f"Using {_name(best)}, since {rivals} isn't clearly faster.")
    return Choice(best, " ".join(parts))


def main() -> None:
    """Run one measurement and print it as JSON, for run_measurement."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("request", help="JSON arguments for measure_steps.")
    measurement = measure_steps(**json.loads(parser.parse_args().request))
    print(json.dumps(dataclasses.asdict(measurement)))


if __name__ == "__main__":
    main()
