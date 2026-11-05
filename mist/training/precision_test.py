"""Time BF16 and FP32 on this GPU and choose which one to train with.

hardware.bf16_supported says a GPU runs BF16 in hardware, not that BF16 is faster
there. TF32 convolutions can make FP32 faster on the same GPU, so the first
mist_train runs a few real training steps in each and saves the choice.

FP32 takes roughly twice the activation memory, so it only wins if it's clearly
faster without shrinking the patch, since a larger patch can help accuracy. It may
use a smaller batch, so the two are compared in samples per second. The steps use
random data, so they leave out data loading.

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
        amp: True for BF16, False for FP32.
        reason: A sentence explaining the choice, to print.
    """

    amp: bool
    reason: str


def measure_steps(
    architecture: str,
    model_kwargs: dict[str, Any],
    batch_size: int,
    amp: bool,
) -> Measurement:
    """Run real training steps on random data, set up as training sets up the GPU.

    Args:
        architecture: Registered model name.
        model_kwargs: Arguments for the model, including in_channels and patch_size.
        batch_size: Batch size per GPU.
        amp: True for BF16, False for FP32.

    Returns:
        The measurement.
    """
    hardware.tune_gpu_math()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model_from_registry(architecture, **model_kwargs).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shape = (batch_size, model_kwargs["in_channels"], *model_kwargs["patch_size"])

    def step() -> None:
        optimizer.zero_grad()
        with hardware.autocast_context(amp):
            output = model(torch.randn(shape, device=device))
            if isinstance(output, dict):
                extras = output.get("deep_supervision") or []
                loss = output["prediction"].mean() + sum(extra.mean() for extra in extras)
            else:
                loss = output.mean()
        loss.backward()
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()

    # The first steps set up cuDNN and the optimizer's state.
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
    amp: bool,
) -> Measurement | None:
    """Run measure_steps in its own process.

    Args:
        architecture: Registered model name.
        model_kwargs: Arguments for the model, including in_channels and patch_size.
        batch_size: Batch size per GPU.
        amp: True for BF16, False for FP32.

    Returns:
        The measurement, or None if it ran out of memory or otherwise failed.
    """
    request = json.dumps(
        {
            "architecture": architecture,
            "model_kwargs": model_kwargs,
            "batch_size": batch_size,
            "amp": amp,
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


def choose_precision(
    architecture: str,
    model_kwargs: dict[str, Any],
    bf16_plan: Plan,
    fp32_plan: Plan,
    gpu_memory_bytes: float,
) -> Choice:
    """Choose BF16 or FP32, keeping BF16 unless FP32 is clearly better.

    Args:
        architecture: Registered model name.
        model_kwargs: Arguments for the model, without the patch size.
        bf16_plan: The plan if training with BF16.
        fp32_plan: The plan if training with FP32.
        gpu_memory_bytes: Memory of the GPU training will run on.

    Returns:
        The choice. Its plan is fp32_plan if amp is False, else bf16_plan.
    """
    if fp32_plan.patch_size != bf16_plan.patch_size:
        return Choice(True, "Using BF16, since FP32 would need a smaller patch.")

    budget = gpu_memory_bytes * constants.MEMORY_FRACTION
    if fp32_plan.estimated_bytes > budget:
        return Choice(True, "Using BF16, since FP32 may not fit in memory at this patch.")

    kwargs = {**model_kwargs, "patch_size": bf16_plan.patch_size}
    bf16 = run_measurement(architecture, kwargs, bf16_plan.batch_size_per_gpu, True)
    fp32 = run_measurement(architecture, kwargs, fp32_plan.batch_size_per_gpu, False)
    if bf16 is None or fp32 is None:
        return Choice(True, "Using BF16, since the speed test couldn't finish.")

    speeds = (
        f"BF16 {bf16.samples_per_second:.2f} and FP32 {fp32.samples_per_second:.2f} "
        "samples per second."
    )
    if fp32.peak_bytes > budget:
        return Choice(True, f"{speeds} Using BF16, since FP32 used too much memory.")
    speedup = fp32.samples_per_second / bf16.samples_per_second
    if speedup < constants.FP32_MIN_SPEEDUP:
        return Choice(True, f"{speeds} Using BF16, since FP32 isn't clearly faster.")
    return Choice(False, f"{speeds} Using FP32, which is {speedup - 1:.0%} faster.")


def main() -> None:
    """Run one measurement and print it as JSON, for run_measurement."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("request", help="JSON arguments for measure_steps.")
    measurement = measure_steps(**json.loads(parser.parse_args().request))
    print(json.dumps(dataclasses.asdict(measurement)))


if __name__ == "__main__":
    main()
