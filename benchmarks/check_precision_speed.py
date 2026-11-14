"""Manual check of why one training precision times slower than another.

mist_train's precision test runs 3 warmup steps and 15 timed steps per precision,
compiled if training compiles, and keeps the median. This runs the same steps
with and without torch.compile, more warmup, and every step's time printed, so a
slow precision can be told apart from slow setup, such as compiling or cuDNN's
algorithm search spilling past the warmup:

    python benchmarks/check_precision_speed.py --results /path/to/results

It reads the network, patch size, and batch size from the results folder's
config.json, so run it after mist_train has planned. Each case runs in its own
process, as in the precision test. --profile prints the slowest GPU kernels for
one case instead, for example --profile bf16 compiled.

Don't run it while something else is using the GPU.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import mist.models  # noqa: F401  # Registers every architecture.
from mist.models.model_registry import get_model_from_registry
from mist.utils import hardware

PRECISIONS = {"fp32": False, "bf16": True, "fp16": hardware.FP16}


def make_step(architecture, model_kwargs, batch_size, amp, compiled):
    """Build a model and return one training step, as the precision test does."""
    hardware.tune_gpu_math()
    device = torch.device("cuda")
    model = get_model_from_registry(architecture, **model_kwargs).to(device).train()
    if compiled:
        model.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shape = (batch_size, model_kwargs["in_channels"], *model_kwargs["patch_size"])
    fp16 = amp == hardware.FP16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    def mean(tensor):
        return tensor.float().mean() if fp16 else tensor.mean()

    def step():
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
        torch.cuda.synchronize()

    return step


def time_case(request):
    """Time every step of one case and print the times as JSON."""
    step = make_step(
        request["architecture"],
        request["model_kwargs"],
        request["batch_size"],
        request["amp"],
        request["compiled"],
    )
    times = []
    for _ in range(request["warmup"] + request["steps"]):
        started = time.perf_counter()
        step()
        times.append(time.perf_counter() - started)
    print(json.dumps({"times": times, "peak_gib": torch.cuda.max_memory_reserved() / 2**30}))


def profile_case(request):
    """Print the GPU kernels that take the most time in a few warmed-up steps."""
    from torch.profiler import ProfilerActivity, profile

    step = make_step(
        request["architecture"],
        request["model_kwargs"],
        request["batch_size"],
        request["amp"],
        request["compiled"],
    )
    for _ in range(request["warmup"]):
        step()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", help="mist_train results folder with a planned config.")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed steps first.")
    parser.add_argument("--steps", type=int, default=20, help="Timed steps.")
    parser.add_argument(
        "--profile",
        nargs=2,
        metavar=("PRECISION", "MODE"),
        help="Profile one case: fp32, bf16, or fp16, then eager or compiled.",
    )
    parser.add_argument("--request", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.request:
        request = json.loads(args.request)
        (profile_case if request.get("profile") else time_case)(request)
        return
    if not args.results:
        parser.error("--results is required.")

    config = json.loads((Path(args.results) / "config.json").read_text())
    base = {
        "architecture": config["model"]["architecture"],
        "model_kwargs": {**config["model"]["params"], **config["spatial_config"]},
        "batch_size": config["training"]["batch_size_per_gpu"],
        "warmup": args.warmup,
        "steps": args.steps,
    }
    if "auto" in (base["model_kwargs"]["patch_size"], base["batch_size"]):
        raise SystemExit("Patch or batch size is still auto; run mist_train once first.")
    print(
        f"{base['architecture']}, patch {base['model_kwargs']['patch_size']}, batch "
        f"{base['batch_size']}, {torch.cuda.get_device_name(0)}, torch {torch.__version__}"
    )

    def run(precision, compiled, profile=False):
        request = {**base, "amp": PRECISIONS[precision], "compiled": compiled}
        if profile:
            request["profile"] = True
        finished = subprocess.run(
            [sys.executable, __file__, "--request", json.dumps(request)],
            capture_output=True,
            text=True,
        )
        if finished.returncode != 0:
            print(f"{precision} {'compiled' if compiled else 'eager'} failed:")
            print(finished.stderr.strip().splitlines()[-1] if finished.stderr else "")
            return None
        return finished.stdout

    if args.profile:
        precision, mode = args.profile
        print(run(precision, mode == "compiled", profile=True))
        return

    print(f"{'case':16} {'first':>7} {'as test':>7} {'median':>7} {'min':>7} {'max':>7}  GiB")
    for compiled in (False, True):
        for precision in PRECISIONS:
            output = run(precision, compiled)
            if output is None:
                continue
            result = json.loads(output.strip().splitlines()[-1])
            times = result["times"]
            # The precision test's median: 15 steps after 3 warmup steps.
            as_test = statistics.median(times[3:18])
            timed = times[args.warmup :]
            name = f"{precision} {'compiled' if compiled else 'eager'}"
            print(
                f"{name:16} {times[0]:7.2f} {as_test:7.3f} "
                f"{statistics.median(timed):7.3f} {min(timed):7.3f} {max(timed):7.3f}  "
                f"{result['peak_gib']:.1f}"
            )
    print(
        "Seconds per step. 'as test' is the median of steps 4 to 18, as the precision "
        "test times them; 'median' is after --warmup steps."
    )


if __name__ == "__main__":
    main()
