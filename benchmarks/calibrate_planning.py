"""Manual check of the patch planner's memory estimate against real GPU memory.

The planner's constants were fitted to one run of one network, so this measures
how far off the estimate is for the others. For each network it trains a few real
steps on one GPU at a planned patch size, then prints peak memory beside the
estimate:

    python benchmarks/calibrate_planning.py

Run it on an idle GPU. A ratio well above 1 means the planner may overfill the GPU
for that network; well below 1 means it leaves memory unused. This only covers
training steps, so it reads lower than a full run with validation and prediction.
"""

import argparse
import gc

import torch

from mist.models.model_registry import get_model_from_registry, list_registered_models
from mist.training import patch_planning as pp
from mist.utils import hardware

GIB = 1024**3
SPACING = [1.0, 1.0, 1.0]
IN_CHANNELS = 1
OUT_CHANNELS = 3


def measure(name: str, patch: list[int], batch: int, amp: bool, steps: int) -> float:
    """Return peak reserved memory in bytes over a few real training steps."""
    model = get_model_from_registry(
        name,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_size=patch,
        target_spacing=SPACING,
    ).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        optimizer.zero_grad()
        with hardware.autocast_context(amp):
            output = model(torch.randn(batch, IN_CHANNELS, *patch).cuda())
            if isinstance(output, dict):
                extras = output.get("deep_supervision") or []
                loss = output["prediction"].mean() + sum(e.mean() for e in extras)
            else:
                loss = output.mean()
        loss.backward()
        optimizer.step()
    peak = float(torch.cuda.max_memory_reserved())
    del model, optimizer, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=list_registered_models())
    parser.add_argument("--target-gb", type=float, default=8.0, help="Memory to plan each for.")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    hardware.tune_gpu_math()  # As in MIST's training.
    memory = float(torch.cuda.get_device_properties(0).total_memory)
    amp = hardware.resolve_amp(True, warn=False)
    print(f"{torch.cuda.get_device_name(0)}, {memory / GIB:.1f} GiB, {'BF16' if amp else 'FP32'}")
    print(f"{'network':18} {'patch':18} {'batch':>5} {'estimate':>9} {'actual':>7} {'ratio':>6}")
    for name in args.models:
        plan = pp.plan_patch_and_batch(
            architecture=name,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            median_size=[512, 512, 512],
            target_spacing=SPACING,
            gpu_memory_bytes=min(args.target_gb * pp._GIB, memory),
            amp=amp,
            num_cases=1000,
        )
        try:
            actual = measure(name, plan.patch_size, plan.batch_size_per_gpu, amp, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"{name:18} {plan.patch_size!s:18} out of memory", flush=True)
            torch.cuda.empty_cache()
            continue
        print(
            f"{name:18} {plan.patch_size!s:18} {plan.batch_size_per_gpu:5} "
            f"{plan.estimated_bytes / GIB:8.1f}G {actual / GIB:6.1f}G "
            f"{actual / plan.estimated_bytes:6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
