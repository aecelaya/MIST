"""Manual check that MIST's default pipeline trains and predicts on real data.

Runs mist_analyze, mist_preprocess, and a short mist_train on one fold in new
folders, then prints PASS, WARN, or FAIL for each check along with timing and
GPU memory. Use a small dataset, since every case in fold 0 gets predicted:

    python benchmarks/check_training.py \\
        --data /path/to/dataset.json --out /path/to/check_training

This shows that the pipeline runs and does what it's designed to. It doesn't show
that accuracy improved: a few epochs are far too few, so the Dice it prints is
only there to catch something badly wrong. Compare accuracy with full runs.

Exits with status 1 if any check fails.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mist.models.model_registry import get_model_from_registry

FOLD = 0
LOG_TAIL_LINES = 25
MEMORY_WARN_FRACTION = 0.95


class Report:
    """Collect and print check results."""

    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add(self, status: str, message: str) -> bool:
        """Record and print one result; return whether it isn't a failure."""
        self.counts[status] += 1
        print(f"  {status:4}  {message}", flush=True)
        return status != "FAIL"

    def check(self, passed: bool, message: str, failure: str | None = None) -> bool:
        """Record PASS or FAIL for a condition."""
        return self.add("PASS" if passed else "FAIL", message if passed else failure or message)

    def info(self, message: str) -> None:
        """Print a line that isn't a check."""
        print(f"        {message}", flush=True)


class MemoryWatcher:
    """Track each GPU's peak memory by polling nvidia-smi.

    This counts everything on the GPU, including other users' jobs, and PyTorch's
    reserved memory rather than only what tensors use. That's the number that
    decides whether a job fits.
    """

    QUERY = "--query-gpu=memory.used,memory.total"

    def __init__(self, interval: float = 2.0):
        self.available = shutil.which("nvidia-smi") is not None
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.baseline = self._sample() if self.available else []
        self.peak = list(self.baseline)

    def _sample(self) -> list[tuple[int, int]]:
        output = subprocess.run(
            ["nvidia-smi", self.QUERY, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return [tuple(int(v) for v in line.split(",")) for line in output.strip().splitlines()]

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sample = self._sample()
            if len(sample) == len(self.peak):
                self.peak = [
                    (max(old[0], new[0]), new[1])
                    for old, new in zip(self.peak, sample, strict=True)
                ]

    def __enter__(self):
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *_):
        if self.available:
            self._stop.set()
            self._thread.join()


def run_step(name: str, module: str, entry: str, args: list[str], logs: Path) -> tuple[bool, float]:
    """Run a MIST entry point in its own process, logging its output to a file."""
    command = [
        sys.executable,
        "-c",
        f"from mist.cli.{module} import {entry}; {entry}()",
        *args,
    ]
    log_path = logs / f"{name}.log"
    start = time.perf_counter()
    with open(log_path, "w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    elapsed = time.perf_counter() - start
    if code != 0:
        text = log_path.read_text(errors="replace")
        if "out of memory" in text.lower():
            print("  FAIL  ran out of memory", flush=True)
        print(f"\n--- last lines of {log_path} ---", flush=True)
        print("\n".join(text.splitlines()[-LOG_TAIL_LINES:]), flush=True)
    return code == 0, elapsed


def edit_config(config_path: Path, steps_per_epoch: int) -> None:
    """Shorten epochs and skip test-time flips; neither has a CLI flag."""
    config = json.loads(config_path.read_text())
    config["training"]["min_steps_per_epoch"] = steps_per_epoch
    config["inference"]["tta"]["enabled"] = False
    config_path.write_text(json.dumps(config, indent=2))


def scalars(log_dir: Path, tag: str) -> list:
    """Read one TensorBoard scalar series, or an empty list if it's missing."""
    if not log_dir.is_dir():
        return []
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator.Scalars(tag) if tag in accumulator.Tags()["scalars"] else []


def check_losses(report: Report, results: Path, epochs: int) -> None:
    """Check the logged training loss: all there, finite, and falling."""
    log_dir = results / "logs" / f"fold_{FOLD}"
    train = [e.value for e in scalars(log_dir / "losses_train", "losses")]
    report.check(
        len(train) == epochs,
        f"logged training loss for all {epochs} epochs",
        f"logged training loss for {len(train)} of {epochs} epochs",
    )
    if not train:
        return
    report.info("training loss: " + ", ".join(f"{v:.3f}" for v in train))
    report.check(all(math.isfinite(v) for v in train), "training loss stayed finite")
    if train[-1] < train[0]:
        report.add("PASS", f"training loss fell ({train[0]:.3f} to {train[-1]:.3f})")
    else:
        report.add(
            "WARN",
            f"training loss didn't fall ({train[0]:.3f} to {train[-1]:.3f}); "
            "this can happen in a very short run, so try more --epochs",
        )
    validation = [e.value for e in scalars(log_dir / "losses_validation", "losses")]
    report.check(
        len(validation) == 1,
        "validation ran on the last epoch only, as --validate-every asked",
        f"expected 1 validation, found {len(validation)}",
    )


def report_step_time(report: Report, results: Path, config: dict, num_train: int) -> None:
    """Report seconds per step from epochs that had no validation."""
    times = [e.wall_time for e in scalars(results / "logs" / f"fold_{FOLD}", "learning_rate")]
    # The first logged time is after epoch 1, and the last epoch also validated.
    gaps = np.diff(times[:-1])
    training = config["training"]
    hardware = training["hardware"]
    batch = training["batch_size_per_gpu"] * max(1, hardware.get("num_gpus") or 1)
    steps = max(training["min_steps_per_epoch"], math.ceil(num_train / batch))
    if len(gaps):
        report.info(
            f"training speed: {float(np.mean(gaps)) / steps:.3f} s/step "
            f"({float(np.mean(gaps)):.1f} s per {steps}-step epoch, without validation)"
        )
    else:
        report.info("training speed: not measured; needs at least 3 epochs")


def check_saved_model(report: Report, results: Path, config: dict, epochs: int) -> None:
    """Check the saved model is the final epoch's and loads into the network."""
    model_path = results / "models" / f"fold_{FOLD}.pt"
    checkpoint_path = results / "checkpoints" / f"fold_{FOLD}_checkpoint.pt"
    if not report.check(model_path.is_file(), f"saved {model_path.name}"):
        return
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    model = get_model_from_registry(
        config["model"]["architecture"], **config["model"]["params"], **config["spatial_config"]
    )
    model.load_state_dict(state, strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    report.add("PASS", f"weights load into the network ({parameters / 1e6:.1f}M parameters)")

    if not report.check(checkpoint_path.is_file(), f"saved {checkpoint_path.name}"):
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report.check(
        checkpoint["epoch"] == epochs - 1,
        f"checkpoint is from the last epoch ({epochs})",
        f"checkpoint is from epoch {checkpoint['epoch'] + 1}, not {epochs}",
    )
    final = checkpoint["model_state_dict"]
    report.check(
        state.keys() == final.keys() and all(torch.equal(state[k], final[k]) for k in state),
        "saved model holds the final epoch's weights (val_percent is 0)",
        "saved model differs from the final epoch's weights",
    )


def check_predictions(report: Report, results: Path, config: dict) -> None:
    """Check fold 0's predictions: all there, on the right grid, with valid labels."""
    paths = pd.read_csv(results / "train_paths.csv")
    cases = paths[paths["fold"] == FOLD]
    labels = set(config["dataset_info"]["labels"])
    prediction_dir = results / "predictions" / "train" / "raw"

    missing, wrong_grid, wrong_labels, dice = [], [], [], []
    for _, case in cases.iterrows():
        path = prediction_dir / f"{case['id']}.nii.gz"
        if not path.is_file():
            missing.append(case["id"])
            continue
        prediction = sitk.ReadImage(str(path))
        truth = sitk.ReadImage(str(case["mask"]))
        same_grid = (
            prediction.GetSize() == truth.GetSize()
            and np.allclose(prediction.GetSpacing(), truth.GetSpacing())
            and np.allclose(prediction.GetOrigin(), truth.GetOrigin())
            and np.allclose(prediction.GetDirection(), truth.GetDirection())
        )
        if not same_grid:
            wrong_grid.append(case["id"])
            continue
        predicted = sitk.GetArrayFromImage(prediction)
        if not set(np.unique(predicted).tolist()) <= labels:
            wrong_labels.append(case["id"])
        predicted_foreground = predicted > 0
        true_foreground = sitk.GetArrayFromImage(truth) > 0
        total = predicted_foreground.sum() + true_foreground.sum()
        if total:
            dice.append(2 * (predicted_foreground & true_foreground).sum() / total)

    report.check(
        not missing,
        f"predicted all {len(cases)} cases in fold {FOLD}",
        f"missing predictions for {len(missing)} of {len(cases)} cases: {missing[:5]}",
    )
    report.check(
        not wrong_grid,
        "predictions match their original images' size, spacing, origin, and direction",
        f"predictions off the original grid for: {wrong_grid[:5]}",
    )
    report.check(
        not wrong_labels,
        f"predictions only use the dataset's labels {sorted(labels)}",
        f"unexpected label values in: {wrong_labels[:5]}",
    )
    if dice:
        report.info(
            f"foreground Dice after this short run: {float(np.mean(dice)):.3f} "
            "(not a measure of final accuracy)"
        )


def report_memory(report: Report, watcher: MemoryWatcher) -> None:
    """Report each GPU's peak memory, warning if it came close to the limit."""
    if not watcher.available:
        report.info("GPU memory: not measured; nvidia-smi isn't available")
        return
    for index, ((peak, total), (base, _)) in enumerate(
        zip(watcher.peak, watcher.baseline, strict=True)
    ):
        message = (
            f"GPU {index} peak memory {peak / 1024:.1f} of {total / 1024:.1f} GiB "
            f"({base / 1024:.1f} GiB in use before training)"
        )
        if peak >= MEMORY_WARN_FRACTION * total:
            report.add("WARN", message + "; close to the limit")
        else:
            report.add("PASS", message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Folder for all outputs.")
    parser.add_argument("--model", help="Architecture to train (default: the analyzer's).")
    parser.add_argument("--epochs", type=int, default=5, help="At least 3, to measure speed.")
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4, help="For analysis and preprocessing.")
    parser.add_argument("--nfolds", type=int, help="Folds, if the dataset has fewer than 5 cases.")
    args = parser.parse_args()

    results = args.out / "results"
    numpy_dir = args.out / "numpy"
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = Report()

    print("Analyzing and preprocessing", flush=True)
    passed, seconds = run_step(
        "analyze",
        "analyze_entrypoint",
        "analyze_entry",
        [
            "--data",
            str(args.data),
            "--results",
            str(results),
            "--num-workers-analyze",
            str(args.workers),
            "--overwrite",
            *(["--nfolds", str(args.nfolds)] if args.nfolds else []),
        ],
        logs,
    )
    if not report.check(passed, f"mist_analyze finished in {seconds:.0f} s", "mist_analyze failed"):
        return summarize(report)

    config_path = results / "config.json"
    config = json.loads(config_path.read_text())
    expected = args.model or "nnunet-plain"
    if args.model is None:
        report.check(
            config["model"]["architecture"] == expected,
            f"analyzer chose the default architecture, {expected}",
            f"analyzer chose {config['model']['architecture']}, not {expected}",
        )
    report.info(f"spacing {config['spatial_config']['target_spacing']}")
    edit_config(config_path, args.steps_per_epoch)

    passed, seconds = run_step(
        "preprocess",
        "preprocess_entrypoint",
        "preprocess_entry",
        [
            "--results",
            str(results),
            "--numpy",
            str(numpy_dir),
            "--num-workers-preprocess",
            str(args.workers),
            "--overwrite",
        ],
        logs,
    )
    if not report.check(
        passed, f"mist_preprocess finished in {seconds:.0f} s", "mist_preprocess failed"
    ):
        return summarize(report)

    print(
        f"Training fold {FOLD} for {args.epochs} epochs of at least {args.steps_per_epoch} steps, "
        "then predicting it",
        flush=True,
    )
    train_args = [
        "--results",
        str(results),
        "--numpy",
        str(numpy_dir),
        "--folds",
        str(FOLD),
        "--epochs",
        str(args.epochs),
        "--warmup-epochs",
        "0",
        "--validate-every",
        str(args.epochs),
        "--overwrite",
    ]
    if args.model:
        train_args += ["--model", args.model]
    with MemoryWatcher() as watcher:
        passed, seconds = run_step("train", "train_entrypoint", "train_entry", train_args, logs)
    report_memory(report, watcher)
    if not report.check(passed, f"mist_train finished in {seconds:.0f} s", "mist_train failed"):
        return summarize(report)

    config = json.loads(config_path.read_text())
    planning = config["planning"]
    report.info(
        f"planned patch {config['spatial_config']['patch_size']}, batch size "
        f"{config['training']['batch_size_per_gpu']} per GPU, for "
        f"{planning['gpu_memory_gb']} GiB; estimated {planning['estimated_gpu_memory_gb']} GiB"
    )
    report.check(
        config["model"]["architecture"] == expected,
        f"trained {expected}",
        f"trained {config['model']['architecture']}, not {expected}",
    )
    loader = config["training"]["hardware"]["data_loader"]
    if loader == "generic" and shutil.which("nvidia-smi"):
        report.add("WARN", "used the generic loader on an NVIDIA machine; is DALI installed?")
    else:
        report.info(f"data loader: {loader}")

    paths = pd.read_csv(results / "train_paths.csv")
    check_losses(report, results, args.epochs)
    report_step_time(report, results, config, int((paths["fold"] != FOLD).sum()))
    check_saved_model(report, results, config, args.epochs)
    check_predictions(report, results, config)
    return summarize(report)


def summarize(report: Report) -> int:
    """Print the totals and return the process's exit status."""
    counts = report.counts
    print(
        f"\nResult: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed. "
        "This checks that the pipeline works, not that accuracy improved.",
        flush=True,
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
