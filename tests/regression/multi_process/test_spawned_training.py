"""Train on two real processes, since multi-GPU bugs don't show on one.

Some bugs only appear across spawned processes, such as settings that don't
reach them, or weights that DDP finds unused. This runs real DDP over gloo, so
it can't catch anything specific to GPUs or NCCL.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from mist.utils import io
from tests.regression.ants_sitk.fixtures import generate_dataset
from tests.regression.ants_sitk.harness import run_pipeline
from tests.regression.multi_process import spawn_training


# At this patch, resenc has more decoder stages than deep supervision uses, which
# once crashed DDP on the second step. Smaller patches have no such stage.
@pytest.mark.parametrize("model", ["nnunet-plain", "resenc"])
def test_training_completes_on_two_spawned_processes(tmp_path: Path, model: str) -> None:
    """A fold trains and validates under DDP, and rank 0 saves the model."""
    dataset = generate_dataset(tmp_path / "dataset")
    outputs = run_pipeline(
        dataset.dataset_json,
        tmp_path / "results",
        tmp_path / "numpy",
        nfolds=2,
    )
    config_path = outputs.results_dir / "config.json"
    config = io.read_json_file(str(config_path))
    config["training"]["min_steps_per_epoch"] = 2
    config["training"]["folds"] = [0]
    config["training"]["hardware"]["num_cpu_workers"] = 1
    config["spatial_config"]["patch_size"] = [64, 64, 64]
    config["training"]["batch_size_per_gpu"] = 2
    io.write_json_file(config_path, config)

    completed = subprocess.run(
        [
            sys.executable,
            spawn_training.__file__,
            "--results",
            str(outputs.results_dir),
            "--numpy",
            str(outputs.numpy_dir),
            "--model",
            model,
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert (outputs.results_dir / "models" / "fold_0.pt").is_file()
