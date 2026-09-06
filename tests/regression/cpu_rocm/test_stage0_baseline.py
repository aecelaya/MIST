"""Stage 0 baseline for the CPU / AMD ROCm support plan.

Originally captured what already worked on CPU-only hardware (analyze,
preprocess) and what didn't (mist_train couldn't even be *imported* without
nvidia-dali-cuda120 installed, because patch_3d_trainer.py had an unguarded
top-level `from mist.data_loading import dali_loader`), so later stages had
a concrete baseline to diff their own progress against -- per
cpu_rocm_support_plan.md's Stage 0.

Stage 4 fixed the import blocker: patch_3d_trainer.py (and
inference_runners.py::test_on_fold, found along the way -- train_entry()
calls it right after training) now go through a registry lookup
(mist.data_loading.data_loader_registry) instead of importing dali_loader
directly, so mist.cli.train_entrypoint imports fine without DALI installed.
test_train_entrypoint_cannot_be_imported_without_dali flipped to
test_train_entrypoint_can_be_imported_without_dali below to make that fix a
real regression guard, per this file's own original note that it should.

The train-side check deliberately runs in a real subprocess rather than
in-process: tests/conftest.py globally stubs `nvidia.dali` in sys.modules so
the rest of the suite can exercise patch_3d_trainer.py's own logic without
the real (CUDA-only, heavy) DALI package installed. That's the right call
for those tests, but it would make an in-process import here silently
succeed via the fake stub even before Stage 4's fix -- masking the failure
this test originally existed to document. A subprocess gets a fresh
interpreter that never loads conftest.py, so it sees exactly what a real
user without DALI installed sees.
"""

import subprocess
import sys
from pathlib import Path

from tests.regression.ants_sitk.fixtures import generate_dataset
from tests.regression.ants_sitk.harness import run_pipeline


def test_analyze_and_preprocess_succeed_on_cpu(tmp_path: Path) -> None:
    """analyze + preprocess already work on CPU-only hardware today.

    Neither stage touches DALI or a blocking torch.cuda call, so this is
    expected to keep passing unchanged through every later stage of the
    CPU/ROCm plan -- it's a regression guard, not something Stage 1+ should
    need to fix.
    """
    dataset = generate_dataset(tmp_path / "dataset")
    outputs = run_pipeline(
        dataset.dataset_json,
        tmp_path / "results",
        tmp_path / "numpy",
    )

    assert (outputs.results_dir / "config.json").is_file()
    assert (outputs.results_dir / "train_paths.csv").is_file()
    assert (outputs.results_dir / "fg_bboxes.csv").is_file()
    assert any((outputs.numpy_dir / "images").glob("*.npy"))
    assert any((outputs.numpy_dir / "labels").glob("*.npy"))


def test_train_entrypoint_can_be_imported_without_dali() -> None:
    """mist_train now imports fine on a machine without DALI installed.

    Was test_train_entrypoint_cannot_be_imported_without_dali until Stage 4:
    patch_3d_trainer.py's unguarded top-level `from mist.data_loading import
    dali_loader` -- the single hardest blocker CPU/ROCm support had to clear
    -- is gone, replaced by a `data_loader_registry` lookup that only
    resolves to the real DALI module when `training.hardware.data_loader`
    is actually `"dali"`. Run in a real subprocess (see module docstring)
    so a regression here (some other file reintroducing an unguarded DALI
    import into mist_train's own import graph) can't be masked by
    tests/conftest.py's global `nvidia.dali` stub.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import mist.cli.train_entrypoint"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"mist.cli.train_entrypoint failed to import without DALI installed "
        f"-- a regression in the data-loader registry wiring (Stage 4). "
        f"stderr:\n{result.stderr}"
    )
