"""Stage 0 baseline for the CPU / AMD ROCm support plan.

Captures what already works on CPU-only hardware today (analyze, preprocess)
and what doesn't (mist_train can't even be *imported* without
nvidia-dali-cuda120 installed), so later stages have a concrete baseline to
diff their own progress against -- per cpu_rocm_support_plan.md's Stage 0.

The train-side check deliberately runs in a real subprocess rather than
in-process: tests/conftest.py globally stubs `nvidia.dali` in sys.modules so
the rest of the suite can exercise patch_3d_trainer.py's own logic without
the real (CUDA-only, heavy) DALI package installed. That's the right call
for those tests, but it would make an in-process import here silently
succeed via the fake stub -- masking exactly the failure this test exists to
document. A subprocess gets a fresh interpreter that never loads conftest.py,
so it sees what a real user without DALI installed actually sees.
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


def test_train_entrypoint_cannot_be_imported_without_dali() -> None:
    """Today, mist_train fails at import time on a machine without DALI.

    patch_3d_trainer.py has an unguarded top-level `from mist.data_loading
    import dali_loader`, so merely importing mist.cli.train_entrypoint (let
    alone running it) requires nvidia-dali-cuda120 to be installed -- on
    CUDA hardware only. This is the single hardest blocker CPU/ROCm support
    has to clear; Stage 2-4 of the plan replace this with a registry lookup
    that only imports DALI when it's actually selected. Once that lands,
    this test's assertion flips (or this test is deleted) -- it exists to
    make that change visible as a real regression check, not to protect the
    current failure.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import mist.cli.train_entrypoint"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        "mist.cli.train_entrypoint imported successfully -- if CPU/ROCm "
        "support has landed, update or remove this baseline test rather "
        "than leaving it silently green."
    )
    assert "ModuleNotFoundError" in result.stderr
    assert "nvidia" in result.stderr
