"""Stage 4 release gate: a full CPU-only MIST run using the generic loader.

Runs analyze -> preprocess -> train (a couple epochs, all folds) -> predict
end to end on synthetic data, on this real CPU-only machine, confirming
`training.hardware.data_loader` resolves to `"generic"` and the whole
pipeline completes without ever touching DALI -- this is the actual
"does this work" check promised by `cpu_rocm_support_plan.md`'s Stage 4
release gate, not just mocked unit tests. It's also the first point in the
whole CPU/ROCm plan where CPU *training* (not just analyze/preprocess, see
`test_stage0_baseline.py`) actually runs.
"""

from pathlib import Path

import pandas as pd

from mist.cli.inference_entrypoint import inference_entry
from mist.cli.train_entrypoint import train_entry
from mist.utils import io
from tests.regression.ants_sitk.fixtures import generate_dataset
from tests.regression.ants_sitk.harness import run_pipeline


def test_full_cpu_pipeline_trains_and_predicts_with_generic_loader(tmp_path: Path) -> None:
    """analyze -> preprocess -> train -> predict, all on CPU, via "generic"."""
    dataset = generate_dataset(tmp_path / "dataset")
    outputs = run_pipeline(
        dataset.dataset_json,
        tmp_path / "results",
        tmp_path / "numpy",
        compute_dtms=True,
        nfolds=2,
    )
    results_dir, numpy_dir = outputs.results_dir, outputs.numpy_dir
    config_path = results_dir / "config.json"

    # analyze's default is unresolved -- confirms the resolution below is
    # actually doing something, not just reflecting an already-"generic"
    # default.
    config = io.read_json_file(str(config_path))
    assert config["training"]["hardware"]["data_loader"] == "auto"

    # Keep the run fast: a handful of steps per epoch instead of the
    # min_steps_per_epoch=250 default. There's no CLI flag for this --
    # deliberately hand-edit-only, like patch_size or grad_clip_norm (see
    # analyzer_utils.py::build_base_config()).
    config["training"]["min_steps_per_epoch"] = 2
    io.write_json_file(config_path, config)

    train_entry(
        [
            "--results",
            str(results_dir),
            "--numpy",
            str(numpy_dir),
            "--model",
            "nnunet-pocket",
            "--epochs",
            "2",
            "--warmup-epochs",
            "0",
            "--batch-size-per-gpu",
            "1",
            "--num-workers-evaluate",
            "1",
            "--overwrite",
        ]
    )

    # The resolved loader is persisted at train time, once, in the parent
    # process -- confirm it actually picked "generic" on this CPU-only host,
    # and that it's the same value read back from disk (not just an
    # in-memory artifact of this process).
    trained_config = io.read_json_file(str(config_path))
    assert trained_config["training"]["hardware"]["data_loader"] == "generic"

    for fold in range(2):
        assert (results_dir / "models" / f"fold_{fold}.pt").is_file()
    assert (results_dir / "results.csv").is_file()

    # mist_predict on the original (raw) images -- a separate code path from
    # training/evaluation, confirming the end-to-end CPU story holds for
    # inference too, per the Stage 4 release gate.
    paths_csv = tmp_path / "predict_paths.csv"
    pd.DataFrame(
        {
            "id": dataset.patient_ids,
            "ct": [str(dataset.train_data / pid / "image.nii.gz") for pid in dataset.patient_ids],
        }
    ).to_csv(paths_csv, index=False)

    predict_output = tmp_path / "predictions"
    inference_entry(
        [
            "--models-dir",
            str(results_dir / "models"),
            "--config",
            str(config_path),
            "--paths-csv",
            str(paths_csv),
            "--output",
            str(predict_output),
            "--device",
            "cpu",
        ]
    )

    predictions = list(predict_output.glob("*.nii.gz"))
    assert len(predictions) == len(dataset.patient_ids)
