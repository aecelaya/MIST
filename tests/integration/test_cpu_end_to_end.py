"""A full CPU-only MIST run: analyze, preprocess, train, evaluate, predict.

Runs the whole pipeline end to end on synthetic data (a couple of epochs, all
folds), confirming `training.hardware.data_loader` resolves to `"generic"` and
every stage completes on a machine without a GPU, with real code rather than
mocks.
"""

from pathlib import Path

import pandas as pd
import pytest

from mist.cli.inference_entrypoint import inference_entry
from mist.cli.train_entrypoint import train_entry
from mist.utils import io
from tests.fixtures.fixtures import generate_dataset
from tests.fixtures.harness import run_pipeline


@pytest.mark.parametrize("model", [None, "nnunet-pocket"], ids=["default", "pocket"])
def test_full_cpu_pipeline_trains_and_predicts_with_generic_loader(
    tmp_path: Path, model: str | None
) -> None:
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

    # analyze writes the generic loader on every machine.
    config = io.read_json_file(str(config_path))
    assert config["training"]["hardware"]["data_loader"] == "generic"

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
            *(["--model", model] if model else []),
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

    # The loader is persisted at train time, once, in the parent process --
    # confirm it's still "generic" when read back from disk.
    trained_config = io.read_json_file(str(config_path))
    assert trained_config["training"]["hardware"]["data_loader"] == "generic"
    assert trained_config["model"]["architecture"] == (model or "nnunet-plain")

    for fold in range(2):
        assert (results_dir / "models" / f"fold_{fold}.pt").is_file()
    assert (results_dir / "results.csv").is_file()

    # mist_predict on the original (raw) images -- a separate code path from
    # training/evaluation, so inference on CPU is checked too.
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
