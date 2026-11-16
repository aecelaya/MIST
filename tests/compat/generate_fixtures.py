"""Build inference-compatibility fixtures with an installed release of MIST.

MIST 3.0 promises that results folders from 2.x (2.3.0 on) keep working for
inference. This script trains tiny models with whatever `mist-medical` is
installed, saves that version's own predictions next to them, and
tests/compat/test_inference_compat.py checks that the current code reproduces
those predictions.

Run it in a fresh environment with the old release installed, from a MIST
checkout, by path (not with `python -m`, which would import the checkout's
`mist` instead of the installed one)::

    pip install "mist-medical[dali]==2.4.1"      # or ==2.5.0rc0
    python tests/compat/generate_fixtures.py --out compat/2.4.1

On a GPU it takes a few minutes per model. On a CPU it works too, but 2.4.1 then
records the generic loader instead of DALI, so the "dali" config shim is only
covered by unit tests. Pack each version's folder and attach it to the
`compat-fixtures-v1` release of mist-medical/MIST, which CI downloads::

    tar -czf mist-compat-2.4.1.tar.gz -C compat 2.4.1

Locally, unpack the tarballs into tests/compat/fixtures/ (ignored by git), or
point MIST_COMPAT_FIXTURES at the folder that holds them.

The folder it writes::

    <out>/
      manifest.json             MIST and torch versions, device, models, cases
      dataset/                  synthetic dataset.json and NIfTI images
      numpy/                    preprocessed arrays, read by test_on_fold
      <model>/results/          config.json, models/fold_0.pt, CSVs, and
                                predictions/train/raw/ from test_on_fold
      <model>/predict/          mist_predict on the raw images of fold 0

This file imports nothing from MIST, so it runs against any installed release.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

NUM_CASES = 8
NFOLDS = 2
PATCH_SIZE = (32, 32, 32)
LABELS = [0, 1, 2]

# (size_xyz, spacing_xyz, rotation in degrees). Different spacings make
# preprocessing resample, which is where inference changed in 2.5.0.
_GEOMETRIES = [
    ((48, 48, 40), (1.0, 1.0, 1.0), 0.0),
    ((52, 48, 24), (0.8, 0.8, 2.0), 0.0),
    ((40, 44, 36), (1.2, 1.1, 1.3), 15.0),
    ((48, 52, 40), (1.0, 1.0, 1.0), 0.0),
    ((56, 56, 20), (0.7, 0.7, 2.5), 0.0),
    ((44, 44, 44), (1.0, 1.0, 1.0), -10.0),
    ((48, 40, 32), (0.9, 1.0, 1.4), 0.0),
    ((40, 40, 40), (1.1, 1.1, 1.1), 0.0),
]


def _direction(angle_deg: float) -> tuple[float, ...]:
    """A rotation about z, as a row-major SimpleITK direction."""
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return tuple(float(v) for v in rot.flatten())


def _make_case(index: int) -> tuple[sitk.Image, sitk.Image]:
    """A CT-like image with a bright sphere (label 1) and a dark one (label 2)."""
    size, spacing, angle = _GEOMETRIES[index]
    rng = np.random.default_rng(1000 + index)
    shape_zyx = (size[2], size[1], size[0])
    zz, yy, xx = np.meshgrid(
        *(np.arange(n) * s for n, s in zip(shape_zyx, spacing[::-1], strict=True)),
        indexing="ij",
    )
    extent = np.array([n * s for n, s in zip(shape_zyx, spacing[::-1], strict=True)])

    image = rng.normal(40.0, 15.0, shape_zyx).astype(np.float32)  # Soft tissue.
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    margin = 0.3 * extent
    for label, value, radius in ((1, 200.0, 9.0), (2, -150.0, 5.0)):
        center = margin + rng.random(3) * (extent - 2 * margin)
        inside = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
        inside = inside <= (radius * rng.uniform(0.8, 1.2)) ** 2
        mask[inside] = label
        image[inside] = value + rng.normal(0.0, 15.0, int(inside.sum()))

    def to_sitk(array: np.ndarray, pixel: int) -> sitk.Image:
        out = sitk.Cast(sitk.GetImageFromArray(array), pixel)
        out.SetSpacing(spacing)
        out.SetOrigin((float(index), -2.0 * index, 0.5 * index))
        out.SetDirection(_direction(angle))
        return out

    return to_sitk(image, sitk.sitkFloat32), to_sitk(mask, sitk.sitkUInt8)


def write_dataset(root: Path) -> list[str]:
    """Write the synthetic dataset and its dataset.json. Returns the case ids."""
    ids = []
    for index in range(NUM_CASES):
        case_id = f"case_{index}"
        case_dir = root / "train-data" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        image, mask = _make_case(index)
        sitk.WriteImage(image, str(case_dir / "image.nii.gz"))
        sitk.WriteImage(mask, str(case_dir / "mask.nii.gz"))
        ids.append(case_id)
    dataset = {
        "task": "mist-compat",
        "modality": "ct",
        "train-data": str((root / "train-data").resolve()),
        "mask": ["mask.nii.gz"],
        "images": {"ct": ["image.nii.gz"]},
        "labels": LABELS,
        "final_classes": {"bright": [1], "dark": [2]},
    }
    (root / "dataset.json").write_text(json.dumps(dataset, indent=2))
    return ids


def run(*command: str) -> None:
    """Run one MIST command, printing it first.

    The command comes from the same environment as this Python, so the venv
    doesn't need to be activated.
    """
    local = Path(sys.executable).parent / command[0]
    if local.is_file():
        command = (str(local), *command[1:])
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def default_models(version: str) -> list[str]:
    """The 2.x default network under its 2.x name, resenc from 2.4, nnunet-plain from 2.5."""
    models = ["nnunet"]
    major, minor = (int(part) for part in version.split(".")[:2])
    if (major, minor) >= (2, 4):
        models.append("resenc")
    if (major, minor) >= (2, 5):
        models.append("nnunet-plain")
    return models


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Folder to create.")
    parser.add_argument("--models", nargs="+", help="Networks to train (default: by version).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if found).")
    args = parser.parse_args()

    version = importlib.metadata.version("mist-medical")
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists.")
    device = args.device or default_device()
    models = args.models or default_models(version)
    print(f"mist-medical {version}, device {device}, models {models}", flush=True)

    dataset_dir = out / "dataset"
    numpy_dir = out / "numpy"
    base_results = out / "_analyze"
    ids = write_dataset(dataset_dir)
    dataset_json = str(dataset_dir / "dataset.json")
    run(
        "mist_analyze",
        "--data", dataset_json,
        "--results", str(base_results),
        "--nfolds", str(NFOLDS),
    )  # fmt: skip
    run(
        "mist_preprocess",
        "--results", str(base_results),
        "--numpy", str(numpy_dir),
    )  # fmt: skip

    train_paths = pd.read_csv(base_results / "train_paths.csv")
    fold_ids = sorted(train_paths.loc[train_paths["fold"] == 0, "id"])
    predict_csv = out / "predict_paths.csv"
    pd.DataFrame(
        {
            "id": fold_ids,
            "ct": [str(dataset_dir / "train-data" / i / "image.nii.gz") for i in fold_ids],
        }
    ).to_csv(predict_csv, index=False)

    for model in models:
        results = out / model / "results"
        shutil.copytree(base_results, results)
        config_path = results / "config.json"
        config = json.loads(config_path.read_text())
        training = config["training"]
        training["min_steps_per_epoch"] = args.steps_per_epoch
        training["val_percent"] = 0.0
        training["hardware"]["num_cpu_workers"] = 2
        config_path.write_text(json.dumps(config, indent=2))
        run(
            "mist_train",
            "--results", str(results),
            "--numpy", str(numpy_dir),
            "--model", model,
            "--patch-size", *(str(p) for p in PATCH_SIZE),
            "--batch-size-per-gpu", "2",
            "--epochs", str(args.epochs),
            "--warmup-epochs", "0",
            "--folds", "0",
        )  # fmt: skip
        run(
            "mist_predict",
            "--models-dir", str(results / "models"),
            "--config", str(config_path),
            "--paths-csv", str(predict_csv),
            "--output", str(out / model / "predict"),
            "--device", device,
        )  # fmt: skip
        # Logs and resume checkpoints (weights plus optimizer state) aren't
        # needed for inference and triple the size.
        shutil.rmtree(results / "logs", ignore_errors=True)
        shutil.rmtree(results / "checkpoints", ignore_errors=True)

    shutil.rmtree(base_results)
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    manifest = {
        "mist_version": version,
        "torch_version": torch_version,
        "device": device,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(dataset_dir),
        "models": models,
        "case_ids": ids,
        "fold_0_ids": fold_ids,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
