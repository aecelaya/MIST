"""Results folders from older MIST releases still predict the same labels.

Each fixture is a results folder trained by a released MIST (2.3.0, 2.4.0,
2.4.1, 2.5.0rc0)
with that version's own predictions saved next to it; generate_fixtures.py
builds them. These tests run the current mist_predict, test_on_fold, and
mist_evaluate on those folders, on the CPU.

Releases before 2.5.0 resampled predicted labels back to the original image;
2.5.0 resamples probabilities instead, which can move boundary voxels. So those
fixtures get a looser tolerance. Fixtures made on a GPU also differ from CPU
inference in the last bits, which can flip a few boundary voxels.

The fixtures are too large for git. Unpack them into tests/compat/fixtures/ or
set MIST_COMPAT_FIXTURES to their folder. Without them these tests skip, unless
MIST_REQUIRE_COMPAT=1, which CI sets once the fixtures are published.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from mist.cli.evaluation_entrypoint import evaluation_entry
from mist.cli.inference_entrypoint import inference_entry
from mist.inference import inference_runners

_DEFAULT_ROOT = Path(__file__).parent / "fixtures"

# Fraction of voxels that must match the saved prediction, and the lowest Dice
# per label, by whether the fixture's version resampled probabilities (2.5.0+).
_TOLERANCE = {
    True: {"agreement": 0.995, "dice": 0.97},
    False: {"agreement": 0.98, "dice": 0.90},
}
# Labels smaller than this in the saved prediction aren't held to the Dice
# threshold, since a few voxels move it a lot.
_MIN_VOXELS_FOR_DICE = 50


def _fixture_root() -> Path:
    return Path(os.environ.get("MIST_COMPAT_FIXTURES", _DEFAULT_ROOT))


def _discover() -> list[tuple[Path, str]]:
    """(version folder, model) for every fixture found."""
    root = _fixture_root()
    if not root.is_dir():
        return []
    found = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for model in manifest["models"]:
            found.append((manifest_path.parent, model))
    return found


_FIXTURES = _discover()


def _params():
    if _FIXTURES:
        return [pytest.param(f, m, id=f"{f.name}-{m}") for f, m in _FIXTURES]
    reason = (
        f"No inference compatibility fixtures in {_fixture_root()}; see "
        "tests/compat/generate_fixtures.py."
    )
    if os.environ.get("MIST_REQUIRE_COMPAT") == "1":
        return [pytest.param(None, None, id="missing")]
    return [pytest.param(None, None, id="missing", marks=pytest.mark.skip(reason=reason))]


def _require(fixture: Path | None) -> None:
    if fixture is None:
        pytest.fail(
            f"MIST_REQUIRE_COMPAT=1 but no fixtures were found in {_fixture_root()}.",
            pytrace=False,
        )


def _resamples_probabilities(version: str) -> bool:
    major, minor = (int(part) for part in version.split(".")[:2])
    return (major, minor) >= (2, 5)


def _manifest(fixture: Path) -> dict:
    return json.loads((fixture / "manifest.json").read_text())


def _relocate(text: str, manifest: dict, fixture: Path) -> str:
    """Point paths from the machine that made the fixture at this copy."""
    return text.replace(manifest["dataset_root"], str(fixture / "dataset"))


def _copy_results(fixture: Path, model: str, manifest: dict, tmp_path: Path) -> Path:
    """A writable copy of a fixture's results folder, with paths fixed."""
    results = tmp_path / "results"
    shutil.copytree(fixture / model / "results", results)
    for name in ("config.json", "train_paths.csv", "test_paths.csv"):
        path = results / name
        if path.is_file():
            path.write_text(_relocate(path.read_text(), manifest, fixture))
    return results


def _compare(saved: Path, new: Path, resampled_probabilities: bool) -> list[str]:
    """Problems with a new prediction compared to the saved one."""
    saved_image, new_image = sitk.ReadImage(str(saved)), sitk.ReadImage(str(new))
    if saved_image.GetSize() != new_image.GetSize():
        return [f"{new.name}: size {new_image.GetSize()} != {saved_image.GetSize()}"]
    problems = []
    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        a, b = getattr(saved_image, name)(), getattr(new_image, name)()
        if not np.allclose(a, b, atol=1e-4):
            problems.append(f"{new.name}: {name[3:].lower()} {b} != {a}")

    expected = sitk.GetArrayFromImage(saved_image)
    actual = sitk.GetArrayFromImage(new_image)
    tolerance = _TOLERANCE[resampled_probabilities]
    agreement = float((expected == actual).mean())
    if agreement < tolerance["agreement"]:
        problems.append(f"{new.name}: {agreement:.4f} of voxels match")
    for label in np.unique(expected[expected > 0]):
        a, b = expected == label, actual == label
        if a.sum() < _MIN_VOXELS_FOR_DICE:
            continue
        dice = 2 * float((a & b).sum()) / float(a.sum() + b.sum())
        if dice < tolerance["dice"]:
            problems.append(f"{new.name}: label {label} Dice {dice:.3f}")
    return problems


def _compare_folders(saved_dir: Path, new_dir: Path, ids: list[str], version: str) -> None:
    resampled = _resamples_probabilities(version)
    problems = []
    foreground = 0
    for case_id in ids:
        saved, new = saved_dir / f"{case_id}.nii.gz", new_dir / f"{case_id}.nii.gz"
        assert saved.is_file(), f"The fixture has no saved prediction {saved}."
        if not new.is_file():
            problems.append(f"{case_id}: no prediction written")
            continue
        foreground += int((sitk.GetArrayFromImage(sitk.ReadImage(str(saved))) > 0).sum())
        problems += _compare(saved, new, resampled)
    assert foreground > 0, (
        "The fixture's saved predictions have no foreground, so they can't show a "
        "difference. Regenerate it with more --epochs."
    )
    assert not problems, f"Predictions differ from MIST {version}'s:\n" + "\n".join(problems)


@pytest.mark.parametrize(("fixture", "model"), _params())
def test_mist_predict_reproduces_saved_predictions(fixture, model, tmp_path):
    """mist_predict on an old results folder gives that version's labels."""
    _require(fixture)
    manifest = _manifest(fixture)
    results = _copy_results(fixture, model, manifest, tmp_path)
    ids = manifest["fold_0_ids"]
    paths_csv = tmp_path / "predict_paths.csv"
    pd.DataFrame(
        {
            "id": ids,
            "ct": [str(fixture / "dataset" / "train-data" / i / "image.nii.gz") for i in ids],
        }
    ).to_csv(paths_csv, index=False)
    output = tmp_path / "predict"

    inference_entry(
        [
            "--models-dir",
            str(results / "models"),
            "--config",
            str(results / "config.json"),
            "--paths-csv",
            str(paths_csv),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    _compare_folders(fixture / model / "predict", output, ids, manifest["mist_version"])


@pytest.mark.parametrize(("fixture", "model"), _params())
def test_test_on_fold_reproduces_saved_predictions(fixture, model, tmp_path):
    """test_on_fold reads an old config.json and preprocessed data as before."""
    _require(fixture)
    manifest = _manifest(fixture)
    results = _copy_results(fixture, model, manifest, tmp_path)
    shutil.rmtree(results / "predictions", ignore_errors=True)
    args = argparse.Namespace(results=str(results), numpy=str(fixture / "numpy"))

    # 2.x configs may say "dali", which now warns and uses the generic loader.
    inference_runners.test_on_fold(args, 0, device="cpu")

    _compare_folders(
        fixture / model / "results" / "predictions" / "train" / "raw",
        results / "predictions" / "train" / "raw",
        manifest["fold_0_ids"],
        manifest["mist_version"],
    )


@pytest.mark.parametrize(("fixture", "model"), _params())
def test_mist_evaluate_runs_on_old_config(fixture, model, tmp_path):
    """mist_evaluate takes an old config.json and scores every case."""
    _require(fixture)
    manifest = _manifest(fixture)
    results = _copy_results(fixture, model, manifest, tmp_path)
    ids = manifest["fold_0_ids"]
    paths_csv = tmp_path / "evaluate_paths.csv"
    pd.DataFrame(
        {
            "id": ids,
            "mask": [str(fixture / "dataset" / "train-data" / i / "mask.nii.gz") for i in ids],
            "prediction": [str(fixture / model / "predict" / f"{i}.nii.gz") for i in ids],
        }
    ).to_csv(paths_csv, index=False)
    output_csv = tmp_path / "evaluation.csv"

    evaluation_entry(
        [
            "--config",
            str(results / "config.json"),
            "--paths-csv",
            str(paths_csv),
            "--output-csv",
            str(output_csv),
        ]  # fmt: skip
    )

    scores = pd.read_csv(output_csv)
    assert set(ids) <= set(scores["id"].astype(str))
