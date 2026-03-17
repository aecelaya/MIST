"""Tests for converting MSD datasets to MIST format."""
import json
import shutil
import numpy as np
import pytest
import SimpleITK as sitk
from pathlib import Path

# MIST imports.
from mist.utils import io, progress_bar
from mist.conversion_tools import conversion_utils
from mist.conversion_tools.msd import convert_msd, copy_msd_data
from tests.conversion_tools.helpers import DummyProgressBar


@pytest.fixture
def temp_msd_dir(tmp_path):
    """Fixture to create a temporary MSD dataset directory structure."""
    source = tmp_path / "msd_source"
    dest = tmp_path / "mist_dest"
    source.mkdir()
    (source / "imagesTr").mkdir()
    (source / "labelsTr").mkdir()
    (source / "imagesTs").mkdir()

    # Dummy 4D image (2 channels) and mask.
    mask_data = np.ones((10, 10, 10), dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_data)

    img1 = sitk.AdditiveGaussianNoise(
        sitk.Image(10, 10, 10, sitk.sitkFloat32), mean=0.0, standardDeviation=1.0
    )
    img2 = sitk.AdditiveGaussianNoise(
        sitk.Image(10, 10, 10, sitk.sitkFloat32), mean=0.0, standardDeviation=1.0
    )
    image = sitk.JoinSeries([img1, img2])

    sitk.WriteImage(image, str(source / "imagesTr" / "patient_001.nii.gz"))
    sitk.WriteImage(mask, str(source / "labelsTr" / "patient_001.nii.gz"))
    sitk.WriteImage(image, str(source / "imagesTs" / "patient_002.nii.gz"))

    dataset_json = {
        "name": "MSD Task",
        "modality": {"0": "T1", "1": "T2"},
        "labels": {"0": "background", "1": "tumor"},
        "training": [{
            "image": "imagesTr/patient_001.nii.gz",
            "label": "labelsTr/patient_001.nii.gz",
        }],
        "test": ["imagesTs/patient_002.nii.gz"],
    }
    with open(source / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f)

    return source, dest


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """Patch utility functions to avoid actual file operations."""
    monkeypatch.setattr(
        progress_bar, "get_progress_bar", lambda msg: DummyProgressBar()
    )
    monkeypatch.setattr(
        conversion_utils,
        "copy_image_from_source_to_dest",
        lambda src, dst: shutil.copy(src, dst),
    )

    def fake_write_json_file(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    monkeypatch.setattr(io, "write_json_file", fake_write_json_file)


def test_convert_msd_creates_correct_output(temp_msd_dir):
    """Tests convert_msd creates the correct directory structure and files."""
    source, dest = temp_msd_dir
    convert_msd(source, dest)

    # Check training output.
    patient_train_dir = dest / "raw" / "train" / "patient_001"
    assert (patient_train_dir / "T1.nii.gz").exists()
    assert (patient_train_dir / "T2.nii.gz").exists()
    assert (patient_train_dir / "mask.nii.gz").exists()

    # Check test output.
    patient_test_dir = dest / "raw" / "test" / "patient_002"
    assert (patient_test_dir / "T1.nii.gz").exists()
    assert (patient_test_dir / "T2.nii.gz").exists()

    # Check MIST config uses relative paths.
    with open(dest / "dataset.json", encoding="utf-8") as f:
        config = json.load(f)
    assert config["modality"] == "other"
    assert config["labels"] == [0, 1]
    assert "tumor" in config["final_classes"]
    assert config["train-data"] == "raw/train"
    assert config["test-data"] == "raw/test"


def test_convert_msd_accepts_path_objects(temp_msd_dir):
    """convert_msd accepts pathlib.Path inputs without error."""
    source, dest = temp_msd_dir
    convert_msd(Path(source), Path(dest))
    assert (dest / "dataset.json").exists()


def test_convert_msd_raises_if_source_missing(tmp_path):
    """Tests convert_msd raises FileNotFoundError if missing source dir."""
    with pytest.raises(FileNotFoundError):
        convert_msd(tmp_path / "missing_dir", tmp_path / "dest")


def test_convert_msd_raises_if_dataset_json_missing(tmp_path):
    """Tests convert_msd raises FileNotFoundError if dataset.json is missing."""
    (tmp_path / "imagesTr").mkdir()
    with pytest.raises(FileNotFoundError):
        convert_msd(tmp_path, tmp_path / "dest")


def test_copy_msd_data_skips_missing_files(tmp_path):
    """Tests that copy_msd_data skips patients with missing image files."""
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()
    msd_json = {
        "training": [{
            "image": "imagesTr/missing.nii.gz",
            "label": "labelsTr/missing.nii.gz",
        }]
    }
    copy_msd_data(source, dest, msd_json, {0: "ct"}, "training", "Skip test")

    assert not (dest / "raw" / "train" / "missing").exists()


def test_copy_msd_data_skips_missing_mask(tmp_path):
    """Tests that copy_msd_data skips when mask is missing but image exists."""
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()

    sitk.WriteImage(
        sitk.Image(10, 10, 10, sitk.sitkFloat32),
        str(source / "imagesTr" / "patient_missing_mask.nii.gz"),
    )

    msd_json = {
        "training": [{
            "image": "imagesTr/patient_missing_mask.nii.gz",
            "label": "labelsTr/patient_missing_mask.nii.gz",
        }]
    }
    copy_msd_data(
        source, dest, msd_json, {0: "ct"}, "training", "Missing mask test"
    )

    assert not (dest / "raw" / "train" / "patient_missing_mask").exists()


def test_copy_msd_data_single_modality_copy(tmp_path):
    """Tests that copy_msd_data directly copies image if only one modality."""
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()

    sitk.WriteImage(
        sitk.Image(10, 10, 10, sitk.sitkFloat32),
        str(source / "imagesTr" / "patient_single_mod.nii.gz"),
    )
    sitk.WriteImage(
        sitk.Image(10, 10, 10, sitk.sitkUInt8),
        str(source / "labelsTr" / "patient_single_mod.nii.gz"),
    )

    msd_json = {
        "training": [{
            "image": "imagesTr/patient_single_mod.nii.gz",
            "label": "labelsTr/patient_single_mod.nii.gz",
        }]
    }
    copy_msd_data(
        source, dest, msd_json, {0: "ct"}, "training", "Single modality test"
    )

    patient_dir = dest / "raw" / "train" / "patient_single_mod"
    assert patient_dir.exists()
    assert (patient_dir / "ct.nii.gz").exists()
    assert (patient_dir / "mask.nii.gz").exists()


def test_convert_msd_no_test_data_creates_correct_json(tmp_path):
    """Tests convert_msd when no test data is present (imagesTs missing)."""
    source = tmp_path / "msd_source_no_test"
    dest = tmp_path / "mist_dest_no_test"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir(parents=True)

    sitk.WriteImage(
        sitk.Image(10, 10, 10, sitk.sitkFloat32),
        str(source / "imagesTr" / "patient.nii.gz"),
    )
    sitk.WriteImage(
        sitk.Image(10, 10, 10, sitk.sitkUInt8),
        str(source / "labelsTr" / "patient.nii.gz"),
    )

    with open(source / "dataset.json", "w", encoding="utf-8") as f:
        json.dump({
            "name": "MSD Task No Test",
            "modality": {"0": "ct"},
            "labels": {"0": "background", "1": "tumor"},
            "training": [{
                "image": "imagesTr/patient.nii.gz",
                "label": "labelsTr/patient.nii.gz",
            }],
        }, f)

    convert_msd(source, dest)

    with open(dest / "dataset.json", encoding="utf-8") as f:
        config = json.load(f)

    assert "test-data" not in config
    assert config["task"] == "MSD Task No Test"
    assert config["modality"] == "ct"
    assert config["labels"] == [0, 1]
    assert config["train-data"] == "raw/train"
