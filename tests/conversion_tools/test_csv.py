"""Tests for converting CSV files to MIST format."""
import json
import shutil
import pandas as pd
import pytest

# MIST imports.
from mist.utils import io, progress_bar
from mist.conversion_tools import conversion_utils
from mist.conversion_tools.csv import (
    _validate_csv_columns,
    convert_csv,
    copy_csv_data,
)
from tests.conversion_tools.helpers import DummyProgressBar


@pytest.fixture
def temp_csv_data(tmp_path):
    """Fixture to create temporary CSV files and directories for testing."""
    img_path = tmp_path / "img.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    img_path.write_text("dummy image")
    mask_path.write_text("dummy mask")

    # Training CSV format: id, mask, image.
    train_csv = tmp_path / "train.csv"
    pd.DataFrame({
        "id": [0],
        "mask": [str(mask_path)],
        "ct": [str(img_path)],
    }).to_csv(train_csv, index=False)

    # Testing CSV format: id, image.
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({
        "id": [1],
        "ct": [str(img_path)],
    }).to_csv(test_csv, index=False)

    return train_csv, test_csv, tmp_path


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


def test_convert_csv_creates_correct_structure(temp_csv_data):
    """Tests convert_csv function creates the correct directory structure."""
    train_csv, test_csv, tmp_path = temp_csv_data
    output_dir = tmp_path / "output"

    convert_csv(train_csv, output_dir, test_csv)

    # Check directory structure.
    assert (output_dir / "raw" / "train" / "0").exists()
    assert (output_dir / "raw" / "test" / "1").exists()

    # Check copied files.
    assert (output_dir / "raw" / "train" / "0" / "mask.nii.gz").exists()
    assert (output_dir / "raw" / "train" / "0" / "ct.nii.gz").exists()
    assert (output_dir / "raw" / "test" / "1" / "ct.nii.gz").exists()

    # Check dataset.json structure and relative paths.
    dataset_json_path = output_dir / "dataset.json"
    assert dataset_json_path.exists()
    with open(dataset_json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["mask"] == ["mask.nii.gz"]
    assert data["train-data"] == "raw/train"
    assert data["test-data"] == "raw/test"


def test_convert_csv_without_test_csv(temp_csv_data):
    """Tests convert_csv function without a test CSV file."""
    train_csv, _, tmp_path = temp_csv_data
    output_dir = tmp_path / "output_no_test"
    convert_csv(train_csv, output_dir)

    with open(output_dir / "dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    assert "test-data" not in data
    assert data["train-data"] == "raw/train"


def test_convert_csv_accepts_path_objects(temp_csv_data):
    """convert_csv accepts pathlib.Path inputs without error."""
    train_csv, test_csv, tmp_path = temp_csv_data
    output_dir = tmp_path / "output_path_obj"
    convert_csv(train_csv, output_dir, test_csv)
    assert (output_dir / "dataset.json").exists()


def test_convert_csv_raises_if_train_missing(tmp_path):
    """Tests convert_csv raises FileNotFoundError if train CSV is missing."""
    with pytest.raises(FileNotFoundError):
        convert_csv(tmp_path / "nonexistent.csv", tmp_path / "out")


def test_convert_csv_raises_if_test_missing(tmp_path, temp_csv_data):
    """Tests convert_csv raises FileNotFoundError if test CSV is missing."""
    train_csv, _, _ = temp_csv_data
    with pytest.raises(FileNotFoundError):
        convert_csv(train_csv, tmp_path, tmp_path / "no_test.csv")


# ---------------------------------------------------------------------------
# _validate_csv_columns
# ---------------------------------------------------------------------------

class TestValidateCsvColumns:
    """Tests for csv._validate_csv_columns."""

    def test_valid_training_csv_passes(self):
        """A well-formed training CSV does not raise."""
        df = pd.DataFrame({"id": [], "mask": [], "ct": []})
        _validate_csv_columns(df, "training")  # no exception

    def test_valid_test_csv_passes(self):
        """A well-formed test CSV does not raise."""
        df = pd.DataFrame({"id": [], "ct": []})
        _validate_csv_columns(df, "test")  # no exception

    @pytest.mark.parametrize(
        "columns, match",
        [
            pytest.param(
                ["id", "mask"],
                "at least 3 columns",
                id="training_too_few_columns",
            ),
            pytest.param(
                ["patient", "mask", "ct"],
                "first column must be 'id'",
                id="training_wrong_first_column",
            ),
            pytest.param(
                ["id", "ct", "mask"],
                "second column must be 'mask'",
                id="training_wrong_second_column",
            ),
        ],
    )
    def test_invalid_training_csv_raises(self, columns, match):
        """Invalid training CSV structures raise ValueError."""
        df = pd.DataFrame(columns=columns)
        with pytest.raises(ValueError, match=match):
            _validate_csv_columns(df, "training")

    @pytest.mark.parametrize(
        "columns, match",
        [
            pytest.param(
                ["id"],
                "at least 2 columns",
                id="test_too_few_columns",
            ),
            pytest.param(
                ["patient", "ct"],
                "first column must be 'id'",
                id="test_wrong_first_column",
            ),
        ],
    )
    def test_invalid_test_csv_raises(self, columns, match):
        """Invalid test CSV structures raise ValueError."""
        df = pd.DataFrame(columns=columns)
        with pytest.raises(ValueError, match=match):
            _validate_csv_columns(df, "test")

    def test_convert_csv_raises_on_bad_column_order(self, tmp_path):
        """convert_csv raises ValueError if training CSV columns are wrong."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({
            "id": [0], "ct": ["/img.nii.gz"], "mask": ["/mask.nii.gz"]
        }).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="second column must be 'mask'"):
            convert_csv(bad_csv, tmp_path / "out")


def test_copy_csv_data_skips_missing_files(tmp_path):
    """Tests copy_csv_data skips patients where files are missing."""
    df = pd.DataFrame({
        "id": [0],
        "mask": [str(tmp_path / "missing_mask.nii.gz")],
        "ct": [str(tmp_path / "missing_img.nii.gz")],
    })
    out_dir = tmp_path / "mist"
    copy_csv_data(df, out_dir, "training", "Testing copy logic")

    patient_dir = out_dir / "0"
    assert not (patient_dir / "ct.nii.gz").exists()
    assert not (patient_dir / "mask.nii.gz").exists()


def test_copy_csv_data_prints_error_summary_on_failures(tmp_path, monkeypatch):
    """copy_csv_data prints a 'N of M patients had errors' summary."""
    df = pd.DataFrame({
        "id": [0, 1],
        "mask": [
            str(tmp_path / "missing_mask_0.nii.gz"),
            str(tmp_path / "missing_mask_1.nii.gz"),
        ],
        "ct": [
            str(tmp_path / "missing_img_0.nii.gz"),
            str(tmp_path / "missing_img_1.nii.gz"),
        ],
    })
    printed = []
    monkeypatch.setattr(
        "mist.conversion_tools.csv.console.print",
        lambda *a, **k: printed.append(str(a[0])),
    )
    copy_csv_data(df, tmp_path / "out", "training", "Test")
    assert any("2 of 2" in msg for msg in printed)


def test_copy_csv_data_test_mode_skips_mask(tmp_path):
    """Test that test mode does not attempt to copy the mask."""
    img_path = tmp_path / "img.nii.gz"
    img_path.write_text("dummy image")

    df = pd.DataFrame({"id": [1], "ct": [str(img_path)]})
    out_dir = tmp_path / "mist_test"
    copy_csv_data(df, out_dir, "test", "Testing test mode")

    patient_dir = out_dir / "1"
    assert (patient_dir / "ct.nii.gz").exists()
    assert not (patient_dir / "mask.nii.gz").exists()


def test_copy_csv_data_skips_missing_images(tmp_path):
    """Test that copy_csv_data skips missing image files but copies the mask."""
    df = pd.DataFrame({
        "id": [0],
        "mask": [str(tmp_path / "existing_mask.nii.gz")],
        "ct": [str(tmp_path / "missing_image.nii.gz")],
    })
    (tmp_path / "existing_mask.nii.gz").write_text("dummy mask")

    out_dir = tmp_path / "mist_output"
    out_dir.mkdir()

    copy_csv_data(df, out_dir, "training", "Testing missing images")

    patient_dir = out_dir / "0"
    assert patient_dir.exists()
    assert (patient_dir / "mask.nii.gz").exists()
    assert not (patient_dir / "ct.nii.gz").exists()
