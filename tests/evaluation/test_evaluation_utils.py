"""Tests for mist.evaluation.evaluation_utils."""
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import pytest

from mist.evaluation import evaluation_utils

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_eval_config(classes=None) -> dict:
    """Return a minimal valid evaluation_config in the new nested format."""
    if classes is None:
        classes = {"tumor": [1]}
    return {
        name: {"labels": labels, "metrics": {"dice": {}}}
        for name, labels in classes.items()
    }


class _FakeImage:
    """Minimal stand-in for an ANTsImage used in validate_mask tests."""

    def __init__(self, arr: np.ndarray):
        """Store the underlying numpy array."""
        self._arr = arr

    def numpy(self) -> np.ndarray:
        """Return the underlying array."""
        return self._arr


# ---------------------------------------------------------------------------
# validate_mask
# ---------------------------------------------------------------------------

class TestValidateMask:
    """Tests for evaluation_utils.validate_mask."""

    def _header(self, ndim=3):
        """Return a fake image header with *ndim* dimensions."""
        return {"dimensions": tuple(range(10, 10 + ndim))}

    def test_valid_3d_integer_mask_returns_none(self, monkeypatch, tmp_path):
        """A 3D mask with integer dtype and known labels returns None."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        arr = np.array([[[0, 1], [1, 0]]], dtype=np.int32)
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(3)
        )
        monkeypatch.setattr(
            ants, "image_read", lambda _: _FakeImage(arr)
        )
        config = _make_eval_config()
        assert evaluation_utils.validate_mask(path, config) is None

    def test_valid_3d_boolean_mask_returns_none(self, monkeypatch, tmp_path):
        """Boolean dtype is accepted as valid."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        arr = np.array([[[True, False]]], dtype=bool)
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(3)
        )
        monkeypatch.setattr(
            ants, "image_read", lambda _: _FakeImage(arr)
        )
        assert evaluation_utils.validate_mask(path, _make_eval_config()) is None

    def test_4d_mask_returns_error(self, monkeypatch, tmp_path):
        """A 4D mask header returns a 'not a 3D image' error string."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(4)
        )
        result = evaluation_utils.validate_mask(path, _make_eval_config())
        assert result is not None
        assert "not a 3D image" in result

    def test_float_dtype_with_fractional_values_returns_error(
        self, monkeypatch, tmp_path
    ):
        """Float dtype with non-integer values (probability map) returns error."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        arr = np.array([[[0.5, 1.0]]], dtype=np.float32)
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(3)
        )
        monkeypatch.setattr(
            ants, "image_read", lambda _: _FakeImage(arr)
        )
        result = evaluation_utils.validate_mask(path, _make_eval_config())
        assert result is not None
        assert "dtype" in result
        assert "non-integer" in result

    def test_float_dtype_with_integer_values_returns_none(
        self, monkeypatch, tmp_path
    ):
        """Float32 mask with only integer-valued labels (e.g. BraTS) is accepted."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        arr = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(3)
        )
        monkeypatch.setattr(
            ants, "image_read", lambda _: _FakeImage(arr)
        )
        assert evaluation_utils.validate_mask(path, _make_eval_config()) is None

    def test_unexpected_labels_returns_error(self, monkeypatch, tmp_path):
        """Labels not in the config produce an 'unexpected labels' error."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        arr = np.array([[[0, 99]]], dtype=np.int32)
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(3)
        )
        monkeypatch.setattr(
            ants, "image_read", lambda _: _FakeImage(arr)
        )
        result = evaluation_utils.validate_mask(path, _make_eval_config())
        assert result is not None
        assert "unexpected labels" in result

    def test_runtime_error_on_read_returns_error(self, monkeypatch, tmp_path):
        """RuntimeError during ants.image_read is caught and reported."""
        path = tmp_path / "mask.nii.gz"
        path.touch()
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(3)
        )
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _: (_ for _ in ()).throw(RuntimeError("corrupt")),
        )
        result = evaluation_utils.validate_mask(path, _make_eval_config())
        assert result is not None
        assert "Could not read" in result

    def test_custom_mask_type_appears_in_error(self, monkeypatch, tmp_path):
        """The mask_type argument is included in error messages."""
        path = tmp_path / "pred.nii.gz"
        path.touch()
        monkeypatch.setattr(
            ants, "image_header_info", lambda _: self._header(4)
        )
        result = evaluation_utils.validate_mask(
            path, _make_eval_config(), mask_type="prediction"
        )
        assert result is not None
        assert "prediction" in result


# ---------------------------------------------------------------------------
# build_evaluation_dataframe
# ---------------------------------------------------------------------------

class TestBuildEvaluationDataframe:
    """Tests for evaluation_utils.build_evaluation_dataframe."""

    @pytest.fixture
    def csv_and_preds(self, tmp_path):
        """Create a train_paths.csv and a matching prediction directory."""
        pred_dir = tmp_path / "preds"
        pred_dir.mkdir()

        (tmp_path / "mask_001.nii.gz").touch()
        (tmp_path / "mask_002.nii.gz").touch()
        (pred_dir / "patient_001.nii.gz").touch()
        (pred_dir / "patient_002.nii.gz").touch()

        csv_path = tmp_path / "train_paths.csv"
        pd.DataFrame({
            "id": ["patient_001", "patient_002"],
            "mask": [
                str(tmp_path / "mask_001.nii.gz"),
                str(tmp_path / "mask_002.nii.gz"),
            ],
        }).to_csv(csv_path, index=False)

        return csv_path, pred_dir

    def test_success_returns_correct_dataframe(self, csv_and_preds):
        """Valid inputs produce a two-row DataFrame with correct columns."""
        csv_path, pred_dir = csv_and_preds
        df, err = evaluation_utils.build_evaluation_dataframe(
            csv_path, pred_dir
        )
        assert err is None
        assert len(df) == 2
        assert list(df.columns) == ["id", "mask", "prediction"]
        assert df.iloc[0]["id"] == "patient_001"
        assert isinstance(df.iloc[0]["mask"], str)
        assert isinstance(df.iloc[0]["prediction"], str)

    def test_accepts_path_objects(self, csv_and_preds):
        """pathlib.Path inputs are accepted without error."""
        csv_path, pred_dir = csv_and_preds
        df, _ = evaluation_utils.build_evaluation_dataframe(
            Path(csv_path), Path(pred_dir)
        )
        assert len(df) == 2

    def test_missing_csv_returns_empty_df_and_error(self, tmp_path):
        """Missing CSV produces an empty DataFrame and an error message."""
        df, err = evaluation_utils.build_evaluation_dataframe(
            tmp_path / "none.csv", tmp_path
        )
        assert df.empty
        assert "No train_paths.csv" in err

    def test_missing_mask_is_skipped(self, csv_and_preds):
        """A patient whose mask is absent is skipped with an error message."""
        csv_path, pred_dir = csv_and_preds
        (csv_path.parent / "mask_001.nii.gz").unlink()
        df, err = evaluation_utils.build_evaluation_dataframe(
            csv_path, pred_dir
        )
        assert len(df) == 1
        assert "patient_001" in err
        assert "mask" in err

    def test_missing_prediction_is_skipped(self, csv_and_preds):
        """A patient whose prediction is absent is skipped."""
        csv_path, pred_dir = csv_and_preds
        (pred_dir / "patient_002.nii.gz").unlink()
        df, err = evaluation_utils.build_evaluation_dataframe(
            csv_path, pred_dir
        )
        assert len(df) == 1
        assert "patient_002" in err
        assert "prediction" in err

    def test_all_missing_returns_empty_df_with_columns(self, csv_and_preds):
        """When all patients are skipped the DataFrame has the right columns."""
        csv_path, pred_dir = csv_and_preds
        (csv_path.parent / "mask_001.nii.gz").unlink()
        (csv_path.parent / "mask_002.nii.gz").unlink()
        df, err = evaluation_utils.build_evaluation_dataframe(
            csv_path, pred_dir
        )
        assert df.empty
        assert list(df.columns) == ["id", "mask", "prediction"]
        assert err is not None

    def test_validate_true_without_config_raises(self, csv_and_preds):
        """validate=True without evaluation_config raises ValueError."""
        csv_path, pred_dir = csv_and_preds
        with pytest.raises(ValueError, match="evaluation_config must be"):
            evaluation_utils.build_evaluation_dataframe(
                csv_path, pred_dir, validate=True
            )

    def test_validate_true_valid_masks_included(
        self, monkeypatch, csv_and_preds
    ):
        """validate=True with passing masks keeps both patients."""
        csv_path, pred_dir = csv_and_preds
        monkeypatch.setattr(
            evaluation_utils, "validate_mask", lambda *_a, **_k: None
        )
        df, err = evaluation_utils.build_evaluation_dataframe(
            csv_path, pred_dir,
            evaluation_config=_make_eval_config(),
            validate=True,
        )
        assert err is None
        assert len(df) == 2

    def test_validate_true_invalid_mask_skipped(
        self, monkeypatch, csv_and_preds
    ):
        """validate=True with a failing mask skips that patient."""
        csv_path, pred_dir = csv_and_preds

        def _failing_validate(path, config, mask_type="mask"):
            """Fail only for the first patient's ground truth mask."""
            if "mask_001" in str(path):
                return "bad dtype"
            return None

        monkeypatch.setattr(
            evaluation_utils, "validate_mask", _failing_validate
        )
        df, err = evaluation_utils.build_evaluation_dataframe(
            csv_path, pred_dir,
            evaluation_config=_make_eval_config(),
            validate=True,
        )
        assert len(df) == 1
        assert "patient_001" in err


# ---------------------------------------------------------------------------
# initialize_results_dataframe
# ---------------------------------------------------------------------------

class TestInitializeResultsDataframe:
    """Tests for evaluation_utils.initialize_results_dataframe."""

    def test_single_class_single_metric(self):
        """Config with one class and one metric produces correct columns."""
        config = {"tumor": {"labels": [1], "metrics": {"dice": {}}}}
        df = evaluation_utils.initialize_results_dataframe(config)
        assert list(df.columns) == ["id", "tumor_dice"]
        assert df.empty

    def test_multiple_classes_multiple_metrics_column_order(self):
        """Columns are ordered class-by-class, metric-by-metric."""
        config = {
            "tumor": {
                "labels": [1],
                "metrics": {"dice": {}, "haus95": {}},
            },
            "edema": {
                "labels": [2],
                "metrics": {"dice": {}, "haus95": {}},
            },
        }
        df = evaluation_utils.initialize_results_dataframe(config)
        assert list(df.columns) == [
            "id",
            "tumor_dice",
            "tumor_haus95",
            "edema_dice",
            "edema_haus95",
        ]

    def test_empty_config_returns_only_id_column(self):
        """An empty config produces a DataFrame with only the 'id' column."""
        df = evaluation_utils.initialize_results_dataframe({})
        assert list(df.columns) == ["id"]
        assert df.empty


# ---------------------------------------------------------------------------
# compute_results_stats
# ---------------------------------------------------------------------------

class TestComputeResultsStats:
    """Tests for evaluation_utils.compute_results_stats."""

    def test_appends_five_stat_rows(self):
        """Five summary rows are appended to the existing data rows."""
        df = pd.DataFrame({
            "id": ["p1", "p2", "p3"],
            "tumor_dice": [0.8, 0.9, 1.0],
        })
        result = evaluation_utils.compute_results_stats(df)
        assert len(result) == 8  # 3 data + 5 stats

    def test_stat_row_labels_are_correct(self):
        """The five appended rows have the expected label strings."""
        df = pd.DataFrame({
            "id": ["p1"],
            "tumor_dice": [0.9],
        })
        result = evaluation_utils.compute_results_stats(df)
        assert result["id"].tail(5).tolist() == [
            "Mean", "Std",
            "25th Percentile", "Median", "75th Percentile",
        ]

    def test_mean_computed_correctly_ignoring_nan(self):
        """nanmean is used, so NaN values are excluded from the mean."""
        df = pd.DataFrame({
            "id": ["p1", "p2", "p3", "p4"],
            "tumor_dice": [0.8, 0.9, 1.0, np.nan],
        })
        result = evaluation_utils.compute_results_stats(df)
        mean_val = result.loc[result["id"] == "Mean", "tumor_dice"].values[0]
        assert mean_val == pytest.approx(0.9)

    def test_all_nan_column_returns_nan_without_error(self):
        """An entirely-NaN column produces NaN stats without crashing."""
        df = pd.DataFrame({
            "id": ["p1", "p2"],
            "tumor_dice": [np.nan, np.nan],
        })
        result = evaluation_utils.compute_results_stats(df)
        mean_val = result.loc[result["id"] == "Mean", "tumor_dice"].values[0]
        assert np.isnan(mean_val)


# ---------------------------------------------------------------------------
# crop_to_union
# ---------------------------------------------------------------------------

class TestCropToUnion:
    """Tests for evaluation_utils.crop_to_union."""

    def test_both_empty_returns_originals_unchanged(self):
        """When both arrays are all-zero the originals are returned."""
        mask = np.zeros((10, 10, 10))
        pred = np.zeros((10, 10, 10))
        out_m, out_p = evaluation_utils.crop_to_union(mask, pred)
        assert out_m is mask
        assert out_p is pred

    def test_crops_to_bounding_box_of_union(self):
        """Non-zero voxels from both arrays define the crop region."""
        mask = np.zeros((20, 20, 20))
        pred = np.zeros((20, 20, 20))
        mask[5:8, 5:8, 5:8] = 1
        pred[10:13, 10:13, 10:13] = 1
        out_m, out_p = evaluation_utils.crop_to_union(mask, pred)
        # Cropped shape must be smaller than original.
        assert out_m.shape[0] < 20
        assert out_p.shape == out_m.shape

    def test_only_mask_nonzero_crops_to_mask_region(self):
        """With only the mask populated the crop covers the mask region."""
        mask = np.zeros((20, 20, 20))
        pred = np.zeros((20, 20, 20))
        mask[3:6, 3:6, 3:6] = 1
        out_m, out_p = evaluation_utils.crop_to_union(mask, pred)
        assert out_m.shape == (3, 3, 3)
        assert out_p.shape == (3, 3, 3)

    def test_only_prediction_nonzero_crops_to_pred_region(self):
        """With only the prediction populated the crop covers that region."""
        mask = np.zeros((20, 20, 20))
        pred = np.zeros((20, 20, 20))
        pred[7:10, 7:10, 7:10] = 1
        out_m, out_p = evaluation_utils.crop_to_union(mask, pred)
        assert out_m.shape == (3, 3, 3)
        assert out_p.shape == (3, 3, 3)
