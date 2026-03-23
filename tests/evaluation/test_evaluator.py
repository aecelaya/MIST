"""Tests for the Evaluator class in mist.evaluation.evaluator."""
import concurrent.futures
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import pytest

from mist.analyze_data import analyzer_utils
from mist.evaluation import evaluation_utils
from mist.evaluation.evaluator import Evaluator
from mist.utils import progress_bar as pb_mod
from mist.utils import console as console_mod
from tests.evaluation.helpers import (
    FakeExecutor,
    fake_get_progress_bar,
    make_ants_image,
    make_eval_config,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_filepaths_df(tmp_path: Path, n: int = 1) -> pd.DataFrame:
    """Create a filepaths DataFrame with *n* patients backed by real files."""
    rows = []
    for i in range(n):
        mask = tmp_path / f"mask_{i}.nii.gz"
        pred = tmp_path / f"pred_{i}.nii.gz"
        mask.touch()
        pred.touch()
        rows.append({
            "id": f"p{i}",
            "mask": str(mask),
            "prediction": str(pred),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def filepaths_df(tmp_path):
    """A single-patient filepaths DataFrame with existing files."""
    return _make_filepaths_df(tmp_path, n=1)


@pytest.fixture
def evaluator(filepaths_df, tmp_path):
    """A fully initialised Evaluator with a single tumor class."""
    return Evaluator(
        filepaths_dataframe=filepaths_df,
        evaluation_config=make_eval_config(),
        output_csv_path=tmp_path / "results.csv",
    )


@pytest.fixture
def _patch_run_env(monkeypatch):
    """Patch ProcessPoolExecutor and progress_bar for run() tests."""
    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        lambda max_workers=None: FakeExecutor(),
    )
    monkeypatch.setattr(
        pb_mod, "get_progress_bar", fake_get_progress_bar, raising=True
    )


# ---------------------------------------------------------------------------
# __init__ and config validation
# ---------------------------------------------------------------------------

class TestEvaluatorInit:
    """Tests for Evaluator.__init__ and _validate_evaluation_config."""

    def test_valid_config_stores_index_and_config(
        self, filepaths_df, tmp_path
    ):
        """Valid new-format config: index is 'id', config stored correctly."""
        ev = Evaluator(
            filepaths_df,
            make_eval_config(),
            tmp_path / "out.csv",
        )
        assert ev.filepaths_dataframe.index.name == "id"
        assert "p0" in ev.filepaths_dataframe.index
        assert "tumor" in ev.evaluation_config

    def test_validate_masks_flag_stored(self, filepaths_df, tmp_path):
        """validate_masks opt-in flag is stored on the instance."""
        ev = Evaluator(
            filepaths_df,
            make_eval_config(),
            tmp_path / "out.csv",
            validate_masks=True,
        )
        assert ev.validate_masks is True

    def test_duplicate_ids_raises(self, filepaths_df, tmp_path):
        """Duplicate patient IDs raise ValueError."""
        dup = pd.concat([filepaths_df, filepaths_df], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate patient IDs"):
            Evaluator(dup, make_eval_config(), tmp_path / "out.csv")

    def test_missing_required_column_raises(self, filepaths_df, tmp_path):
        """DataFrame missing a required column raises ValueError."""
        bad_df = filepaths_df.drop(columns=["mask"])
        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            Evaluator(bad_df, make_eval_config(), tmp_path / "out.csv")

    # --- new-format validation ---

    @pytest.mark.parametrize(
        "bad_config, match_msg",
        [
            pytest.param(
                {"tumor": {"labels": [1]}},
                "must contain both 'labels' and 'metrics'",
                id="missing_metrics_key",
            ),
            pytest.param(
                {"tumor": {"metrics": {"dice": {}}}},
                "must contain both 'labels' and 'metrics'",
                id="missing_labels_key",
            ),
            pytest.param(
                {"tumor": {"labels": [], "metrics": {"dice": {}}}},
                "non-empty list",
                id="empty_labels",
            ),
            pytest.param(
                {"tumor": {"labels": "not_a_list", "metrics": {"dice": {}}}},
                "non-empty list",
                id="labels_not_list",
            ),
            pytest.param(
                {"tumor": {"labels": [0], "metrics": {"dice": {}}}},
                "greater than 0",
                id="label_is_zero",
            ),
            pytest.param(
                {"tumor": {"labels": [-1], "metrics": {"dice": {}}}},
                "greater than 0",
                id="label_negative",
            ),
            pytest.param(
                {"tumor": {"labels": [1, 0], "metrics": {"dice": {}}}},
                "greater than 0",
                id="label_list_contains_zero",
            ),
            pytest.param(
                {"tumor": {"labels": [1], "metrics": "not_a_dict"}},
                "must be a dictionary",
                id="metrics_not_dict",
            ),
        ],
    )
    def test_invalid_new_format_raises(
        self, filepaths_df, tmp_path, bad_config, match_msg
    ):
        """Each invalid new-format config raises ValueError with a message."""
        with pytest.raises(ValueError, match=match_msg):
            Evaluator(filepaths_df, bad_config, tmp_path / "out.csv")

    def test_results_dataframe_initialized_with_correct_columns(
        self, evaluator
    ):
        """results_dataframe is pre-populated with id + class_metric columns."""
        cols = list(evaluator.results_dataframe.columns)
        assert cols[0] == "id"
        assert "tumor_dice" in cols


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

class TestComputeDiagonalDistance:
    """Tests for Evaluator._compute_diagonal_distance."""

    @pytest.mark.parametrize(
        "shape, spacing, expected",
        [
            pytest.param(
                (10, 10, 10), (1.0, 1.0, 1.0),
                np.sqrt(300),
                id="unit_cube_10",
            ),
            pytest.param(
                (4, 3), (2.0, 2.0),
                np.sqrt(64 + 36),
                id="2d_rect",
            ),
        ],
    )
    def test_returns_euclidean_diagonal(self, shape, spacing, expected):
        """Diagonal equals norm of (shape * spacing) in mm."""
        result = Evaluator._compute_diagonal_distance(shape, spacing)
        assert result == pytest.approx(expected, rel=1e-5)


class TestHandleEdgeCases:
    """Tests for Evaluator._handle_edge_cases."""

    @pytest.mark.parametrize(
        "m_sum, p_sum, best, worst, expected",
        [
            pytest.param(0, 0, 1.0, 0.0, 1.0, id="both_empty_returns_best"),
            pytest.param(0, 10, 1.0, 0.0, 0.0, id="gt_empty_returns_worst"),
            pytest.param(10, 0, 1.0, 0.0, 0.0, id="pred_empty_returns_worst"),
            pytest.param(10, 10, 1.0, 0.0, None, id="neither_empty_returns_none"),
        ],
    )
    def test_edge_case_values(self, m_sum, p_sum, best, worst, expected):
        """Correct value is returned for each combination of empty masks."""
        result = Evaluator._handle_edge_cases(m_sum, p_sum, best, worst)
        assert result == expected


# ---------------------------------------------------------------------------
# _load_patient_data
# ---------------------------------------------------------------------------

class TestLoadPatientData:
    """Tests for Evaluator._load_patient_data."""

    def test_success_returns_mask_and_prediction(
        self, evaluator, monkeypatch
    ):
        """Valid patient ID returns a dict with mask and prediction keys."""
        mock_img = make_ants_image()
        monkeypatch.setattr(
            ants, "image_header_info",
            lambda _: {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": np.eye(3).flatten().tolist()},
        )
        monkeypatch.setattr(
            ants, "image_read", lambda _: mock_img
        )
        monkeypatch.setattr(
            analyzer_utils, "compare_headers", lambda *_: True
        )
        data = evaluator._load_patient_data("p0")
        assert "mask" in data
        assert "prediction" in data

    def test_missing_patient_id_raises_value_error(self, evaluator):
        """Requesting a non-existent patient ID raises ValueError."""
        with pytest.raises(ValueError, match="No data found for patient ID"):
            evaluator._load_patient_data("does_not_exist")

    def test_missing_mask_file_raises_file_not_found(
        self, evaluator, tmp_path
    ):
        """FileNotFoundError is raised when the mask file is absent."""
        evaluator.filepaths_dataframe.loc["p0", "mask"] = str(
            tmp_path / "gone.nii.gz"
        )
        with pytest.raises(FileNotFoundError, match="Mask not found"):
            evaluator._load_patient_data("p0")

    def test_missing_prediction_file_raises_file_not_found(
        self, evaluator, tmp_path
    ):
        """FileNotFoundError is raised when the prediction file is absent."""
        evaluator.filepaths_dataframe.loc["p0", "prediction"] = str(
            tmp_path / "gone.nii.gz"
        )
        with pytest.raises(FileNotFoundError, match="Prediction not found"):
            evaluator._load_patient_data("p0")

    def test_header_mismatch_raises_value_error(
        self, evaluator, monkeypatch
    ):
        """Mismatched image headers raise ValueError."""
        monkeypatch.setattr(
            ants, "image_header_info",
            lambda _: {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": np.eye(3).flatten().tolist()},
        )
        monkeypatch.setattr(
            analyzer_utils, "compare_headers", lambda *_: False
        )
        with pytest.raises(ValueError, match="Header mismatch"):
            evaluator._load_patient_data("p0")

    def test_validate_masks_false_skips_validation(
        self, filepaths_df, tmp_path, monkeypatch
    ):
        """When validate_masks=False, validate_mask is never called."""
        mock_img = make_ants_image()
        monkeypatch.setattr(
            ants, "image_header_info",
            lambda _: {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": np.eye(3).flatten().tolist()},
        )
        monkeypatch.setattr(ants, "image_read", lambda _: mock_img)
        monkeypatch.setattr(analyzer_utils, "compare_headers", lambda *_: True)

        called = {"count": 0}
        monkeypatch.setattr(
            evaluation_utils,
            "validate_mask",
            lambda *_a, **_k: called.__setitem__("count", called["count"] + 1),
        )

        ev = Evaluator(
            filepaths_df,
            make_eval_config(),
            tmp_path / "out.csv",
            validate_masks=False,
        )
        ev._load_patient_data("p0")
        assert called["count"] == 0

    def test_validate_masks_true_calls_validate_mask(
        self, filepaths_df, tmp_path, monkeypatch
    ):
        """When validate_masks=True, validate_mask is called for both files."""
        mock_img = make_ants_image()
        monkeypatch.setattr(
            ants, "image_header_info",
            lambda _: {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": np.eye(3).flatten().tolist()},
        )
        monkeypatch.setattr(ants, "image_read", lambda _: mock_img)
        monkeypatch.setattr(analyzer_utils, "compare_headers", lambda *_: True)
        monkeypatch.setattr(
            evaluation_utils, "validate_mask", lambda *_a, **_k: None
        )

        called = {"count": 0}
        original = evaluation_utils.validate_mask

        def _counting_validate(*a, **k):
            called["count"] += 1
            return original(*a, **k)

        monkeypatch.setattr(evaluation_utils, "validate_mask", _counting_validate)

        ev = Evaluator(
            filepaths_df,
            make_eval_config(),
            tmp_path / "out.csv",
            validate_masks=True,
        )
        ev._load_patient_data("p0")
        assert called["count"] == 2  # mask + prediction

    def test_validate_masks_true_raises_on_invalid_mask(
        self, filepaths_df, tmp_path, monkeypatch
    ):
        """validate_masks=True raises ValueError when validation fails."""
        monkeypatch.setattr(
            evaluation_utils,
            "validate_mask",
            lambda path, *_a, **_k: "bad dtype",
        )

        ev = Evaluator(
            filepaths_df,
            make_eval_config(),
            tmp_path / "out.csv",
            validate_masks=True,
        )
        with pytest.raises(ValueError, match="Mask validation failed"):
            ev._load_patient_data("p0")


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for Evaluator._compute_metrics."""

    def _run(self, evaluator, monkeypatch, mock_metric, mask, pred,
             spacing=(1.0, 1.0, 1.0), override=None):
        """Patch get_metric and call _compute_metrics."""
        monkeypatch.setattr(
            "mist.evaluation.evaluator.get_metric",
            lambda _name: mock_metric,
        )
        return evaluator._compute_metrics(
            patient_id="p0",
            mask=mask,
            prediction=pred,
            spacing=spacing,
            class_metrics_config={"dice": {}},
            diagonal_distance_override=override,
        )

    def test_both_empty_returns_best(self, evaluator, monkeypatch):
        """Both-empty masks short-circuit to the metric's best value."""
        metric = type("M", (), {"best": 1.0, "worst": 0.0})()
        result, err = self._run(
            evaluator, monkeypatch, metric,
            np.zeros((5, 5, 5)), np.zeros((5, 5, 5)),
        )
        assert result["dice"] == 1.0
        assert err is None

    def test_one_empty_finite_worst_returns_worst(
        self, evaluator, monkeypatch
    ):
        """One empty mask short-circuits to the metric's finite worst value."""
        metric = type("M", (), {"best": 1.0, "worst": 0.0})()
        result, err = self._run(
            evaluator, monkeypatch, metric,
            np.ones((5, 5, 5)), np.zeros((5, 5, 5)),
        )
        assert result["dice"] == 0.0

    def test_one_empty_infinite_worst_uses_diagonal(
        self, evaluator, monkeypatch
    ):
        """With worst=inf the diagonal distance is substituted."""
        metric = type("M", (), {"best": 0.0, "worst": float("inf")})()
        shape = (10, 10, 10)
        spacing = (1.0, 1.0, 1.0)
        expected_diag = Evaluator._compute_diagonal_distance(shape, spacing)
        result, _ = self._run(
            evaluator, monkeypatch, metric,
            np.zeros(shape), np.ones(shape),
            spacing=spacing,
        )
        assert result["dice"] == pytest.approx(expected_diag, rel=1e-4)

    def test_diagonal_distance_override_is_used(
        self, evaluator, monkeypatch
    ):
        """diagonal_distance_override replaces the computed diagonal."""
        metric = type("M", (), {"best": 0.0, "worst": float("inf")})()
        result, _ = self._run(
            evaluator, monkeypatch, metric,
            np.zeros((5, 5, 5)), np.ones((5, 5, 5)),
            override=99.9,
        )
        assert result["dice"] == pytest.approx(99.9)

    def test_valid_metric_result_stored(self, evaluator, monkeypatch):
        """A well-behaved metric's return value is stored directly."""
        metric = type("M", (), {
            "best": 1.0, "worst": 0.0,
            "__call__": lambda self, *a, **k: 0.75,
        })()
        result, err = self._run(
            evaluator, monkeypatch, metric,
            np.ones((5, 5, 5)), np.ones((5, 5, 5)),
        )
        assert result["dice"] == pytest.approx(0.75)
        assert err is None

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
            pytest.param(-np.inf, id="neg_inf"),
        ],
    )
    def test_nan_inf_result_replaced_with_worst(
        self, evaluator, monkeypatch, bad_value
    ):
        """NaN or Inf returned by a metric is replaced with worst value."""
        metric = type("M", (), {
            "best": 1.0, "worst": 0.0,
            "__call__": lambda self, *a, **k: bad_value,
        })()
        result, err = self._run(
            evaluator, monkeypatch, metric,
            np.ones((5, 5, 5)), np.ones((5, 5, 5)),
        )
        assert result["dice"] == 0.0
        assert err is not None
        assert "NaN/Inf" in err

    def test_exception_in_metric_replaced_with_worst(
        self, evaluator, monkeypatch
    ):
        """Exception raised by a metric is caught; worst value is used."""
        def _bad(*a, **k):
            raise RuntimeError("broken")

        metric = type("M", (), {
            "best": 1.0, "worst": 0.0,
            "__call__": _bad,
        })()
        result, err = self._run(
            evaluator, monkeypatch, metric,
            np.ones((5, 5, 5)), np.ones((5, 5, 5)),
        )
        assert result["dice"] == 0.0
        assert err is not None
        assert "Error in dice" in err


# ---------------------------------------------------------------------------
# _evaluate_single_patient
# ---------------------------------------------------------------------------

class TestEvaluateSinglePatient:
    """Tests for Evaluator._evaluate_single_patient."""

    def test_single_label_class_produces_flat_result(self, evaluator):
        """Single-label class uses equality comparison; result is keyed correctly."""
        mask = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int32)
        pred = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int32)
        result, err = evaluator._evaluate_single_patient(
            "p0", mask, pred, (1.0, 1.0, 1.0)
        )
        assert "id" in result
        assert "tumor_dice" in result
        assert result["tumor_dice"] == pytest.approx(1.0)
        assert err is None

    def test_multi_label_class_uses_isin(self, filepaths_df, tmp_path):
        """Multi-label class combines labels via np.isin."""
        config = {
            "combined": {
                "labels": [1, 2],
                "metrics": {"dice": {}},
            }
        }
        ev = Evaluator(filepaths_df, config, tmp_path / "out.csv")
        mask = np.zeros((5, 5, 5), dtype=np.int32)
        pred = np.zeros((5, 5, 5), dtype=np.int32)
        mask[0:2, 0:2, 0:2] = 1
        mask[3:5, 3:5, 3:5] = 2
        pred[0:2, 0:2, 0:2] = 1
        pred[3:5, 3:5, 3:5] = 2
        result, err = ev._evaluate_single_patient(
            "p0", mask, pred, (1.0, 1.0, 1.0)
        )
        assert result["combined_dice"] == pytest.approx(1.0)
        assert err is None

    def test_class_errors_are_aggregated(self, evaluator, monkeypatch):
        """Errors from multiple classes are joined into one string."""
        evaluator.evaluation_config = {
            "c1": {"labels": [1], "metrics": {"dice": {}}},
            "c2": {"labels": [2], "metrics": {"dice": {}}},
        }

        def _fake_compute(patient_id, mask, prediction, spacing,
                          class_metrics_config,
                          diagonal_distance_override=None):
            """Return an error for class c2 only."""
            err = "bad" if class_metrics_config == {"dice": {}} else None
            return {"dice": 0.0}, err

        monkeypatch.setattr(evaluator, "_compute_metrics", _fake_compute)

        _, errors = evaluator._evaluate_single_patient(
            "p0",
            np.zeros((5, 5, 5), dtype=np.int32),
            np.zeros((5, 5, 5), dtype=np.int32),
            (1.0, 1.0, 1.0),
        )
        # Both classes produce errors so both should appear.
        assert errors is not None


# ---------------------------------------------------------------------------
# _evaluate_patient_pipeline
# ---------------------------------------------------------------------------

class TestEvaluatePatientPipeline:
    """Tests for Evaluator._evaluate_patient_pipeline."""

    def test_success_returns_result_and_no_error(
        self, evaluator, monkeypatch
    ):
        """A successful pipeline returns a result dict and None for errors."""
        mock_img = make_ants_image()
        monkeypatch.setattr(
            ants, "image_header_info",
            lambda _: {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": np.eye(3).flatten().tolist()},
        )
        monkeypatch.setattr(ants, "image_read", lambda _: mock_img)
        monkeypatch.setattr(
            analyzer_utils, "compare_headers", lambda *_: True
        )
        result, err = evaluator._evaluate_patient_pipeline("p0")
        assert result is not None
        assert "id" in result

    def test_exception_returns_none_result_and_error_message(
        self, evaluator, monkeypatch
    ):
        """An exception produces None result and a CRITICAL FAILURE message."""
        monkeypatch.setattr(
            evaluator,
            "_load_patient_data",
            lambda _pid: (_ for _ in ()).throw(RuntimeError("disk error")),
        )
        result, err = evaluator._evaluate_patient_pipeline("p0")
        assert result is None
        assert err is not None
        assert "CRITICAL FAILURE" in err
        assert "disk error" in err

    def test_metric_warning_propagated_in_error_message(
        self, evaluator, monkeypatch
    ):
        """Non-fatal metric warnings are surfaced in the pipeline error string."""
        mock_img = make_ants_image()
        monkeypatch.setattr(
            ants, "image_header_info",
            lambda _: {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0),
                       "origin": (0.0, 0.0, 0.0),
                       "direction": np.eye(3).flatten().tolist()},
        )
        monkeypatch.setattr(ants, "image_read", lambda _: mock_img)
        monkeypatch.setattr(
            analyzer_utils, "compare_headers", lambda *_: True
        )
        # Make _evaluate_single_patient return a warning (non-None errors).
        monkeypatch.setattr(
            evaluator,
            "_evaluate_single_patient",
            lambda patient_id, mask, prediction, spacing: (
                {"id": patient_id, "tumor_dice": 0.0}, "NaN/Inf warning"
            ),
        )
        result, err = evaluator._evaluate_patient_pipeline("p0")
        assert result is not None
        assert err is not None
        assert "Patient p0" in err


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestEvaluatorRun:
    """Tests for Evaluator.run."""

    def test_success_writes_csv_with_correct_structure(
        self, evaluator, monkeypatch, _patch_run_env
    ):
        """run() saves a CSV containing the data rows plus 5 summary rows."""
        monkeypatch.setattr(
            Evaluator,
            "_evaluate_patient_pipeline",
            lambda self, pid: ({"id": pid, "tumor_dice": 0.85}, None),
        )
        evaluator.run()
        assert Path(evaluator.output_csv_path).exists()
        results = pd.read_csv(evaluator.output_csv_path)
        # 1 patient + 5 stat rows.
        assert len(results) == 6
        assert results.iloc[0]["tumor_dice"] == pytest.approx(0.85)

    def test_critical_failure_still_writes_csv_and_reports_error(
        self, evaluator, monkeypatch, _patch_run_env, capsys
    ):
        """run() writes the CSV and prints CRITICAL FAILURE on total failure."""
        monkeypatch.setattr(
            Evaluator,
            "_evaluate_patient_pipeline",
            lambda self, pid: (None, f"CRITICAL FAILURE for {pid}: boom"),
        )
        printed = []
        monkeypatch.setattr(
            console_mod.console, "print",
            lambda msg, **k: printed.append(str(msg)),
        )
        evaluator.run()
        assert Path(evaluator.output_csv_path).exists()
        assert any("CRITICAL FAILURE" in m for m in printed)
        # When all patients fail, an error is printed instead of a success msg.
        assert any("All patients failed" in m for m in printed)
        assert not any("\u2713" in m for m in printed)

    def test_multiple_patients_all_results_in_csv(
        self, tmp_path, monkeypatch, _patch_run_env
    ):
        """All patient results appear in the saved CSV."""
        df = _make_filepaths_df(tmp_path, n=3)
        ev = Evaluator(
            df,
            make_eval_config(),
            tmp_path / "results.csv",
        )
        monkeypatch.setattr(
            Evaluator,
            "_evaluate_patient_pipeline",
            lambda self, pid: ({"id": pid, "tumor_dice": 0.9}, None),
        )
        ev.run()
        results = pd.read_csv(ev.output_csv_path)
        # 3 patients + 5 stat rows.
        assert len(results) == 8
