"""Tests for the Analyzer class and its methods."""
import argparse
import json
import os
from pathlib import Path
from importlib import metadata
from typing import Union

import numpy as np
import pandas as pd
import pytest
import ants

# MIST imports.
from mist.analyze_data.analyzer import Analyzer
from mist.analyze_data.data_dumper import DataDumper
from mist.utils import io as io_mod, progress_bar
from mist.preprocessing import preprocessing_utils
from mist.analyze_data import analyzer_utils as au

# Shared test helpers.
from tests.analyze_data.helpers import (
    FakePB,
    fake_get_progress_bar,
    make_ants_image,
)

# Constants.
TRAIN_N = 5
TEST_N = 3


# ---------------------------------------------------------------------------
# Dataset / filesystem helpers (local — not shared across test files)
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_train_dir_for(path_like: Union[str, Path]) -> Path:
    p = Path(path_like)
    d = _ensure_dir(p.parent / "train_data")
    (d / "placeholder.txt").write_text("x")
    return d


def _ensure_test_dir_for(path_like: Union[str, Path]) -> Path:
    p = Path(path_like)
    d = _ensure_dir(p.parent / "test_data")
    (d / "placeholder.txt").write_text("y")
    return d


def fake_dataset_json(path: Union[str, Path]) -> dict:
    """Return a minimal valid dataset.json dictionary."""
    tdir = _ensure_train_dir_for(path)
    return {
        "task": "segmentation",
        "modality": "ct",
        "train-data": str(tdir),
        "mask": ["mask.nii.gz"],
        "images": {"ct": ["image.nii.gz"]},
        "labels": [0, 1],
        "final_classes": {"background": [0], "foreground": [1]},
    }


def fake_get_files_df(_dataset_json_path: str, split: str) -> pd.DataFrame:
    """Return a fake DataFrame of file paths for training or testing."""
    n = TRAIN_N if split == "train" else TEST_N
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "fold": list(range(n)),
            "mask": [f"{i}_mask.nii.gz" for i in range(n)],
            "ct": [f"{i}_ct.nii.gz" for i in range(n)],
        }
    )


def fake_compare_headers(_h1, _h2) -> bool:
    """Always return True."""
    return True


def fake_is_image_3d(h: dict) -> bool:
    """Return True if header has exactly 3 dimensions."""
    return len(h.get("dimensions", ())) == 3


def fake_get_resampled_image_dimensions(
    curr_dims, curr_spacing, target_spacing
):
    """Compute resampled dimensions by scaling current dims by spacing."""
    scale = np.array(curr_spacing, float) / np.array(target_spacing, float)
    return (np.array(curr_dims, float) * scale).round().astype(int)


def fake_get_float32_example_memory_size(dims, nch, _nlabels) -> int:
    """Return a simplified memory size estimate."""
    return int(np.prod(dims) * max(1, nch) * 4)


def fake_image_header_info(_p: str) -> dict:
    """Return a fake image header."""
    return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}


def fake_reorient_image2(image, _orient):
    """Return image unchanged."""
    return image


def fake_get_fg_mask_bbox(_image) -> dict:
    """Return a fake foreground bounding box."""
    return {
        "x_start": 2,
        "x_end": 4,
        "y_start": 2,
        "y_end": 4,
        "z_start": 2,
        "z_end": 4,
        "x_og_size": 10,
        "y_og_size": 10,
        "z_og_size": 10,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def capture_console(monkeypatch):
    """Capture rich console output; returns a list of message strings."""
    logs = []
    monkeypatch.setattr(
        "rich.console.Console.print",
        lambda self, *a, **k: logs.append(" ".join(str(x) for x in a)),
    )
    return logs


def assert_exclusion_summary(logs, n):
    """Assert the logs contain a summary line about excluding n examples."""
    assert any(f"Excluding {n} example(s) from training." in m for m in logs)


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, tmp_path):
    """Patch the environment for all Analyzer tests."""
    monkeypatch.chdir(tmp_path)

    # IO.
    def _read_json(path: str):
        if "dataset.json" in str(path) or "dummy_dataset" in str(path):
            return fake_dataset_json(path)
        return {}

    monkeypatch.setattr(
        io_mod, "read_json_file", _read_json, raising=True
    )
    monkeypatch.setattr(
        io_mod,
        "write_json_file",
        lambda p, d: Path(p).write_text(
            json.dumps(d), encoding="utf-8"
        ),
        raising=True,
    )

    # Analyzer utils — use real build_base_config so schema stays in sync.
    monkeypatch.setattr(
        au, "get_files_df", fake_get_files_df, raising=True
    )
    monkeypatch.setattr(
        au,
        "add_folds_to_df",
        lambda df, n_splits=None, **__: (
            df
            if "fold" in df.columns
            else df.assign(fold=list(range(len(df))))
        ),
        raising=True,
    )
    monkeypatch.setattr(
        au, "compare_headers", fake_compare_headers, raising=True
    )
    monkeypatch.setattr(
        au, "is_image_3d", fake_is_image_3d, raising=True
    )
    monkeypatch.setattr(
        au,
        "get_resampled_image_dimensions",
        fake_get_resampled_image_dimensions,
        raising=True,
    )
    monkeypatch.setattr(
        au,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size,
        raising=True,
    )
    monkeypatch.setattr(
        au,
        "get_best_patch_size",
        lambda _dims: [16, 16, 16],
        raising=True,
    )

    # ANTs / progress / version.
    monkeypatch.setattr(
        ants, "image_read", lambda _p: make_ants_image(), raising=True
    )
    monkeypatch.setattr(
        ants,
        "image_header_info",
        fake_image_header_info,
        raising=True,
    )
    monkeypatch.setattr(
        ants, "reorient_image2", fake_reorient_image2, raising=True
    )
    monkeypatch.setattr(
        progress_bar,
        "get_progress_bar",
        fake_get_progress_bar,
        raising=True,
    )
    monkeypatch.setattr(
        preprocessing_utils,
        "get_fg_mask_bbox",
        fake_get_fg_mask_bbox,
        raising=True,
    )
    monkeypatch.setattr(
        metadata, "version", lambda _pkg: "0.9.0", raising=True
    )

    # Prevent DataDumper from running during Analyzer tests.
    monkeypatch.setattr(DataDumper, "run", lambda self: None, raising=True)


@pytest.fixture
def args(tmp_path):
    """Return argparse.Namespace with default arguments."""
    return argparse.Namespace(
        data=str(tmp_path / "dataset.json"),
        results=str(_ensure_dir(tmp_path / "results")),
        nfolds=5,
        no_preprocess=False,
        patch_size=None,
        folds=None,
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Initialization and schema validation
# ---------------------------------------------------------------------------

class TestAnalyzerInit:
    """Tests for Analyzer.__init__ and dataset validation."""

    def test_init_and_paths(self, args):
        """Basic initialization populates expected path attributes."""
        a = Analyzer(args)
        assert a.paths_df.columns.tolist()[:4] == [
            "id", "fold", "mask", "ct"
        ]
        assert a.paths_csv.endswith("train_paths.csv")
        assert a.fg_bboxes_csv.endswith("fg_bboxes.csv")
        assert a.config_json.endswith("config.json")

    def test_missing_required_field_raises(self, args, monkeypatch):
        """Missing required field in dataset.json raises KeyError."""
        def _bad(path):
            d = fake_dataset_json(path)
            d.pop("task")
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
        with pytest.raises(KeyError):
            Analyzer(args)

    @pytest.mark.parametrize(
        "key, bad_val, exc",
        [
            pytest.param(
                "mask", "not_a_list", TypeError, id="mask_not_list"
            ),
            pytest.param(
                "mask", [], ValueError, id="mask_empty_list"
            ),
            pytest.param(
                "images", ["not_a_dict"], TypeError,
                id="images_not_dict",
            ),
            pytest.param(
                "images", {}, ValueError, id="images_empty_dict"
            ),
            pytest.param(
                "labels", "nah", TypeError, id="labels_not_list"
            ),
            pytest.param(
                "labels", [], ValueError, id="labels_empty_list"
            ),
            pytest.param(
                "labels", [1, 2], ValueError, id="labels_missing_bg"
            ),
            pytest.param(
                "final_classes", ["dict_expected"], TypeError,
                id="final_classes_not_dict",
            ),
            pytest.param(
                "final_classes", {}, ValueError,
                id="final_classes_empty_dict",
            ),
        ],
    )
    def test_schema_type_and_value_checks(
        self, args, monkeypatch, key, bad_val, exc
    ):
        """Invalid types/values for required fields raise the expected exc."""
        def _bad(path):
            d = fake_dataset_json(path)
            d[key] = bad_val
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
        with pytest.raises(exc):
            Analyzer(args)

    @pytest.mark.parametrize(
        "field",
        [
            pytest.param("task", id="task"),
            pytest.param("modality", id="modality"),
            pytest.param("train-data", id="train-data"),
            pytest.param("mask", id="mask"),
            pytest.param("images", id="images"),
            pytest.param("labels", id="labels"),
            pytest.param("final_classes", id="final_classes"),
        ],
    )
    def test_none_required_field_raises_value_error(
        self, args, monkeypatch, field
    ):
        """Required fields set to None raise ValueError with a description."""
        def _bad(path):
            d = fake_dataset_json(path)
            d[field] = None
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
        with pytest.raises(ValueError) as exc_info:
            Analyzer(args)
        assert f"Got None for '{field}'" in str(exc_info.value)

    def test_missing_train_data_directory_raises(
        self, args, monkeypatch, tmp_path
    ):
        """Non-existent train-data path raises FileNotFoundError."""
        def _missing(path):
            d = fake_dataset_json(path)
            d["train-data"] = str(tmp_path / "does_not_exist")
            return d
        monkeypatch.setattr(
            io_mod, "read_json_file", _missing, raising=True
        )
        with pytest.raises(FileNotFoundError):
            Analyzer(args)

    def test_empty_train_data_directory_raises(
        self, args, monkeypatch, tmp_path
    ):
        """Empty train-data directory raises FileNotFoundError."""
        empty = _ensure_dir(tmp_path / "empty_train")

        def _empty(path):
            d = fake_dataset_json(path)
            d["train-data"] = str(empty)
            return d
        monkeypatch.setattr(
            io_mod, "read_json_file", _empty, raising=True
        )
        with pytest.raises(FileNotFoundError):
            Analyzer(args)

    def test_warns_if_overwriting_config(
        self, args, monkeypatch, capture_console
    ):
        """overwrite=True triggers a warning when config.json already exists."""
        config_path = Path(args.results) / "config.json"
        config_path.write_text("{}")
        Analyzer(args)
        assert any(
            "Overwriting existing configuration at" in m
            and str(config_path) in m
            for m in capture_console
        )

    def test_no_warning_without_overwrite(
        self, args, monkeypatch, capture_console
    ):
        """overwrite=False does not trigger any console warning."""
        (Path(args.results) / "config.json").write_text(
            "{}", encoding="utf-8"
        )
        args.overwrite = False
        Analyzer(args)
        assert capture_console == []

    @pytest.mark.parametrize(
        "modality",
        [
            pytest.param("ct", id="ct"),
            pytest.param("CT", id="CT_uppercase"),
            pytest.param("mr", id="mr"),
            pytest.param("MR", id="MR_uppercase"),
            pytest.param("other", id="other"),
        ],
    )
    def test_valid_modality_accepted(self, args, monkeypatch, modality):
        """Known modality strings (case-insensitive) do not raise."""
        def _mod(path):
            d = fake_dataset_json(path)
            d["modality"] = modality
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _mod, raising=True)
        Analyzer(args)  # must not raise

    def test_invalid_modality_raises_value_error(self, args, monkeypatch):
        """An unrecognised modality string raises ValueError."""
        def _bad(path):
            d = fake_dataset_json(path)
            d["modality"] = "xray"
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
        with pytest.raises(ValueError, match="modality"):
            Analyzer(args)

    def test_final_classes_label_not_in_labels_raises(
        self, args, monkeypatch
    ):
        """A label in final_classes that is absent from labels raises ValueError."""
        def _bad(path):
            d = fake_dataset_json(path)
            # Label 99 is not in d["labels"].
            d["final_classes"] = {"tumor": [1, 99]}
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
        with pytest.raises(ValueError, match="final_classes"):
            Analyzer(args)

    def test_final_classes_all_known_labels_accepted(
        self, args, monkeypatch
    ):
        """final_classes whose labels are all in labels does not raise."""
        def _ok(path):
            d = fake_dataset_json(path)
            d["final_classes"] = {"tumor": [1]}
            return d
        monkeypatch.setattr(io_mod, "read_json_file", _ok, raising=True)
        Analyzer(args)  # must not raise


# ---------------------------------------------------------------------------
# Per-sample analysis helpers
# ---------------------------------------------------------------------------

class TestAnalyzerHelpers:
    """Tests for individual Analyzer helper methods."""

    def test_check_crop_fg_writes_csv_and_returns_shape(self, args):
        """check_crop_fg writes fg_bboxes.csv and returns valid values."""
        a = Analyzer(args)
        crop, cropped = a.check_crop_fg()
        assert isinstance(crop, (bool, np.bool_))
        assert cropped.shape == (len(a.paths_df), 3)
        df = pd.read_csv(a.fg_bboxes_csv)
        assert {
            "id", "x_start", "x_end", "y_start",
            "y_end", "z_start", "z_end",
            "x_og_size", "y_og_size", "z_og_size",
        }.issubset(df.columns)

    def test_check_nz_ratio_sparse_images(self, args, monkeypatch):
        """Sparse first image causes check_nz_ratio to return True."""
        def _sparse_or_dense(p):
            if "ct" in str(p):
                img = make_ants_image(fill=0.0)
                img.numpy()[2:4, 2:4, 2:4] = 1.0
                return img
            return make_ants_image(fill=0.0)
        monkeypatch.setattr(
            ants, "image_read", _sparse_or_dense, raising=True
        )
        assert bool(Analyzer(args).check_nz_ratio())

    def test_check_nz_ratio_dense_images(self, args, monkeypatch):
        """Dense images cause check_nz_ratio to return False."""
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _p: make_ants_image(fill=1.0),
            raising=True,
        )
        assert not bool(Analyzer(args).check_nz_ratio())

    def test_get_target_spacing_handles_anisotropy(
        self, args, monkeypatch
    ):
        """Anisotropic images → target spacing max equals the percentile."""
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _p: make_ants_image(spacing=(1.0, 1.0, 5.0)),
            raising=True,
        )
        monkeypatch.setattr(
            np, "percentile", lambda a, q: 3.0, raising=True
        )
        assert max(Analyzer(args).get_target_spacing()) == 3.0

    def test_check_resampled_dims_warns_when_large(
        self, args, monkeypatch, capture_console
    ):
        """check_resampled_dims logs a warning when exceeding memory limit."""
        monkeypatch.setattr(
            au,
            "get_float32_example_memory_size",
            lambda *_: int(2e10),
            raising=True,
        )
        monkeypatch.setattr(
            au,
            "get_resampled_image_dimensions",
            lambda *_a, **_k: (128, 128, 128),
            raising=True,
        )
        Analyzer(args).check_resampled_dims(
            np.ones((TRAIN_N, 3)) * 5
        )
        assert any(
            "Resampled example is larger than the recommended memory size"
            in m
            for m in capture_console
        )

    def test_check_crop_fg_raises_on_worker_error(self, args, monkeypatch):
        """An exception in the FG-crop worker propagates as RuntimeError."""
        monkeypatch.setattr(
            preprocessing_utils,
            "get_fg_mask_bbox",
            lambda _: (_ for _ in ()).throw(RuntimeError("bad bbox")),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).check_crop_fg()

    def test_check_nz_ratio_raises_on_worker_error(self, args, monkeypatch):
        """An exception in the NZ-ratio worker propagates as RuntimeError."""
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _p: (_ for _ in ()).throw(RuntimeError("bad file")),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).check_nz_ratio()

    def test_get_target_spacing_raises_on_worker_error(
        self, args, monkeypatch
    ):
        """An exception in the spacing worker propagates as RuntimeError."""
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _p: (_ for _ in ()).throw(RuntimeError("bad file")),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).get_target_spacing()

    def test_check_resampled_dims_raises_on_worker_error(
        self, args, monkeypatch
    ):
        """An exception in the resampled-dims worker propagates as RuntimeError."""
        monkeypatch.setattr(
            ants,
            "image_header_info",
            lambda _p: (_ for _ in ()).throw(RuntimeError("bad header")),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).check_resampled_dims(np.ones((TRAIN_N, 3)) * 5)

    def test_check_resampled_dims_raises_if_called_before_analyze_dataset(
        self, args
    ):
        """check_resampled_dims raises RuntimeError when config keys are absent."""
        a = Analyzer(args)
        # Remove the keys that analyze_dataset would have written.
        a.config.get("preprocessing", {}).pop("crop_to_foreground", None)
        a.config.get("preprocessing", {}).pop("target_spacing", None)
        with pytest.raises(RuntimeError, match="analyze_dataset"):
            a.check_resampled_dims(np.ones((TRAIN_N, 3)) * 5)


# ---------------------------------------------------------------------------
# get_ct_normalization_parameters
# ---------------------------------------------------------------------------

class TestGetCtNormalizationParameters:
    """Integration tests for Analyzer.get_ct_normalization_parameters."""

    def _make_ct_analyzer(self, args, monkeypatch, hu_value: float):
        """Return an Analyzer whose image_read always returns a uniform HU image.

        The mask is all-ones (full foreground) so every voxel is sampled.
        """
        def _image_read(path):
            if "mask" in str(path):
                return make_ants_image(fill=1.0)
            return make_ants_image(fill=hu_value)

        monkeypatch.setattr(ants, "image_read", _image_read, raising=True)
        return Analyzer(args)

    def test_output_keys_are_present(self, args, monkeypatch):
        """Return dict contains all four expected keys."""
        result = self._make_ct_analyzer(
            args, monkeypatch, hu_value=0.0
        ).get_ct_normalization_parameters()
        assert set(result) == {"window_min", "window_max", "z_score_mean", "z_score_std"}

    def test_constant_hu_gives_correct_mean(self, args, monkeypatch):
        """Uniform HU images produce the correct mean."""
        result = self._make_ct_analyzer(
            args, monkeypatch, hu_value=100.0
        ).get_ct_normalization_parameters()
        assert result["z_score_mean"] == pytest.approx(100.0, abs=1.0)

    def test_constant_hu_gives_zero_std(self, args, monkeypatch):
        """Uniform HU images produce std ≈ 0."""
        result = self._make_ct_analyzer(
            args, monkeypatch, hu_value=100.0
        ).get_ct_normalization_parameters()
        assert result["z_score_std"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_hu_window_matches_value(self, args, monkeypatch):
        """Both window bounds equal the uniform HU value within 1 HU."""
        hu = 200.0
        result = self._make_ct_analyzer(
            args, monkeypatch, hu_value=hu
        ).get_ct_normalization_parameters()
        assert abs(result["window_min"] - hu) <= 1.0
        assert abs(result["window_max"] - hu) <= 1.0

    def test_window_min_less_than_window_max_with_variance(
        self, args, monkeypatch
    ):
        """When patients have different HU values, window_min < window_max."""
        call_count = [0]

        def _image_read(path):
            if "mask" in str(path):
                return make_ants_image(fill=1.0)
            # Alternate between low and high HU across patients.
            hu = -500.0 if call_count[0] % 2 == 0 else 500.0
            call_count[0] += 1
            return make_ants_image(fill=hu)

        monkeypatch.setattr(ants, "image_read", _image_read, raising=True)
        result = Analyzer(args).get_ct_normalization_parameters()
        assert result["window_min"] < result["window_max"]

    def test_empty_foreground_is_skipped(self, args, monkeypatch):
        """Patients with all-zero masks contribute nothing to statistics."""
        def _image_read(path):
            if "mask" in str(path):
                return make_ants_image(fill=0.0)  # no foreground
            return make_ants_image(fill=100.0)

        monkeypatch.setattr(ants, "image_read", _image_read, raising=True)
        result = Analyzer(args).get_ct_normalization_parameters()
        assert set(result) == {
            "window_min", "window_max", "z_score_mean", "z_score_std"
        }

    def test_out_of_range_hu_emits_warning(
        self, args, monkeypatch, capture_console
    ):
        """Foreground HU values outside the histogram range trigger a warning."""
        def _image_read(path):
            if "mask" in str(path):
                return make_ants_image(fill=1.0)
            return make_ants_image(fill=5000.0)  # above CT_HU_HIST_MAX

        monkeypatch.setattr(ants, "image_read", _image_read, raising=True)
        Analyzer(args).get_ct_normalization_parameters()
        assert any(
            "HU values outside the histogram range" in m
            for m in capture_console
        )

    def test_ct_normalization_raises_on_worker_error(
        self, args, monkeypatch
    ):
        """An exception in the CT stats worker propagates as RuntimeError."""
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _p: (_ for _ in ()).throw(RuntimeError("bad file")),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).get_ct_normalization_parameters()


# ---------------------------------------------------------------------------
# analyze_dataset
# ---------------------------------------------------------------------------

class TestAnalyzerAnalyzeDataset:
    """Tests for Analyzer.analyze_dataset."""

    def test_config_is_fully_populated(self, args, monkeypatch):
        """analyze_dataset populates all expected config fields."""
        monkeypatch.setattr(
            au,
            "get_best_patch_size",
            lambda _d: [24, 24, 24],
            raising=True,
        )
        monkeypatch.setattr(
            au,
            "get_resampled_image_dimensions",
            lambda *_: (10, 10, 10),
            raising=True,
        )
        monkeypatch.setattr(
            au,
            "get_float32_example_memory_size",
            lambda *_: int(1e6),
            raising=True,
        )
        monkeypatch.setattr(
            Analyzer,
            "get_target_spacing",
            lambda self: [1.0, 1.0, 1.0],
            raising=True,
        )
        monkeypatch.setattr(
            Analyzer,
            "check_crop_fg",
            lambda self: (
                True, np.ones((len(self.paths_df), 3)) * 8
            ),
            raising=True,
        )
        monkeypatch.setattr(
            Analyzer,
            "check_resampled_dims",
            lambda self, _d: [10, 10, 10],
            raising=True,
        )
        monkeypatch.setattr(
            Analyzer,
            "check_nz_ratio",
            lambda self: True,
            raising=True,
        )
        monkeypatch.setattr(
            Analyzer,
            "get_ct_normalization_parameters",
            lambda self: {
                "window_min": -1000,
                "window_max": 1000,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
            raising=True,
        )
        monkeypatch.setattr(
            metadata, "version", lambda _pkg: "0.9.0", raising=True
        )

        a = Analyzer(args)
        a.analyze_dataset()
        cfg = a.config

        assert cfg["mist_version"] == "0.9.0"
        assert cfg["dataset_info"]["images"] == ["ct"]
        assert cfg["preprocessing"]["crop_to_foreground"] is True
        assert cfg["preprocessing"]["median_resampled_image_size"] == [
            10, 10, 10
        ]
        assert cfg["model"]["params"]["patch_size"] == [24, 24, 24]
        assert (
            cfg["inference"]["inferer"]["params"]["patch_size"]
            == [24, 24, 24]
        )
        assert cfg["evaluation"] == {
            "background": {
                "labels": [0],
                "metrics": {"dice": {}, "haus95": {}},
            },
            "foreground": {
                "labels": [1],
                "metrics": {"dice": {}, "haus95": {}},
            },
        }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestAnalyzerRun:
    """Tests for Analyzer.run."""

    def test_writes_config_and_paths_csv(self, args, monkeypatch):
        """run() writes config.json and train_paths.csv, respecting nfolds."""
        monkeypatch.setattr(
            au, "get_files_df",
            lambda _d, _s: pd.DataFrame({
                "id": [0, 1],
                "fold": [0, 1],
                "mask": ["m0.nii.gz", "m1.nii.gz"],
                "ct": ["i0.nii.gz", "i1.nii.gz"],
            }),
            raising=True,
        )
        a = Analyzer(args)
        a.run()
        cfg = json.loads(Path(a.config_json).read_text(encoding="utf-8"))
        assert cfg["training"]["nfolds"] == args.nfolds
        assert cfg["training"]["folds"] == list(range(args.nfolds))
        df = pd.read_csv(a.paths_csv)
        assert not df.empty and "fold" in df.columns

    def test_writes_test_paths_csv_when_test_data_provided(
        self, args, monkeypatch
    ):
        """If test-data is provided, run() writes test_paths.csv."""
        train_dir = _ensure_train_dir_for(args.data)
        test_dir = _ensure_test_dir_for(args.data)

        def _read_json(path: str):
            if "dataset.json" in path:
                return {
                    "task": "segmentation",
                    "modality": "ct",
                    "train-data": str(train_dir),
                    "test-data": str(test_dir),
                    "mask": ["mask.nii.gz"],
                    "images": {"ct": ["image.nii.gz"]},
                    "labels": [0, 1],
                    "final_classes": {
                        "background": [0], "foreground": [1]
                    },
                }
            return {}

        monkeypatch.setattr(
            io_mod, "read_json_file", _read_json, raising=True
        )

        calls = []

        def _fake_get_files_df(_d, split):
            """Return a fake DataFrame for either split."""
            calls.append(split)
            n = 2 if split == "train" else 1
            return pd.DataFrame({
                "id": list(range(n)),
                "fold": list(range(n)),
                "mask": [f"m{i}.nii.gz" for i in range(n)],
                "ct": [f"i{i}.nii.gz" for i in range(n)],
            })

        monkeypatch.setattr(
            au, "get_files_df", _fake_get_files_df, raising=True
        )

        Analyzer(args).run()
        assert (Path(args.results) / "test_paths.csv").exists()
        assert "train" in calls and "test" in calls

    def test_raises_if_test_directory_missing(
        self, args, monkeypatch
    ):
        """run() raises FileNotFoundError when test-data dir is absent."""
        train_dir = _ensure_train_dir_for(args.data)
        missing = Path(args.data).parent / "missing_test"

        def _read_json(path: str):
            if "dataset.json" in path:
                return {
                    "task": "segmentation",
                    "modality": "ct",
                    "train-data": str(train_dir),
                    "test-data": str(missing),
                    "mask": ["mask.nii.gz"],
                    "images": {"ct": ["image.nii.gz"]},
                    "labels": [0, 1],
                    "final_classes": {
                        "background": [0], "foreground": [1]
                    },
                }
            return {}

        monkeypatch.setattr(
            io_mod, "read_json_file", _read_json, raising=True
        )
        with pytest.raises(FileNotFoundError):
            Analyzer(args).run()


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

class TestValidateDataset:
    """Tests for Analyzer.validate_dataset.

    validate_dataset is called before add_folds_to_df, so paths_df has
    columns [id, mask, image_1, ...] with no 'fold' column. The autouse
    fixture below overrides fake_get_files_df to reflect this.
    """

    @pytest.fixture(autouse=True)
    def _no_fold_df(self, monkeypatch):
        """Override paths_df to match production state (no fold column)."""
        no_fold = pd.DataFrame({
            "id": list(range(TRAIN_N)),
            "mask": [f"{i}_mask.nii.gz" for i in range(TRAIN_N)],
            "ct": [f"{i}_ct.nii.gz" for i in range(TRAIN_N)],
        })
        monkeypatch.setattr(
            au, "get_files_df", lambda *_a, **_k: no_fold, raising=True
        )

    def test_happy_path_preserves_all_samples(self, args):
        """validate_dataset does not exclude any sample when all are valid."""
        a = Analyzer(args)
        initial = len(a.paths_df)
        a.validate_dataset()
        assert len(a.paths_df) == initial

    def test_illegal_label_excludes_sample(self, args, monkeypatch):
        """Mask with an illegal label value is excluded."""
        def _dispatch(path: str):
            if "mask" in path:
                arr = np.zeros((10, 10, 10), dtype=np.float32)
                arr[2:4, 2:4, 2:4] = (
                    99 if path.startswith("0_") or "/0_mask.nii.gz" in path else 1
                )
                return ants.from_numpy(arr)
            return make_ants_image(fill=1.0)

        monkeypatch.setattr(ants, "image_read", _dispatch, raising=True)
        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1

    def test_non_runtime_error_excludes_one_sample_and_logs(
        self, args, monkeypatch, capture_console
    ):
        """Non-RuntimeError exceptions from ANTs are also caught and exclude the patient."""
        def _read_dispatch(path: str):
            if "0_mask.nii.gz" in str(path):
                raise Exception("ITK internal error")
            return ants.from_numpy(
                np.ones((10, 10, 10), dtype=np.float32)
            )

        monkeypatch.setattr(
            ants, "image_read", _read_dispatch, raising=True
        )
        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any(
            "In 0:" in m and "ITK internal error" in m
            for m in capture_console
        )
        assert_exclusion_summary(capture_console, 1)

    def test_runtime_error_excludes_one_sample_and_logs(
        self, args, monkeypatch, capture_console
    ):
        """RuntimeError on one sample is caught and that sample excluded."""
        def _read_dispatch(path: str):
            if "0_mask.nii.gz" in str(path):
                raise RuntimeError("corrupted NIfTI header")
            return ants.from_numpy(
                np.ones((10, 10, 10), dtype=np.float32)
            )

        monkeypatch.setattr(
            ants, "image_read", _read_dispatch, raising=True
        )
        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any(
            "In 0:" in m and "corrupted NIfTI header" in m
            for m in capture_console
        )
        assert_exclusion_summary(capture_console, 1)

    def test_runtime_error_all_samples_raises(self, args, monkeypatch):
        """RuntimeError on every sample raises RuntimeError."""
        monkeypatch.setattr(
            ants,
            "image_read",
            lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).validate_dataset()

    def test_header_mismatch_excludes_one_and_logs(
        self, args, monkeypatch, capture_console
    ):
        """Sample with mismatched image/mask headers is excluded."""
        def _hdr_with_path(path: str):
            spacing = (
                (2.0, 2.0, 2.0)
                if "0_ct.nii.gz" in path
                else (1.0, 1.0, 1.0)
            )
            return {"dimensions": (10, 10, 10), "spacing": spacing}

        def _compare_by_spacing(h1: dict, h2: dict) -> bool:
            """Compare headers by spacing tuple equality."""
            return (
                tuple(h1.get("spacing", ()))
                == tuple(h2.get("spacing", ()))
            )

        monkeypatch.setattr(
            ants, "image_header_info", _hdr_with_path, raising=True
        )
        monkeypatch.setattr(
            au, "compare_headers", _compare_by_spacing, raising=True
        )

        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any(
            "In 0:" in m
            and "Mismatch between image  and mask header information" in m
            for m in capture_console
        )
        assert_exclusion_summary(capture_console, 1)

    def test_header_mismatch_all_samples_raises(
        self, args, monkeypatch
    ):
        """All-mismatch headers cause validate_dataset to raise."""
        def _hdr_router(path: str):
            if path.endswith(".nii.gz") and "mask" not in path:
                return {
                    "dimensions": (10, 10, 10),
                    "spacing": (2.0, 2.0, 2.0),
                }
            return {
                "dimensions": (10, 10, 10),
                "spacing": (1.0, 1.0, 1.0),
            }

        monkeypatch.setattr(
            ants, "image_header_info", _hdr_router, raising=True
        )
        monkeypatch.setattr(
            au,
            "compare_headers",
            lambda h1, h2: (
                tuple(h1.get("spacing", ()))
                == tuple(h2.get("spacing", ()))
            ),
            raising=True,
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).validate_dataset()

    def test_4d_image_excludes_one_and_logs(
        self, args, monkeypatch, capture_console
    ):
        """4D image is excluded and a descriptive message is logged."""
        def _hdr_router(path: str):
            if "mask" in path:
                return {
                    "dimensions": (10, 10, 10),
                    "spacing": (1.0, 1.0, 1.0),
                }
            if "0_ct.nii.gz" in path or path.startswith("0_"):
                return {
                    "dimensions": (10, 10, 10, 1),
                    "spacing": (1.0, 1.0, 1.0, 1.0),
                }
            return {
                "dimensions": (10, 10, 10),
                "spacing": (1.0, 1.0, 1.0),
            }

        monkeypatch.setattr(
            au, "compare_headers", lambda h1, h2: True, raising=True
        )
        monkeypatch.setattr(
            ants, "image_header_info", _hdr_router, raising=True
        )

        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any(
            "In 0:" in m
            and "Got 4D image, make sure all images are 3D" in m
            for m in capture_console
        )
        assert_exclusion_summary(capture_console, 1)

    def test_4d_mask_excludes_one_and_logs(
        self, args, monkeypatch, capture_console
    ):
        """4D mask is excluded and a descriptive message is logged."""
        def _hdr_router(path: str):
            if "mask" in path:
                if "0_mask.nii.gz" in path or path.startswith("0_"):
                    return {
                        "dimensions": (10, 10, 10, 1),
                        "spacing": (1.0, 1.0, 1.0, 1.0),
                    }
                return {
                    "dimensions": (10, 10, 10),
                    "spacing": (1.0, 1.0, 1.0),
                }
            return {
                "dimensions": (10, 10, 10),
                "spacing": (1.0, 1.0, 1.0),
            }

        monkeypatch.setattr(
            au, "compare_headers", lambda h1, h2: True, raising=True
        )
        monkeypatch.setattr(
            ants, "image_header_info", _hdr_router, raising=True
        )

        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any(
            "In 0:" in m
            and "Got 4D mask" in m
            and "images are 3D" in m
            for m in capture_console
        )
        assert_exclusion_summary(capture_console, 1)

    def test_4d_image_all_raise(self, args, monkeypatch):
        """All-4D images cause validate_dataset to raise RuntimeError."""
        def _hdr_router(path: str):
            if "mask" in path:
                return {
                    "dimensions": (10, 10, 10),
                    "spacing": (1.0, 1.0, 1.0),
                }
            return {
                "dimensions": (10, 10, 10, 1),
                "spacing": (1.0, 1.0, 1.0, 1.0),
            }

        monkeypatch.setattr(
            au, "compare_headers", lambda h1, h2: True, raising=True
        )
        monkeypatch.setattr(
            ants, "image_header_info", _hdr_router, raising=True
        )
        with pytest.raises(RuntimeError):
            Analyzer(args).validate_dataset()

    def test_corrupt_secondary_image_excludes_patient(
        self, args, monkeypatch, capture_console
    ):
        """RuntimeError from ants.image_header_info inside the loop excludes patient."""
        call_count = {"n": 0}

        def _hdr_router(path: str):
            # Raise on the second call for patient 0's image (inside the loop).
            if "0_ct.nii.gz" in path:
                call_count["n"] += 1
                if call_count["n"] > 1:
                    raise RuntimeError("corrupt secondary image header")
            return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}

        monkeypatch.setattr(
            ants, "image_header_info", _hdr_router, raising=True
        )
        monkeypatch.setattr(
            au, "compare_headers", lambda h1, h2: True, raising=True
        )

        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any("In 0:" in m for m in capture_console)

    def test_multi_image_mismatch_excludes_one_and_logs(
        self, args, monkeypatch, capture_console
    ):
        """Patient with mismatched image-to-image headers is excluded."""
        two_image_df = pd.DataFrame({
            "id": list(range(TRAIN_N)),
            "mask": [f"{i}_mask.nii.gz" for i in range(TRAIN_N)],
            "ct": [f"{i}_ct.nii.gz" for i in range(TRAIN_N)],
            "t2": [f"{i}_t2.nii.gz" for i in range(TRAIN_N)],
        })
        monkeypatch.setattr(
            au, "get_files_df", lambda *_a, **_k: two_image_df, raising=True
        )

        def _hdr_router(path: str):
            # Tag headers so the comparison can distinguish mask-vs-image
            # from image-vs-image without relying on spacing alone.
            if "mask" in path:
                return {
                    "dimensions": (10, 10, 10),
                    "spacing": (1.0, 1.0, 1.0),
                    "_type": "mask",
                }
            if "0_t2" in path:
                return {
                    "dimensions": (10, 10, 10),
                    "spacing": (2.0, 2.0, 2.0),
                    "_type": "t2",
                }
            return {
                "dimensions": (10, 10, 10),
                "spacing": (1.0, 1.0, 1.0),
                "_type": "image",
            }

        def _compare(h1: dict, h2: dict) -> bool:
            # Mask-vs-any comparison always passes.
            if h1.get("_type") == "mask" or h2.get("_type") == "mask":
                return True
            # Image-vs-image: compare spacing.
            return (
                tuple(h1.get("spacing", ())) == tuple(h2.get("spacing", ()))
            )

        monkeypatch.setattr(
            ants, "image_header_info", _hdr_router, raising=True
        )
        monkeypatch.setattr(au, "compare_headers", _compare, raising=True)

        a = Analyzer(args)
        a.validate_dataset()
        assert len(a.paths_df) == TRAIN_N - 1
        assert any(
            "In 0:" in m and "Mismatch between" in m and "images" in m
            for m in capture_console
        )
        assert_exclusion_summary(capture_console, 1)
