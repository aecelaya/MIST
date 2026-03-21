"""Tests for mist.inference.inference_runners."""
from typing import Optional
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
import copy
import os
import pytest
import torch
import numpy as np
import pandas as pd

# MIST imports.
from mist.training import training_utils
from mist.inference import inference_runners as ir


# =========================
# Shared fixtures & helpers
# =========================

@pytest.fixture()
def mock_mist_config():
    """Fixture to provide a mock MIST configuration."""
    return {
        "dataset_info": {"modality": "ct", "labels": [0, 1]},
        "spatial_config": {
            "patch_size": [64, 64, 64],
            "target_spacing": [1.0, 1.0, 1.0],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
            "compute_dtms": False,
            "normalize_dtms": True,
        },
        "model": {
            "architecture": "nnunet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
            },
        },
        "training": {
            "seed": 42,
            "hardware": {"num_gpus": 2, "num_cpu_workers": 8},
        },
        "inference": {
            "inferer": {
                "name": "sliding_window",
                "params": {
                    "patch_blend_mode": "gaussian",
                    "patch_overlap": 0.5,
                },
            },
            "ensemble": {"strategy": "mean"},
            "tta": {"enabled": True, "strategy": "all_flips"},
        },
    }


@pytest.fixture
def noop_cuda_tensor_to(monkeypatch):
    """Make Tensor.to('cuda' | torch.device('cuda')) a no-op.

    This is so tests can exercise the 'cuda' code path on machines without CUDA
    builds.
    """
    orig_to = torch.Tensor.to

    def _safe_to(self, *args, **kwargs):
        device_arg = args[0] if args else kwargs.get("device", None)
        dev_type = (
            device_arg if isinstance(device_arg, str)
            else (
                device_arg.type
                if isinstance(device_arg, torch.device)
                else None
            )
        )
        if dev_type == "cuda":
            return self
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _safe_to, raising=True)


class _DummyANTsImage:
    """Minimal stand-in for ANTsImage."""
    def __init__(self, array: Optional[np.ndarray]=None):
        self._array = (
            np.array(0, dtype=np.uint8) if array is None else np.asarray(array)
        )
        self.astype_arg = None
        self.new_like_last_data = None
        self.shape = self._array.shape

    def numpy(self):
        """Return the underlying numpy array."""
        return np.asarray(self._array)

    def astype(self, dtype: str):
        """Simulate ANTsImage astype method."""
        self.astype_arg = dtype
        if dtype == "uint8":
            self._array = self._array.astype(np.uint8)
        return self

    def new_image_like(self, data):
        """Simulate ANTsImage new_image_like method."""
        self.new_like_last_data = np.asarray(data)
        return _DummyANTsImage(self.new_like_last_data)


def _predictor_logits_two_class(_: torch.Tensor) -> torch.Tensor:
    """Return logits so argmax along channel (axis=1) -> all ones."""
    zeros = torch.zeros(1, 1, 2, 2, 2)
    ones = torch.ones(1, 1, 2, 2, 2)
    return torch.cat([zeros, ones], dim=1)


class _PB:
    """Minimal progress bar stub with context and .track()."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def track(self, it):
        """Yield items from the iterable."""
        return it


class _DummyModel:
    """Minimal model stub with eval() and to() methods."""
    def __init__(self):
        self.device = None

    def eval(self):
        """No-op eval method."""
        return self

    def to(self, device):
        """Record device and return self."""
        self.device = device
        return self


class _DummyLoader:
    """Minimal DALI-like loader with .next() returning [{'image': tensor}]."""
    def __init__(self, n: int):
        self.n = n
        self.i = 0
        self.batch = [{"image": torch.zeros(1, 1, 2, 2, 2)}]

    def next(self):
        """Return a batch until n is reached, then keep returning last batch."""
        if self.i < self.n:
            self.i += 1
        return self.batch


def _make_train_df(fold: int, image_key: str = "image"):
    """Create a minimal DataFrame with patient id, fold, and image path."""
    return pd.DataFrame(
        [{"id": "p1", "fold": fold, image_key: "/tmp/p1.nii.gz"}]
    )


def _make_bbox_df():
    """Create a minimal DataFrame with foreground bounding box for a patient."""
    return pd.DataFrame(
        [{"id": "p1", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}]
    )


def _prep_dirs(tmp_path: Path):
    """Prepare filesystem layout for test_on_fold test."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions" / "train" / "raw").mkdir(
        parents=True, exist_ok=True
    )
    (numpy_dir / "images").mkdir(parents=True, exist_ok=True)
    return results_dir, numpy_dir


def _df_single_case(tmp_path: Path):
    """Minimal dataframe with a single patient id + image path."""
    return pd.DataFrame(
        [{"id": "p1", "image": str((tmp_path / "images" / "p1.nii.gz"))}]
    )


def _ensure_dir(p: Path):
    """Ensure directory exists, create if not."""
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================
# predict_single_example tests
# ============================

@patch("mist.inference.inference_utils.back_to_original_space")
def test_predict_single_example_no_remap_no_crop(
    mock_back_to_original_space,
    mock_mist_config,
    monkeypatch,
):
    """No remap when labels match and no crop; returns uint8 ANTs-like image."""
    cfg = copy.deepcopy(mock_mist_config)
    cfg["model"]["params"]["out_channels"] = 2
    cfg["preprocessing"]["crop_to_foreground"] = False
    cfg["dataset_info"]["labels"] = [0, 1]

    monkeypatch.setattr(
        ir, "ic", SimpleNamespace(ARGMAX_AXIS=1, BATCH_AXIS=0), raising=False
    )

    mocked_ants_img = _DummyANTsImage(np.ones((2, 2, 2), dtype=np.int64))
    mock_back_to_original_space.return_value = mocked_ants_img

    pre_img = torch.randn(1, 1, 2, 2, 2)
    orig_ants = _DummyANTsImage(np.zeros((2, 2, 2), dtype=np.int64))

    out = ir.predict_single_example(
        preprocessed_image=pre_img,
        original_ants_image=orig_ants,
        mist_configuration=cfg,
        predictor=_predictor_logits_two_class,
        foreground_bounding_box=None,
    )

    call_kwargs = mock_back_to_original_space.call_args.kwargs
    assert call_kwargs["original_ants_image"] is orig_ants
    np.testing.assert_array_equal(
        call_kwargs["raw_prediction"], np.ones((2, 2, 2))
    )
    assert call_kwargs["training_labels"] == [0, 1]
    assert call_kwargs["foreground_bounding_box"] is None
    assert isinstance(out, _DummyANTsImage)
    assert out.astype_arg == "uint8"
    assert out.numpy().dtype == np.uint8


@patch("mist.inference.inference_utils.remap_mask_labels")
@patch("mist.inference.inference_utils.back_to_original_space")
@patch("mist.preprocessing.preprocessing_utils.get_fg_mask_bbox")
def test_predict_single_example_with_crop_and_remap(
    mock_get_fg_bbox,
    mock_back_to_original_space,
    mock_remap_labels,
    mock_mist_config,
    monkeypatch,
):
    """Crop bbox computed and labels remapped when original labels differ."""
    cfg = copy.deepcopy(mock_mist_config)
    cfg["model"]["params"]["out_channels"] = 2
    cfg["preprocessing"]["crop_to_foreground"] = True
    cfg["dataset_info"]["labels"] = [0, 2]

    monkeypatch.setattr(
        ir, "ic", SimpleNamespace(ARGMAX_AXIS=1, BATCH_AXIS=0), raising=False
    )

    bbox = {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}
    mock_get_fg_bbox.return_value = bbox

    ants_after_restore = _DummyANTsImage(np.full((2, 2, 2), 1, dtype=np.int64))
    mock_back_to_original_space.return_value = ants_after_restore

    remapped = np.full((2, 2, 2), 2, dtype=np.int64)
    mock_remap_labels.return_value = remapped

    pre_img = torch.randn(1, 1, 2, 2, 2)
    orig_ants = _DummyANTsImage(np.zeros((2, 2, 2), dtype=np.int64))

    out = ir.predict_single_example(
        preprocessed_image=pre_img,
        original_ants_image=orig_ants,
        mist_configuration=cfg,
        predictor=_predictor_logits_two_class,
        foreground_bounding_box=None,
    )

    mock_get_fg_bbox.assert_called_once_with(orig_ants)
    call_kwargs = mock_back_to_original_space.call_args.kwargs
    assert call_kwargs["foreground_bounding_box"] == bbox
    assert call_kwargs["training_labels"] == [0, 1]
    mock_remap_labels.assert_called_once()
    np.testing.assert_array_equal(
        mock_remap_labels.call_args.args[0], ants_after_restore.numpy()
    )
    assert mock_remap_labels.call_args.args[1] == [0, 2]
    assert isinstance(out, _DummyANTsImage)
    assert orig_ants.new_like_last_data is not None
    np.testing.assert_array_equal(orig_ants.new_like_last_data, remapped)
    assert out.astype_arg == "uint8"


# ==================
# test_on_fold tests
# ==================

@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.ants.image_read", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.io.read_json_file")
def test_test_on_fold_success_no_crop_tta_enabled(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with one case, no cropping, TTA enabled, and writes output."""
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 0
    _make_train_df(fold).to_csv(results_dir / "train_paths.csv", index=False)
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = False
    cfg["inference"]["tta"]["enabled"] = True
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    model = _DummyModel()
    mock_load_model.return_value = model

    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")
    mock_get_strategy.return_value = lambda: SimpleNamespace(name="tta")
    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        training_utils,
        "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )

    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device=None)

    mock_get_test_dataset.assert_called_once()
    mock_load_model.assert_called_once()
    mock_predict_single.assert_called_once()
    mock_image_write.assert_called_once()

    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(
        os.path.join("predictions", "train", "raw", "p1.nii.gz")
    )

    # Ensure TTA branch used the strategy from config.
    assert any(
        c.args == (cfg["inference"]["tta"]["strategy"],)
        for c in mock_get_strategy.mock_calls
    )


@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.ants.image_read", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.io.read_json_file")
def test_test_on_fold_crop_to_foreground_bbox_passed(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    _mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """crop_to_foreground=True, bbox from CSV is forwarded to prediction."""
    monkeypatch.setattr(
        training_utils,
        "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )

    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 2
    _make_train_df(fold).to_csv(results_dir / "train_paths.csv", index=False)
    bbox_df = _make_bbox_df()
    bbox_df.to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = True
    cfg["preprocessing"]["skip"] = False
    cfg["inference"]["tta"]["enabled"] = False
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    model = _DummyModel()
    mock_load_model.return_value = model

    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")

    def _strategy(name):
        return (
            (lambda: SimpleNamespace(name="none"))
            if name == "none"
            else (lambda: SimpleNamespace(name=name))
        )

    mock_get_strategy.side_effect = _strategy

    captured = {}

    def _capture_kwargs(**kwargs):
        captured.update(kwargs)
        return _DummyANTsImage()

    mock_predict_single.side_effect = _capture_kwargs

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )

    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device="cpu")

    assert "foreground_bounding_box" in captured
    assert captured["foreground_bounding_box"] == bbox_df.iloc[0].to_dict()
    assert any(
        (c.args and c.args[0] == "none") for c in mock_get_strategy.mock_calls
    )


@pytest.mark.parametrize(
    "cuda_available, explicit_device, expected",
    [(False, None, "cpu"), (True, None, "cuda"), (True, "cpu", "cpu"), (False, "cuda", "cuda")],
)
@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.predict_single_example", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy", return_value=lambda: SimpleNamespace(name="tta"))
@patch("mist.inference.inference_runners.get_ensembler", return_value=SimpleNamespace(name="ensembler"))
@patch("mist.inference.inference_runners.get_inferer", return_value=(lambda **_: SimpleNamespace(name="inferer")))
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.ants.image_read", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.io.read_json_file")
def test_test_on_fold_device_resolution(
    mock_read_json,
    _mock_image_write,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    _mock_predict_single,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
    cuda_available,
    explicit_device,
    expected,
    noop_cuda_tensor_to,
):
    """Test explicit device overrides; otherwise cuda-to-cpu fallback."""
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 1
    _make_train_df(fold).to_csv(results_dir / "train_paths.csv", index=False)
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    model = _DummyModel()
    mock_load_model.return_value = model

    observed = {}

    class _Pred:
        def __init__(self, **kwargs):
            observed["device"] = kwargs["device"]

    mock_Predictor.side_effect = _Pred

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        training_utils,
        "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )

    ir.test_on_fold(
        mist_args=mist_args, fold_number=fold, device=explicit_device
    )

    assert observed["device"] == expected


@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor", return_value=MagicMock())
@patch("mist.inference.inference_runners.get_strategy", return_value=lambda: SimpleNamespace(name="tta"))
@patch("mist.inference.inference_runners.get_ensembler", return_value=SimpleNamespace(name="ensembler"))
@patch("mist.inference.inference_runners.get_inferer", return_value=(lambda **_: SimpleNamespace(name="inferer")))
@patch("mist.inference.inference_runners.model_loader.load_model_from_config", return_value=_DummyModel())
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.ants.image_read", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.io.read_json_file")
def test_test_on_fold_error_message_collected_and_printed_single(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    _mock_load_model,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    _mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """If prediction fails for a case, the formatted error is printed."""
    # FS layout & inputs (one case).
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 0
    pd.DataFrame([{"id": "pX", "fold": fold, "image": "/tmp/pX.nii.gz"}]).to_csv(
        results_dir / "train_paths.csv", index=False
    )
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    # Cause prediction to fail.
    mock_predict_single.side_effect = RuntimeError("boom")

    # Capture printed output.
    printed = []
    console = MagicMock()
    console.print.side_effect = lambda msg: printed.append(str(msg))
    mock_console_cls.return_value = console

    # Misc patches.
    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )
    monkeypatch.setattr(
        training_utils,
        "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Run.
    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device=None)

    # Assert nothing written and error printed in the expected format.
    mock_image_write.assert_not_called()
    assert any(
        "[red][Error] Prediction failed for pX: boom[/red]" in m
        for m in printed
    )


@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor", return_value=MagicMock())
@patch("mist.inference.inference_runners.get_strategy", return_value=lambda: SimpleNamespace(name="tta"))
@patch("mist.inference.inference_runners.get_ensembler", return_value=SimpleNamespace(name="ensembler"))
@patch("mist.inference.inference_runners.get_inferer", return_value=(lambda **_: SimpleNamespace(name="inferer")))
@patch("mist.inference.inference_runners.model_loader.load_model_from_config", return_value=_DummyModel())
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.ants.image_read", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.io.read_json_file")
def test_test_on_fold_error_messages_multiple_printed(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    _mock_load_model,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    _mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """If multiple cases fail, each formatted error is printed."""
    # Two failing cases.
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 1
    two_df = pd.DataFrame(
        [
            {"id": "pA", "fold": fold, "image": "/tmp/pA.nii.gz"},
            {"id": "pB", "fold": fold, "image": "/tmp/pB.nii.gz"},
        ]
    )
    two_df.to_csv(results_dir / "train_paths.csv", index=False)
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=2)

    # Both predictions raise different exceptions to prove formatting.
    mock_predict_single.side_effect = [
        FileNotFoundError("missing.nii.gz"),
        ValueError("bad shape"),
    ]

    printed = []
    console = MagicMock()
    console.print.side_effect = lambda msg: printed.append(str(msg))
    mock_console_cls.return_value = console

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )
    monkeypatch.setattr(
        training_utils,
        "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device=None)

    # No outputs written and both error lines printed.
    mock_image_write.assert_not_called()
    assert any(
        "[red][Error] Prediction failed for pA: missing.nii.gz[/red]" in m
        for m in printed
    )
    assert any(
        "[red][Error] Prediction failed for pB: bad shape[/red]" in m
        for m in printed
    )


# ===========================
# infer_from_dataframe tests
# ===========================

@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inferers.inferer_registry.get_inferer")
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.inference_utils.validate_inference_images")
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_success_preprocess_path_tta_enabled(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test preprocessing branch (skip=False), TTA enabled, writes output."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])
    img_np = np.zeros((2, 2, 2), dtype=np.float32)
    bbox = {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}
    mock_preprocess_example.return_value = {"image": img_np, "fg_bbox": bbox}

    mock_load_models.return_value = [MagicMock(name="model0")]
    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")
    mock_get_strategy.return_value = lambda: SimpleNamespace(name="tta")
    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device=None,
    )

    mock_load_models.assert_called_once()
    mock_validate_images.assert_called_once()
    mock_preprocess_example.assert_called_once()
    mock_predict_single.assert_called_once()
    mock_image_write.assert_called_once()

    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(os.path.join("out", "p1.nii.gz"))
    assert any(
        c.args == (cfg["inference"]["tta"]["strategy"],)
        for c in mock_get_strategy.mock_calls
    )


@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inferers.inferer_registry.get_inferer")
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.inference_utils.validate_inference_images")
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_success_tta_disabled(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with no preprocessing, TTA disabled, and writes output."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))

    cfg = copy.deepcopy(mock_mist_config)
    cfg["inference"]["tta"]["enabled"] = False

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])
    mock_preprocess_example.return_value = {
        "image": np.zeros((2, 2, 2), dtype=np.float32),
        "fg_bbox": None,
    }

    mock_load_models.return_value = [MagicMock(name="model0")]
    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")

    def _strategy(name):
        return (
            (lambda: SimpleNamespace(name="none"))
            if name == "none"
            else (lambda: SimpleNamespace(name=name))
        )

    mock_get_strategy.side_effect = _strategy
    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device="cpu",
    )

    mock_predict_single.assert_called_once()
    mock_image_write.assert_called_once()
    assert any(
        (c.args and c.args[0] == "none") for c in mock_get_strategy.mock_calls
    )


@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy", return_value=lambda: SimpleNamespace(name="tta"))
@patch("mist.inference.inference_runners.get_ensembler", return_value=SimpleNamespace(name="ensembler"))
@patch("mist.inference.inferers.inferer_registry.get_inferer", return_value=(lambda **_: SimpleNamespace(name="inferer")))
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.inference_utils.validate_inference_images")
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_postprocess_applied_and_messages_printed(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with postprocessing and check messages are printed after run."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    post_file = tmp_path / "post.json"
    post_file.write_text("{}", encoding="utf-8")

    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])
    mock_preprocess_example.return_value = {
        "image": np.zeros((2, 2, 2), dtype=np.float32),
        "fg_bbox": None,
    }

    mock_load_models.return_value = [MagicMock(name="model0")]
    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()

    with patch("mist.inference.inference_runners.Postprocessor") as MockPP:
        pp = MagicMock()
        pp.apply_strategy_to_single_example.return_value = (
            _DummyANTsImage(),
            ["warn: something"],
        )
        MockPP.return_value = pp

        printed = []
        console = MagicMock()
        console.print.side_effect = lambda msg: printed.append(str(msg))
        mock_console_cls.return_value = console

        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory=out_dir,
            mist_configuration=cfg,
            models_directory=models_dir,
            postprocessing_strategy_filepath=str(post_file),
            device="cpu",
        )

        MockPP.assert_called_once()
        pp.apply_strategy_to_single_example.assert_called_once()
        assert any(
            "Inference completed with the following messages:" in m
            for m in printed
        )
        assert any("warn: something" in m for m in printed)
        mock_image_write.assert_called_once()


@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inferers.inferer_registry.get_inferer")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
def test_infer_from_dataframe_postprocess_file_missing_raises(
    mock_console,
    _mock_pb,
    _mock_predictor,
    _mock_get_strategy,
    _mock_get_ensembler,
    _mock_get_inferer,
    _mock_load_models,
    tmp_path,
    mock_mist_config,
):
    """If postprocess strategy file path is provided but missing, raise."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    missing_path = tmp_path / "nope.json"
    cfg = copy.deepcopy(mock_mist_config)

    with pytest.raises(FileNotFoundError):
        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory=out_dir,
            mist_configuration=cfg,
            models_directory=models_dir,
            postprocessing_strategy_filepath=str(missing_path),
            device="cpu",
        )


@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy", return_value=lambda: SimpleNamespace(name="tta"))
@patch("mist.inference.inference_runners.get_ensembler", return_value=SimpleNamespace(name="ensembler"))
@patch("mist.inference.inferers.inferer_registry.get_inferer", return_value=(lambda **_: SimpleNamespace(name="inferer")))
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.inference_utils.validate_inference_images")
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_logs_errors_and_continues_then_summarizes(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """First patient fails; second succeeds; prints error summary."""
    df = pd.DataFrame(
        [
            {"id": "pA", "image": str((tmp_path / "images" / "pA.nii.gz"))},
            {"id": "pB", "image": str((tmp_path / "images" / "pB.nii.gz"))},
        ]
    )
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])
    mock_preprocess_example.return_value = {
        "image": np.zeros((2, 2, 2), dtype=np.float32),
        "fg_bbox": None,
    }
    mock_load_models.return_value = [MagicMock(name="model0")]

    mock_predict_single.side_effect = [RuntimeError("boom"), _DummyANTsImage()]

    printed = []
    console = MagicMock()
    console.print.side_effect = lambda msg: printed.append(str(msg))
    mock_console_cls.return_value = console

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device="cpu",
    )

    assert any("Prediction failed for pA" in m for m in printed)
    assert any(
        "Inference completed with the following messages:" in m for m in printed
    )
    assert mock_image_write.call_count == 1
    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(os.path.join("out", "pB.nii.gz"))


@pytest.mark.parametrize(
    "cuda_available, explicit_device, expected",
    [(False, None, "cpu"), (True, None, "cuda"), (True, "cpu", "cpu"), (False, "cuda", "cuda")],
)
@patch("mist.inference.inference_runners.progress_bar.get_progress_bar", return_value=_PB())
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.predict_single_example", return_value=_DummyANTsImage())
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy", return_value=lambda: SimpleNamespace(name="tta"))
@patch("mist.inference.inference_runners.get_ensembler", return_value=SimpleNamespace(name="ensembler"))
@patch("mist.inference.inferers.inferer_registry.get_inferer", return_value=lambda **_: SimpleNamespace(name="inferer"))
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models", return_value=[MagicMock(name="model0")])
@patch("mist.inference.inference_runners.inference_utils.validate_inference_images", return_value=(_DummyANTsImage(), ["X"]))
@patch(
    "mist.inference.inference_runners.preprocess.preprocess_example",
    return_value={"image": np.zeros((2, 2, 2), dtype=np.float32), "fg_bbox": None},
)
@patch("mist.inference.inference_runners.ants.image_write")
def test_infer_from_dataframe_device_resolution(
    _mock_image_write,
    _mock_preprocess_example,
    _mock_validate_images,
    _mock_load_models,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    _mock_predict_single,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
    cuda_available,
    explicit_device,
    expected,
    noop_cuda_tensor_to,
):
    """Test device selection behavior in infer_from_dataframe."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    observed = {}

    class _Pred:
        def __init__(self, **kwargs):
            observed["device"] = kwargs["device"]

    mock_Predictor.side_effect = _Pred

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device=explicit_device,
    )

    assert observed["device"] == expected
