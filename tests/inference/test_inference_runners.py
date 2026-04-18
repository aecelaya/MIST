"""Tests for mist.inference.inference_runners."""
from typing import Optional
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
import copy
import importlib
import os
import sys
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

    def __init__(self, array: Optional[np.ndarray] = None):
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


@pytest.fixture
def fold_runner(mock_mist_config, monkeypatch, tmp_path):
    """Patches all external deps for test_on_fold tests.

    Returns a SimpleNamespace with:
      .run(fold, cfg, device, train_df, bbox_df) — calls ir.test_on_fold
      .image_write — MagicMock for ants.image_write
      .predict_single — MagicMock for predict_single_example
      .printed — list of strings passed to console.print
    """
    results_dir, numpy_dir = _prep_dirs(tmp_path)

    image_write = MagicMock()
    predict_single = MagicMock(return_value=_DummyANTsImage())
    printed = []
    mock_read_json = MagicMock()
    mock_get_test_dataset = MagicMock(return_value=_DummyLoader(n=1))

    monkeypatch.setattr(ir.ants, "image_read", MagicMock(return_value=_DummyANTsImage()))
    monkeypatch.setattr(ir.ants, "image_write", image_write)
    monkeypatch.setattr(ir.progress_bar, "get_progress_bar", MagicMock(return_value=_PB()))
    monkeypatch.setattr("mist.utils.console.console.print", lambda msg: printed.append(str(msg)))
    monkeypatch.setattr(ir, "predict_single_example", predict_single)
    monkeypatch.setattr(ir, "Predictor", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ir, "get_strategy", MagicMock(return_value=lambda: SimpleNamespace(name="tta")))
    monkeypatch.setattr(ir, "get_ensembler", MagicMock(return_value=SimpleNamespace(name="ensembler")))
    monkeypatch.setattr(ir, "get_inferer", MagicMock(return_value=lambda **_: SimpleNamespace(name="inferer")))
    monkeypatch.setattr(ir.model_loader, "load_model_from_config", MagicMock(return_value=_DummyModel()))
    mock_dali_module = MagicMock()
    mock_dali_module.get_test_dataset = mock_get_test_dataset
    monkeypatch.setattr(ir, "dali_loader", mock_dali_module)
    monkeypatch.setattr(ir.io, "read_json_file", mock_read_json)
    monkeypatch.setattr(ir, "ic", SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}))
    monkeypatch.setattr(
        training_utils, "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def run(fold=0, cfg=None, device=None, train_df=None, bbox_df=None):
        cfg = cfg or copy.deepcopy(mock_mist_config)
        mock_read_json.return_value = cfg

        df = train_df if train_df is not None else _make_train_df(fold)
        bb = bbox_df if bbox_df is not None else _make_bbox_df()
        df.to_csv(results_dir / "train_paths.csv", index=False)
        bb.to_csv(results_dir / "fg_bboxes.csv", index=False)
        (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")
        (results_dir / "config.json").write_text("{}", encoding="utf-8")
        mock_get_test_dataset.return_value = _DummyLoader(
            n=len(df[df["fold"] == fold])
        )

        mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
        ir.test_on_fold(mist_args=mist_args, fold_number=fold, device=device)

    return SimpleNamespace(
        run=run,
        results_dir=results_dir,
        image_write=image_write,
        predict_single=predict_single,
        printed=printed,
    )


@pytest.fixture
def infer_runner(mock_mist_config, monkeypatch, tmp_path):
    """Patches all external deps for infer_from_dataframe tests.

    Returns a SimpleNamespace with:
      .run(df, cfg, device, postprocessing_strategy_filepath) — calls ir.infer_from_dataframe
      .image_write — MagicMock for ants.image_write
      .predict_single — MagicMock for predict_single_example
      .validate — MagicMock for inference_utils.validate_inference_images
      .preprocess — MagicMock for preprocess.preprocess_example
      .printed — list of strings passed to console.print
      .out_dir, .models_dir — path strings
    """
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))

    image_write = MagicMock()
    predict_single = MagicMock(return_value=_DummyANTsImage())
    mock_validate = MagicMock(return_value=(_DummyANTsImage(), ["x"]))
    mock_preprocess = MagicMock(return_value={
        "image": np.zeros((2, 2, 2), dtype=np.float32),
        "fg_bbox": None,
    })
    printed = []

    monkeypatch.setattr(ir.ants, "image_write", image_write)
    monkeypatch.setattr(ir.progress_bar, "get_progress_bar", MagicMock(return_value=_PB()))
    monkeypatch.setattr("mist.utils.console.console.print", lambda msg: printed.append(str(msg)))
    monkeypatch.setattr(ir, "predict_single_example", predict_single)
    monkeypatch.setattr(ir, "Predictor", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ir, "get_strategy", MagicMock(return_value=lambda: SimpleNamespace(name="tta")))
    monkeypatch.setattr(ir, "get_ensembler", MagicMock(return_value=SimpleNamespace(name="ensembler")))
    monkeypatch.setattr(ir, "get_inferer", MagicMock(return_value=lambda **_: SimpleNamespace(name="inferer")))
    monkeypatch.setattr(
        ir.inference_utils, "load_test_time_models",
        MagicMock(return_value=[MagicMock(name="model0")]),
    )
    monkeypatch.setattr(ir.inference_utils, "validate_inference_images", mock_validate)
    monkeypatch.setattr(ir.preprocess, "preprocess_example", mock_preprocess)
    monkeypatch.setattr(ir, "ic", SimpleNamespace(
        NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
        NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def run(df=None, cfg=None, device="cpu", postprocessing_strategy_filepath=None):
        df = df if df is not None else _df_single_case(tmp_path)
        cfg = cfg or copy.deepcopy(mock_mist_config)
        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory=out_dir,
            mist_configuration=cfg,
            models_directory=models_dir,
            postprocessing_strategy_filepath=postprocessing_strategy_filepath,
            device=device,
        )

    return SimpleNamespace(
        run=run,
        out_dir=out_dir,
        models_dir=models_dir,
        image_write=image_write,
        predict_single=predict_single,
        validate=mock_validate,
        preprocess=mock_preprocess,
        printed=printed,
    )


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


@patch("mist.inference.inference_utils.back_to_original_space")
def test_predict_single_example_skip_true_bypasses_spatial_restore(
    mock_back_to_original_space,
    mock_mist_config,
    monkeypatch,
):
    """When skip=True, back_to_original_space is NOT called.

    skip=True means images were read as-is with no spatial transforms.
    The prediction is already in the original image's voxel space, so we
    copy the original header directly via new_image_like — no reorient,
    no resample, no back_to_original_space.
    """
    cfg = copy.deepcopy(mock_mist_config)
    cfg["model"]["params"]["out_channels"] = 2
    cfg["preprocessing"]["skip"] = True
    cfg["preprocessing"]["crop_to_foreground"] = False
    cfg["dataset_info"]["labels"] = [0, 1]

    monkeypatch.setattr(
        ir, "ic", SimpleNamespace(ARGMAX_AXIS=1, BATCH_AXIS=0), raising=False
    )

    pre_img = torch.randn(1, 1, 2, 2, 2)
    orig_ants = _DummyANTsImage(np.zeros((2, 2, 2), dtype=np.int64))

    out = ir.predict_single_example(
        preprocessed_image=pre_img,
        original_ants_image=orig_ants,
        mist_configuration=cfg,
        predictor=_predictor_logits_two_class,
        foreground_bounding_box=None,
    )

    # back_to_original_space must NOT be called when skip=True.
    mock_back_to_original_space.assert_not_called()

    # new_image_like must have been called to copy the original header.
    assert orig_ants.new_like_last_data is not None
    assert out.astype_arg == "uint8"


# ==================
# test_on_fold tests
# ==================

def test_test_on_fold_raises_if_dali_not_available(
    monkeypatch, tmp_path, mock_mist_config
):
    """test_on_fold raises RuntimeError when dali_loader is None (DALI not installed)."""
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    _make_train_df(0).to_csv(results_dir / "train_paths.csv", index=False)
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ir, "dali_loader", None)
    monkeypatch.setattr(ir.io, "read_json_file", lambda _: copy.deepcopy(mock_mist_config))
    monkeypatch.setattr(
        training_utils, "get_npy_paths",
        lambda data_dir, patient_ids: [
            os.path.join(str(data_dir), f"{pid}.npy") for pid in patient_ids
        ],
    )

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    with pytest.raises(RuntimeError, match="NVIDIA DALI is required"):
        ir.test_on_fold(mist_args=mist_args, fold_number=0, device=torch.device("cpu"))


def test_dali_import_error_sets_dali_loader_none():
    """inference_runners.dali_loader is None when the DALI import fails."""
    import mist.data_loading
    import mist.inference.inference_runners as runners_mod

    original_sys = sys.modules.get("mist.data_loading.dali_loader")
    original_attr = getattr(mist.data_loading, "dali_loader", None)

    try:
        # Setting sys.modules entry to None causes ImportError on import.
        sys.modules["mist.data_loading.dali_loader"] = None  # type: ignore[assignment]
        if hasattr(mist.data_loading, "dali_loader"):
            delattr(mist.data_loading, "dali_loader")
        importlib.reload(runners_mod)
        assert runners_mod.dali_loader is None
    finally:
        # Restore sys.modules and the package attribute.
        if original_sys is None:
            sys.modules.pop("mist.data_loading.dali_loader", None)
        else:
            sys.modules["mist.data_loading.dali_loader"] = original_sys
        if original_attr is not None:
            mist.data_loading.dali_loader = original_attr
        importlib.reload(runners_mod)


def test_test_on_fold_success_no_crop_tta_enabled(fold_runner, mock_mist_config):
    """One case, no cropping, TTA enabled — output written to correct path."""
    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = False
    cfg["inference"]["tta"]["enabled"] = True

    fold_runner.run(fold=0, cfg=cfg)

    fold_runner.predict_single.assert_called_once()
    fold_runner.image_write.assert_called_once()
    out_path = fold_runner.image_write.call_args.args[1]
    assert out_path.endswith(
        os.path.join("predictions", "train", "raw", "p1.nii.gz")
    )


def test_test_on_fold_crop_to_foreground_bbox_passed(fold_runner, mock_mist_config):
    """crop_to_foreground=True: bbox from CSV is forwarded to prediction."""
    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = True
    cfg["preprocessing"]["skip"] = False
    cfg["inference"]["tta"]["enabled"] = False

    bbox_df = _make_bbox_df()
    fold_runner.run(fold=2, cfg=cfg, bbox_df=bbox_df)

    call_kwargs = fold_runner.predict_single.call_args.kwargs
    assert "foreground_bounding_box" in call_kwargs
    assert call_kwargs["foreground_bounding_box"] == bbox_df.iloc[0].to_dict()


def test_test_on_fold_error_message_collected_and_printed_single(
    fold_runner, mock_mist_config
):
    """If prediction fails for a case, the formatted error is printed."""
    fold_runner.predict_single.side_effect = RuntimeError("boom")

    train_df = pd.DataFrame(
        [{"id": "pX", "fold": 0, "image": "/tmp/pX.nii.gz"}]
    )
    fold_runner.run(fold=0, train_df=train_df)

    fold_runner.image_write.assert_not_called()
    assert any(
        "Prediction failed for pX: boom" in m
        for m in fold_runner.printed
    )


def test_test_on_fold_error_messages_multiple_printed(
    fold_runner, mock_mist_config
):
    """If multiple cases fail, each formatted error is printed."""
    fold_runner.predict_single.side_effect = [
        FileNotFoundError("missing.nii.gz"),
        ValueError("bad shape"),
    ]

    train_df = pd.DataFrame([
        {"id": "pA", "fold": 1, "image": "/tmp/pA.nii.gz"},
        {"id": "pB", "fold": 1, "image": "/tmp/pB.nii.gz"},
    ])
    fold_runner.run(fold=1, train_df=train_df)

    fold_runner.image_write.assert_not_called()
    assert any(
        "Prediction failed for pA: missing.nii.gz" in m
        for m in fold_runner.printed
    )
    assert any(
        "Prediction failed for pB: bad shape" in m
        for m in fold_runner.printed
    )


# ===========================
# infer_from_dataframe tests
# ===========================

def test_infer_from_dataframe_success_preprocess_path_tta_enabled(
    infer_runner, mock_mist_config
):
    """Preprocessing branch (skip=False), TTA enabled — output written."""
    infer_runner.run(cfg=copy.deepcopy(mock_mist_config))

    infer_runner.image_write.assert_called_once()
    out_path = infer_runner.image_write.call_args.args[1]
    assert out_path.endswith(os.path.join("out", "p1.nii.gz"))


def test_infer_from_dataframe_success_tta_disabled(infer_runner, mock_mist_config):
    """TTA disabled — output is still written."""
    cfg = copy.deepcopy(mock_mist_config)
    cfg["inference"]["tta"]["enabled"] = False

    infer_runner.run(cfg=cfg)

    infer_runner.predict_single.assert_called_once()
    infer_runner.image_write.assert_called_once()


def test_infer_from_dataframe_postprocess_applied_and_messages_printed(
    infer_runner, mock_mist_config, tmp_path
):
    """With postprocessing, warnings from the postprocessor are printed."""
    post_file = tmp_path / "post.json"
    post_file.write_text("{}", encoding="utf-8")

    with patch("mist.inference.inference_runners.Postprocessor") as MockPP:
        pp = MagicMock()
        pp.apply_strategy_to_single_example.return_value = (
            _DummyANTsImage(),
            ["warn: something"],
        )
        MockPP.return_value = pp

        infer_runner.run(postprocessing_strategy_filepath=str(post_file))

        MockPP.assert_called_once()
        pp.apply_strategy_to_single_example.assert_called_once()
        assert any(
            "Inference completed with the following messages:" in m
            for m in infer_runner.printed
        )
        assert any("warn: something" in m for m in infer_runner.printed)
        infer_runner.image_write.assert_called_once()


def test_infer_from_dataframe_postprocess_file_missing_raises(
    infer_runner, tmp_path
):
    """If postprocess strategy file path is provided but missing, raise."""
    with pytest.raises(FileNotFoundError):
        infer_runner.run(
            postprocessing_strategy_filepath=str(tmp_path / "nope.json")
        )


def test_infer_from_dataframe_logs_errors_and_continues_then_summarizes(
    infer_runner, mock_mist_config, tmp_path
):
    """First patient fails; second succeeds; prints error summary."""
    df = pd.DataFrame([
        {"id": "pA", "image": str((tmp_path / "images" / "pA.nii.gz"))},
        {"id": "pB", "image": str((tmp_path / "images" / "pB.nii.gz"))},
    ])
    infer_runner.predict_single.side_effect = [
        RuntimeError("boom"), _DummyANTsImage()
    ]

    infer_runner.run(df=df)

    assert any("Prediction failed for pA" in m for m in infer_runner.printed)
    assert any(
        "Inference completed with the following messages:" in m
        for m in infer_runner.printed
    )
    assert infer_runner.image_write.call_count == 1
    out_path = infer_runner.image_write.call_args.args[1]
    assert out_path.endswith(os.path.join("out", "pB.nii.gz"))


@pytest.mark.parametrize(
    "cuda_available, explicit_device, expected",
    [(False, None, "cpu"), (True, None, "cuda"), (True, "cpu", "cpu"), (False, "cuda", "cuda")],
)
def test_infer_from_dataframe_device_resolution(
    infer_runner,
    mock_mist_config,
    monkeypatch,
    cuda_available,
    explicit_device,
    expected,
    noop_cuda_tensor_to,
):
    """Device selection: explicit device overrides; otherwise cuda-to-cpu fallback."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    observed = {}

    class _Pred:
        def __init__(self, **kwargs):
            observed["device"] = kwargs["device"]

    monkeypatch.setattr(ir, "Predictor", _Pred)

    infer_runner.run(cfg=copy.deepcopy(mock_mist_config), device=explicit_device)

    assert observed["device"] == expected
