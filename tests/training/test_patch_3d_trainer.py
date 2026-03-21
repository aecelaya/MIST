"""Unit tests for Patch3DTrainer."""
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any, Dict, Tuple
import pandas as pd
import pytest
import torch
from torch import nn

# MIST imports.
from mist.training.trainers import base_trainer as bt
from mist.data_loading import dali_loader as dl
from mist.training.trainers.patch_3d_trainer import Patch3DTrainer


class DummyIter:
    """Simple DALI-style iterator with .next()[0] and .reset()."""

    def __init__(self, batch: Dict[str, torch.Tensor], steps: int):
        """Initialize the iterator.

        Args:
            batch: Single batch to return each time.
            steps: How many steps before wrapping around.
        """
        self._batch = batch
        self._steps = max(0, int(steps))
        self._i = 0

    def next(self) -> Tuple[Dict[str, torch.Tensor]]:
        """Return a batch and advance the internal counter."""
        if self._i >= self._steps:
            self._i = 0
        self._i += 1
        return (self._batch,)

    def reset(self) -> None:
        """Reset the internal step counter."""
        self._i = 0


class DummyModel(nn.Module):
    """Tiny model that returns dict outputs with 'prediction'."""

    def __init__(self) -> None:
        """Initialize a single linear layer."""
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward that mimics MIST dict outputs."""
        pred = self.fc(x)
        return {"prediction": pred, "deep_supervision": None}


@contextmanager
def no_op_autocast() -> Any:
    """CPU-safe replacement for torch.autocast."""
    yield


class FakeScaler:
    """Fake GradScaler with a toggle for AMP enabled/disabled."""

    def __init__(self, enabled: bool) -> None:
        """Initialize with an enabled flag."""
        self.enabled = enabled
        self._scaled = 0
        self._stepped = 0
        self._updated = 0
        self._unscaled = 0 # NEW: Track unscale calls

    def scale(self, loss: torch.Tensor) -> "FakeScaler":
        """Pretend to scale the loss; return self for chaining .backward()."""
        self._scaled += 1
        self._loss = loss
        return self

    def backward(self) -> None:
        """Call backward on the stored loss."""
        self._loss.backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Mock the unscale_ method."""
        self._unscaled += 1

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Record an optimizer step."""
        optimizer.step()
        self._stepped += 1

    def update(self) -> None:
        """Record an update call."""
        self._updated += 1


@pytest.fixture(autouse=True)
def patch_cuda(monkeypatch):
    """Pretend CUDA exists and make .to(...) a no-op (CPU-only tests)."""
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: True, raising=False
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda _i: None, raising=False
    )
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )


@pytest.fixture
def tmp_pipeline(tmp_path):
    """Create minimal results/numpy directories with required files."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results / "models").mkdir(parents=True)
    (results / "logs").mkdir(parents=True)
    numpy_dir.mkdir(parents=True)

    # train_paths.csv.
    df = pd.DataFrame({"id": ["p0", "p1"], "fold": [0, 1]})
    df.to_csv(results / "train_paths.csv", index=False)

    # Config with sections used by Patch3DTrainer.
    cfg = {
        "dataset_info": {"labels": [0, 1, 2]},
        "spatial_config": {
            "patch_size": [32, 32, 32],
            "target_spacing": [1.0, 1.0, 1.0],
        },
        "model": {
            "architecture": "nnunet",
            "params": {}
        },
        "training": {
            "seed": 42,
            "nfolds": 5,
            "folds": [0, 1, 2, 3, 4],
            "val_percent": 0.0,
            "epochs": 1000,
            "min_steps_per_epoch": 250,
            "batch_size_per_gpu": 2,
            "dali_foreground_prob": 0.6,
            "loss": {
                "name": "dice_ce",
                "composite_loss_weighting": None,
            },
            "optimizer": "adam",
            "learning_rate": 0.001,
            "lr_scheduler": "cosine",
            "l2_penalty": 0.00001,
            "amp": True,
            "oversampling": 0.6,
            "augmentation": {
                "enabled": True,
                "transforms": {
                    "flips": True,
                    "zoom": True,
                    "noise": True,
                    "blur": True,
                    "brightness": True,
                    "contrast": True
                }
            },
            "hardware": {
                "num_gpus": 2,
                "num_cpu_workers": 8,
                "master_addr": "localhost",
                "master_port": 12345,
                "communication_backend": "nccl"
            }
        },
        "inference": {
            "inferer": {
                "name": "sliding_window",
                "params": {
                    "patch_blend_mode": "gaussian",
                    "patch_overlap": 0.5
                }
            },
        }
    }
    (results / "config.json").write_text(json.dumps(cfg))
    return results, numpy_dir


@pytest.fixture
def mist_args(tmp_pipeline):
    """Return a CLI-like namespace for the trainer."""
    results, numpy_dir = tmp_pipeline
    return SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        model=None,
        patch_size=None,
        pocket=False,
        folds=None,
        epochs=None,
        batch_size_per_gpu=None,
        loss=None,
        use_dtms=False,
        composite_loss_weighting=None,
        optimizer=None,
        l2_penalty=None,
        learning_rate=None,
        lr_scheduler=None,
        val_percent=None,
    )


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch, tmp_pipeline):
    """Stub out get_npy_paths and align nfolds with the tiny test dataset."""
    # 1) Make nfolds consistent with train_paths.csv (ids p0,p1 -> folds {0,1}).
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["nfolds"] = 2
    cfg["training"]["folds"] = [0, 1]
    cfg_path.write_text(json.dumps(cfg))

    # 2) Stub training_utils.get_npy_paths so it doesn't require files to exist.
    def fake_get_npy_paths(data_dir, patient_ids, **kwargs):
        base = Path(data_dir).expanduser().resolve()
        return [str((base / f"{pid}.npy").resolve()) for pid in patient_ids]

    monkeypatch.setattr(bt.training_utils, "get_npy_paths", fake_get_npy_paths)


def test_build_dataloaders_passes_expected_args(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test build_dataloaders forwards the correct arguments to dali_loader."""
    # Capture calls.
    captured = {}

    def fake_get_training_dataset(**kwargs: Any) -> DummyIter:
        """Capture training args and return a dummy iterator."""
        captured["train"] = kwargs
        batch = {"image": torch.zeros(1, 4), "label": torch.zeros(1, 4)}
        return DummyIter(batch, steps=2)

    def fake_get_validation_dataset(**kwargs: Any) -> DummyIter:
        """Capture validation args and return a dummy iterator."""
        captured["val"] = kwargs
        batch = {"image": torch.zeros(1, 4), "label": torch.zeros(1, 4)}
        return DummyIter(batch, steps=1)

    monkeypatch.setattr(dl, "get_training_dataset", fake_get_training_dataset)
    monkeypatch.setattr(
        dl, "get_validation_dataset", fake_get_validation_dataset
    )

    # Trainer instance (BaseTrainer __init__ will read config & set folds).
    t = Patch3DTrainer(mist_args)

    # Fold data we pass directly (use keys expected by Patch3DTrainer).
    fold_data = {
        "train_images": ["images/p0.npy", "images/p2.npy"],
        "train_labels": ["labels/p0.npy", "labels/p2.npy"],
        "train_dtms": None,
        "val_images": ["images/p1.npy"],
        "val_labels": ["labels/p1.npy"],
    }

    train_loader, val_loader = t.build_dataloaders(
        fold_data, rank=0, world_size=1
    )
    assert isinstance(train_loader, DummyIter)
    assert isinstance(val_loader, DummyIter)

    # Check a few key args.
    tr = captured["train"]
    assert tr["extract_patches"] is True
    assert tr["batch_size"] == t.config["training"]["batch_size_per_gpu"]
    assert tr["roi_size"] == t.config["spatial_config"]["patch_size"]
    assert tr["rank"] == 0 and tr["world_size"] == 1

    vl = captured["val"]
    assert vl["rank"] == 0 and vl["world_size"] == 1
    assert "image_paths" in vl and "label_paths" in vl


@pytest.mark.parametrize("amp_enabled", [False, True])
def test_training_step_criterion_optimizer_and_scaler(
    tmp_pipeline, mist_args, monkeypatch, amp_enabled,
):
    """Exercise Patch3DTrainer.training_step with/without AMP."""
    t = Patch3DTrainer(mist_args)

    # Model and optimizer.
    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Spy on optimizer.step to ensure it fires exactly once.
    step_calls = {"count": 0}
    _orig_step = opt.step

    def _spy_step(*args, **kwargs):
        step_calls["count"] += 1
        return _orig_step(*args, **kwargs)

    monkeypatch.setattr(opt, "step", _spy_step)

    # Fake criterion that records inputs and returns a simple scalar loss.
    called = {}
    def fake_criterion(*, y_true, y_pred, y_supervision, alpha, dtm):
        called["y_true"] = y_true
        called["y_pred"] = y_pred
        called["y_sup"] = y_supervision
        called["alpha"] = alpha
        called["dtm"] = dtm
        return (y_pred - y_true).pow(2).mean()

    # Scaler and (for AMP) a CPU-safe autocast no-op.
    if amp_enabled:
        scaler = FakeScaler(enabled=True)

        class _NoOpAutocast:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False

        monkeypatch.setattr(torch, "autocast", lambda *a, **k: _NoOpAutocast())
    else:
        scaler = None

    state = {
        "model": model,
        "optimizer": opt,
        "scaler": scaler,
        "loss_function": fake_criterion,
        "composite_loss_weighting": None,
        "epoch": 0,
        "global_step": 0,
    }

    batch = {
        "image": torch.zeros(2, 4, dtype=torch.float32),
        "label": torch.zeros(2, 4, dtype=torch.float32),
        "dtm": None,
    }

    loss = t.training_step(state=state, data=batch)

    # Assertions common to both branches.
    assert isinstance(loss, torch.Tensor)
    assert "y_pred" in called and "y_true" in called

    # FIX: Expect 0.5 (the new default), not None
    assert called["alpha"] == 0.5 

    assert step_calls["count"] == 1

    # Branch-specific assertions.
    if amp_enabled:
        assert isinstance(state["scaler"], FakeScaler)
        assert state["scaler"].enabled is True
        assert state["scaler"]._scaled == 1
        assert state["scaler"]._stepped == 1
        assert state["scaler"]._updated == 1
        assert state["scaler"]._unscaled == 1 
    else:
        assert state["scaler"] is None


@pytest.mark.parametrize("amp_enabled", [False, True])
def test_training_step_sequence_enforcement(
    tmp_pipeline, mist_args, monkeypatch, amp_enabled
):
    """Verify that unscale happens BEFORE clip, and clip BEFORE step."""
    t = Patch3DTrainer(mist_args)
    model = DummyModel()

    # Create a shared event log.
    events = []

    # 1. Mock Optimizer Step.
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(opt, "step", lambda: events.append("opt_step"))

    # 2. Mock Gradient Clipping.
    # We patch the torch.nn.utils function directly.
    monkeypatch.setattr(
        torch.nn.utils, 
        "clip_grad_norm_", 
        lambda params, max_norm: events.append("clip_grad")
    )

    # 3. Enhanced FakeScaler that logs events.
    class EventScaler:
        """Scaler that logs unscale and step events."""
        def __init__(self):
            self.enabled = True
        def scale(self, loss):
            """Scale the loss and return self."""
            return self
        def backward(self):
            """Dummy backward."""
            pass
        def update(self):
            """Dummy update."""
            pass
        def unscale_(self, optimizer):
            """Dummy unscale that logs event."""
            events.append("scaler_unscale")
        def step(self, optimizer):
            """Dummy step that logs event."""
            events.append("scaler_step")
            optimizer.step()

    # Setup environment.
    scaler = EventScaler() if amp_enabled else None

    if amp_enabled:
        class _NoOpAutocast:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        monkeypatch.setattr(torch, "autocast", lambda *a, **k: _NoOpAutocast())

    # Mock Criterion
    def fake_criterion(**kwargs):
        """Dummy criterion that returns a constant loss."""
        return torch.tensor(0.5, requires_grad=True)

    state = {
        "model": model, "optimizer": opt, "scaler": scaler,
        "loss_function": fake_criterion, "composite_loss_weighting": None,
        "epoch": 0
    }
    batch = {"image": torch.randn(1, 4), "label": torch.randn(1, 4)}

    # Run
    t.training_step(state=state, data=batch)

    # ASSERT THE EXACT ORDER.
    if amp_enabled:
        # Expected: Unscale -> Clip -> Scaler Step -> Optimizer Step
        assert events == [
            "scaler_unscale", "clip_grad", "scaler_step", "opt_step"
        ]
    else:
        # Expected: Clip -> Optimizer Step.
        assert events == ["clip_grad", "opt_step"]


@patch("mist.training.trainers.patch_3d_trainer.sliding_window_inference")
def test_validation_step_calls_sliding_window_and_validation_loss(
    mock_swi, tmp_pipeline, mist_args,
):
    """Validate sliding-window inference wiring and validation loss usage.

    Ensures:
      - `sliding_window_inference` is called with expected keyword arguments.
      - The model instance is passed as the `predictor`.
      - The ROI size and overlap come from the trainer config.
      - The trainer's `validation_loss` is called with the prediction.
    """
    t = Patch3DTrainer(mist_args)
    model = DummyModel()

    # Make the patched SWI return a tensor shaped like the inputs.
    mock_swi.side_effect = lambda **kwargs: torch.zeros_like(kwargs["inputs"])

    # Replace the trainer's validation loss with a simple capturing function.
    captured = {"y_true": None, "y_pred": None}

    def validation_loss(y_true, y_pred):
        """Capture arguments and return a scalar tensor."""
        captured["y_true"] = y_true
        captured["y_pred"] = y_pred
        return (y_pred - y_true).pow(2).mean()

    t.validation_loss = validation_loss  # type: ignore[attr-defined]

    state = {"model": model}
    batch = {
        "image": torch.ones(1, 4, dtype=torch.float32),
        "label": torch.zeros(1, 4, dtype=torch.float32),
    }

    # Run validation step.
    val_loss = t.validation_step(state=state, data=batch)

    # Basic assertions.
    assert isinstance(val_loss, torch.Tensor)

    # Check the patched SWI was called exactly once with expected kwargs.
    mock_swi.assert_called_once()
    kwargs = mock_swi.call_args.kwargs
    assert torch.equal(kwargs["inputs"], batch["image"])
    assert kwargs["roi_size"] == t.config["spatial_config"]["patch_size"]
    expected_overlap = (
        t.config["inference"]["inferer"]["params"]["patch_overlap"]
    )
    assert kwargs["overlap"] == pytest.approx(expected_overlap)
    assert kwargs["sw_batch_size"] == 1
    assert kwargs["predictor"] is model
    assert kwargs["device"] == batch["image"].device

    # Ensure validation loss consumed y_true/y_pred.
    assert captured["y_true"] is not None
    assert captured["y_pred"] is not None
