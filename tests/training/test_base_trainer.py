"""Tests for the BaseTrainer implementation."""
import json
import os
import math
import pickle
import rich
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import pandas as pd
import pytest
import torch
from torch import nn

# MIST imports.
from mist.training.trainers import base_trainer as bt

# Setup base trainer for tests.
BaseTrainer = bt.BaseTrainer


class DummyIter:
    """DALI-style loader with .next()[0] and .reset()."""

    def __init__(self, batch, length: int):
        self._batch = batch
        self._length = max(0, length)
        self._i = 0

    def next(self):
        """Return the next batch, cycling through."""
        if self._i >= self._length:
            self._i = 0
        self._i += 1
        return (self._batch,)

    def reset(self):
        """Reset the iterator to the start."""
        self._i = 0


class DummyModel(nn.Module):
    """Dummy model with a single linear layer for testing."""

    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2, 2)

    def forward(self, x):
        """Forward pass through the dummy linear layer."""
        return self.l(x)


class DummyLoss(nn.Module):
    """Dummy loss that returns a constant value."""

    def __init__(self, value: float = 1.0, **kwargs):
        super().__init__()  # <-- important
        self.value = value

    def forward(self, y_true, y_pred):
        """Return a constant loss value."""
        # Ensure it's a proper tensor on the same device as predictions.
        dev = y_pred.device if isinstance(y_pred, torch.Tensor) else "cpu"
        return torch.tensor(self.value, device=dev, dtype=torch.float32)


class DummyDDP(nn.Module):
    """Dummy DDP wrapper that does nothing but pass through the"""

    def __init__(self, module, device_ids=None, find_unused_parameters=False):
        super().__init__()
        self.module = module

    def forward(self, *a, **k):
        """Forward pass that calls the wrapped module."""
        return self.module(*a, **k)


class DummySummaryWriter:
    """Dummy SummaryWriter that collects scalars in memory."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.closed = False

    def add_scalars(self, tag, scalars, step):
        """Collect scalars in memory."""
        self.scalars.append((tag, dict(scalars), int(step)))

    def flush(self):
        """Flush the collected scalars."""
        pass

    def close(self):
        """Close the writer, marking it as closed."""
        self.closed = True


class DummyProgressCtx:
    """Dummy context manager for progress bars that does nothing."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, **kwargs):
        pass


class DummyTrainer(BaseTrainer):
    """Concrete subclass providing minimal train/val steps and loaders."""

    def __init__(self, mist_args, train_loss_value=1.0, val_loss_value=2.0):
        self._train_loss_value = train_loss_value
        self._val_loss_value = val_loss_value
        super().__init__(mist_args)

    def build_dataloaders(self, fold_data, rank, world_size):
        """Build dummy dataloaders for training and validation."""
        train_len = int(fold_data["steps_per_epoch"])
        val_len = max(
            1, math.ceil(len(fold_data["val_images"]) / max(1, world_size))
        )
        batch = torch.zeros(1, 2)
        return DummyIter(batch, train_len), DummyIter(batch, val_len)

    def training_step(self, **kwargs):
        """Perform a dummy training step."""
        state = kwargs["state"]
        dev = next(state["model"].parameters()).device
        return torch.tensor(self._train_loss_value, device=dev)

    def validation_step(self, **kwargs):
        """Perform a dummy validation step."""
        state = kwargs["state"]
        dev = next(state["model"].parameters()).device
        return torch.tensor(self._val_loss_value, device=dev)


@pytest.fixture(autouse=True)
def patch_paths(monkeypatch):
    """Patch training_utils.get_npy_paths so tests don't hit the FS."""
    def fake_get_npy_paths(data_dir, patient_ids, **kwargs):
        base = Path(data_dir).resolve()
        # Return absolute-looking paths; existence isn't required by the tests.
        return [str((base / f"{pid}.npy").resolve()) for pid in patient_ids]

    monkeypatch.setattr(bt.training_utils, "get_npy_paths", fake_get_npy_paths)
    # JSON read/write pass-through is fine.


@pytest.fixture(autouse=True)
def patch_cuda_and_moves(monkeypatch):
    """Patch CUDA availability and device moves so tests run CPU-only."""
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )

    # Make .to(device) a no-op so models stay on CPU.
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    # Inside the module under test, make torch.tensor(...) ignore device kwarg.
    _orig_tensor = torch.tensor

    def _cpu_tensor(*args, **kwargs):
        """Create a tensor on CPU, ignoring device."""
        kwargs.pop("device", None)
        return _orig_tensor(*args, **kwargs)

    monkeypatch.setattr(bt.torch, "tensor", _cpu_tensor, raising=False)


@pytest.fixture
def tmp_pipeline(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a temporary results and numpy directory with required files."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results / "models").mkdir(parents=True)
    (results / "logs").mkdir(parents=True)
    numpy_dir.mkdir(parents=True)

    # Minimal train_paths.csv with fold assignments.
    df = pd.DataFrame({"id": ["p0", "p1", "p2", "p3"], "fold": [0, 1, 0, 1]})
    df.to_csv(results / "train_paths.csv", index=False)

    config = {
        "spatial_config": {
            "patch_size": [16, 16, 16],
            "target_spacing": [1.0, 1.0, 1.0],
        },
        "model": {
            "architecture": "dummy",
            "params": {}
        },
        "preprocessing": {},
        "training": {
            "nfolds": 2,
            "folds": [0],
            "epochs": 1,
            "batch_size_per_gpu": 1,
            "learning_rate": 0.01,
            "hardware": {
                "master_addr": "127.0.0.1",
                "master_port": 12355,
                "communication_backend": "nccl",
                "num_gpus": 1,
            },
            "min_steps_per_epoch": 2,
            "val_percent": 0.0,
            "seed": 0,
            "loss": {
                "name": "dummy_loss",
                "params": {
                    "use_dtms": False,
                    "composite_loss_weighting": None,
                },
            },
            "optimizer": "sgd",
            "l2_penalty": 0.0,
            "lr_scheduler": "constant",
            "amp": False,
        },
        "inference": {
            "inferer": {
                "name": "sliding_window",
                "params": {
                    "patch_blend_mode": "gaussian",
                    "patch_overlap": 0.5,
                },
            },
            "ensemble": {
                "strategy": "mean",
            },
            "tta": {
                "enabled": True,
                "strategy": "all_flips",
            },
        }
    }
    (results / "config.json").write_text(json.dumps(config))

    return results, numpy_dir


@pytest.fixture
def mist_args(tmp_pipeline):
    """Create a mist_args object with minimal configuration."""
    results, numpy_dir = tmp_pipeline
    return SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        model=None,
        patch_size=None,
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
def patch_path_resolver(monkeypatch):
    """Patch BaseTrainer's path builder to avoid real filesystem checks."""
    def fake_get_paths(data_dir, patient_ids):
        base = Path(data_dir)
        return [str(base / f"{pid}.npy") for pid in patient_ids]

    # Important: keep it a staticmethod on the class
    monkeypatch.setattr(
        bt.BaseTrainer,
        "_get_numpy_file_paths_list",
        staticmethod(fake_get_paths),
        raising=False,
    )
    # We leave JSON read/write alone.


@pytest.fixture(autouse=True)
def patch_registries(monkeypatch):
    """Patch registries: model, loss, optimizer, and LR scheduler."""
    monkeypatch.setattr(bt, "get_model_from_registry", lambda arch, **p: DummyModel())
    monkeypatch.setattr(bt, "get_loss", lambda name: DummyLoss)
    monkeypatch.setattr(bt, "get_alpha_scheduler", lambda cfg: object())

    def fake_get_optimizer(
        name, params, weight_decay, eps, learning_rate=None, l2_penalty=None
    ):
        """Fake optimizer that returns a dummy SGD."""
        lr = 0.1 if learning_rate is None else float(learning_rate)
        wd = float(weight_decay if l2_penalty is None else l2_penalty)
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)

    monkeypatch.setattr(bt, "get_optimizer", fake_get_optimizer)

    class DummyScheduler:
        """Dummy scheduler that does nothing."""

        def step(self):
            """Dummy step method that does nothing."""
            pass

    monkeypatch.setattr(
        bt, "get_lr_scheduler", lambda name, optimizer, epochs: DummyScheduler()
    )


@pytest.fixture(autouse=True)
def patch_ddp_and_tb_and_save(monkeypatch):
    """Patch DDP, TensorBoard, and torch.save to use dummy classes."""
    monkeypatch.setattr(bt, "DDP", DummyDDP)
    monkeypatch.setattr(bt, "SummaryWriter", DummySummaryWriter)
    monkeypatch.setattr(bt.progress_bar, "TrainProgressBar", DummyProgressCtx)
    monkeypatch.setattr(
        bt.progress_bar, "ValidationProgressBar", DummyProgressCtx
    )
    monkeypatch.setattr(torch, "save", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def patch_dist(monkeypatch):
    """Patch torch.distributed to be inert but count calls."""
    calls = {
        "init": 0, "destroy": 0, "all_reduce": 0, "broadcast": 0, "barrier": 0
    }

    class FakeDist:
        """Fake distributed module to track calls."""
        _initialized = False
        _rank = 0
        _world_size = 1

        @staticmethod
        def is_available():
            """Check if distributed is available."""
            return True

        @staticmethod
        def is_initialized():
            """Check if distributed is initialized."""
            return FakeDist._initialized

        @staticmethod
        def get_rank():
            """Get the current process rank."""
            return FakeDist._rank

        @staticmethod
        def get_world_size():
            """Get the total number of processes."""
            return FakeDist._world_size

        @staticmethod
        def init_process_group(backend, rank, world_size):
            """Initialize the process group."""
            calls["init"] += 1
            FakeDist._initialized = True
            FakeDist._rank = int(rank)
            FakeDist._world_size = int(world_size)

        @staticmethod
        def destroy_process_group():
            """Destroy the process group."""
            calls["destroy"] += 1
            FakeDist._initialized = False

        @staticmethod
        def all_reduce(t, op=None):
            """Perform all-reduce operation."""
            calls["all_reduce"] += 1

        @staticmethod
        def broadcast(t, src):
            """Broadcast tensor to all processes."""
            calls["broadcast"] += 1

        @staticmethod
        def barrier():
            """Synchronize all processes."""
            calls["barrier"] += 1

        ReduceOp = SimpleNamespace(SUM=0)

    monkeypatch.setattr(bt, "dist", FakeDist)
    return calls


def test_getstate_drops_console(tmp_pipeline, mist_args, monkeypatch):
    """__getstate__ should drop/neutralize the unpicklable console attribute."""
    trainer = DummyTrainer(mist_args)
    # Sanity: console is created during __init__.
    assert trainer.console is not None

    state = trainer.__getstate__()
    # console must be removed/neutralized for pickling.
    assert "console" in state
    assert state["console"] is None

    # Ensure other representative fields survive.
    assert "config" in state
    assert isinstance(state["config"], dict)
    assert "training" in state["config"]


def test_pickling_roundtrip_recreates_console(tmp_pipeline, mist_args):
    """Pickle/unpickle round-trip should recreate a proper rich Console."""
    trainer = DummyTrainer(mist_args)

    # Round-trip through pickle (simulating mp.spawn serialization).
    blob = pickle.dumps(trainer)
    restored = pickle.loads(blob)

    # Console must be reconstructed on load.
    assert restored.console is not None
    assert isinstance(restored.console, rich.console.Console)

    # And core state should be preserved.
    assert restored.config == trainer.config
    assert restored.batch_size == trainer.batch_size
    assert type(restored) is type(trainer)


def test_setstate_respects_non_none_console(tmp_pipeline, mist_args):
    """__setstate__ should NOT overwrite an existing non-None console."""
    trainer = DummyTrainer(mist_args)

    # Build a fake state that already carries a (non-None) console sentinel.
    sentinel_console = object()
    state = trainer.__dict__.copy()
    state["console"] = sentinel_console

    # Apply setstate on an already-constructed object.
    trainer.__setstate__(state)

    # Because console in state was not None, it must be preserved.
    assert trainer.console is sentinel_console


def test_update_num_gpus_and_batchsize(tmp_pipeline, mist_args, monkeypatch):
    """Test that num_gpus and batch size are set correctly."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    assert trainer.config["training"]["hardware"]["num_gpus"] == 1
    assert (
        trainer.batch_size ==
        trainer.config["training"]["batch_size_per_gpu"] * 1
    )


def test_setup_folds_no_valsplit(tmp_pipeline, mist_args, monkeypatch):
    """Test setup_folds with no validation split."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    fold0 = trainer.folds[0]
    expected = max(
        trainer.config["training"]["min_steps_per_epoch"],
        math.ceil(len(fold0["train_images"]) / max(1, trainer.batch_size)),
    )
    assert fold0["steps_per_epoch"] == expected
    assert fold0["train_dtms"] is None


def test_setup_folds_with_dtms_and_valsplit(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test setup_folds with DTM data and validation split."""
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"]["loss"]["params"]["use_dtms"] = True
    cfg["training"]["val_percent"] = 0.5
    (Path(results) / "config.json").write_text(json.dumps(cfg))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    fold0 = trainer.folds[0]
    assert isinstance(fold0["train_dtms"], list)
    assert len(fold0["val_images"]) > 0


def test_build_components_single_gpu(tmp_pipeline, mist_args, monkeypatch):
    """Test build_components with single GPU, no DDP."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    assert isinstance(state["model"], DummyModel)


def test_build_components_multi_gpu_wraps_with_ddp(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test build_components with multiple GPUs, DDP wrapping."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"]["amp"] = True
    (Path(results) / "config.json").write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=2)
    assert isinstance(state["model"], DummyDDP)
    assert state["scaler"].is_enabled() is True


def test_setup_initializes_process_group_once(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Ensure process group is initialized only once."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    trainer = DummyTrainer(mist_args)
    trainer.setup(rank=0, world_size=2)
    trainer.setup(rank=0, world_size=2)
    assert patch_dist["init"] == 1


def test_train_fold_runs_full_epoch(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Test that train_fold runs a full epoch with correct steps."""
    """With world_size=1, no collectives are called in the new code path."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # New behavior: no DDP => no collectives.
    assert patch_dist["broadcast"] == 0
    assert patch_dist["barrier"] == 0
    assert patch_dist["all_reduce"] == 0


def test_train_fold_early_stop_on_nan(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """NaN training loss should trigger early stop and cleanup (DDP case)."""
    # Use 2 GPUs so DDP is engaged, ensuring cleanup() destroys the process
    # group.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    trainer = DummyTrainer(
        mist_args, train_loss_value=float("nan"), val_loss_value=2.0
    )
    trainer.train_fold(fold=0, rank=0, world_size=2)
    assert patch_dist["destroy"] >= 1


def test_overwrite_config_from_args(tmp_pipeline, mist_args, monkeypatch):
    """Test that mist_args overwrite config values correctly."""
    mist_args.model = "myarch"
    mist_args.patch_size = [32, 32, 32]
    mist_args.folds = [0]
    mist_args.epochs = 3
    mist_args.batch_size_per_gpu = 2
    mist_args.loss = "my_loss"
    mist_args.use_dtms = True
    mist_args.composite_loss_weighting = "linear"
    mist_args.optimizer = "adamw"
    mist_args.l2_penalty = 0.01
    mist_args.learning_rate = 0.005
    mist_args.lr_scheduler = "cosine"
    mist_args.val_percent = 0.025

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)

    cfg = trainer.config
    assert cfg["model"]["architecture"] == "myarch"
    assert cfg["spatial_config"]["patch_size"] == [32, 32, 32]
    assert cfg["training"]["epochs"] == 3
    assert cfg["training"]["batch_size_per_gpu"] == 2
    assert cfg["training"]["loss"]["name"] == "my_loss"
    assert cfg["training"]["loss"]["params"]["use_dtms"] is True
    assert (
        cfg["training"]["loss"]["params"]["composite_loss_weighting"] ==
        "linear"
    )
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["training"]["l2_penalty"] == pytest.approx(0.01)
    assert cfg["training"]["learning_rate"] == pytest.approx(0.005)
    assert cfg["training"]["lr_scheduler"] == "cosine"
    assert cfg["training"]["val_percent"] == pytest.approx(0.025)


def test_fit_single_gpu_calls_run_directly(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test that fit with single GPU calls run directly."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    called = {"run": 0}

    def spy_run(rank, world_size):
        assert rank == 0 and world_size == 1
        called["run"] += 1

    trainer = DummyTrainer(mist_args)
    trainer.run_cross_validation = spy_run
    trainer.fit()
    assert called["run"] == 1


def test_fit_multi_gpu_uses_spawn(tmp_pipeline, mist_args, monkeypatch):
    """Test that fit with multiple GPUs uses spawn."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    spawned = {"count": 0}

    def fake_spawn(fn, args, nprocs, join):
        assert nprocs == 2
        assert isinstance(args, tuple) and args[0] == 2
        spawned["count"] += 1

    monkeypatch.setattr(bt.mp, "spawn", fake_spawn)
    trainer = DummyTrainer(mist_args)
    trainer.fit()
    assert spawned["count"] == 1


def test_invalid_folds_subset_raises(tmp_pipeline, mist_args):
    """Test that invalid folds subset raises ValueError."""
    results, _ = tmp_pipeline

    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["nfolds"] = 2
    cfg_path.write_text(json.dumps(cfg))

    mist_args.folds = [0, 2]

    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)

    msg = str(excinfo.value)
    assert "subset of [0, 1, ..., nfolds-1]" in msg
    assert "Found folds: [0, 2]" in msg


def test_update_num_gpus_raises_when_cuda_unavailable(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test that update_num_gpus raises when CUDA is unavailable."""
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: False, raising=False
    )

    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)

    msg = str(excinfo.value)
    assert "CUDA is not available" in msg


def test_update_num_gpus_raises_when_zero_devices(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test that update_num_gpus raises when device_count is zero."""
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0, raising=False)

    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)

    msg = str(excinfo.value)
    assert "device_count() == 0" in msg
    assert "CUDA_VISIBLE_DEVICES" in msg


def test_update_num_gpus_sets_config_and_persists(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test that update_num_gpus sets config and persists to disk."""
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)

    results, _ = tmp_pipeline
    trainer = DummyTrainer(mist_args)

    assert trainer.config["training"]["hardware"]["num_gpus"] == 2

    cfg_path = Path(results) / "config.json"
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["training"]["hardware"]["num_gpus"] == 2


def test_train_fold_raises_when_val_images_less_than_world_size(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test that train_fold raises when val_images < world_size."""
    trainer = DummyTrainer(mist_args)

    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "build_components", lambda *a, **k: None)

    trainer.folds = [
        {
            "val_images": ["val_img_0"],  # Length 1.
            "train_images": ["train_img_0", "train_img_1"],
            "steps_per_epoch": 1,
        }
    ]

    with pytest.raises(ValueError) as excinfo:
        trainer.train_fold(fold=0, rank=0, world_size=2)

    msg = str(excinfo.value)
    assert "Not enough validation data" in msg
    assert "reduce the number of GPUs" in msg


def test_train_loop_else_branch_rank_nonzero(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Test train_fold else branch for rank > 0."""
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.folds[0]["steps_per_epoch"] = 2
    trainer.folds[0]["val_images"] = ["val0", "val1"]
    trainer.folds[0]["train_images"] = ["tr0", "tr1", "tr2"]

    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    trainer.train_fold(fold=0, rank=1, world_size=2)

    assert patch_dist["all_reduce"] >= 3
    assert patch_dist["broadcast"] >= 1
    assert patch_dist["barrier"] >= 1


def test_validation_else_branch_rank_nonzero(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Test validation step else branch for rank > 0."""
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)

    trainer.folds[0]["steps_per_epoch"] = 0
    trainer.folds[0]["val_images"] = ["val0", "val1"]
    trainer.folds[0]["train_images"] = ["tr0", "tr1"]

    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    before = patch_dist["all_reduce"]
    trainer.train_fold(fold=0, rank=1, world_size=2)
    after = patch_dist["all_reduce"]

    assert (after - before) >= 1
    assert patch_dist["broadcast"] >= 1
    assert patch_dist["barrier"] >= 1


def test_validation_no_improvement_message(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test that validation step logs no improvement message."""
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    trainer = DummyTrainer(
        mist_args, train_loss_value=1.0, val_loss_value=float("inf")
    )

    trainer.folds[0]["steps_per_epoch"] = 1
    trainer.folds[0]["val_images"] = ["val0", "val1"]

    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    buf = StringIO()
    monkeypatch.setattr(
        trainer.console, "print", lambda msg: buf.write(str(msg))
    )

    trainer.train_fold(fold=0, rank=0, world_size=1)

    output = buf.getvalue()
    assert "Validation loss did not improve" in output


def test_run_cross_validation_rank0_prints_and_calls_all_folds(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test run_cross_validation on rank 0 prints and calls all folds."""
    results, _ = tmp_pipeline

    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["folds"] = [0, 1]
    cfg_path.write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)

    out = []
    monkeypatch.setattr(
        trainer.console, "print", lambda msg: out.append(str(msg))
    )

    calls = []

    def spy_train_fold(fold, rank, world_size):
        calls.append((fold, rank, world_size))

    monkeypatch.setattr(trainer, "train_fold", spy_train_fold)

    trainer.run_cross_validation(rank=0, world_size=2)

    assert any("Starting training" in s for s in out)
    assert calls == [(0, 0, 2), (1, 0, 2)]


def test_run_cross_validation_nonzero_rank_no_print_but_calls_folds(
    tmp_pipeline, mist_args, monkeypatch
):
    """Test run_cross_validation on non-zero rank does not print but calls."""
    results, _ = tmp_pipeline

    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["folds"] = [0, 1]
    cfg_path.write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)

    out = []
    monkeypatch.setattr(
        trainer.console, "print", lambda msg: out.append(str(msg))
    )

    calls = []

    def spy_train_fold(fold, rank, world_size):
        calls.append((fold, rank, world_size))

    monkeypatch.setattr(trainer, "train_fold", spy_train_fold)

    trainer.run_cross_validation(rank=1, world_size=2)

    assert not any("Starting training" in s for s in out)
    assert calls == [(0, 1, 2), (1, 1, 2)]


@pytest.mark.parametrize("schedule_cfg", [None, "linear"])
def test_build_components_composite_loss_scheduler(
    tmp_pipeline, mist_args, monkeypatch, schedule_cfg
):
    """Test that build_components sets composite_loss_weighting correctly."""
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["loss"]["params"]["composite_loss_weighting"] = schedule_cfg
    cfg_path.write_text(json.dumps(cfg))

    calls = {"args": []}
    sentinel = object()

    def spy_get_alpha_scheduler(schedule, **kwargs):
        # record both the schedule and any named kwargs (e.g., num_epochs)
        calls["args"].append({"schedule": schedule, **kwargs})
        return sentinel

    monkeypatch.setattr(bt, "get_alpha_scheduler", spy_get_alpha_scheduler)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    if schedule_cfg is None:
        assert state["composite_loss_weighting"] is None
        assert not calls["args"]
    else:
        assert state["composite_loss_weighting"] is sentinel
        assert calls["args"][0]["schedule"] == schedule_cfg
        # Optional sanity checks (keep or drop as you like):
        assert "num_epochs" in calls["args"][0]
        assert isinstance(calls["args"][0]["num_epochs"], int)


def test_set_seed_swallows_dist_errors(tmp_pipeline, mist_args, monkeypatch):
    """_set_seed should ignore exceptions from torch.distributed calls."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)

    class BadDist:
        """Fake distributed module that raises in is_initialized()."""

        def is_initialized(self):
            """Raise an error to simulate a bad distributed state."""
            raise RuntimeError("kaboom")

    # Override the module's dist with one that raises inside is_initialized().
    monkeypatch.setattr(bt, "dist", BadDist())

    # Ensure default env path (rank from env not used).
    os.environ.pop("RANK", None)

    # Should NOT raise despite the deliberate exception.
    trainer._set_seed(123)

    # Verify we proceeded past the except: env var was set.
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_validation_rank0_ddp_allreduce_and_mean(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """With DDP, rank 0 validation uses all_reduce and divides by world_size."""
    # Enable multi-GPU (DDP path).
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)

    # Capture the SummaryWriter instance used during training to inspect logged
    # scalars.
    created_writers = []

    class CapturingWriter(DummySummaryWriter):
        def __init__(self, log_dir):
            super().__init__(log_dir)
            created_writers.append(self)

    # Override the writer for this test (autouse fixture sets a default; we
    # override it here).
    monkeypatch.setattr(bt, "SummaryWriter", CapturingWriter)

    # Build trainer with a known validation loss.
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=4.0)

    # Run one epoch on rank 0 with world_size=2 (DDP enabled).
    trainer.train_fold(fold=0, rank=0, world_size=2)

    # Ensure an all_reduce occurred somewhere (train and/or val). We
    # specifically exercised the rank-0 validation branch that calls all_reduce
    # if use_ddp.
    assert patch_dist["all_reduce"] >= 1

    # Inspect the logged scalars. Validation mean should be val_loss/world_size.
    assert created_writers, "SummaryWriter was not instantiated"
    tag, scalars, step = created_writers[-1].scalars[-1]
    assert tag == "losses"
    # With FakeDist all_reduce as a no-op, the code divides by world_size
    # explicitly.
    assert scalars["validation"] == pytest.approx(4.0 / 2)
    # (Optional) train mean is also divided by world_size on rank 0 in DDP.
    assert scalars["train"] == pytest.approx(1.0 / 2)
