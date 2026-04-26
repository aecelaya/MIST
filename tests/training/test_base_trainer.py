"""Tests for the BaseTrainer implementation."""
import json
import os
import math
import pickle
import rich
from mist.utils import console as console_mod
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from torch import nn

# MIST imports.
from mist.training.trainers import base_trainer as bt

# Setup base trainer for tests.
BaseTrainer = bt.BaseTrainer

# Capture real implementations before any autouse fixtures patch them.
_real_torch_save = torch.save
_real_torch_load = torch.load
_real_save_checkpoint = bt.BaseTrainer.save_checkpoint


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
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        """Forward pass through the dummy linear layer."""
        return self.fc(x)


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

    def add_scalar(self, tag, value, step):
        """Collect a single scalar in memory."""
        self.scalars.append((tag, float(value), int(step)))

    def add_scalars(self, tag, scalars, step):
        """Collect scalars in memory."""
        self.scalars.append((tag, dict(scalars), int(step)))

    def flush(self):
        """Flush the collected scalars."""

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
def tmp_pipeline(tmp_path: Path) -> tuple[Path, Path]:
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
                "composite_loss_weighting": None,
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
        composite_loss_weighting=None,
        optimizer=None,
        l2_penalty=None,
        learning_rate=None,
        lr_scheduler=None,
        warmup_epochs=None,
        val_percent=None,
        resume=False,
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
    def _fake_registry(arch, **p):
        assert "patch_size" in p, (
            "get_model_from_registry must receive patch_size from spatial_config"
        )
        assert "target_spacing" in p, (
            "get_model_from_registry must receive target_spacing from spatial_config"
        )
        return DummyModel()
    monkeypatch.setattr(bt, "get_model_from_registry", _fake_registry)
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

        def state_dict(self):
            """Return empty state dict."""
            return {}

        def load_state_dict(self, state):
            """Load state dict (no-op)."""

    monkeypatch.setattr(
        bt, "get_lr_scheduler",
        lambda name, optimizer, epochs, warmup_epochs=0: DummyScheduler()
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
    monkeypatch.setattr(
        bt.BaseTrainer, "save_checkpoint", lambda *a, **k: None
    )


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
    cfg["training"]["loss"]["name"] = "gsl"
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
    clw = cfg["training"]["loss"]["composite_loss_weighting"]
    assert clw["name"] == "linear"
    assert "init_pause" in clw["params"]
    assert "start_val" in clw["params"]
    assert "end_val" in clw["params"]
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


def test_fit_sets_cudnn_conv_fp32_precision_when_available(
    tmp_pipeline, mist_args, monkeypatch
):
    """fit() sets cudnn.conv.fp32_precision='tf32' on PyTorch >= 2.5."""
    fake_conv = SimpleNamespace(fp32_precision=None)
    monkeypatch.setattr(torch.backends.cudnn, "conv", fake_conv, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    trainer.run_cross_validation = lambda rank, world_size: None
    trainer.fit()
    assert fake_conv.fp32_precision == "tf32"


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

    out = []
    monkeypatch.setattr(
        console_mod.console, "print", lambda msg: out.append(str(msg))
    )

    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert any("Validation loss did not improve" in s for s in out)


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
        console_mod.console, "print", lambda msg: out.append(str(msg))
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
        console_mod.console, "print", lambda msg: out.append(str(msg))
    )

    calls = []

    def spy_train_fold(fold, rank, world_size):
        calls.append((fold, rank, world_size))

    monkeypatch.setattr(trainer, "train_fold", spy_train_fold)

    trainer.run_cross_validation(rank=1, world_size=2)

    assert not any("Starting training" in s for s in out)
    assert calls == [(0, 1, 2), (1, 1, 2)]


@pytest.mark.parametrize("clw_cfg", [
    None,
    {"name": "linear", "params": {"init_pause": 5}},
])
def test_build_components_composite_loss_scheduler(
    tmp_pipeline, mist_args, monkeypatch, clw_cfg
):
    """Test that build_components sets composite_loss_weighting correctly.

    Uses a composite loss name so the COMPOSITE_LOSSES guard is satisfied.
    """
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    # Use a composite loss so the scheduler guard fires.
    cfg["training"]["loss"]["name"] = "bl"
    cfg["training"]["loss"]["composite_loss_weighting"] = clw_cfg
    cfg_path.write_text(json.dumps(cfg))

    calls = {"args": []}
    sentinel = object()

    def spy_get_alpha_scheduler(name, num_epochs, **params):
        calls["args"].append({"name": name, "num_epochs": num_epochs, **params})
        return sentinel

    monkeypatch.setattr(bt, "get_alpha_scheduler", spy_get_alpha_scheduler)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    if clw_cfg is None:
        assert state["composite_loss_weighting"] is None
        assert not calls["args"]
    else:
        assert state["composite_loss_weighting"] is sentinel
        assert calls["args"][0]["name"] == clw_cfg["name"]
        assert isinstance(calls["args"][0]["num_epochs"], int)
        # Params from the config are unpacked and forwarded.
        for k, v in clw_cfg["params"].items():
            assert calls["args"][0][k] == v


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


def test_resume_raises_on_incompatible_model_override(
    tmp_pipeline, mist_args, monkeypatch
):
    """--resume + --model mismatch raises ValueError before training starts."""
    mist_args.resume = True
    mist_args.model = "different_arch"

    with pytest.raises(ValueError, match="incompatible with the saved checkpoint"):
        DummyTrainer(mist_args)


def test_resume_raises_on_incompatible_patch_size_override(
    tmp_pipeline, mist_args, monkeypatch
):
    """--resume + --patch-size mismatch raises ValueError before training starts."""
    mist_args.resume = True
    mist_args.patch_size = [32, 32, 32]  # config default is [16, 16, 16]

    with pytest.raises(ValueError, match="incompatible with the saved checkpoint"):
        DummyTrainer(mist_args)


def test_resume_warns_on_loss_override(tmp_pipeline, mist_args, monkeypatch):
    """--resume + --loss override prints a yellow warning."""
    mist_args.resume = True
    mist_args.loss = "dice"  # config default is dummy_loss

    printed = []
    monkeypatch.setattr(
        console_mod.console, "print", lambda msg: printed.append(str(msg))
    )

    DummyTrainer(mist_args)

    assert any("Warning" in s and "--resume" in s for s in printed)
    assert any("--loss" in s for s in printed)


def test_resume_no_warning_when_no_overrides(tmp_pipeline, mist_args, monkeypatch):
    """--resume with no config overrides produces no warning."""
    mist_args.resume = True

    printed = []
    monkeypatch.setattr(
        console_mod.console, "print", lambda msg: printed.append(str(msg))
    )

    DummyTrainer(mist_args)

    assert not any("Warning" in s for s in printed)


def test_save_and_load_checkpoint_roundtrip(
    tmp_pipeline, mist_args, monkeypatch
):
    """save_checkpoint followed by load_checkpoint restores state exactly."""
    # Use real torch.save/load for this test.
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    # Mutate state to non-default values to verify restoration.
    state["epoch"] = 3
    state["global_step"] = 150
    state["best_val_loss"] = 0.42

    trainer.save_checkpoint(fold=0, state=state)
    assert trainer._checkpoint_path(0).exists()
    # Temp file must not linger after a successful save.
    assert not trainer._checkpoint_path(0).with_suffix(".tmp").exists()

    # Build a fresh state and load into it.
    fresh_state = trainer.build_components(rank=0, world_size=1)
    loaded = trainer.load_checkpoint(fold=0, state=fresh_state)

    assert loaded is True
    # Epoch is incremented by 1 so the training loop resumes on the next epoch.
    assert fresh_state["epoch"] == 4
    assert fresh_state["global_step"] == 150
    assert fresh_state["best_val_loss"] == pytest.approx(0.42)


def test_load_checkpoint_returns_false_when_missing(
    tmp_pipeline, mist_args, monkeypatch
):
    """load_checkpoint returns False and leaves state unchanged when no file."""
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    state["best_val_loss"] = 1.23

    loaded = trainer.load_checkpoint(fold=0, state=state)

    assert loaded is False
    assert state["best_val_loss"] == pytest.approx(1.23)


def test_train_fold_saves_checkpoint_each_epoch(
    tmp_pipeline, mist_args, monkeypatch
):
    """train_fold should call save_checkpoint once per completed epoch."""
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert trainer._checkpoint_path(0).exists()
    checkpoint = torch.load(trainer._checkpoint_path(0), weights_only=False)
    total_epochs = trainer.config["training"]["epochs"]
    # After one epoch (epoch index 0), saved epoch should be 0.
    assert checkpoint["epoch"] == total_epochs - 1
    assert checkpoint["fold"] == 0


def test_resume_loads_checkpoint_and_prints_message(
    tmp_pipeline, mist_args, monkeypatch
):
    """With --resume and an existing checkpoint, train_fold loads it and prints."""
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    # First run: train and save a checkpoint.
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer.train_fold(fold=0, rank=0, world_size=1)
    assert trainer._checkpoint_path(0).exists()

    # Second run: resume=True.
    mist_args.resume = True
    mist_args.epochs = 3
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["epochs"] = 3
    cfg_path.write_text(json.dumps(cfg))

    trainer2 = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    # Share the same checkpoints dir so the checkpoint is found.
    trainer2.checkpoints_dir = trainer.checkpoints_dir

    out = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))
    trainer2.train_fold(fold=0, rank=0, world_size=1)

    assert any("Resuming fold 0" in s for s in out)


def test_resume_warns_when_no_checkpoint(
    tmp_pipeline, mist_args, monkeypatch
):
    """With --resume but no checkpoint, train_fold warns and starts fresh."""
    mist_args.resume = True
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)

    out = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))
    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert any("No checkpoint found" in s for s in out)


def test_run_cross_validation_skips_completed_fold(
    tmp_pipeline, mist_args, monkeypatch
):
    """With --resume, completed folds (epoch >= epochs-1) are skipped."""
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["folds"] = [0, 1]
    cfg_path.write_text(json.dumps(cfg))

    # First run: train fold 0 to completion and save checkpoint.
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # Second run: resume=True, both folds requested.
    mist_args.resume = True
    trainer2 = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer2.checkpoints_dir = trainer.checkpoints_dir

    out = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))

    called_folds = []
    original_train_fold = trainer2.train_fold

    def spy_train_fold(fold, rank, world_size):
        called_folds.append(fold)
        original_train_fold(fold=fold, rank=rank, world_size=world_size)

    monkeypatch.setattr(trainer2, "train_fold", spy_train_fold)
    trainer2.run_cross_validation(rank=0, world_size=1)

    # Fold 0 was completed so it should be skipped; fold 1 should run.
    assert 0 not in called_folds
    assert 1 in called_folds
    assert any("already complete" in s for s in out)


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
    writer = created_writers[-1]
    losses_entry = next(
        (e for e in writer.scalars if e[0] == "losses"), None
    )
    assert losses_entry is not None, "'losses' scalars were not logged"
    _, scalars, _ = losses_entry
    # With FakeDist all_reduce as a no-op, the code divides by world_size
    # explicitly.
    assert scalars["validation"] == pytest.approx(4.0 / 2)
    # (Optional) train mean is also divided by world_size on rank 0 in DDP.
    assert scalars["train"] == pytest.approx(1.0 / 2)


# =============================================
# Coverage gap: warmup_epochs CLI override
# =============================================

def test_init_warmup_epochs_override(tmp_pipeline, monkeypatch):
    """warmup_epochs CLI arg is applied to config when not None."""
    results, numpy_dir = tmp_pipeline
    args = SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        model=None,
        patch_size=None,
        folds=None,
        epochs=None,
        batch_size_per_gpu=None,
        loss=None,
        composite_loss_weighting=None,
        optimizer=None,
        l2_penalty=None,
        learning_rate=None,
        lr_scheduler=None,
        warmup_epochs=5,
        val_percent=None,
        resume=False,
    )
    trainer = DummyTrainer(args)
    assert trainer.config["training"]["warmup_epochs"] == 5


# =============================================
# Coverage gap: _check_resume_overrides warnings
# =============================================

def test_check_resume_overrides_warns_on_all_training_diffs(
    tmp_pipeline, monkeypatch
):
    """All six warning branches fire when every overridable field differs."""
    results, numpy_dir = tmp_pipeline
    # Start with resume=True and overrides that differ from the saved config.
    args = SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        model=None,
        patch_size=None,
        folds=None,
        epochs=None,
        batch_size_per_gpu=None,
        loss="cross_entropy",               # differs from "dummy_loss"
        composite_loss_weighting="linear",  # differs from None
        optimizer="adam",                   # differs from "sgd"
        l2_penalty=0.01,                    # differs from 0.0
        learning_rate=0.001,                # differs from 0.01
        lr_scheduler="cosine",              # differs from "constant"
        warmup_epochs=3,                    # differs from missing/0
        val_percent=None,
        resume=True,
    )

    printed = []
    monkeypatch.setattr(
        console_mod.console, "print", lambda msg: printed.append(str(msg))
    )

    DummyTrainer(args)

    combined = "\n".join(printed)
    assert "--loss" in combined
    assert "--composite-loss-weighting" in combined
    assert "--optimizer" in combined
    assert "--learning-rate" in combined
    assert "--lr-scheduler" in combined
    assert "--warmup-epochs" in combined
    assert "--l2-penalty" in combined


# =============================================
# Coverage gap: _validate_pretrained_config
# =============================================

def test_validate_pretrained_config_warns_when_no_config_path(
    tmp_pipeline, mist_args, monkeypatch
):
    """pretrained_weights set but no config path emits a Python warning."""
    mist_args.pretrained_weights = "/fake/weights.pt"
    # pretrained_config is intentionally not set (absent == None via getattr)

    with pytest.warns(UserWarning, match="--pretrained-config was not provided"):
        DummyTrainer(mist_args)


def test_validate_pretrained_config_calls_validator_when_both_set(
    tmp_pipeline, mist_args, monkeypatch
):
    """When both pretrained_weights and pretrained_config are given, the
    encoder compatibility validator is called with the loaded source config."""
    mist_args.pretrained_weights = "/fake/weights.pt"
    mist_args.pretrained_config = "/fake/source_config.json"

    source_cfg = {"model": {"architecture": "nnunet"}}
    _real_read = bt.io.read_json_file
    monkeypatch.setattr(
        bt.io, "read_json_file",
        lambda path: source_cfg if path == "/fake/source_config.json" else _real_read(path),
    )

    calls = []
    monkeypatch.setattr(
        bt, "validate_encoder_compatibility",
        lambda src, dst: calls.append((src, dst)),
    )

    DummyTrainer(mist_args)

    assert len(calls) == 1
    assert calls[0][0] is source_cfg


# =============================================
# Coverage gap: build_components pretrained encoder loading
# =============================================

def test_build_components_loads_pretrained_encoder(
    tmp_pipeline, mist_args, monkeypatch
):
    """When pretrained_weights is set, load_pretrained_encoder is called and
    the summary is printed on rank 0."""
    mist_args.pretrained_weights = "/fake/encoder.pt"
    mist_args.pretrained_config = "/fake/source_config.json"

    dummy_summary = {
        "loaded": list(range(10)),
        "channel_strategy_applied": list(range(2)),
        "skipped": list(range(1)),
    }

    _real_read = bt.io.read_json_file
    monkeypatch.setattr(
        bt.io, "read_json_file",
        lambda path: {} if path == "/fake/source_config.json" else _real_read(path),
    )
    monkeypatch.setattr(
        bt, "load_pretrained_encoder",
        lambda model, path, strategy: (model, dummy_summary),
    )
    monkeypatch.setattr(
        bt, "validate_encoder_compatibility", lambda *a: None
    )

    printed = []
    monkeypatch.setattr(
        console_mod.console, "print", lambda msg: printed.append(str(msg))
    )

    trainer = DummyTrainer(mist_args)
    trainer.build_components(rank=0, world_size=1)

    combined = "\n".join(printed)
    assert "Pretrained encoder loaded" in combined
    assert "10" in combined   # loaded count
    assert "2" in combined    # channel_strategy_applied count


# =============================================
# Coverage gap: build_components spacing-aware loss
# =============================================

def test_build_components_spacing_aware_loss_injects_spacing(
    tmp_pipeline, mist_args, monkeypatch
):
    """A spacing-aware loss name causes sddl_spacing_xyz to be passed to the
    loss constructor."""
    spacing_loss = next(iter(bt.TrainerConstants.SPACING_AWARE_LOSSES))

    results, numpy_dir = tmp_pipeline
    config_path = Path(results) / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["loss"]["name"] = spacing_loss
    config_path.write_text(json.dumps(config))

    received_params = {}

    def capturing_loss_cls(**kwargs):
        received_params.update(kwargs)
        return DummyLoss()

    monkeypatch.setattr(bt, "get_loss", lambda name: capturing_loss_cls)

    trainer = DummyTrainer(mist_args)
    trainer.build_components(rank=0, world_size=1)

    assert "sddl_spacing_xyz" in received_params
    assert received_params["sddl_spacing_xyz"] == [1.0, 1.0, 1.0]


# =============================================
# Coverage gap: load_checkpoint with AMP scaler
# =============================================

def test_load_checkpoint_restores_scaler_state(
    tmp_pipeline, mist_args, monkeypatch
):
    """When state has a scaler and the checkpoint has scaler_state_dict,
    the scaler's load_state_dict is called."""
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    # Write a checkpoint that includes a fake scaler_state_dict.
    path = trainer._checkpoint_path(0)
    model = state["model"]
    fake_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": state["optimizer"].state_dict(),
        "lr_scheduler_state_dict": state["lr_scheduler"].state_dict(),
        "scaler_state_dict": {"scale": 65536.0},
        "epoch": 2,
        "global_step": 100,
        "best_val_loss": 0.3,
    }
    _real_torch_save(fake_checkpoint, path)

    # Attach a fake scaler with a load_state_dict tracker.
    scaler_loads = []

    class FakeScaler:
        def load_state_dict(self, sd):
            scaler_loads.append(sd)

    state["scaler"] = FakeScaler()

    loaded = trainer.load_checkpoint(fold=0, state=state)

    assert loaded is True
    assert len(scaler_loads) == 1
    assert scaler_loads[0] == {"scale": 65536.0}


# =============================================
# Coverage gap: alpha TensorBoard logging (line 934)
# =============================================

def test_train_fold_logs_alpha_for_composite_loss(
    tmp_pipeline, mist_args, monkeypatch
):
    """train_fold logs 'alpha' to TensorBoard when composite_loss_weighting is set.

    This covers the ``if state["composite_loss_weighting"] is not None`` branch
    in the TensorBoard logging block (base_trainer.py line 933-936).
    """
    results, _ = tmp_pipeline

    # Set the config to use a composite loss with a weighting schedule.
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["loss"]["name"] = "cldice"
    cfg["training"]["loss"]["composite_loss_weighting"] = {
        "name": "constant",
        "params": {"value": 0.7},
    }
    cfg_path.write_text(json.dumps(cfg))

    # Stub get_alpha_scheduler to return a simple callable — no real scheduler
    # needed; we just need state["composite_loss_weighting"] to be non-None.
    monkeypatch.setattr(
        bt, "get_alpha_scheduler",
        lambda name, num_epochs, **kw: lambda epoch: 0.7
    )

    # Capture the SummaryWriter used during training.
    created_writers = []

    class CapturingWriter(DummySummaryWriter):
        def __init__(self, log_dir):
            super().__init__(log_dir)
            created_writers.append(self)

    monkeypatch.setattr(bt, "SummaryWriter", CapturingWriter)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert created_writers, "SummaryWriter was not instantiated"
    writer = created_writers[-1]

    alpha_entries = [e for e in writer.scalars if e[0] == "alpha"]
    assert alpha_entries, "'alpha' was never logged to TensorBoard"
    # Each logged alpha value should be the fixed 0.7 returned by the stub.
    for _, value, _ in alpha_entries:
        assert value == pytest.approx(0.7)
