"""Tests for the BaseTrainer implementation."""

import json
import math
import os
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import rich
import torch
from torch import nn

# MIST imports.
from mist.training.trainers import base_trainer as bt
from mist.utils import console as console_mod

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
        val_len = max(1, math.ceil(len(fold_data["val_images"]) / max(1, world_size)))
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
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0), raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: None, raising=False)

    # Make .to(device) a no-op so models stay on CPU.
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: self, raising=False)

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
        "model": {"architecture": "dummy", "params": {}},
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
        },
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

    def fake_get_optimizer(name, params, weight_decay, eps, learning_rate=None, l2_penalty=None):
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
        bt,
        "get_lr_scheduler",
        lambda name, optimizer, epochs, warmup_epochs=0: DummyScheduler(),
    )


@pytest.fixture(autouse=True)
def patch_ddp_and_tb_and_save(monkeypatch):
    """Patch DDP, TensorBoard, and torch.save to use dummy classes."""
    monkeypatch.setattr(bt, "DDP", DummyDDP)
    monkeypatch.setattr(bt, "SummaryWriter", DummySummaryWriter)
    monkeypatch.setattr(bt.progress_bar, "TrainProgressBar", DummyProgressCtx)
    monkeypatch.setattr(bt.progress_bar, "ValidationProgressBar", DummyProgressCtx)
    monkeypatch.setattr(torch, "save", lambda *a, **k: None)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", lambda *a, **k: None)


_NO_DEVICE_ID = object()


@pytest.fixture(autouse=True)
def patch_dist(monkeypatch):
    """Patch torch.distributed to be inert but count calls."""
    calls = {
        "init": 0,
        "destroy": 0,
        "all_reduce": 0,
        "broadcast": 0,
        "barrier": 0,
        # Sentinel so tests can tell "device_id kwarg omitted" (CPU) apart
        # from "device_id=None passed explicitly" -- setup() never does the
        # latter, but the fake shouldn't quietly conflate the two.
        "last_device_id": _NO_DEVICE_ID,
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
        def init_process_group(backend, rank, world_size, device_id=_NO_DEVICE_ID):
            """Initialize the process group."""
            calls["init"] += 1
            calls["last_device_id"] = device_id
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
    assert trainer.batch_size == trainer.config["training"]["batch_size_per_gpu"] * 1


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


def test_setup_folds_with_dtms_and_valsplit(tmp_pipeline, mist_args, monkeypatch):
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


def test_build_components_multi_gpu_wraps_with_ddp(tmp_pipeline, mist_args, monkeypatch):
    """Test build_components with multiple GPUs, DDP wrapping."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"]["amp"] = True
    (Path(results) / "config.json").write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=2)
    assert isinstance(state["model"], DummyDDP)
    assert "scaler" not in state


def _is_compiled(model):
    """Return whether a model was compiled in place."""
    return getattr(model, "_compiled_call_impl", None) is not None


def test_build_components_does_not_compile_by_default(tmp_pipeline, mist_args):
    """Without training.compile, the model isn't compiled."""
    state = DummyTrainer(mist_args).build_components(rank=0, world_size=1)
    assert not _is_compiled(state["model"])


def test_build_components_old_config_without_compile_key(tmp_pipeline, mist_args):
    """A config from before the setting trains uncompiled."""
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"].pop("compile", None)
    (Path(results) / "config.json").write_text(json.dumps(cfg))
    state = DummyTrainer(mist_args).build_components(rank=0, world_size=1)
    assert not _is_compiled(state["model"])


@pytest.mark.parametrize("world_size", [1, 2])
def test_build_components_compiles_in_place(tmp_pipeline, mist_args, monkeypatch, world_size):
    """Compiling keeps the model's type and weight names, and comes before DDP."""
    monkeypatch.setattr(bt.hardware, "resolve_compile", lambda requested, warn=True: True)
    state = DummyTrainer(mist_args).build_components(rank=0, world_size=world_size)
    model = state["model"].module if world_size > 1 else state["model"]
    assert isinstance(model, DummyModel)
    assert _is_compiled(model)
    assert not any(key.startswith("_orig_mod.") for key in model.state_dict())


def test_setup_initializes_process_group_once(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """Ensure process group is initialized only once."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    trainer = DummyTrainer(mist_args)
    trainer.setup(rank=0, world_size=2)
    trainer.setup(rank=0, world_size=2)
    assert patch_dist["init"] == 1


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
def test_setup_passes_device_id_on_cuda_and_rocm(
    tmp_pipeline, mist_args, monkeypatch, patch_dist, accelerator
):
    """setup() passes an explicit device_id on CUDA/ROCm (for NCCL/RCCL)."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: accelerator)
    trainer = DummyTrainer(mist_args)
    trainer.setup(rank=0, world_size=2)
    assert patch_dist["last_device_id"] == torch.device("cuda", 0)


def test_setup_omits_device_id_on_cpu(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """setup() omits device_id on CPU -- gloo has no GPU device concept."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "cpu")
    trainer = DummyTrainer(mist_args)
    trainer.setup(rank=0, world_size=2)
    assert patch_dist["last_device_id"] is _NO_DEVICE_ID


def test_train_fold_skips_cuda_set_device_on_cpu(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """train_fold() must not call torch.cuda.set_device on CPU-only hardware."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "cpu")
    called = {"count": 0}
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a, **k: called.__setitem__("count", called["count"] + 1)
    )
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.train_fold(fold=0, rank=0, world_size=1)
    assert called["count"] == 0


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
def test_train_fold_calls_cuda_set_device_on_cuda_and_rocm(
    tmp_pipeline, mist_args, monkeypatch, patch_dist, accelerator
):
    """train_fold() still calls torch.cuda.set_device on CUDA/ROCm."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: accelerator)
    called = {"count": 0}
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a, **k: called.__setitem__("count", called["count"] + 1)
    )
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.train_fold(fold=0, rank=0, world_size=1)
    assert called["count"] == 1


def test_train_fold_runs_full_epoch(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """Test that train_fold runs a full epoch with correct steps."""
    """With world_size=1, no collectives are called in the new code path."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # New behavior: no DDP => no collectives.
    assert patch_dist["broadcast"] == 0
    assert patch_dist["barrier"] == 0
    assert patch_dist["all_reduce"] == 0


def test_train_fold_early_stop_on_nan(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """NaN training loss should trigger early stop (DDP case)."""
    # Use 2 GPUs so DDP is engaged.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    trainer = DummyTrainer(mist_args, train_loss_value=float("nan"), val_loss_value=2.0)
    trainer.train_fold(fold=0, rank=0, world_size=2)
    # The process group is shared across folds and is only destroyed once
    # run_cross_validation's fold loop exits, not by train_fold itself -- see
    # the comments in train_fold for why per-fold teardown is unsafe.
    assert patch_dist["destroy"] == 0
    assert patch_dist["init"] >= 1


def test_run_cross_validation_cleans_up_process_group_once_after_nan(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """cleanup() runs exactly once, after run_cross_validation's fold loop,
    even when a fold aborts early on a NaN loss."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    trainer = DummyTrainer(mist_args, train_loss_value=float("nan"), val_loss_value=2.0)
    trainer.run_cross_validation(rank=0, world_size=2)
    assert patch_dist["destroy"] == 1


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


@pytest.mark.parametrize(
    ("accelerator", "expected_backend"),
    [("cuda", "nccl"), ("rocm", "nccl"), ("cpu", "gloo")],
)
def test_overwrite_config_from_args_resolves_auto_communication_backend(
    tmp_pipeline, mist_args, monkeypatch, accelerator, expected_backend
):
    """ "auto" is resolved and persisted to config.json at train time.

    Mirrors training.amp's resolve-and-persist pattern: the analyze-time
    machine that wrote "auto" may not be the train-time machine, so
    resolution has to happen here, not in analyzer_utils.py.
    """
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["hardware"]["communication_backend"] = "auto"
    cfg_path.write_text(json.dumps(cfg))

    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: accelerator)
    trainer = DummyTrainer(mist_args)

    assert trainer.config["training"]["hardware"]["communication_backend"] == expected_backend
    persisted = json.loads(cfg_path.read_text())
    assert persisted["training"]["hardware"]["communication_backend"] == expected_backend


def test_overwrite_config_from_args_leaves_explicit_communication_backend(
    tmp_pipeline, mist_args, monkeypatch
):
    """An explicit (non-"auto") backend from config.json is never overridden."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "cpu")
    trainer = DummyTrainer(mist_args)
    # tmp_pipeline's base config.json already hardcodes "nccl" explicitly
    # (not "auto"), even though this test pins the accelerator to "cpu" --
    # confirming resolution is skipped entirely for explicit values.
    assert trainer.config["training"]["hardware"]["communication_backend"] == "nccl"


def test_fit_single_gpu_calls_run_directly(tmp_pipeline, mist_args, monkeypatch):
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


def test_run_cross_validation_sets_cudnn_conv_fp32_precision_when_available(
    tmp_pipeline, mist_args, monkeypatch
):
    """run_cross_validation sets cudnn.conv.fp32_precision='tf32' on PyTorch >= 2.5."""
    fake_conv = SimpleNamespace(fp32_precision=None)
    monkeypatch.setattr(torch.backends.cudnn, "conv", fake_conv, raising=False)
    trainer = DummyTrainer(mist_args)
    monkeypatch.setattr(trainer, "train_fold", lambda **_: None)
    trainer.run_cross_validation(rank=0, world_size=1)
    assert fake_conv.fp32_precision == "tf32"


def test_run_cross_validation_skips_cudnn_tuning_on_cpu(tmp_pipeline, mist_args, monkeypatch):
    """run_cross_validation must not touch cudnn/TF32 settings on CPU-only hardware."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "cpu")
    fake_conv = SimpleNamespace(fp32_precision=None)
    monkeypatch.setattr(torch.backends.cudnn, "conv", fake_conv, raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False, raising=False)
    trainer = DummyTrainer(mist_args)
    monkeypatch.setattr(trainer, "train_fold", lambda **_: None)
    trainer.run_cross_validation(rank=0, world_size=1)
    assert fake_conv.fp32_precision is None
    assert torch.backends.cudnn.benchmark is False


def test_init_warns_about_mistakes_in_config_file(tmp_pipeline, mist_args, monkeypatch):
    """A misspelled key in config.json gets a warning, and training still starts."""
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["compil"] = True
    cfg_path.write_text(json.dumps(cfg))
    warnings = []
    monkeypatch.setattr(bt, "print_warning", warnings.append)

    DummyTrainer(mist_args)

    assert any("training.compil'" in warning for warning in warnings)


@pytest.mark.parametrize("rank", [0, 1])
def test_run_cross_validation_tunes_gpu_math_before_training_on_every_rank(
    tmp_pipeline, mist_args, monkeypatch, rank
):
    """Each process sets its own GPU math, since mp.spawn's don't inherit it."""
    events = []
    monkeypatch.setattr(bt.hardware, "tune_gpu_math", lambda: events.append("tune"))
    trainer = DummyTrainer(mist_args)
    monkeypatch.setattr(trainer, "train_fold", lambda **_: events.append("train"))
    trainer.run_cross_validation(rank=rank, world_size=2)
    assert events[0] == "tune"
    assert events.count("tune") == 1
    assert "train" in events


def test_fit_does_not_tune_gpu_math_in_the_parent_process(tmp_pipeline, mist_args, monkeypatch):
    """With spawn, the parent's settings wouldn't reach the training processes."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "cuda")
    monkeypatch.setattr(bt.mp, "spawn", lambda *args, **kwargs: None)
    tuned = []
    monkeypatch.setattr(bt.hardware, "tune_gpu_math", lambda: tuned.append(True))
    DummyTrainer(mist_args).fit()
    assert not tuned


def test_fit_world_size_is_one_on_cpu_regardless_of_device_count(
    tmp_pipeline, mist_args, monkeypatch
):
    """Regression guard for the world_size = torch.cuda.device_count() bug.

    torch.cuda.device_count() is 0 on CPU-only hardware, which used to flow
    straight into run_cross_validation and break its world_size == 1 single-
    process assumptions. Also covers a stale/misreported device_count > 0 on
    a machine get_accelerator_type() has determined is CPU-only -- fit()
    must still run as a single process.
    """
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    called = {}

    def spy_run(rank, world_size):
        called["rank"] = rank
        called["world_size"] = world_size

    trainer = DummyTrainer(mist_args)
    trainer.run_cross_validation = spy_run
    trainer.fit()
    assert called == {"rank": 0, "world_size": 1}


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


def test_update_num_gpus_is_zero_on_cpu(tmp_pipeline, mist_args, monkeypatch):
    """num_gpus is 0 (not a raise) on CPU-only hardware.

    Regression guard for a real bug: this used to unconditionally raise
    "CUDA is not available" here, which meant CPU-only training could never
    even construct a trainer -- found via a real end-to-end CPU training run
    (cpu_rocm_support_plan.md Stage 4), not by this mocked test alone.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

    trainer = DummyTrainer(mist_args)

    assert trainer.config["training"]["hardware"]["num_gpus"] == 0
    assert trainer.batch_size == trainer.config["training"]["batch_size_per_gpu"]


def test_update_num_gpus_raises_when_zero_devices(tmp_pipeline, mist_args, monkeypatch):
    """Test that update_num_gpus raises when device_count is zero."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0, raising=False)

    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)

    msg = str(excinfo.value)
    assert "device_count() == 0" in msg
    assert "CUDA_VISIBLE_DEVICES" in msg


def test_update_num_gpus_sets_config_and_persists(tmp_pipeline, mist_args, monkeypatch):
    """Test that update_num_gpus sets config and persists to disk."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)

    results, _ = tmp_pipeline
    trainer = DummyTrainer(mist_args)

    assert trainer.config["training"]["hardware"]["num_gpus"] == 2

    cfg_path = Path(results) / "config.json"
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["training"]["hardware"]["num_gpus"] == 2


def test_update_num_gpus_sets_config_on_rocm(tmp_pipeline, mist_args, monkeypatch):
    """num_gpus is set from device_count() on ROCm too, same as CUDA."""
    monkeypatch.setattr(bt.hardware, "get_accelerator_type", lambda: "rocm")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)

    trainer = DummyTrainer(mist_args)

    assert trainer.config["training"]["hardware"]["num_gpus"] == 2


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


def test_train_loop_else_branch_rank_nonzero(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """Test train_fold else branch for rank > 0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: None, raising=False)
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: self, raising=False)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.folds[0]["steps_per_epoch"] = 2
    trainer.folds[0]["val_images"] = ["val0", "val1"]
    trainer.folds[0]["train_images"] = ["tr0", "tr1", "tr2"]

    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    trainer.train_fold(fold=0, rank=1, world_size=2)

    assert patch_dist["all_reduce"] >= 3
    assert patch_dist["broadcast"] >= 1
    assert patch_dist["barrier"] >= 1


def test_validation_else_branch_rank_nonzero(tmp_pipeline, mist_args, monkeypatch, patch_dist):
    """Test validation step else branch for rank > 0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: None, raising=False)
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: self, raising=False)

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


def test_validation_no_improvement_message(tmp_pipeline, mist_args, monkeypatch):
    """With a validation split, a non-improving epoch logs a message."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: None, raising=False)
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: self, raising=False)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=float("inf"))

    trainer.folds[0]["steps_per_epoch"] = 1
    trainer.folds[0]["val_images"] = ["val0", "val1"]
    trainer.config["training"]["val_percent"] = 0.5

    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    out = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))

    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert any("Smoothed validation loss did not improve" in s for s in out)


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
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))

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
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))

    calls = []

    def spy_train_fold(fold, rank, world_size):
        calls.append((fold, rank, world_size))

    monkeypatch.setattr(trainer, "train_fold", spy_train_fold)

    trainer.run_cross_validation(rank=1, world_size=2)

    assert not any("Starting training" in s for s in out)
    assert calls == [(0, 1, 2), (1, 1, 2)]


@pytest.mark.parametrize(
    "clw_cfg",
    [
        None,
        {"name": "linear", "params": {"init_pause": 5}},
    ],
)
def test_build_components_composite_loss_scheduler(tmp_pipeline, mist_args, monkeypatch, clw_cfg):
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


def test_resume_raises_on_incompatible_model_override(tmp_pipeline, mist_args, monkeypatch):
    """--resume + --model mismatch raises ValueError before training starts."""
    mist_args.resume = True
    mist_args.model = "different_arch"

    with pytest.raises(ValueError, match="incompatible with the saved checkpoint"):
        DummyTrainer(mist_args)


def test_resume_raises_on_incompatible_patch_size_override(tmp_pipeline, mist_args, monkeypatch):
    """--resume + --patch-size mismatch raises ValueError before training starts."""
    mist_args.resume = True
    mist_args.patch_size = [32, 32, 32]  # config default is [16, 16, 16]

    with pytest.raises(ValueError, match="incompatible with the saved checkpoint"):
        DummyTrainer(mist_args)


@pytest.mark.parametrize("saved, override", [("nnunet", "nnunet-res"), ("nnunet-res", "nnunet")])
def test_resume_accepts_architecture_alias(tmp_pipeline, mist_args, monkeypatch, saved, override):
    """--resume treats an alias and the name it stands for as the same network."""
    results, _ = tmp_pipeline
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model"]["architecture"] = saved
    cfg_path.write_text(json.dumps(cfg))

    mist_args.resume = True
    mist_args.model = override
    DummyTrainer(mist_args)


def test_resume_warns_on_loss_override(tmp_pipeline, mist_args, monkeypatch):
    """--resume + --loss override prints a yellow warning."""
    mist_args.resume = True
    mist_args.loss = "dice"  # config default is dummy_loss

    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))

    DummyTrainer(mist_args)

    assert any("Warning" in s and "--resume" in s for s in printed)
    assert any("--loss" in s for s in printed)


def test_resume_no_warning_when_no_overrides(tmp_pipeline, mist_args, monkeypatch):
    """--resume with no config overrides produces no warning."""
    mist_args.resume = True

    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))

    DummyTrainer(mist_args)

    assert not any("Warning" in s for s in printed)


def test_save_and_load_checkpoint_roundtrip(tmp_pipeline, mist_args, monkeypatch):
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


def test_load_checkpoint_returns_false_when_missing(tmp_pipeline, mist_args, monkeypatch):
    """load_checkpoint returns False and leaves state unchanged when no file."""
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    state["best_val_loss"] = 1.23

    loaded = trainer.load_checkpoint(fold=0, state=state)

    assert loaded is False
    assert state["best_val_loss"] == pytest.approx(1.23)


def test_load_checkpoint_uses_map_location_cpu(tmp_pipeline, mist_args, monkeypatch):
    """load_checkpoint passes map_location="cpu" to torch.load.

    Regression guard: without this, torch.load tries to recreate tensors on
    whatever device type they were saved from, which crashes outright if
    that device type isn't available on the current machine -- e.g. resuming
    a checkpoint saved during CUDA or ROCm training on a CPU-only machine.
    Can't reproduce the actual crash without real CUDA/ROCm hardware to
    produce a genuinely non-CPU checkpoint, so this instead confirms the
    kwarg itself is always passed, which is what prevents it.
    """
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    trainer.save_checkpoint(fold=0, state=state)

    calls = []
    real_load = torch.load

    def spy_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    trainer.load_checkpoint(fold=0, state=trainer.build_components(rank=0, world_size=1))

    assert len(calls) == 1
    assert calls[0].get("map_location") == "cpu"


def test_run_cross_validation_resume_skip_check_uses_map_location_cpu(
    tmp_pipeline, mist_args, monkeypatch
):
    """The resume-skip check in run_cross_validation also uses map_location="cpu".

    Same regression as test_load_checkpoint_uses_map_location_cpu, but for
    the second, separate torch.load call site -- the one that only reads
    checkpoint["epoch"] to decide whether to skip an already-complete fold,
    but still has to fully deserialize the tensors to get there.
    """
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer.train_fold(fold=0, rank=0, world_size=1)
    assert trainer._checkpoint_path(0).exists()

    mist_args.resume = True
    trainer2 = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)
    trainer2.checkpoints_dir = trainer.checkpoints_dir

    calls = []
    real_load = torch.load

    def spy_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    trainer2.run_cross_validation(rank=0, world_size=1)

    assert len(calls) >= 1
    assert all(call.get("map_location") == "cpu" for call in calls)


def test_train_fold_saves_checkpoint_each_epoch(tmp_pipeline, mist_args, monkeypatch):
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


def test_resume_loads_checkpoint_and_prints_message(tmp_pipeline, mist_args, monkeypatch):
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


def test_resume_warns_when_no_checkpoint(tmp_pipeline, mist_args, monkeypatch):
    """With --resume but no checkpoint, train_fold warns and starts fresh."""
    mist_args.resume = True
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=0.5)

    out = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: out.append(str(msg)))
    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert any("No checkpoint found" in s for s in out)


def test_run_cross_validation_skips_completed_fold(tmp_pipeline, mist_args, monkeypatch):
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


def test_validation_rank0_ddp_allreduce_and_mean(tmp_pipeline, mist_args, monkeypatch, patch_dist):
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
    losses_entry = next((e for e in writer.scalars if e[0] == "losses"), None)
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


def test_check_resume_overrides_warns_on_all_training_diffs(tmp_pipeline, monkeypatch):
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
        loss="cross_entropy",  # differs from "dummy_loss"
        composite_loss_weighting="linear",  # differs from None
        optimizer="adam",  # differs from "sgd"
        l2_penalty=0.01,  # differs from 0.0
        learning_rate=0.001,  # differs from 0.01
        lr_scheduler="cosine",  # differs from "constant"
        warmup_epochs=3,  # differs from missing/0
        val_percent=0.5,  # differs from 0.0
        validate_every=2,  # differs from missing/1
        resume=True,
    )

    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))

    DummyTrainer(args)

    combined = "\n".join(printed)
    assert "--loss" in combined
    assert "--composite-loss-weighting" in combined
    assert "--optimizer" in combined
    assert "--learning-rate" in combined
    assert "--lr-scheduler" in combined
    assert "--warmup-epochs" in combined
    assert "--l2-penalty" in combined
    assert "--val-percent" in combined
    assert "--validate-every" in combined


# =============================================
# Coverage gap: _validate_pretrained_config
# =============================================


def test_validate_pretrained_config_warns_when_no_config_path(tmp_pipeline, mist_args, monkeypatch):
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
        bt.io,
        "read_json_file",
        lambda path: source_cfg if path == "/fake/source_config.json" else _real_read(path),
    )

    calls = []
    monkeypatch.setattr(
        bt,
        "validate_encoder_compatibility",
        lambda src, dst: calls.append((src, dst)),
    )

    DummyTrainer(mist_args)

    assert len(calls) == 1
    assert calls[0][0] is source_cfg


# =============================================
# Coverage gap: build_components pretrained encoder loading
# =============================================


def test_build_components_loads_pretrained_encoder(tmp_pipeline, mist_args, monkeypatch):
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
        bt.io,
        "read_json_file",
        lambda path: {} if path == "/fake/source_config.json" else _real_read(path),
    )
    monkeypatch.setattr(
        bt,
        "load_pretrained_encoder",
        lambda model, path, strategy: (model, dummy_summary),
    )
    monkeypatch.setattr(bt, "validate_encoder_compatibility", lambda *a: None)

    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))

    trainer = DummyTrainer(mist_args)
    trainer.build_components(rank=0, world_size=1)

    combined = "\n".join(printed)
    assert "Pretrained encoder loaded" in combined
    assert "10" in combined  # loaded count
    assert "2" in combined  # channel_strategy_applied count


class _AliasedConvModel(nn.Module):
    """Registers one convolution under two names, as ResEnc's ConvNormAct does."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3)  # 108 weights + 4 biases
        self.all_modules = nn.Sequential(self.conv)


def test_build_components_counts_shared_tensors_once(tmp_pipeline, mist_args, monkeypatch):
    """The transfer summary's parameter count counts aliased keys once."""
    mist_args.pretrained_weights = "/fake/encoder.pt"
    aliased = _AliasedConvModel()
    keys = list(aliased.state_dict())  # Four keys, two distinct tensors.
    assert len(keys) == 4

    monkeypatch.setattr(
        bt,
        "load_pretrained_encoder",
        lambda model, path, strategy: (
            aliased,
            {"loaded": keys, "channel_strategy_applied": [], "skipped": []},
        ),
    )
    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))

    trainer = DummyTrainer(mist_args)
    trainer.build_components(rank=0, world_size=1)

    combined = "\n".join(printed)
    assert "4 tensors (112 scalar parameters)" in combined


# =============================================
# Coverage gap: build_components spacing-aware loss
# =============================================


def test_build_components_spacing_aware_loss_injects_spacing(tmp_pipeline, mist_args, monkeypatch):
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
# Coverage gap: alpha TensorBoard logging (line 934)
# =============================================


def test_train_fold_logs_alpha_for_composite_loss(tmp_pipeline, mist_args, monkeypatch):
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
    monkeypatch.setattr(bt, "get_alpha_scheduler", lambda name, num_epochs, **kw: lambda epoch: 0.7)

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


# =============================================
# Model selection and validation frequency
# =============================================


class SequenceValTrainer(DummyTrainer):
    """DummyTrainer whose validation loss follows a per-epoch sequence."""

    def __init__(self, mist_args, val_losses):
        self._val_losses = list(val_losses)
        self.val_calls = []
        super().__init__(mist_args)

    def validation_step(self, **kwargs):
        """Return this epoch's validation loss and record the call."""
        epoch = kwargs["state"]["epoch"]
        self.val_calls.append(epoch)
        return torch.tensor(self._val_losses[epoch])


def _set_epochs(results, epochs, **training):
    """Rewrite config.json with the given epoch count and training keys."""
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["epochs"] = epochs
    cfg["training"].update(training)
    cfg_path.write_text(json.dumps(cfg))


@pytest.fixture
def saved_model_epochs(monkeypatch):
    """Record the epoch at which each inference model is saved."""
    saved = []
    monkeypatch.setattr(
        bt.BaseTrainer,
        "_save_model",
        staticmethod(lambda state, model_name: saved.append(state["epoch"])),
    )
    return saved


def test_final_model_saved_once_on_last_epoch(tmp_pipeline, mist_args, saved_model_epochs):
    """Without a validation split, only the final-epoch model is saved."""
    results, _ = tmp_pipeline
    _set_epochs(results, 3)

    # Validation improves every epoch, which used to save each time.
    trainer = SequenceValTrainer(mist_args, val_losses=[0.9, 0.5, 0.1])
    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert saved_model_epochs == [2]


def test_final_model_saved_even_if_validation_never_improves(
    tmp_pipeline, mist_args, saved_model_epochs
):
    """The final-epoch model is saved regardless of the validation loss."""
    results, _ = tmp_pipeline
    _set_epochs(results, 2)

    trainer = SequenceValTrainer(mist_args, val_losses=[float("inf"), float("inf")])
    trainer.train_fold(fold=0, rank=0, world_size=1)

    assert saved_model_epochs == [1]


def test_best_model_uses_smoothed_validation_loss(tmp_pipeline, mist_args, saved_model_epochs):
    """With a validation split, the best smoothed loss picks the saved model."""
    results, _ = tmp_pipeline
    _set_epochs(results, 4)

    # A single low epoch (0.1) moves the smoothed loss only a little.
    trainer = SequenceValTrainer(mist_args, val_losses=[1.0, 0.1, 1.0, 1.0])
    trainer.config["training"]["val_percent"] = 0.5
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # Smoothed: 1.0, 0.91, 0.919, 0.9271 -> improves at epochs 0 and 1 only.
    assert saved_model_epochs == [0, 1]


def test_update_best_model_smoothing_math(tmp_pipeline, mist_args, saved_model_epochs):
    """The smoothed loss is an EMA with the configured decay."""
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    trainer._update_best_model(state, 1.0, "unused.pt")
    assert state["smoothed_val_loss"] == pytest.approx(1.0)
    assert state["best_val_loss"] == pytest.approx(1.0)

    trainer._update_best_model(state, 0.0, "unused.pt")
    decay = bt.TrainerConstants.VAL_LOSS_EMA_DECAY
    assert state["smoothed_val_loss"] == pytest.approx(decay)
    assert state["best_val_loss"] == pytest.approx(decay)
    assert len(saved_model_epochs) == 2


def test_update_best_model_restarts_after_non_finite(tmp_pipeline, mist_args, saved_model_epochs):
    """A non-finite smoothed loss restarts smoothing at the next value."""
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    trainer._update_best_model(state, float("inf"), "unused.pt")
    assert saved_model_epochs == []

    trainer._update_best_model(state, 0.3, "unused.pt")
    assert state["smoothed_val_loss"] == pytest.approx(0.3)
    assert len(saved_model_epochs) == 1


def test_validate_every_skips_epochs_but_always_runs_last(tmp_pipeline, mist_args, monkeypatch):
    """Validation runs every N epochs and on the last epoch only."""
    results, _ = tmp_pipeline
    _set_epochs(results, 5, validate_every=2)

    writers = []

    class CapturingWriter(DummySummaryWriter):
        def __init__(self, log_dir):
            super().__init__(log_dir)
            writers.append(self)

    monkeypatch.setattr(bt, "SummaryWriter", CapturingWriter)

    trainer = SequenceValTrainer(mist_args, val_losses=[0.5] * 5)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # Fold 0 has two validation cases, so each validation makes two calls.
    val_steps = len(trainer.folds[0]["val_images"])
    assert trainer.val_calls == [e for e in (1, 3, 4) for _ in range(val_steps)]

    losses = [e for e in writers[-1].scalars if e[0] == "losses"]
    logged_val_steps = [step for _, values, step in losses if "validation" in values]
    assert logged_val_steps == [2, 4, 5]
    assert all("train" in values for _, values, _ in losses)


def test_small_validation_split_warns(tmp_pipeline, mist_args, monkeypatch):
    """A validation split with too few cases prints a warning."""
    results, _ = tmp_pipeline
    _set_epochs(results, 1, val_percent=0.5)

    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))

    DummyTrainer(mist_args)

    combined = "\n".join(printed)
    assert "holds out only 1 of 2" in combined


def test_overwrite_config_sets_compile(tmp_pipeline, mist_args):
    """--compile overrides training.compile."""
    mist_args.compile = True
    trainer = DummyTrainer(mist_args)
    assert trainer.config["training"]["compile"] is True


def test_overwrite_config_keeps_compile_without_the_flag(tmp_pipeline, mist_args):
    """Without --compile, the config's value stands."""
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"]["compile"] = True
    (Path(results) / "config.json").write_text(json.dumps(cfg))
    mist_args.compile = False
    trainer = DummyTrainer(mist_args)
    assert trainer.config["training"]["compile"] is True


def test_overwrite_config_sets_validate_every(tmp_pipeline, mist_args):
    """--validate-every overrides training.validate_every."""
    mist_args.validate_every = 3
    trainer = DummyTrainer(mist_args)
    assert trainer.config["training"]["validate_every"] == 3


def test_checkpoint_roundtrips_smoothed_val_loss(tmp_pipeline, mist_args, monkeypatch):
    """The smoothed validation loss survives save and load."""
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    state["smoothed_val_loss"] = 0.37
    trainer.save_checkpoint(fold=0, state=state)

    fresh_state = trainer.build_components(rank=0, world_size=1)
    trainer.load_checkpoint(fold=0, state=fresh_state)
    assert fresh_state["smoothed_val_loss"] == pytest.approx(0.37)


def test_load_pre_2_5_checkpoint_without_smoothed_loss(tmp_pipeline, mist_args, monkeypatch):
    """Checkpoints without a smoothed loss load with smoothing reset."""
    monkeypatch.setattr(torch, "save", _real_torch_save)
    monkeypatch.setattr(torch, "load", _real_torch_load)
    monkeypatch.setattr(bt.BaseTrainer, "save_checkpoint", _real_save_checkpoint)

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    trainer.save_checkpoint(fold=0, state=state)

    path = trainer._checkpoint_path(0)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["smoothed_val_loss"]
    torch.save(checkpoint, path)

    fresh_state = trainer.build_components(rank=0, world_size=1)
    fresh_state["smoothed_val_loss"] = 0.5
    trainer.load_checkpoint(fold=0, state=fresh_state)
    assert fresh_state["smoothed_val_loss"] is None


# =============================================
# Planning the patch and batch size at training
# =============================================


def _set_auto(results, **overrides):
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["spatial_config"]["patch_size"] = overrides.get("patch", "auto")
    cfg["training"]["batch_size_per_gpu"] = overrides.get("batch", "auto")
    cfg["preprocessing"]["median_resampled_image_size"] = [64, 64, 64]
    cfg["model"]["params"] = {"in_channels": 1, "out_channels": 2}
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def _fake_plan(monkeypatch, calls):
    def plan(config, num_cases, num_gpus):
        calls.append((num_cases, num_gpus))
        return bt.patch_planning.Plan([32, 32, 16], 3, 2 * 1024**3)

    monkeypatch.setattr(bt.patch_planning, "plan_for_config", plan)
    monkeypatch.setattr(bt.patch_planning, "detect_gpu_memory_bytes", lambda: (8.0 * 1024**3, 1))


def test_first_training_run_plans_and_saves(tmp_pipeline, mist_args, monkeypatch):
    """An "auto" config is planned once and written back to config.json."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto(results)
    calls = []
    _fake_plan(monkeypatch, calls)
    trainer = DummyTrainer(mist_args)

    assert len(calls) == 1 and calls[0][0] == 4  # All 4 cases in train_paths.csv.
    saved = json.loads(cfg_path.read_text())
    assert saved["spatial_config"]["patch_size"] == [32, 32, 16]
    assert saved["training"]["batch_size_per_gpu"] == 3
    assert saved["planning"]["gpu_memory_gb"] == 8.0
    assert trainer.batch_size == 3


def test_later_runs_reuse_the_saved_plan_and_only_warn(tmp_pipeline, mist_args, monkeypatch):
    """A saved plan is kept, whichever GPU the later run has."""
    results, _ = tmp_pipeline
    _set_auto(results)
    calls = []
    _fake_plan(monkeypatch, calls)
    DummyTrainer(mist_args)

    checked = []
    monkeypatch.setattr(
        bt.patch_planning, "check_plan_fits", lambda config, patch_was_set: checked.append(1)
    )
    trainer = DummyTrainer(mist_args)
    assert len(calls) == 1 and checked == [1]
    assert trainer.config["spatial_config"]["patch_size"] == [32, 32, 16]


def test_hand_set_values_reach_the_planner_unchanged(tmp_pipeline, mist_args, monkeypatch):
    """--patch-size is in the config by the time planning reads it."""
    results, _ = tmp_pipeline
    _set_auto(results)
    seen = []

    def plan(config, num_cases, num_gpus):
        seen.append(config["spatial_config"]["patch_size"])
        return bt.patch_planning.Plan(config["spatial_config"]["patch_size"], 2, 1024**3)

    monkeypatch.setattr(bt.patch_planning, "plan_for_config", plan)
    monkeypatch.setattr(bt.patch_planning, "detect_gpu_memory_bytes", lambda: (8.0 * 1024**3, 1))
    mist_args.patch_size = [48, 48, 48]
    DummyTrainer(mist_args)
    assert seen == [[48, 48, 48]]


def test_configs_with_real_values_are_never_replanned(tmp_pipeline, mist_args, monkeypatch):
    """Configs from earlier versions keep their patch and batch size."""
    calls = []
    _fake_plan(monkeypatch, calls)
    trainer = DummyTrainer(mist_args)
    assert calls == []
    assert trainer.config["spatial_config"]["patch_size"] == [16, 16, 16]


def _planned_run(tmp_pipeline, mist_args, monkeypatch):
    """Run first-time planning with a fake plan; return the config path and calls."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto(results)
    calls = []
    _fake_plan(monkeypatch, calls)
    DummyTrainer(mist_args)
    assert len(calls) == 1
    return cfg_path, calls


def test_changing_the_network_plans_again(tmp_pipeline, mist_args, monkeypatch):
    """A saved plan for another network is replaced, with the new one recorded."""
    cfg_path, calls = _planned_run(tmp_pipeline, mist_args, monkeypatch)
    mist_args.model = "other"
    printed = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg: printed.append(str(msg)))
    DummyTrainer(mist_args)
    assert len(calls) == 2
    assert json.loads(cfg_path.read_text())["planning"]["architecture"] == "other"
    assert any("planning again for other" in message for message in printed)


def test_same_network_keeps_the_saved_plan(tmp_pipeline, mist_args, monkeypatch):
    """Naming the network the plan was made for doesn't replan."""
    _, calls = _planned_run(tmp_pipeline, mist_args, monkeypatch)
    mist_args.model = "dummy"
    DummyTrainer(mist_args)
    assert len(calls) == 1


def test_an_alias_of_the_planned_network_keeps_the_saved_plan(tmp_pipeline, mist_args, monkeypatch):
    """nnunet and nnunet-res are the same network."""
    cfg_path, calls = _planned_run(tmp_pipeline, mist_args, monkeypatch)
    cfg = json.loads(cfg_path.read_text())
    cfg["model"]["architecture"] = "nnunet"
    cfg["planning"]["architecture"] = "nnunet"
    cfg_path.write_text(json.dumps(cfg))
    mist_args.model = "nnunet-res"
    DummyTrainer(mist_args)
    assert len(calls) == 1


def test_changing_the_network_keeps_a_patch_set_on_the_command_line(
    tmp_pipeline, mist_args, monkeypatch
):
    """--patch-size with a new --model stays, and only the batch is planned again."""
    cfg_path, calls = _planned_run(tmp_pipeline, mist_args, monkeypatch)
    seen = []
    real = bt.patch_planning.plan_for_config

    def plan(config, num_cases, num_gpus):
        seen.append(
            (config["spatial_config"]["patch_size"], config["training"]["batch_size_per_gpu"])
        )
        return real(config, num_cases, num_gpus)

    monkeypatch.setattr(bt.patch_planning, "plan_for_config", plan)
    mist_args.model = "other"
    mist_args.patch_size = [48, 48, 48]
    DummyTrainer(mist_args)
    assert seen == [([48, 48, 48], "auto")]


def test_changing_the_network_keeps_a_batch_edited_in_config(tmp_pipeline, mist_args, monkeypatch):
    """A batch edited after planning isn't the plan's, so it stays."""
    cfg_path, calls = _planned_run(tmp_pipeline, mist_args, monkeypatch)
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["batch_size_per_gpu"] = 7
    cfg_path.write_text(json.dumps(cfg))
    seen = []
    real = bt.patch_planning.plan_for_config

    def plan(config, num_cases, num_gpus):
        seen.append(
            (config["spatial_config"]["patch_size"], config["training"]["batch_size_per_gpu"])
        )
        return real(config, num_cases, num_gpus)

    monkeypatch.setattr(bt.patch_planning, "plan_for_config", plan)
    mist_args.model = "other"
    DummyTrainer(mist_args)
    assert seen == [("auto", 7)]


def test_config_without_a_plan_keeps_its_values_for_a_new_network(
    tmp_pipeline, mist_args, monkeypatch
):
    """Configs from earlier versions have no plan to replace."""
    calls = []
    _fake_plan(monkeypatch, calls)
    mist_args.model = "other"
    trainer = DummyTrainer(mist_args)
    assert calls == []
    assert trainer.config["spatial_config"]["patch_size"] == [16, 16, 16]


def test_resume_with_a_changed_network_still_raises(tmp_pipeline, mist_args, monkeypatch):
    """--resume can't replan a running job; the architecture check stops it first."""
    _, calls = _planned_run(tmp_pipeline, mist_args, monkeypatch)
    mist_args.model = "other"
    mist_args.resume = True
    with pytest.raises(ValueError, match="incompatible with the saved checkpoint"):
        DummyTrainer(mist_args)
    assert len(calls) == 1


# =============================================
# What planning prints
# =============================================


@pytest.fixture
def printed(monkeypatch):
    """Capture everything printed through the shared console."""
    messages = []
    monkeypatch.setattr(console_mod.console, "print", lambda msg="": messages.append(str(msg)))
    return messages


def test_first_run_announces_planning_then_the_result(
    tmp_pipeline, mist_args, monkeypatch, printed
):
    """Planning shows a header, what it's planning for, and what it chose, in order."""
    results, _ = tmp_pipeline
    _set_auto(results)
    _fake_plan(monkeypatch, [])
    DummyTrainer(mist_args)

    combined = "\n".join(printed)
    header = combined.index("Planning patch and batch size")
    target = combined.index("Planning dummy for 8.0 GiB")
    result = combined.index("Patch [32, 32, 16], batch size 3 per GPU")
    assert header < target < result
    assert "Saved to config.json" in combined


def test_first_run_without_a_gpu_says_so(tmp_pipeline, mist_args, monkeypatch, printed):
    """The default memory isn't presented as if it were a real GPU."""
    results, _ = tmp_pipeline
    _set_auto(results)
    _fake_plan(monkeypatch, [])
    monkeypatch.setattr(bt.patch_planning, "detect_gpu_memory_bytes", lambda: (None, 0))
    DummyTrainer(mist_args)
    assert any("No GPU found, so planning dummy" in message for message in printed)


def test_later_runs_say_they_reuse_the_saved_values(tmp_pipeline, mist_args, monkeypatch, printed):
    """Reusing a plan prints the values, with no planning header."""
    results, _ = tmp_pipeline
    _set_auto(results)
    _fake_plan(monkeypatch, [])
    DummyTrainer(mist_args)
    printed.clear()

    DummyTrainer(mist_args)
    combined = "\n".join(printed)
    assert "Using patch [32, 32, 16] and batch size 3 per GPU from config.json" in combined
    assert "Planning patch and batch size" not in combined


# =============================================
# Choosing the precision
# =============================================


def _set_auto_amp(results, **overrides):
    cfg_path = _set_auto(results, **overrides)
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["amp"] = "auto"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def _fake_precision(monkeypatch, amp, bf16_supported=True):
    """Fake the BF16 check, the two plans, and the speed test; record what's tested."""
    tested = []

    def plan(config, num_cases, num_gpus):
        batch = 8 if config["training"]["amp"] else 4
        return bt.patch_planning.Plan([32, 32, 16], batch, 2 * 1024**3)

    def choose(architecture, model_kwargs, bf16_plan, fp32_plan, memory):
        tested.append((architecture, bf16_plan.batch_size_per_gpu, fp32_plan.batch_size_per_gpu))
        return bt.precision_test.Choice(amp, "Speeds were measured.")

    monkeypatch.setattr(bt.patch_planning, "plan_for_config", plan)
    monkeypatch.setattr(bt.patch_planning, "detect_gpu_memory_bytes", lambda: (8.0 * 1024**3, 1))
    monkeypatch.setattr(bt.hardware, "bf16_supported", lambda: bf16_supported)
    monkeypatch.setattr(bt.precision_test, "choose_precision", choose)
    return tested


def test_auto_amp_uses_the_winning_precision_and_its_plan(
    tmp_pipeline, mist_args, monkeypatch, printed
):
    """FP32 winning saves amp false with the FP32 plan's batch size."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto_amp(results)
    tested = _fake_precision(monkeypatch, amp=False)
    trainer = DummyTrainer(mist_args)

    assert tested == [("dummy", 8, 4)]
    assert trainer.config["training"]["amp"] is False
    assert trainer.config["training"]["batch_size_per_gpu"] == 4
    saved = json.loads(cfg_path.read_text())
    assert saved["training"]["amp"] is False
    assert saved["planning"]["amp"] is False
    assert saved["planning"]["amp_auto"] is True

    combined = "\n".join(printed)
    assert "BF16 or FP32" in combined
    assert combined.index("Speeds were measured") < combined.index("batch size 4 per GPU")


def test_auto_amp_keeps_the_bf16_plan_when_bf16_wins(tmp_pipeline, mist_args, monkeypatch):
    """BF16 winning saves amp true with the BF16 plan's batch size."""
    results, _ = tmp_pipeline
    _set_auto_amp(results)
    _fake_precision(monkeypatch, amp=True)
    trainer = DummyTrainer(mist_args)
    assert trainer.config["training"]["amp"] is True
    assert trainer.config["training"]["batch_size_per_gpu"] == 8


def test_auto_amp_runs_the_speed_test_once(tmp_pipeline, mist_args, monkeypatch):
    """Later runs reuse the saved choice."""
    results, _ = tmp_pipeline
    _set_auto_amp(results)
    tested = _fake_precision(monkeypatch, amp=False)
    DummyTrainer(mist_args)
    trainer = DummyTrainer(mist_args)
    assert len(tested) == 1
    assert trainer.config["training"]["amp"] is False


def test_auto_amp_without_bf16_hardware_is_fp32_without_a_test_or_warning(
    tmp_pipeline, mist_args, monkeypatch
):
    """A pre-Ampere GPU has nothing to compare, and "auto" didn't ask for BF16."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto_amp(results)
    tested = _fake_precision(monkeypatch, amp=True, bf16_supported=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        DummyTrainer(mist_args)
    assert tested == []
    assert json.loads(cfg_path.read_text())["training"]["amp"] is False


def test_auto_amp_without_bf16_hardware_is_saved_with_a_hand_set_plan(
    tmp_pipeline, mist_args, monkeypatch
):
    """With nothing to plan, the resolved precision still reaches config.json."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto_amp(results, patch=[16, 16, 16], batch=2)
    _fake_precision(monkeypatch, amp=True, bf16_supported=False)
    DummyTrainer(mist_args)
    assert json.loads(cfg_path.read_text())["training"]["amp"] is False


def test_auto_amp_is_tested_even_with_a_hand_set_plan(tmp_pipeline, mist_args, monkeypatch):
    """A hand-set patch and batch stay, and the precision is still chosen."""
    results, _ = tmp_pipeline
    _set_auto_amp(results, patch=[16, 16, 16], batch=2)
    tested = _fake_precision(monkeypatch, amp=False)

    def keep(config, num_cases, num_gpus):
        return bt.patch_planning.Plan(
            config["spatial_config"]["patch_size"],
            config["training"]["batch_size_per_gpu"],
            1024**3,
        )

    monkeypatch.setattr(bt.patch_planning, "plan_for_config", keep)
    trainer = DummyTrainer(mist_args)
    assert len(tested) == 1
    assert trainer.config["training"]["amp"] is False
    assert trainer.config["spatial_config"]["patch_size"] == [16, 16, 16]
    assert trainer.config["training"]["batch_size_per_gpu"] == 2


@pytest.mark.parametrize("amp", [True, False])
def test_amp_set_by_hand_skips_the_speed_test(tmp_pipeline, mist_args, monkeypatch, amp):
    """true and false are kept as they are, including in configs from earlier versions."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto(results)
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["amp"] = amp
    cfg_path.write_text(json.dumps(cfg))
    tested = _fake_precision(monkeypatch, amp=not amp)
    trainer = DummyTrainer(mist_args)
    assert tested == []
    assert trainer.config["training"]["amp"] is amp
    assert json.loads(cfg_path.read_text())["planning"]["amp_auto"] is False


def test_changing_the_network_tests_a_chosen_precision_again(tmp_pipeline, mist_args, monkeypatch):
    """Which precision is faster can depend on the network."""
    results, _ = tmp_pipeline
    _set_auto_amp(results)
    tested = _fake_precision(monkeypatch, amp=False)
    DummyTrainer(mist_args)
    mist_args.model = "other"
    DummyTrainer(mist_args)
    assert [entry[0] for entry in tested] == ["dummy", "other"]


def test_changing_the_network_keeps_a_precision_set_by_hand(tmp_pipeline, mist_args, monkeypatch):
    """A hand-set amp survives replanning for another network."""
    results, _ = tmp_pipeline
    cfg_path = _set_auto(results)
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["amp"] = True
    cfg_path.write_text(json.dumps(cfg))
    tested = _fake_precision(monkeypatch, amp=False)
    DummyTrainer(mist_args)
    mist_args.model = "other"
    trainer = DummyTrainer(mist_args)
    assert tested == []
    assert trainer.config["training"]["amp"] is True
