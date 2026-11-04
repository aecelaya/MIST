"""Every trainable parameter must get a gradient, or multi-GPU training crashes.

DistributedDataParallel raises an error on the second step if a trainable
parameter received no gradient, so single-GPU and CPU runs never show this.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from mist.models.model_registry import get_model_from_registry, list_registered_models

# Stage count depends on patch size, so cover shallow and deep networks.
ADAPTIVE_DECODERS = ["nnunet-plain", "resenc"]


def _build(name, patch_size):
    return get_model_from_registry(
        name,
        in_channels=1,
        out_channels=2,
        patch_size=list(patch_size),
        target_spacing=[1.0, 1.0, 1.0],
    )


def _training_loss(model, patch_size):
    output = model(torch.zeros(1, 1, *patch_size))
    if not isinstance(output, dict):
        return output.mean()
    return output["prediction"].mean() + sum(d.mean() for d in output["deep_supervision"] or [])


def _unused(model, patch_size):
    model.train()
    _training_loss(model, patch_size).backward()
    return [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


@pytest.mark.parametrize("name", list_registered_models())
def test_every_architecture_uses_all_trainable_parameters(name):
    """No registered network leaves a trainable parameter without a gradient."""
    # SwinUNETR's bottleneck needs more than one voxel per axis.
    patch_size = (64, 64, 64) if name.startswith("swinunetr") else (32, 32, 32)
    assert _unused(_build(name, patch_size), patch_size) == []


@pytest.mark.parametrize("name", ADAPTIVE_DECODERS)
@pytest.mark.parametrize("patch_size", [(16, 16, 16), (32, 32, 8), (64, 64, 64), (96, 96, 32)])
def test_deep_decoders_use_all_trainable_parameters(name, patch_size):
    """Holds for networks with fewer and more stages than deep supervision uses."""
    assert _unused(_build(name, patch_size), patch_size) == []


@pytest.mark.parametrize("name", ADAPTIVE_DECODERS)
def test_frozen_seg_layers_stay_in_the_state_dict(name):
    """Freezing unused layers leaves checkpoints loadable by the same keys."""
    model = _build(name, (64, 64, 64))
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(n.startswith("decoder.seg_layers.") for n in frozen)
    assert all(n in model.state_dict() for n in frozen)
    # The full-resolution layer and the two deep supervision heads still train.
    assert all(p.requires_grad for p in model.decoder.seg_layers[-3:].parameters())


def _ddp_steps(rank, name, patch_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    model = DDP(_build(name, patch_size).train())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        _training_loss(model, patch_size).backward()
        optimizer.step()
    dist.destroy_process_group()


@pytest.mark.parametrize("name, port", [("nnunet-plain", 29611), ("resenc", 29612)])
def test_real_ddp_trains_past_the_second_step(name, port):
    """Two processes take several steps; the original crash came on the second."""
    mp.spawn(_ddp_steps, args=(name, (64, 64, 64), port), nprocs=2, start_method="fork")
