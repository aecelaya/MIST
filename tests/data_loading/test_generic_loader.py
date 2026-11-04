"""Tests for the generic, accelerator-agnostic data loader in MIST.

Split into three groups:

* Pure unit tests (no I/O) for the Stage 2 helper functions -- sharding,
  padding, anchor placement, foreground selection.
* Pure unit tests (no I/O) for the Stage 3 augmentation functions -- flips,
  zoom, noise, blur, brightness, contrast.
* An integration/smoke test running the full get_training_dataset /
  get_validation_dataset / get_test_dataset entry points against real
  preprocessed fixture data (reusing tests/regression/ants_sitk's harness),
  since that's the only way to exercise the actual .next()/.reset() iterator
  contract end to end -- including a synthetic "does it train" check with
  augmentation enabled, per the Stage 3 release gate.

See cpu_rocm_support_plan.md Stages 2-3 for the design this is validating
against: same signatures/contract as dali_loader.py, foreground-oversampling
rate (not RNG stream) equivalence, DDP sharding with no drops/duplicates, and
augmentations matching data_loading_constants.py's probabilities/ranges (not
DALI's exact numerical output).
"""

import gc
import multiprocessing
import os
import subprocess
import sys
import time
import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mist.data_loading import augmentation
from mist.data_loading import generic_loader as gl
from mist.utils import hardware
from tests.regression.ants_sitk.fixtures import generate_dataset
from tests.regression.ants_sitk.harness import run_pipeline

# --------------------------------------------------------------------------- #
# _shard_indices
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("num_examples", "world_size"),
    [(0, 1), (1, 1), (4, 1), (4, 2), (5, 2), (7, 3), (1, 4), (10, 4)],
)
def test_shard_indices_partitions_with_no_drops_or_duplicates(num_examples, world_size):
    """Every index lands in exactly one rank's shard; sizes differ by <= 1."""
    shards = [gl._shard_indices(num_examples, rank, world_size) for rank in range(world_size)]

    all_indices = [i for shard in shards for i in shard]
    assert sorted(all_indices) == list(range(num_examples))  # No drops, no dupes.

    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1


# --------------------------------------------------------------------------- #
# _random_anchor
# --------------------------------------------------------------------------- #


def test_random_anchor_without_center_stays_in_bounds():
    """A plain random anchor is always a valid crop start."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        anchor = gl._random_anchor(dim_size=20, roi=6, rng=rng, center=None)
        assert 0 <= anchor <= 20 - 6


def test_random_anchor_with_center_keeps_center_in_patch():
    """A foreground-centered anchor always places `center` inside the patch."""
    rng = np.random.default_rng(0)
    dim_size, roi, center = 20, 6, 15
    for _ in range(50):
        anchor = gl._random_anchor(dim_size, roi, rng, center=center)
        assert 0 <= anchor <= dim_size - roi
        assert anchor <= center < anchor + roi


# --------------------------------------------------------------------------- #
# _foreground_anchor_center
# --------------------------------------------------------------------------- #


def test_foreground_anchor_center_returns_none_without_foreground():
    """No foreground voxels anywhere -> None (caller falls back to random)."""
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    rng = np.random.default_rng(0)
    locations = gl._foreground_locations(label, [1, 2], rng)
    assert locations == {}
    assert gl._foreground_anchor_center(locations, labels=[1, 2], rng=rng) is None


def test_foreground_anchor_center_finds_the_only_voxel():
    """A single foreground voxel is found deterministically."""
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    label[3, 4, 5, 0] = 1
    rng = np.random.default_rng(0)
    locations = gl._foreground_locations(label, [1, 2], rng)
    assert gl._foreground_anchor_center(locations, labels=[1, 2], rng=rng) == (3, 4, 5)


def test_foreground_anchor_center_falls_back_to_any_nonzero_voxel():
    """A class not present in `labels` still yields a fallback voxel."""
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    label[1, 2, 3, 0] = 9  # Not in the configured `labels` list below.
    rng = np.random.default_rng(0)
    locations = gl._foreground_locations(label, [1, 2], rng)
    assert gl._foreground_anchor_center(locations, labels=[1, 2], rng=rng) == (1, 2, 3)


def test_foreground_locations_caps_each_class(monkeypatch):
    """Large classes keep a bounded subset of real, distinct locations."""
    monkeypatch.setattr(gl, "_MAX_FOREGROUND_LOCATIONS", 50)
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    label[:5] = 1  # 500 voxels, over the cap.
    label[9, 9, 9, 0] = 2  # One voxel, under it.
    locations = gl._foreground_locations(label, [1, 2], np.random.default_rng(0))
    assert locations[1].shape == (50, 3)
    assert len({tuple(c) for c in locations[1].tolist()}) == 50
    assert np.all(locations[1][:, 0] < 5)
    assert locations[2].tolist() == [[9, 9, 9]]


def test_foreground_anchor_center_weights_classes_equally():
    """A tiny class is picked as often as a huge one, as in dali_loader.py."""
    label = np.zeros((20, 20, 20, 1), dtype=np.uint8)
    label[:10] = 1
    label[19, 19, 19, 0] = 2
    rng = np.random.default_rng(0)
    locations = gl._foreground_locations(label, [1, 2], rng)
    picks = [gl._foreground_anchor_center(locations, [1, 2], rng) for _ in range(1000)]
    assert 0.4 < np.mean([p == (19, 19, 19) for p in picks]) < 0.6


def test_dataset_scans_each_label_once(tmp_path, monkeypatch):
    """Foreground locations are found once per case, then reused."""
    array = np.zeros((10, 10, 10, 1), dtype=np.float32)
    array[5, 5, 5, 0] = 1
    paths = [str(tmp_path / f"{i}.npy") for i in range(2)]
    for path in paths:
        np.save(path, array)

    calls = []
    real = gl._foreground_locations
    monkeypatch.setattr(
        gl, "_foreground_locations", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    dataset = gl._PatchTrainingDataset(paths, paths, None, (4, 4, 4), [1], 1.0, True)
    for _ in range(5):
        dataset[0]
        dataset[1]
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# _extract_patch
# --------------------------------------------------------------------------- #


def test_extract_patch_returns_roi_shaped_arrays():
    """The extracted patch always has exactly roi_size spatial dims."""
    rng = np.random.default_rng(0)
    image = np.random.default_rng(1).random((10, 10, 10, 1)).astype(np.float32)
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    label[5, 5, 5, 0] = 1
    dtm = np.random.default_rng(2).random((10, 10, 10, 3)).astype(np.float32)

    out_image, out_label, out_dtm = gl._extract_patch(
        image, label, dtm, roi_size=(4, 4, 4), labels=[1], oversampling=0.5, rng=rng
    )
    assert out_image.shape == (4, 4, 4, 1)
    assert out_label.shape == (4, 4, 4, 1)
    assert out_dtm.shape == (4, 4, 4, 3)


def test_extract_patch_with_oversampling_one_always_includes_foreground():
    """oversampling=1.0 with a known foreground voxel always captures it."""
    label = np.zeros((20, 20, 20, 1), dtype=np.uint8)
    label[10, 10, 10, 0] = 1
    image = np.zeros((20, 20, 20, 1), dtype=np.float32)

    for trial in range(20):
        rng = np.random.default_rng(trial)
        _, out_label, _ = gl._extract_patch(
            image, label, None, roi_size=(6, 6, 6), labels=[1], oversampling=1.0, rng=rng
        )
        assert np.any(out_label == 1), f"trial {trial} missed the foreground voxel"


def test_extract_patch_pads_when_image_smaller_than_roi():
    """A patch larger than the source image triggers padding, not a crash."""
    rng = np.random.default_rng(0)
    image = np.ones((4, 4, 4, 1), dtype=np.float32)
    label = np.zeros((4, 4, 4, 1), dtype=np.uint8)

    out_image, out_label, out_dtm = gl._extract_patch(
        image, label, None, roi_size=(8, 8, 8), labels=None, oversampling=None, rng=rng
    )
    assert out_image.shape == (8, 8, 8, 1)
    assert out_label.shape == (8, 8, 8, 1)
    assert out_dtm is None


# --------------------------------------------------------------------------- #
# _to_channels_first
# --------------------------------------------------------------------------- #


def test_to_channels_first_moves_channel_axis_to_front():
    """(D, H, W, C) becomes (C, D, H, W)."""
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32)
    tensor = gl._to_channels_first(array)
    assert tensor.shape == (5, 2, 3, 4)
    assert torch.equal(tensor, torch.from_numpy(np.moveaxis(array, -1, 0)))


def test_to_channels_first_never_warns_on_a_real_memory_mapped_input(tmp_path):
    """No writable-tensor warning, given a genuine memmap (not just a copy).

    Regression guard: np.ascontiguousarray is a no-op (returns the *same*
    array, no copy) whenever its input is already C-contiguous -- which a
    moveaxis view over `_load_case`'s memory-mapped (mmap_mode="r") reads
    can be, e.g. for `_FullVolumeDataset`, which never crops. Using
    ascontiguousarray there used to silently hand back a read-only array
    wrapped in a torch.Tensor, and PyTorch warns loudly about exactly that
    ("writing to this tensor will result in undefined behavior").

    This must use a *real* memmap, not a plain array with its writeable flag
    toggled off: torch.from_numpy() reports the resulting tensor as writable
    either way (that's the "undefined behavior" part -- the flag doesn't
    propagate reliably), so the only safe, reliable signal is whether
    PyTorch's own warning fires -- and actually attempting to write to a
    tensor genuinely backed by a read-only mmap crashes the whole process
    with SIGBUS rather than raising a catchable Python exception, so this
    test must never attempt that write, only check for the warning.
    """
    path = tmp_path / "single_channel.npy"
    np.save(path, np.arange(2 * 3 * 4).reshape(2, 3, 4, 1).astype(np.float32))
    array = gl._load_case(str(path))
    assert np.moveaxis(array, -1, 0).flags["C_CONTIGUOUS"] is True  # the trigger condition

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gl._to_channels_first(array)


def test_load_case_returns_a_memory_mapped_array(tmp_path):
    """_load_case mmaps rather than fully reading the file into memory.

    Regression guard for the fix that stopped re-reading whole volumes from
    disk on every single patch draw (see _load_case's docstring) -- confirms
    it's actually memory-mapped, not just that the values happen to match.
    """
    path = tmp_path / "case.npy"
    np.save(path, np.zeros((4, 5, 6, 1), dtype=np.float32))
    array = gl._load_case(str(path))
    assert isinstance(array, np.memmap)
    assert array.flags["WRITEABLE"] is False


# --------------------------------------------------------------------------- #
# _validate_train_and_eval_inputs
# --------------------------------------------------------------------------- #


def test_validate_train_and_eval_inputs_rejects_empty_images():
    """No images raises, mirroring dali_loader.py's validation."""
    with pytest.raises(ValueError, match="No images found"):
        gl._validate_train_and_eval_inputs([], ["a.npy"])


def test_validate_train_and_eval_inputs_rejects_empty_labels():
    """No labels raises."""
    with pytest.raises(ValueError, match="No labels found"):
        gl._validate_train_and_eval_inputs(["a.npy"], [])


def test_validate_train_and_eval_inputs_rejects_length_mismatch():
    """Mismatched image/label counts raise."""
    with pytest.raises(ValueError, match="Number of images and labels do not match"):
        gl._validate_train_and_eval_inputs(["a.npy", "b.npy"], ["a.npy"])


def test_validate_train_and_eval_inputs_rejects_dtm_length_mismatch():
    """Mismatched image/DTM counts raise."""
    with pytest.raises(ValueError, match="Number of images and DTMs do not match"):
        gl._validate_train_and_eval_inputs(["a.npy", "b.npy"], ["a.npy", "b.npy"], ["a.npy"])


# --------------------------------------------------------------------------- #
# _target_device
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("accelerator", "expected"),
    [
        ("cpu", torch.device("cpu")),
        ("cuda", torch.device("cuda", 1)),
        ("rocm", torch.device("cuda", 1)),
    ],
)
def test_target_device_resolves_per_accelerator(monkeypatch, accelerator, expected):
    """CPU has no per-rank device; CUDA/ROCm both target "cuda:<rank>"."""
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    assert gl._target_device(rank=1) == expected


# --------------------------------------------------------------------------- #
# _multiprocessing_context
# --------------------------------------------------------------------------- #


def test_multiprocessing_context_none_when_no_workers():
    """num_workers <= 0 means no worker processes, so context is irrelevant."""
    assert gl._multiprocessing_context(0) is None
    assert gl._multiprocessing_context(-1) is None


def test_multiprocessing_context_fork_on_non_windows(monkeypatch):
    """ "fork" is requested explicitly on Linux/macOS -- see module docstring
    for why this matters on macOS specifically (a confirmed indefinite hang
    with the platform default "spawn", found via a real Stage 4 CPU run)."""
    monkeypatch.setattr(gl.platform, "system", lambda: "Darwin")
    assert gl._multiprocessing_context(4) == "fork"

    monkeypatch.setattr(gl.platform, "system", lambda: "Linux")
    assert gl._multiprocessing_context(4) == "fork"


def test_multiprocessing_context_none_on_windows(monkeypatch):
    """Windows has no fork(); fall back to torch's own default (spawn)."""
    monkeypatch.setattr(gl.platform, "system", lambda: "Windows")
    assert gl._multiprocessing_context(4) is None


# --------------------------------------------------------------------------- #
# _pin_memory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("accelerator", "expected"),
    [("cpu", False), ("cuda", True), ("rocm", True)],
)
def test_pin_memory_only_on_a_real_accelerator(monkeypatch, accelerator, expected):
    """pin_memory is requested on CUDA/ROCm, never on CPU-only hardware.

    Meaningless (and PyTorch warns about it) with no accelerator to pin
    memory for -- gated the same way _target_device is.
    """
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    assert gl._pin_memory() is expected


# --------------------------------------------------------------------------- #
# _prefetch_factor
# --------------------------------------------------------------------------- #


def test_prefetch_factor_none_when_no_workers():
    """num_workers <= 0 means no worker processes to prefetch ahead of --
    PyTorch raises a ValueError if prefetch_factor is set in that case."""
    assert gl._prefetch_factor(0) is None
    assert gl._prefetch_factor(-1) is None


def test_prefetch_factor_set_when_workers_present():
    """A positive prefetch_factor is requested whenever workers exist."""
    assert gl._prefetch_factor(4) == 4


# --------------------------------------------------------------------------- #
# Long-lived training workers
# --------------------------------------------------------------------------- #


def test_endless_sampler_covers_every_index_each_pass():
    """Each pass holds every index once, and passes keep coming in new orders."""
    sampler = gl._EndlessShuffleSampler(5, torch.Generator().manual_seed(0))
    drawn = list(islice(iter(sampler), 15))
    passes = [drawn[:5], drawn[5:10], drawn[10:]]
    assert all(sorted(p) == list(range(5)) for p in passes)
    assert passes[0] != passes[1] or passes[1] != passes[2]


class _PidDataset(Dataset):
    """Report which process produced each item."""

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"pid": torch.tensor(os.getpid())}


def _endless_iterator(num_workers):
    return gl.GenericIterator(
        DataLoader(
            _PidDataset(),
            batch_size=2,
            sampler=gl._EndlessShuffleSampler(3, torch.Generator().manual_seed(0)),
            num_workers=num_workers,
            multiprocessing_context=gl._multiprocessing_context(num_workers),
        ),
        device=torch.device("cpu"),
        endless=True,
    )


def test_endless_iterator_keeps_its_workers_across_passes_and_resets():
    """The same worker processes serve many passes and epoch boundaries."""
    iterator = _endless_iterator(num_workers=2)
    pids = set()
    for _ in range(3):
        for _ in range(6):  # Two passes' worth of batches.
            pids.update(iterator.next()[0]["pid"].tolist())
        iterator.reset()
    assert 1 <= len(pids) <= 2
    assert os.getpid() not in pids


def test_endless_iterator_works_without_workers():
    """With no workers, batches come from this process and stay full-sized."""
    iterator = _endless_iterator(num_workers=0)
    for _ in range(5):
        batch = iterator.next()[0]
        assert batch["pid"].tolist() == [os.getpid()] * 2
    iterator.reset()
    assert iterator.next()[0]["pid"].shape == (2,)


def test_dropping_an_endless_iterator_stops_its_workers():
    """Workers don't outlive their fold's iterator."""
    before = set(multiprocessing.active_children())
    iterator = _endless_iterator(num_workers=2)
    iterator.next()
    assert set(multiprocessing.active_children()) - before
    del iterator
    gc.collect()
    deadline = time.monotonic() + 20
    while set(multiprocessing.active_children()) - before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not set(multiprocessing.active_children()) - before


def test_process_with_endless_workers_exits_cleanly(tmp_path):
    """Two folds' worth of workers, then a normal exit: no hang at shutdown."""
    script = tmp_path / "folds.py"
    script.write_text(
        """
import numpy as np
from mist.data_loading import generic_loader as gl

array = np.zeros((10, 10, 10, 1), dtype=np.float32)
array[5, 5, 5, 0] = 1
paths = []
for i in range(3):
    paths.append(f"case{i}.npy")
    np.save(paths[-1], array)

for fold in range(2):
    loader = gl.get_training_dataset(
        paths, paths, None, 2, (4, 4, 4), [1], 0.5, fold, 2, 0, 1
    )
    for _ in range(5):
        loader.next()
    loader.reset()
    loader.next()
print("done")
"""
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "done" in result.stdout


# --------------------------------------------------------------------------- #
# _extract_patch with a spatial-augmentation margin
# --------------------------------------------------------------------------- #


def test_slice_with_padding_zero_pads_outside_bounds():
    """Slicing past the edges pads with zeros and keeps the inside values."""
    array = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4, 1)
    out = gl._slice_with_padding(array, starts=(-2, 1, 2), shape=(6, 2, 4))
    assert out.shape == (6, 2, 4, 1)
    np.testing.assert_array_equal(out[2:6, :, :2], array[0:4, 1:3, 2:4])
    assert np.all(out[:2] == 0)
    assert np.all(out[:, :, 2:] == 0)


def test_slice_with_padding_never_reads_outside_the_slice(monkeypatch):
    """A patch past the edge of a case neither pads nor reads the whole array."""

    def fail(*args, **kwargs):
        raise AssertionError("np.pad would read the whole memory-mapped case")

    monkeypatch.setattr(gl.np, "pad", fail)
    array = np.ones((4, 4, 4, 2), dtype=np.float32)
    out = gl._slice_with_padding(array, starts=(2, -1, 0), shape=(4, 4, 4))
    assert out.shape == (4, 4, 4, 2)
    assert out.dtype == array.dtype
    assert out.sum() == 2 * 3 * 4 * 2


def test_slice_with_padding_inside_bounds_returns_a_view():
    """A slice inside the array copies nothing."""
    array = np.ones((4, 4, 4, 1), dtype=np.float32)
    out = gl._slice_with_padding(array, starts=(0, 1, 1), shape=(2, 2, 2))
    assert np.shares_memory(out, array)


def test_slice_with_padding_entirely_outside_is_all_zeros():
    """A slice that misses the array comes back as zeros of the right shape."""
    array = np.ones((4, 4, 4, 1), dtype=np.float32)
    assert not gl._slice_with_padding(array, starts=(9, 0, 0), shape=(2, 2, 2)).any()
    assert not gl._slice_with_padding(array, starts=(-5, 0, 0), shape=(2, 2, 2)).any()


def test_extract_patch_with_margin_centers_the_unmargined_patch():
    """The enlarged patch's center is the patch the loader would have cut."""
    shape = (20, 20, 20)
    image = np.random.default_rng(0).normal(size=(*shape, 1)).astype(np.float32)
    label = np.zeros((*shape, 1), dtype=np.float32)

    plain = gl._extract_patch(image, label, None, (8, 8, 8), None, None, np.random.default_rng(5))[
        0
    ]
    margined = gl._extract_patch(
        image, label, None, (8, 8, 8), None, None, np.random.default_rng(5), (3, 0, 2)
    )[0]
    assert margined.shape == (14, 8, 12, 1)
    np.testing.assert_array_equal(margined[3:11, :, 2:10], plain)


# --------------------------------------------------------------------------- #
# Integration: real preprocessed data through the public entry points
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def preprocessed(tmp_path_factory) -> dict:
    """Run analyze -> preprocess once and return everything the tests need."""
    root = tmp_path_factory.mktemp("generic_loader_fixture")
    dataset = generate_dataset(root / "dataset")
    outputs = run_pipeline(dataset.dataset_json, root / "results", root / "numpy")

    import json

    import pandas as pd

    config = json.loads((outputs.results_dir / "config.json").read_text())
    train_df = pd.read_csv(outputs.results_dir / "train_paths.csv")
    patient_ids = list(train_df["id"])

    def npy_paths(subdir: str) -> list[str]:
        return [str(outputs.numpy_dir / subdir / f"{pid}.npy") for pid in patient_ids]

    return {
        "config": config,
        "patient_ids": patient_ids,
        "image_paths": npy_paths("images"),
        "label_paths": npy_paths("labels"),
        "dtm_paths": npy_paths("dtms"),
    }


def test_get_training_dataset_yields_roi_shaped_batches_and_cycles(preprocessed):
    """.next() yields correctly-shaped batches, more times than there are cases."""
    patch_size = tuple(preprocessed["config"]["spatial_config"]["patch_size"])
    labels = preprocessed["config"]["dataset_info"]["labels"][1:]  # Foreground only.
    num_cases = len(preprocessed["image_paths"])

    loader = gl.get_training_dataset(
        image_paths=preprocessed["image_paths"],
        label_paths=preprocessed["label_paths"],
        dtm_paths=preprocessed["dtm_paths"],
        batch_size=2,
        roi_size=patch_size,
        labels=labels,
        oversampling=0.8,
        seed=0,
        num_workers=0,
        rank=0,
        world_size=1,
    )

    # Call .next() more times than a single pass over the dataset would allow
    # (num_cases // batch_size), to exercise the auto-cycling contract.
    for _ in range(num_cases * 3):
        batch = loader.next()[0]
        assert batch["image"].shape[1:] == (1, *patch_size)
        assert batch["label"].shape[1:] == (1, *patch_size)
        assert batch["dtm"].shape[1:] == (3, *patch_size)
        assert batch["image"].device == gl._target_device(0)

    loader.reset()
    batch = loader.next()[0]
    assert batch["image"].shape[1:] == (1, *patch_size)


def test_get_validation_dataset_yields_full_size_unpatched_images(preprocessed):
    """Validation batches are full-size (not cropped to patch_size)."""
    loader = gl.get_validation_dataset(
        image_paths=preprocessed["image_paths"],
        label_paths=preprocessed["label_paths"],
        seed=0,
        num_workers=0,
        rank=0,
        world_size=1,
    )
    seen_shapes = set()
    for _ in range(len(preprocessed["image_paths"])):
        batch = loader.next()[0]
        assert batch["image"].shape[0] == 1  # batch_size=1, always.
        seen_shapes.add(tuple(batch["image"].shape))
    # The four fixtures have four distinct sizes -- confirms these are full
    # images, not uniformly-cropped patches.
    assert len(seen_shapes) == len(preprocessed["image_paths"])


def test_get_test_dataset_preserves_input_order(preprocessed):
    """Test batches come back in the exact order of the input path list."""
    loader = gl.get_test_dataset(
        image_paths=preprocessed["image_paths"],
        seed=0,
        num_workers=0,
    )
    for expected_path in preprocessed["image_paths"]:
        batch = loader.next()[0]
        expected = gl._to_channels_first(np.load(expected_path)).unsqueeze(0)
        assert torch.equal(batch["image"], expected)


def test_get_test_dataset_rejects_empty_input():
    """No images raises ValueError, matching dali_loader.py."""
    with pytest.raises(ValueError, match="No images found"):
        gl.get_test_dataset(image_paths=[], seed=0, num_workers=0)


def test_training_dataset_shards_across_ranks_cover_all_cases(preprocessed):
    """Sharded training loaders across 2 ranks cover every case, no overlap."""
    world_size = 2
    seen_images = []
    for rank in range(world_size):
        loader = gl.get_training_dataset(
            image_paths=preprocessed["image_paths"],
            label_paths=preprocessed["label_paths"],
            dtm_paths=None,
            batch_size=1,
            roi_size=tuple(preprocessed["config"]["spatial_config"]["patch_size"]),
            labels=None,
            oversampling=0.0,
            seed=0,
            num_workers=0,
            rank=rank,
            world_size=world_size,
        )
        # One full pass over this rank's shard.
        shard_size = len(gl._shard_indices(len(preprocessed["image_paths"]), rank, world_size))
        for _ in range(shard_size):
            loader.next()
        seen_images.append(shard_size)

    assert sum(seen_images) == len(preprocessed["image_paths"])


@pytest.mark.parametrize("with_dtm", [True, False])
def test_get_training_dataset_with_margin_and_augmentation(preprocessed, with_dtm):
    """Enlarged patches, augmented on the device, come out patch-shaped."""
    roi_size = (8, 8, 8)
    settings = augmentation.AugmentationSettings()
    loader = gl.get_training_dataset(
        image_paths=preprocessed["image_paths"],
        label_paths=preprocessed["label_paths"],
        dtm_paths=preprocessed["dtm_paths"] if with_dtm else None,
        batch_size=2,
        roi_size=roi_size,
        labels=preprocessed["config"]["dataset_info"]["labels"][1:],
        oversampling=0.5,
        seed=0,
        num_workers=0,
        rank=0,
        world_size=1,
        patch_margin=augmentation.spatial_margin(roi_size, settings),
    )
    raw = loader.next()[0]
    assert raw["image"].shape[2:] == tuple(
        r + 2 * m for r, m in zip(roi_size, augmentation.spatial_margin(roi_size, settings))
    )

    augmented = augmentation.AugmentedIterator(
        loader, augmentation.BatchAugmenter(roi_size, settings, seed=0)
    )
    for _ in range(6):
        batch = augmented.next()[0]
        assert batch["image"].shape[1:] == (1, *roi_size)
        assert batch["label"].shape[1:] == (1, *roi_size)
        if with_dtm:
            assert batch["dtm"].shape[1:] == (3, *roi_size)


def test_training_with_augmentation_enabled_reduces_loss(preprocessed):
    """A model can actually train on augmented batches.

    Just a check that augmented batches are well-formed enough for a tiny
    model to fit this small, fixed dataset. Patch placement in the loader is
    unseeded by design (see generic_loader.py's module docstring), so the
    assertion is on the best late loss rather than an exact value.
    """
    torch.manual_seed(0)

    roi_size = (12, 12, 12)
    settings = augmentation.AugmentationSettings()
    loader = augmentation.AugmentedIterator(
        gl.get_training_dataset(
            image_paths=preprocessed["image_paths"],
            label_paths=preprocessed["label_paths"],
            dtm_paths=None,
            batch_size=2,
            roi_size=roi_size,
            labels=preprocessed["config"]["dataset_info"]["labels"][1:],
            oversampling=0.8,
            seed=0,
            num_workers=0,
            rank=0,
            world_size=1,
            patch_margin=augmentation.spatial_margin(roi_size, settings),
        ),
        augmentation.BatchAugmenter(roi_size, settings, seed=1234),
    )

    model = nn.Sequential(
        nn.Conv3d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(8, 1, 3, padding=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(40):
        batch = loader.next()[0]
        image = batch["image"]
        target = (batch["label"] > 0).float()

        optimizer.zero_grad()
        loss = criterion(model(image), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    early = sum(losses[:5]) / 5
    best_late = min(losses[-10:])
    assert best_late < early, f"loss did not decrease: early={early:.4f}, best_late={best_late:.4f}"
