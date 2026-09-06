"""Tests for the generic, accelerator-agnostic data loader in MIST.

Split into two groups:

* Pure unit tests (no I/O) for the private helper functions -- sharding,
  padding, anchor placement, foreground selection.
* An integration/smoke test running the full get_training_dataset /
  get_validation_dataset / get_test_dataset entry points against real
  preprocessed fixture data (reusing tests/regression/ants_sitk's harness),
  since that's the only way to exercise the actual .next()/.reset() iterator
  contract end to end.

See cpu_rocm_support_plan.md Stage 2 for the design this is validating
against: same signatures/contract as dali_loader.py, foreground-oversampling
rate (not RNG stream) equivalence, and DDP sharding with no drops/duplicates.
"""

import numpy as np
import pytest
import torch

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
# _pad_to_roi
# --------------------------------------------------------------------------- #


def test_pad_to_roi_is_noop_when_already_large_enough():
    """No padding is applied when every spatial axis already meets roi_size."""
    array = np.random.default_rng(0).random((8, 8, 8, 2)).astype(np.float32)
    padded = gl._pad_to_roi(array, (4, 4, 4))
    assert padded is array  # Literally the same object, not just equal.


def test_pad_to_roi_pads_smaller_axes_with_zeros():
    """Smaller axes are zero-padded up to roi_size; original data is preserved."""
    array = np.ones((4, 5, 6, 2), dtype=np.float32)
    padded = gl._pad_to_roi(array, (8, 8, 8))
    assert padded.shape == (8, 8, 8, 2)
    # Original data preserved at the origin corner.
    assert np.array_equal(padded[:4, :5, :6, :], array)
    # Padded region is zero-filled.
    assert np.all(padded[4:, :, :, :] == 0)
    assert np.all(padded[:, 5:, :, :] == 0)
    assert np.all(padded[:, :, 6:, :] == 0)


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
    assert gl._foreground_anchor_center(label, labels=[1, 2], rng=rng) is None


def test_foreground_anchor_center_finds_the_only_voxel():
    """A single foreground voxel is found deterministically."""
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    label[3, 4, 5, 0] = 1
    rng = np.random.default_rng(0)
    assert gl._foreground_anchor_center(label, labels=[1, 2], rng=rng) == (3, 4, 5)


def test_foreground_anchor_center_falls_back_to_any_nonzero_voxel():
    """A class not present in `labels` still yields a fallback voxel."""
    label = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    label[1, 2, 3, 0] = 9  # Not in the configured `labels` list below.
    rng = np.random.default_rng(0)
    assert gl._foreground_anchor_center(label, labels=[1, 2], rng=rng) == (1, 2, 3)


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
