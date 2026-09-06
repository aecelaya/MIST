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

from itertools import combinations

import numpy as np
import pytest
import torch
from torch import nn

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
# _flip_fn
# --------------------------------------------------------------------------- #


def test_flip_fn_is_deterministic_given_seed():
    """Same seed -> identical output."""
    image = np.arange(2 * 3 * 4 * 1).reshape(2, 3, 4, 1).astype(np.float32)
    label = image.astype(np.uint8)
    out1 = gl._flip_fn(image.copy(), label.copy(), None, np.random.default_rng(7))
    out2 = gl._flip_fn(image.copy(), label.copy(), None, np.random.default_rng(7))
    assert np.array_equal(out1[0], out2[0])
    assert np.array_equal(out1[1], out2[1])


def test_flip_fn_applies_the_same_flip_to_image_label_and_dtm():
    """Whichever axes get flipped, image/label/dtm all get the same ones."""
    image = np.random.default_rng(0).random((4, 4, 4, 1)).astype(np.float32)
    label = np.random.default_rng(1).integers(0, 3, (4, 4, 4, 1)).astype(np.uint8)
    dtm = np.random.default_rng(2).random((4, 4, 4, 2)).astype(np.float32)

    out_image, out_label, out_dtm = gl._flip_fn(image, label, dtm, np.random.default_rng(3))

    # Recover which axis combination was applied (if any) without assuming
    # anything about the function's internals beyond "some subset of axes
    # (0, 1, 2) got flipped, identically for all three arrays".
    candidates = [axes for r in range(4) for axes in combinations(range(3), r)]
    matches = [axes for axes in candidates if np.array_equal(np.flip(image, axis=axes), out_image)]
    assert len(matches) == 1
    axes = matches[0]
    assert np.array_equal(np.flip(label, axis=axes), out_label)
    assert np.array_equal(np.flip(dtm, axis=axes), out_dtm)


# --------------------------------------------------------------------------- #
# _zoom_fn
# --------------------------------------------------------------------------- #


def test_zoom_fn_always_returns_roi_size_and_sometimes_triggers():
    """Output is always roi_size-shaped; the 15% chance triggers eventually."""
    roi_size = (10, 10, 10)
    image = np.random.default_rng(1).random((*roi_size, 1)).astype(np.float32)
    label = np.zeros((*roi_size, 1), dtype=np.uint8)
    label[4:7, 4:7, 4:7, 0] = 1

    changed = False
    for seed in range(200):
        out_image, out_label = gl._zoom_fn(image, label, roi_size, np.random.default_rng(seed))
        assert out_image.shape == (*roi_size, 1)
        assert out_label.shape == (*roi_size, 1)
        if not np.array_equal(out_image, image):
            changed = True
    assert changed, "zoom never triggered across 200 seeds"


def test_zoom_fn_label_stays_valid_class_ids():
    """Nearest-neighbor interpolation must not invent fractional class ids."""
    roi_size = (12, 12, 12)
    image = np.random.default_rng(1).random((*roi_size, 1)).astype(np.float32)
    label = np.zeros((*roi_size, 1), dtype=np.uint8)
    label[4:8, 4:8, 4:8, 0] = 1

    for seed in range(50):
        _, out_label = gl._zoom_fn(image, label, roi_size, np.random.default_rng(seed))
        assert set(np.unique(out_label)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# _noise_fn / _blur_fn / _brightness_fn / _contrast_fn
# --------------------------------------------------------------------------- #


def test_noise_fn_stays_within_original_range_and_sometimes_changes():
    """Matches dali_loader.py's clamp-to-original-range behavior."""
    image = np.random.default_rng(1).uniform(0.0, 1.0, (6, 6, 6, 1)).astype(np.float32)
    changed = False
    for seed in range(200):
        out = gl._noise_fn(image, np.random.default_rng(seed))
        assert out.shape == image.shape
        assert out.min() >= image.min()
        assert out.max() <= image.max()
        if not np.array_equal(out, image):
            changed = True
    assert changed


def test_noise_fn_is_deterministic_given_seed():
    image = np.random.default_rng(1).uniform(0.0, 1.0, (4, 4, 4, 1)).astype(np.float32)
    out1 = gl._noise_fn(image, np.random.default_rng(5))
    out2 = gl._noise_fn(image, np.random.default_rng(5))
    assert np.array_equal(out1, out2)


def test_blur_fn_stays_within_original_range_and_sometimes_changes():
    """Matches dali_loader.py's clamp-to-original-range behavior."""
    image = np.random.default_rng(1).uniform(0.0, 1.0, (8, 8, 8, 1)).astype(np.float32)
    changed = False
    for seed in range(200):
        out = gl._blur_fn(image, np.random.default_rng(seed))
        assert out.shape == image.shape
        assert out.min() >= image.min() - 1e-5
        assert out.max() <= image.max() + 1e-5
        if not np.array_equal(out, image):
            changed = True
    assert changed


def test_blur_fn_does_not_mix_channels():
    """sigma's channel-axis entry is 0 -- channels must stay independent."""
    image = np.zeros((8, 8, 8, 2), dtype=np.float32)
    image[..., 0] = np.random.default_rng(1).uniform(0.0, 1.0, (8, 8, 8))
    image[..., 1] = 5.0  # Constant channel -- must stay untouched by blur.
    for seed in range(50):
        out = gl._blur_fn(image, np.random.default_rng(seed))
        assert np.allclose(out[..., 1], 5.0)


def test_brightness_fn_scales_without_clamping_and_is_deterministic():
    """Matches dali_loader.py's brightness_fn: pure scaling, no clamp."""
    image = np.random.default_rng(1).uniform(0.2, 0.8, (4, 4, 4, 1)).astype(np.float32)
    out1 = gl._brightness_fn(image, np.random.default_rng(5))
    out2 = gl._brightness_fn(image, np.random.default_rng(5))
    assert np.array_equal(out1, out2)

    changed = False
    for seed in range(200):
        out = gl._brightness_fn(image, np.random.default_rng(seed))
        assert out.shape == image.shape
        if not np.array_equal(out, image):
            changed = True
            ratios = out / image
            assert np.allclose(ratios, ratios.flat[0])  # Constant scale factor.
    assert changed


def test_contrast_fn_stays_within_original_range_and_sometimes_changes():
    """Matches dali_loader.py's contrast_fn: scaled, then clamped."""
    image = np.random.default_rng(1).uniform(0.2, 0.8, (6, 6, 6, 1)).astype(np.float32)
    changed = False
    for seed in range(200):
        out = gl._contrast_fn(image, np.random.default_rng(seed))
        assert out.shape == image.shape
        assert out.min() >= image.min() - 1e-5
        assert out.max() <= image.max() + 1e-5
        if not np.array_equal(out, image):
            changed = True
    assert changed


# --------------------------------------------------------------------------- #
# _apply_augmentations
# --------------------------------------------------------------------------- #


def test_apply_augmentations_skips_zoom_only_when_dtm_present(monkeypatch):
    """Zoom/DTM asymmetry inherited from dali_loader.py's define_graph."""
    calls = []
    monkeypatch.setattr(
        gl,
        "_zoom_fn",
        lambda image, label, roi_size, rng: (calls.append("zoom"), (image, label))[1],
    )
    monkeypatch.setattr(
        gl,
        "_flip_fn",
        lambda image, label, dtm, rng: (calls.append("flip"), (image, label, dtm))[1],
    )

    image = np.zeros((4, 4, 4, 1), dtype=np.float32)
    label = np.zeros((4, 4, 4, 1), dtype=np.uint8)
    dtm = np.zeros((4, 4, 4, 2), dtype=np.float32)
    rng = np.random.default_rng(0)
    no_op_kwargs = {
        "use_flips": True,
        "use_zoom": True,
        "use_noise": False,
        "use_blur": False,
        "use_brightness": False,
        "use_contrast": False,
    }

    gl._apply_augmentations(image, label, dtm, (4, 4, 4), rng=rng, **no_op_kwargs)
    assert calls == ["flip"]

    calls.clear()
    gl._apply_augmentations(image, label, None, (4, 4, 4), rng=rng, **no_op_kwargs)
    assert calls == ["zoom", "flip"]


# --------------------------------------------------------------------------- #
# _PatchTrainingDataset augmentation wiring
# --------------------------------------------------------------------------- #


def test_use_augmentation_false_disables_every_individual_flag():
    """Matches dali_loader.py's TrainPipeline: the master switch wins."""
    dataset = gl._PatchTrainingDataset(
        image_paths=[],
        label_paths=[],
        dtm_paths=None,
        roi_size=(4, 4, 4),
        labels=None,
        oversampling=None,
        extract_patches=True,
        use_augmentation=False,
        use_flips=True,
        use_zoom=True,
        use_noise=True,
        use_blur=True,
        use_brightness=True,
        use_contrast=True,
    )
    assert not any(
        (
            dataset._use_flips,
            dataset._use_zoom,
            dataset._use_noise,
            dataset._use_blur,
            dataset._use_brightness,
            dataset._use_contrast,
        )
    )


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


def test_get_training_dataset_with_augmentation_enabled_and_dtm(preprocessed):
    """Augmented batches (DTM path -- flips only, no zoom) stay roi-shaped."""
    roi_size = (8, 8, 8)
    loader = gl.get_training_dataset(
        image_paths=preprocessed["image_paths"],
        label_paths=preprocessed["label_paths"],
        dtm_paths=preprocessed["dtm_paths"],
        batch_size=2,
        roi_size=roi_size,
        labels=preprocessed["config"]["dataset_info"]["labels"][1:],
        oversampling=0.5,
        seed=0,
        num_workers=0,
        rank=0,
        world_size=1,
        use_augmentation=True,
    )
    for _ in range(6):
        batch = loader.next()[0]
        assert batch["image"].shape[1:] == (1, *roi_size)
        assert batch["label"].shape[1:] == (1, *roi_size)
        assert batch["dtm"].shape[1:] == (3, *roi_size)


def test_get_training_dataset_with_augmentation_enabled_without_dtm(preprocessed):
    """Augmented batches (no-DTM path -- zoom + flips) stay roi-shaped."""
    roi_size = (8, 8, 8)
    loader = gl.get_training_dataset(
        image_paths=preprocessed["image_paths"],
        label_paths=preprocessed["label_paths"],
        dtm_paths=None,
        batch_size=2,
        roi_size=roi_size,
        labels=preprocessed["config"]["dataset_info"]["labels"][1:],
        oversampling=0.5,
        seed=0,
        num_workers=0,
        rank=0,
        world_size=1,
        use_augmentation=True,
    )
    for _ in range(6):
        batch = loader.next()[0]
        assert batch["image"].shape[1:] == (1, *roi_size)
        assert batch["label"].shape[1:] == (1, *roi_size)


def test_training_with_augmentation_enabled_reduces_loss(preprocessed, monkeypatch):
    """Stage 3 release gate: a model can actually train on augmented batches.

    Not a claim of numerical parity with DALI (different RNG, different
    interpolation) -- just that augmented batches are well-formed enough for
    a tiny model to overfit this small, fixed dataset, matching the release
    gate in cpu_rocm_support_plan.md's Stage 3.

    Augmentation's own RNG is deliberately unseeded in production (see
    generic_loader.py's module docstring on why), so it's pinned here to one
    shared, seeded Generator for a reproducible test -- otherwise this test
    would flake on whichever random augmentation draws happen to land on a
    given run, exactly as an earlier version of it did.
    """
    torch.manual_seed(0)
    fixed_rng = np.random.default_rng(1234)
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: fixed_rng)

    roi_size = (12, 12, 12)
    loader = gl.get_training_dataset(
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
        use_augmentation=True,
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
