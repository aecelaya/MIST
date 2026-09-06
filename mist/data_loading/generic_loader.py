"""Generic, accelerator-agnostic data loader for MIST.

`dali_loader.py` requires an NVIDIA GPU (`nvidia-dali-cuda120`), so it can't
run on CPU-only or AMD ROCm hardware. This module is the CPU/ROCm-friendly
alternative: it runs entirely on the CPU (NumPy for patch extraction,
`torch.utils.data.Dataset`/`DataLoader` for batching and worker-process
parallelism) and hands a finished batch off to the target accelerator only
once, inside `.next()`.

`get_training_dataset`/`get_validation_dataset`/`get_test_dataset` match
`dali_loader.py`'s signatures and its `.next()`/`.reset()` iterator contract
exactly -- `.next()` returns `[{"image": ..., "label": ..., "dtm": ...}]`
already on the target device, and `.reset()` starts a fresh pass -- so
nothing downstream (`patch_3d_trainer.py`, `inference_runners.py`) needs to
know or care which loader backend is active. See `cpu_rocm_support_plan.md`
Stage 2.

This is deliberately *not* a byte-for-byte reimplementation of DALI's
pipeline. Patch extraction and foreground-oversampled patch placement are
careful approximations of `dali_loader.py`'s `random_object_bbox`/
`roi_random_crop` combination -- matching the same foreground-sampling
*rate*, not the same RNG stream or connected-component-object semantics,
since the two loaders were never going to share an RNG stream anyway (see
`cpu_rocm_support_plan.md`'s Stage 2 release gate: a statistical, not
exact-match, equivalence check). For the same reason, patch placement here
uses a fresh, unseeded `numpy.random.Generator` per `__getitem__` call rather
than one seeded from `training.seed`: a `DataLoader` with `num_workers > 0`
forks/spawns separate worker processes, and a single `Generator` seeded once
in the parent would otherwise be copied into every worker with identical
state, making all of them draw the same "random" sequence independently.
`training.seed` still governs the (much simpler, single-process) shuffling
of case order between epochs, which doesn't have that pitfall.

This stage (Stage 2) covers loading, DDP sharding, and patch extraction
only -- no augmentation. That's Stage 3: the `use_*` augmentation flags
below are accepted for call-site parity with `dali_loader.py` (so Stage 4
doesn't need to change any call site when the generic loader is selected),
but they currently have no effect regardless of value.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mist.utils import hardware

# Deliberately not imported from data_loading_utils.py: that module imports
# nvidia.dali at the top for its DALI-graph-building helpers, so importing it
# here would defeat the entire point of this module (running without DALI
# installed). This is the same validation dali_loader.py's pipelines run via
# data_loading_utils.validate_train_and_eval_inputs, just duplicated instead
# of shared to keep this module import-safe with no NVIDIA/DALI dependency.


def _validate_train_and_eval_inputs(
    image_paths: list[str],
    label_paths: list[str],
    dtm_paths: list[str] | None = None,
) -> None:
    """Validate that the input data is correct.

    Args:
        image_paths: List of image file paths.
        label_paths: List of label file paths.
        dtm_paths: Optional list of DTM data file paths.

    Raises:
        ValueError: If the number of images, labels, or DTMs are incorrect.
    """
    if not image_paths:
        raise ValueError("No images found!")
    if not label_paths:
        raise ValueError("No labels found!")
    if len(image_paths) != len(label_paths):
        raise ValueError("Number of images and labels do not match!")
    if dtm_paths is not None:
        if not dtm_paths:
            raise ValueError("No DTM data found!")
        if len(image_paths) != len(dtm_paths):
            raise ValueError("Number of images and DTMs do not match!")


def _target_device(rank: int) -> torch.device:
    """Resolve the device .next() should move a finished batch to.

    Mirrors dali_loader.py's device_id=rank convention: on CUDA/ROCm this is
    "cuda:<rank>" (the same torch.cuda compatibility shim used everywhere
    else in MIST); CPU-only hardware has no per-rank device to target.
    """
    if hardware.get_accelerator_type() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda", rank)


def _shard_indices(num_examples: int, rank: int, world_size: int) -> list[int]:
    """Partition [0, num_examples) across ranks with no drops or duplicates.

    Round-robin (rank::world_size) rather than contiguous chunks or
    torch.utils.data.distributed.DistributedSampler's default (which pads
    the tail with duplicated samples so every rank sees an equal count):
    every index is assigned to exactly one rank, and shard sizes differ by
    at most one across ranks with nothing needing to be duplicated to
    achieve that.
    """
    return list(range(rank, num_examples, world_size))


def _load_case(path: str) -> np.ndarray:
    """Load one preprocessed .npy array (image, label, or DTM) from disk."""
    return np.load(path)


def _pad_to_roi(array: np.ndarray, roi_size: tuple[int, int, int]) -> np.ndarray:
    """Zero-pad the spatial (first 3) axes of a (D, H, W, C) array to roi_size.

    Matches dali_loader.py's `fn.pad(image, axes=(0, 1, 2), shape=roi_size)`:
    a no-op on any axis already >= the corresponding roi_size entry.
    """
    pad_widths = [(0, max(0, roi - array.shape[axis])) for axis, roi in enumerate(roi_size)]
    pad_widths.append((0, 0))  # Channel axis is never padded.
    if all(width == (0, 0) for width in pad_widths):
        return array
    return np.pad(array, pad_widths, mode="constant", constant_values=0)


def _random_anchor(
    dim_size: int,
    roi: int,
    rng: np.random.Generator,
    center: int | None,
) -> int:
    """Pick a valid crop start along one axis, optionally covering `center`.

    With `center` given (a foreground voxel's coordinate along this axis),
    the returned anchor is jittered within the patch so the crop still
    contains it -- this is the foreground-oversampling placement. Without
    it, the anchor is uniform over the entire valid range, matching a plain
    random crop.
    """
    max_start = max(0, dim_size - roi)
    if center is None:
        return int(rng.integers(0, max_start + 1))
    jitter = int(rng.integers(0, roi))
    return int(np.clip(center - jitter, 0, max_start))


def _foreground_anchor_center(
    label: np.ndarray,
    labels: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, int, int] | None:
    """Pick a random foreground voxel's (d, h, w) coordinate to center on.

    Mirrors dali_loader.py's equal per-class weighting (its label_weights =
    [1 / len(labels)] * len(labels)): shuffle the foreground classes into a
    random order, then return a random voxel of the first one actually
    present in this example. Falls back to any nonzero voxel if none of the
    configured classes are present, and to None (caller falls back to a
    plain random crop) if there is no foreground at all.
    """
    class_order = list(labels)
    rng.shuffle(class_order)
    label_channel = label[..., 0]
    for class_id in class_order:
        coords = np.argwhere(label_channel == class_id)
        if coords.size:
            return tuple(int(c) for c in coords[rng.integers(0, coords.shape[0])])

    coords = np.argwhere(label_channel != 0)
    if coords.size:
        return tuple(int(c) for c in coords[rng.integers(0, coords.shape[0])])
    return None


def _extract_patch(
    image: np.ndarray,
    label: np.ndarray,
    dtm: np.ndarray | None,
    roi_size: tuple[int, int, int],
    labels: list[int] | None,
    oversampling: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract one (D, H, W, C)-shaped patch, foreground-oversampled.

    Approximates dali_loader.py's biased_crop_fn: pads up to roi_size, then
    -- with probability `oversampling` -- centers the crop on a random
    foreground voxel (falling back to a plain random crop if the example has
    no foreground at all); otherwise the crop is placed uniformly at random.
    See the module docstring for why this matches DALI statistically, not
    exactly.
    """
    image = _pad_to_roi(image, roi_size)
    label = _pad_to_roi(label, roi_size)
    if dtm is not None:
        dtm = _pad_to_roi(dtm, roi_size)

    center = None
    if labels and oversampling and rng.random() < oversampling:
        center = _foreground_anchor_center(label, labels, rng)

    anchors = [
        _random_anchor(
            label.shape[axis],
            roi_size[axis],
            rng,
            center=center[axis] if center is not None else None,
        )
        for axis in range(3)
    ]
    slices = tuple(
        slice(anchor, anchor + roi) for anchor, roi in zip(anchors, roi_size, strict=True)
    )

    image = image[slices]
    label = label[slices]
    if dtm is not None:
        dtm = dtm[slices]
    return image, label, dtm


def _to_channels_first(array: np.ndarray) -> torch.Tensor:
    """Move a (D, H, W, C) array to (C, D, H, W).

    Matches dali_loader.py's final `fn.transpose(image, perm=[3, 0, 1, 2])`.
    """
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0)))


class _PatchTrainingDataset(Dataset):
    """CPU-side dataset yielding one (optionally patch-extracted) example.

    All reading and patch extraction happens here, on the CPU, in worker
    processes managed by the DataLoader -- see the module docstring for why
    patch placement uses a fresh, unseeded RNG per call rather than one
    seeded up front.
    """

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        dtm_paths: list[str] | None,
        roi_size: tuple[int, int, int],
        labels: list[int] | None,
        oversampling: float | None,
        extract_patches: bool,
    ):
        self._image_paths = image_paths
        self._label_paths = label_paths
        self._dtm_paths = dtm_paths
        self._roi_size = tuple(roi_size)
        self._labels = labels
        self._oversampling = oversampling
        self._extract_patches = extract_patches

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = _load_case(self._image_paths[index])
        label = _load_case(self._label_paths[index])
        dtm = _load_case(self._dtm_paths[index]) if self._dtm_paths else None

        if self._extract_patches:
            rng = np.random.default_rng()
            image, label, dtm = _extract_patch(
                image,
                label,
                dtm,
                self._roi_size,
                self._labels,
                self._oversampling,
                rng,
            )

        batch = {
            "image": _to_channels_first(image),
            "label": _to_channels_first(label),
        }
        if dtm is not None:
            batch["dtm"] = _to_channels_first(dtm)
        return batch


class _FullVolumeDataset(Dataset):
    """CPU-side dataset yielding one full (unpatched) example per index.

    Used for validation and test, matching dali_loader.py's EvalPipeline and
    TestPipeline: no patch extraction, no shuffling.
    """

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str] | None = None,
    ):
        self._image_paths = image_paths
        self._label_paths = label_paths

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = {"image": _to_channels_first(_load_case(self._image_paths[index]))}
        if self._label_paths is not None:
            batch["label"] = _to_channels_first(_load_case(self._label_paths[index]))
        return batch


class GenericIterator:
    """DALI-compatible iterator: `.next()` -> `[batch on device]`, `.reset()`.

    This is the "chokepoint" mentioned in the module docstring -- the only
    place a CPU-loaded batch is moved to the target accelerator -- which is
    what keeps `patch_3d_trainer.py` and `inference_runners.py` backend-
    agnostic.
    """

    def __init__(self, data_loader: DataLoader, device: torch.device):
        self._data_loader = data_loader
        self._device = device
        self._iterator = iter(self._data_loader)

    def next(self) -> list[dict[str, torch.Tensor]]:
        """Return the next batch, auto-restarting when the pass is exhausted.

        Mirrors dali_loader.py's DALIGenericIterator, which is constructed
        there without an explicit `size`/`reader_name`: MIST's training loop
        can call `.next()` more times per "epoch" than the dataset has
        batches for (see `training.min_steps_per_epoch`), so this must never
        raise `StopIteration` on its own -- only an explicit `.reset()` call
        marks an epoch boundary.
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._data_loader)
            batch = next(self._iterator)
        return [{key: value.to(self._device, non_blocking=True) for key, value in batch.items()}]

    def reset(self) -> None:
        """Start a fresh pass over the dataset (reshuffles a shuffling loader)."""
        self._iterator = iter(self._data_loader)


def get_training_dataset(
    image_paths: list[str],
    label_paths: list[str],
    dtm_paths: list[str] | None,
    batch_size: int,
    roi_size: tuple[int, int, int],
    labels: list[int] | None,
    oversampling: float | None,
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
    extract_patches: bool = True,
    use_augmentation: bool = True,
    use_flips: bool = True,
    use_zoom: bool = True,
    use_noise: bool = True,
    use_blur: bool = True,
    use_brightness: bool = True,
    use_contrast: bool = True,
) -> GenericIterator:
    """Generic-loader equivalent of dali_loader.get_training_dataset.

    Same signature and `.next()`/`.reset()` contract as dali_loader.py (see
    the module docstring). The `use_*` augmentation flags are accepted here
    purely for call-site parity -- Stage 3 of `cpu_rocm_support_plan.md`
    implements them, so today they have no effect regardless of value.

    Args:
        image_paths: List of file paths to the image data.
        label_paths: List of file paths to the label data.
        dtm_paths: List of file paths to the DTM data, or None.
        batch_size: The batch size for training.
        roi_size: The patch size used for training.
        labels: List of foreground labels in the dataset (background
            excluded), used for foreground-oversampled patch placement.
        oversampling: Probability of centering a patch on a foreground voxel
            rather than placing it uniformly at random.
        seed: Random seed for shuffling case order between epochs.
        num_workers: Number of DataLoader worker processes.
        rank: The rank of the current process.
        world_size: The total number of processes.
        extract_patches: Whether to extract a roi_size patch from each
            example. If False, the entire (already roi_size-shaped) example
            is returned unmodified.
        use_augmentation: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).
        use_flips: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).
        use_zoom: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).
        use_noise: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).
        use_blur: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).
        use_brightness: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).
        use_contrast: Accepted for parity with dali_loader.py; not yet
            implemented (Stage 3).

    Returns:
        A GenericIterator over training batches, already on the target
        device.
    """
    _validate_train_and_eval_inputs(image_paths, label_paths, dtm_paths)

    shard = _shard_indices(len(image_paths), rank, world_size)
    dataset = _PatchTrainingDataset(
        image_paths=[image_paths[i] for i in shard],
        label_paths=[label_paths[i] for i in shard],
        dtm_paths=[dtm_paths[i] for i in shard] if dtm_paths else None,
        roi_size=tuple(roi_size),
        labels=labels,
        oversampling=oversampling,
        extract_patches=extract_patches,
    )
    shuffle_generator = torch.Generator().manual_seed(seed + rank)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=shuffle_generator,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return GenericIterator(data_loader, device=_target_device(rank))


def get_validation_dataset(
    image_paths: list[str],
    label_paths: list[str],
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> GenericIterator:
    """Generic-loader equivalent of dali_loader.get_validation_dataset.

    Streams full (unpatched) images and labels, sharded across ranks but not
    shuffled -- matching dali_loader.py's EvalPipeline (shuffle_input=False).

    Args:
        image_paths: List of file paths to the image data.
        label_paths: List of file paths to the label data.
        seed: Unused here (validation is never shuffled); accepted for
            parity with dali_loader.py, which also takes it unconditionally.
        num_workers: Number of DataLoader worker processes.
        rank: The rank of the current process.
        world_size: The total number of processes.

    Returns:
        A GenericIterator over validation batches, already on the target
        device.
    """
    _validate_train_and_eval_inputs(image_paths, label_paths)

    shard = _shard_indices(len(image_paths), rank, world_size)
    dataset = _FullVolumeDataset(
        image_paths=[image_paths[i] for i in shard],
        label_paths=[label_paths[i] for i in shard],
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return GenericIterator(data_loader, device=_target_device(rank))


def get_test_dataset(
    image_paths: list[str],
    seed: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
) -> GenericIterator:
    """Generic-loader equivalent of dali_loader.get_test_dataset.

    Streams full (unpatched) images in the exact order given -- no shuffling,
    no sharding by default (rank/world_size default to 0/1, matching
    dali_loader.py and inference_runners.py::test_on_fold's assumption that
    the loader yields batches in the same order as its input file list).

    Args:
        image_paths: List of file paths to the image data.
        seed: Unused here (test is never shuffled); accepted for parity with
            dali_loader.py, which also takes it unconditionally.
        num_workers: Number of DataLoader worker processes.
        rank: The rank of the current process. Defaults to 0.
        world_size: The total number of processes. Defaults to 1.

    Returns:
        A GenericIterator over test batches, already on the target device.

    Raises:
        ValueError: If no images are found in the input data.
    """
    if len(image_paths) == 0:
        raise ValueError("No images found!")

    shard = _shard_indices(len(image_paths), rank, world_size)
    dataset = _FullVolumeDataset(image_paths=[image_paths[i] for i in shard])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return GenericIterator(data_loader, device=_target_device(rank))
