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
already on the target device, and `.reset()` marks an epoch boundary -- so
nothing downstream (`patch_3d_trainer.py`, `inference_runners.py`) needs to
know or care which loader backend is active. See `cpu_rocm_support_plan.md`
Stages 2-3.

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

Augmentation is not done here: like dali_loader.py, this loader only
extracts patches (enlarged by `patch_margin` when rotation or scaling is
enabled), and mist.data_loading.augmentation augments each batch in PyTorch
on the training device, so both loaders share one implementation.

Every `DataLoader` built here with `num_workers > 0` explicitly requests
the "fork" multiprocessing context (see `_multiprocessing_context()`).
Found necessary during Stage 4's real end-to-end CPU run: on macOS, whose
default start method is "spawn", a real `mist_train` process that had used
any multi-worker `DataLoader` here would hang *indefinitely* at interpreter
shutdown -- confirmed via a stack sample showing the main thread stuck in
`Py_FinalizeEx -> ... -> os.waitpid() -> __wait4`, well past the point
training had already finished and every result had already been written to
disk. Forcing "fork" (Linux's default anyway, so a no-op there) fixed it
outright and, unlike falling back to `num_workers=0` everywhere, keeps real
worker parallelism.
"""

import platform
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from mist.preprocessing import foreground_locations
from mist.utils import hardware

# Training patches that run past the image edge are sent as only the part inside
# the image, and padded with zeros on the training device. Patches enlarged for
# spatial augmentation are often mostly padding on small images, such as BraTS,
# and each sent byte is copied in the main process before it reaches the device.
# False pads in the workers instead, as before 2.5.0; it's here for comparison.
_PAD_ON_DEVICE = True

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


def _multiprocessing_context(num_workers: int) -> str | None:
    """Pick a DataLoader multiprocessing_context for worker processes.

    "fork" is requested explicitly wherever the platform supports it
    (everywhere except Windows) instead of relying on each platform's own
    default -- see the module docstring for why: macOS's default "spawn"
    was found to hang the whole process indefinitely at interpreter exit.
    Linux already defaults to "fork" so this is a no-op there, just made
    explicit. Returns None (irrelevant either way) when num_workers == 0,
    since no worker processes get created at all in that case.
    """
    if num_workers <= 0 or platform.system() == "Windows":
        return None
    return "fork"


def _pin_memory() -> bool:
    """Whether the DataLoader should allocate batches in page-locked memory.

    GenericIterator.next() already does `.to(device, non_blocking=True)` --
    but non_blocking only actually enables an asynchronous H2D copy when the
    source tensor is in pinned memory; without pin_memory=True on the
    DataLoader, that non_blocking=True was silently degrading to a regular
    blocking copy, giving none of the overlap it was asking for. Meaningless
    (and PyTorch warns about it) with no accelerator, so gated on hardware.
    """
    return hardware.get_accelerator_type() != "cpu"


def _prefetch_factor(num_workers: int) -> int | None:
    """Batches per worker to prefetch ahead, or None (num_workers == 0).

    PyTorch's own default (2) is a reasonable floor; bumping it modestly
    gives the consumer a bigger buffer to draw from before it catches up to
    (and has to wait on) the producers -- directly aimed at the bursty
    "fast for a few steps, then stalls" pattern a data-starved GPU sees.
    This alone doesn't change the underlying producer throughput, so it's a
    smoothing knob, not a fix for a genuinely slower-than-the-GPU pipeline.
    Must be left at the DataLoader default (None) when num_workers == 0 --
    PyTorch raises a ValueError otherwise, since there's no worker to
    prefetch ahead of the main process in that case.
    """
    return 4 if num_workers > 0 else None


def _target_device(rank: int) -> torch.device:
    """Resolve the device .next() should move a finished batch to.

    Mirrors dali_loader.py's device_id=rank convention. Delegates to
    hardware.get_device_for_rank() (added in Stage 4 once base_trainer.py
    needed the identical logic at its own model.to()/tensor-device call
    sites) rather than duplicating it here.
    """
    return hardware.get_device_for_rank(rank)


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
    """Memory-map one preprocessed .npy array (image, label, or DTM).

    ``mmap_mode="r"`` avoids reading the whole file into a fresh in-memory
    array on every call -- the OS page cache handles caching transparently
    instead, with no explicit allocation and no unbounded-growth risk (the
    kernel evicts pages under memory pressure the same way it already does
    for any file read, unlike an app-level cache that would just keep
    growing). This matters a lot here: `_PatchTrainingDataset` calls this
    once per patch drawn, and MIST's epochs are step-count-based
    (`min_steps_per_epoch`), not dataset-length-based (the training
    `DataLoader` cycles through the cases endlessly),
    so on any dataset where `steps_per_epoch * batch_size` exceeds the case
    count -- true for most real training runs -- the same case's full
    volume gets reloaded from scratch many times within a single epoch. A
    plain `np.load()` pays that cost in full every time; mmap only pages in
    the bytes actually touched, which for `_extract_patch`'s final crop
    is roughly the patch's own size, not the whole volume (see
    `_slice_with_padding`). Confirmed as the real bottleneck, not a CPU/worker-count
    one, on a real 2x A100 run: doubling `num_cpu_workers` (8 -> 16) made no
    difference, which rules out augmentation-compute throughput and points
    at disk read throughput instead.

    Safe to do unconditionally: every augmentation function in this module
    (`_extract_patch` and its helpers) already only reads its input and returns a new array
    rather than mutating in place, and `_to_channels_first` forces a real
    writable, contiguous copy (`np.ascontiguousarray`) as the last step
    before wrapping in a `torch.Tensor` -- so a read-only memmap view never
    hits an in-place write anywhere in this pipeline.
    """
    return np.load(path, mmap_mode="r")


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
    locations: dict[int | None, np.ndarray],
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
    for class_id in [*class_order, None]:
        coords = locations.get(class_id)
        if coords is not None:
            return tuple(int(c) for c in coords[rng.integers(0, coords.shape[0])])
    return None


def _slice_with_padding(
    array: np.ndarray,
    starts: Sequence[int],
    shape: Sequence[int],
) -> np.ndarray:
    """Slice a (D, H, W, C) array, zero-padding any part outside its bounds.

    Matches dali_loader.py's `fn.slice(..., out_of_bounds_policy="pad")`. Only
    the part of the slice inside the array is read, so a memory-mapped case is
    not loaded in full just because a patch runs past its edge.
    """
    inside = []
    target = []
    for start, size, dim in zip(starts, shape, array.shape[:3], strict=True):
        low = min(max(0, start), dim)
        high = max(min(dim, start + size), low)
        inside.append(slice(low, high))
        target.append(slice(low - start, low - start + high - low))

    inside = tuple(inside)
    if tuple(s.stop - s.start for s in inside) == tuple(shape):
        return array[inside]

    patch = np.zeros((*shape, array.shape[-1]), dtype=array.dtype)
    patch[tuple(target)] = array[inside]
    return patch


def _slice_inside(
    array: np.ndarray,
    starts: Sequence[int],
    shape: Sequence[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Slice the part of a (D, H, W, C) patch inside the array, and say how to pad it.

    Padding the result with zeros by the returned amounts gives exactly what
    _slice_with_padding returns, without making the padding here.

    Returns:
        The slice inside the array, and (before, after) zero padding per axis.
    """
    inside = []
    pad = []
    for start, size, dim in zip(starts, shape, array.shape[:3], strict=True):
        low = min(max(0, start), dim)
        high = max(min(dim, start + size), low)
        inside.append(slice(low, high))
        # A patch wholly outside the array along an axis is all padding.
        before = min(max(low - start, 0), size - (high - low))
        pad.append((before, size - before - (high - low)))
    return array[tuple(inside)], pad


def _pad_on_device(array: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Zero-pad a (C, D, H, W) tensor by (before, after) amounts per spatial axis."""
    widths = [int(amount) for axis in reversed(range(3)) for amount in pad[axis]]
    return F.pad(array, widths) if any(widths) else array


def _patch_window(
    label: np.ndarray,
    roi_size: tuple[int, int, int],
    labels: list[int] | None,
    oversampling: float | None,
    rng: np.random.Generator,
    patch_margin: tuple[int, int, int],
    get_locations: Callable[[], dict[int | None, np.ndarray]] | None,
) -> tuple[list[int], list[int]]:
    """Choose where a patch starts and its shape, including the margin."""
    center = None
    if labels and oversampling and rng.random() < oversampling:
        locations = (
            get_locations() if get_locations else foreground_locations.find(label, labels, rng)
        )
        center = _foreground_anchor_center(locations, labels, rng)

    anchors = [
        _random_anchor(
            label.shape[axis],
            roi_size[axis],
            rng,
            center=center[axis] if center is not None else None,
        )
        for axis in range(3)
    ]
    starts = [anchor - margin for anchor, margin in zip(anchors, patch_margin, strict=True)]
    shape = [roi + 2 * margin for roi, margin in zip(roi_size, patch_margin, strict=True)]
    return starts, shape


def _extract_patch(
    image: np.ndarray,
    label: np.ndarray,
    dtm: np.ndarray | None,
    roi_size: tuple[int, int, int],
    labels: list[int] | None,
    oversampling: float | None,
    rng: np.random.Generator,
    patch_margin: tuple[int, int, int] = (0, 0, 0),
    get_locations: Callable[[], dict[int | None, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract one (D, H, W, C)-shaped patch, foreground-oversampled.

    Approximates dali_loader.py's biased_crop_fn: with probability
    `oversampling`, centers the crop on a random foreground voxel (falling back
    to a plain random crop if the example has no foreground at all); otherwise
    the crop is placed uniformly at random. See the module docstring for why
    this matches DALI statistically, not exactly. The crop is then extended by
    `patch_margin` voxels on each side, keeping the same center. Anything
    outside the image, including along an axis smaller than roi_size, is
    zero-padded. `get_locations` supplies the case's foreground locations, so
    the caller can cache them; by default they're found from `label`.
    """
    starts, shape = _patch_window(
        label, roi_size, labels, oversampling, rng, patch_margin, get_locations
    )
    image = _slice_with_padding(image, starts, shape)
    label = _slice_with_padding(label, starts, shape)
    if dtm is not None:
        dtm = _slice_with_padding(dtm, starts, shape)
    return image, label, dtm


def _to_channels_first(array: np.ndarray) -> torch.Tensor:
    """Move a (D, H, W, C) array to (C, D, H, W).

    Matches dali_loader.py's final `fn.transpose(image, perm=[3, 0, 1, 2])`.

    Uses `np.array(..., order="C")` (default `copy=True`) rather than
    `np.ascontiguousarray`, which is a no-op -- returns the input unchanged,
    no copy -- whenever it's already C-contiguous. That's not just a missed
    optimization: a moveaxis view over `_load_case`'s memory-mapped (read-
    only) `.npy` reads can already be C-contiguous (e.g. `_FullVolumeDataset`,
    which never crops), so `ascontiguousarray` would silently hand back a
    read-only array wrapped in a torch.Tensor -- PyTorch warns loudly about
    exactly this ("writing to this tensor will result in undefined
    behavior"). `np.array()`'s default `copy=True` guarantees an actually
    new, writable buffer regardless of the input's contiguity.
    """
    return torch.from_numpy(np.array(np.moveaxis(array, -1, 0), order="C"))


class _PatchTrainingDataset(Dataset):
    """CPU-side dataset yielding one (optionally patch-extracted) example.

    All reading and patch extraction happens here, on the CPU, in worker
    processes managed by the DataLoader -- see the module docstring for why
    patch placement uses a fresh, unseeded RNG per call rather than one
    seeded up front. Each worker keeps its own cache of foreground locations,
    filled the first time it needs a case's: from the file mist_preprocess saved
    if there is one, otherwise by scanning the label.
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
        patch_margin: tuple[int, int, int] = (0, 0, 0),
    ):
        self._image_paths = image_paths
        self._label_paths = label_paths
        self._dtm_paths = dtm_paths
        self._roi_size = tuple(roi_size)
        self._labels = labels
        self._oversampling = oversampling
        self._extract_patches = extract_patches
        self._patch_margin = tuple(int(m) for m in patch_margin)
        self._foreground_cache: dict[int, dict[int | None, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._image_paths)

    def _cached_locations(
        self,
        index: int,
        label: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[int | None, np.ndarray]:
        """Return a case's foreground locations, finding them only once."""
        if index not in self._foreground_cache:
            saved = foreground_locations.load(
                foreground_locations.path_for(self._label_paths[index]), self._labels
            )
            if saved is None:
                saved = foreground_locations.find(label, self._labels, rng)
            self._foreground_cache[index] = saved
        return self._foreground_cache[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = _load_case(self._image_paths[index])
        label = _load_case(self._label_paths[index])
        dtm = _load_case(self._dtm_paths[index]) if self._dtm_paths else None

        rng = np.random.default_rng()

        if self._extract_patches and _PAD_ON_DEVICE:
            starts, shape = _patch_window(
                label,
                self._roi_size,
                self._labels,
                self._oversampling,
                rng,
                self._patch_margin,
                lambda: self._cached_locations(index, label, rng),
            )
            image, pad = _slice_inside(image, starts, shape)
            label, _ = _slice_inside(label, starts, shape)
            batch = {
                "image": _to_channels_first(image),
                "label": _to_channels_first(label),
                "pad": torch.tensor(pad, dtype=torch.int64),
            }
            if dtm is not None:
                batch["dtm"] = _to_channels_first(_slice_inside(dtm, starts, shape)[0])
            return batch

        if self._extract_patches:
            image, label, dtm = _extract_patch(
                image,
                label,
                dtm,
                self._roi_size,
                self._labels,
                self._oversampling,
                rng,
                self._patch_margin,
                get_locations=lambda: self._cached_locations(index, label, rng),
            )

        batch = {
            "image": _to_channels_first(image),
            "label": _to_channels_first(label),
        }
        if dtm is not None:
            batch["dtm"] = _to_channels_first(dtm)
        return batch


def _collate_patches(samples: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """Batch training patches, leaving any that need padding for the device.

    Patches entirely inside their images stack as usual. Otherwise the patches
    differ in shape, so they're kept apart, and GenericIterator pads and stacks
    them on the training device.
    """
    if all(not sample["pad"].any() for sample in samples):
        return default_collate([{k: v for k, v in s.items() if k != "pad"} for s in samples])
    return {"_unpadded": samples}


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


class _EndlessShuffleSampler(Sampler[int]):
    """Yield shuffled passes over a dataset's indices, forever.

    The DataLoader's iterator then never ends, so one pool of workers serves a
    whole fold (batchgenerators' infinite data loaders work the same way).
    Every case still appears once per pass, and a batch that spans two passes
    is full-sized rather than cut short at the end of a pass.
    """

    def __init__(self, num_examples: int, generator: torch.Generator):
        self._num_examples = num_examples
        self._generator = generator

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from torch.randperm(self._num_examples, generator=self._generator).tolist()

    def __len__(self) -> int:
        return self._num_examples


class GenericIterator:
    """DALI-compatible iterator: `.next()` -> `[batch on device]`, `.reset()`.

    This is the "chokepoint" mentioned in the module docstring -- the only
    place a CPU-loaded batch is moved to the target accelerator -- which is
    what keeps `patch_3d_trainer.py` and `inference_runners.py` backend-
    agnostic.

    With `endless=True`, the data loader never runs out and `.reset()` does
    nothing, so its worker processes and their caches last until this iterator
    is dropped at the end of the fold, which shuts them down.
    """

    def __init__(self, data_loader: DataLoader, device: torch.device, endless: bool = False):
        self._data_loader = data_loader
        self._device = device
        self._endless = endless
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
        if "_unpadded" in batch:
            samples = batch["_unpadded"]
            keys = [key for key in samples[0] if key != "pad"]
            return [
                {
                    key: torch.stack(
                        [
                            _pad_on_device(
                                sample[key].to(self._device, non_blocking=True), sample["pad"]
                            )
                            for sample in samples
                        ]
                    )
                    for key in keys
                }
            ]
        return [{key: value.to(self._device, non_blocking=True) for key, value in batch.items()}]

    def reset(self) -> None:
        """Start a fresh pass over the dataset, unless the data loader is endless."""
        if not self._endless:
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
    patch_margin: tuple[int, int, int] = (0, 0, 0),
) -> GenericIterator:
    """Generic-loader equivalent of dali_loader.get_training_dataset.

    Same signature and `.next()`/`.reset()` contract as dali_loader.py (see
    the module docstring). Like dali_loader.py, this only extracts patches;
    augmentation happens afterwards in mist.data_loading.augmentation.

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
        seed: Random seed for shuffling case order between passes.
        num_workers: Number of DataLoader worker processes. They last for the
            whole fold, not one pass over the data.
        rank: The rank of the current process.
        world_size: The total number of processes.
        extract_patches: Whether to extract a roi_size patch from each
            example. If False, the entire (already roi_size-shaped) example
            is returned unmodified.
        patch_margin: Voxels added to each side of the patch along each
            axis, so spatial augmentation has real image content at the
            borders.

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
        patch_margin=patch_margin,
    )
    shuffle_generator = torch.Generator().manual_seed(seed + rank)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=_EndlessShuffleSampler(len(dataset), shuffle_generator),
        collate_fn=_collate_patches if extract_patches and _PAD_ON_DEVICE else None,
        num_workers=num_workers,
        multiprocessing_context=_multiprocessing_context(num_workers),
        pin_memory=_pin_memory(),
        prefetch_factor=_prefetch_factor(num_workers),
    )
    return GenericIterator(data_loader, device=_target_device(rank), endless=True)


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
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=_multiprocessing_context(num_workers),
        pin_memory=_pin_memory(),
    )
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
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=_multiprocessing_context(num_workers),
        pin_memory=_pin_memory(),
    )
    return GenericIterator(data_loader, device=_target_device(rank))
