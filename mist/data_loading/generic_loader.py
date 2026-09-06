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

Augmentations (Stage 3) run on CPU tensors/NumPy inside
`Dataset.__getitem__`, matching `data_loading_utils.py`'s DALI-side
transforms one for one -- same constants (`data_loading_constants.py`,
which has no DALI dependency and is safe to import here), same
probabilities and ranges -- with one asymmetry inherited directly from
`dali_loader.py`'s own `TrainPipeline.define_graph`: zoom only ever runs
when there is no DTM. DALI's own pipeline skips zoom outright whenever
DTMs are present (rather than resizing the DTM to match), so this mirrors
that rather than "fixing" it -- Stage 3's job is parity with the existing
DALI transforms, not new behavior. No claim of exact numerical parity with
DALI's own transforms is made or needed (different RNG stream, and likely
different interpolation kernels for zoom) -- see the Stage 3 release gate
in `cpu_rocm_support_plan.md`.

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
from collections.abc import Sequence

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from mist.data_loading.data_loading_constants import DataLoadingConstants as constants
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


def _match_roi_size(array: np.ndarray, roi_size: tuple[int, int, int]) -> np.ndarray:
    """Guarantee exactly roi_size spatial dims after a scipy zoom's rounding.

    `scipy.ndimage.zoom` computes its output shape as
    `round(input_shape * zoom_factor)` per axis; floating-point zoom
    factors can, in principle, land one voxel short or long of the intended
    `roi_size`. Padding (if short) then trimming (if long) forces an exact
    match rather than let a shape mismatch propagate downstream, where it
    would break stacking patches into a batch.
    """
    array = _pad_to_roi(array, roi_size)
    return array[tuple(slice(0, roi) for roi in roi_size)]


def _flip_fn(
    image: np.ndarray,
    label: np.ndarray,
    dtm: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Independently flip each spatial axis with probability 0.5.

    Matches dali_loader.py's flips_fn: three independent coin flips --
    depthwise (axis 0), vertical (axis 1), horizontal (axis 2) -- applied
    identically to image, label, and DTM (when present).
    """
    axes = [
        axis
        for axis, probability in enumerate(
            (
                constants.DEPTH_FLIP_PROBABILITY,
                constants.VERTICAL_FLIP_PROBABILITY,
                constants.HORIZONTAL_FLIP_PROBABILITY,
            )
        )
        if rng.random() < probability
    ]
    if not axes:
        return image, label, dtm
    image = np.flip(image, axis=axes).copy()
    label = np.flip(label, axis=axes).copy()
    if dtm is not None:
        dtm = np.flip(dtm, axis=axes).copy()
    return image, label, dtm


def _zoom_fn(
    image: np.ndarray,
    label: np.ndarray,
    roi_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly zoom in on the patch, resizing back to roi_size.

    Matches dali_loader.py's zoom_fn: with probability `ZOOM_FN_PROBABILITY`,
    center-crop to `scale` (Uniform[ZOOM_FN_RANGE_MIN, ZOOM_FN_RANGE_MAX])
    times `roi_size` along every axis, then resize back up to `roi_size` --
    cubic interpolation for the image, nearest-neighbor for the label so its
    class values aren't corrupted by interpolation.

    Only ever called on the DTM-free path -- see the module docstring's note
    on the zoom/DTM asymmetry inherited from dali_loader.py.
    """
    if rng.random() >= constants.ZOOM_FN_PROBABILITY:
        return image, label

    scale = rng.uniform(constants.ZOOM_FN_RANGE_MIN, constants.ZOOM_FN_RANGE_MAX)
    cropped_size = tuple(max(1, int(round(scale * roi))) for roi in roi_size)
    starts = tuple((roi_size[axis] - cropped_size[axis]) // 2 for axis in range(3))
    crop = tuple(
        slice(start, start + size) for start, size in zip(starts, cropped_size, strict=True)
    )
    zoom_factors = tuple(roi_size[axis] / cropped_size[axis] for axis in range(3)) + (1.0,)

    image = ndimage.zoom(image[crop], zoom_factors, order=3)
    label = ndimage.zoom(label[crop], zoom_factors, order=0)
    return _match_roi_size(image, roi_size), _match_roi_size(label, roi_size)


def _noise_fn(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise, clamped to the original value range.

    Matches dali_loader.py's noise_fn: stddev ~ Uniform(NOISE_FN_RANGE_MIN,
    NOISE_FN_RANGE_MAX), applied with probability NOISE_FN_PROBABILITY.
    """
    if rng.random() >= constants.NOISE_FN_PROBABILITY:
        return image
    stddev = rng.uniform(constants.NOISE_FN_RANGE_MIN, constants.NOISE_FN_RANGE_MAX)
    noise = rng.normal(loc=0.0, scale=stddev, size=image.shape).astype(image.dtype)
    return np.clip(image + noise, image.min(), image.max())


def _blur_fn(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian blur, clamped to the original value range.

    Matches dali_loader.py's blur_fn: sigma ~ Uniform(BLUR_FN_RANGE_MIN,
    BLUR_FN_RANGE_MAX), applied with probability BLUR_FN_PROBABILITY, over
    the spatial axes only (never across the channel axis).
    """
    if rng.random() >= constants.BLUR_FN_PROBABILITY:
        return image
    sigma = rng.uniform(constants.BLUR_FN_RANGE_MIN, constants.BLUR_FN_RANGE_MAX)
    blurred = ndimage.gaussian_filter(image, sigma=(sigma, sigma, sigma, 0))
    return np.clip(blurred, image.min(), image.max())


def _brightness_fn(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale intensities by a random brightness factor.

    Matches dali_loader.py's brightness_fn: scale ~
    Uniform(BRIGHTNESS_FN_RANGE_MIN, BRIGHTNESS_FN_RANGE_MAX), applied with
    probability BRIGHTNESS_FN_PROBABILITY (identity otherwise).
    """
    if rng.random() >= constants.BRIGHTNESS_FN_PROBABILITY:
        return image
    scale = rng.uniform(constants.BRIGHTNESS_FN_RANGE_MIN, constants.BRIGHTNESS_FN_RANGE_MAX)
    return image * scale


def _contrast_fn(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale intensities about their own range, clamped back to it.

    Matches dali_loader.py's contrast_fn: scale ~
    Uniform(CONTRAST_FN_RANGE_MIN, CONTRAST_FN_RANGE_MAX), applied with
    probability CONTRAST_FN_PROBABILITY (identity otherwise).
    """
    if rng.random() >= constants.CONTRAST_FN_PROBABILITY:
        return image
    min_, max_ = image.min(), image.max()
    scale = rng.uniform(constants.CONTRAST_FN_RANGE_MIN, constants.CONTRAST_FN_RANGE_MAX)
    return np.clip(image * scale, min_, max_)


def _apply_augmentations(
    image: np.ndarray,
    label: np.ndarray,
    dtm: np.ndarray | None,
    roi_size: tuple[int, int, int],
    use_flips: bool,
    use_zoom: bool,
    use_noise: bool,
    use_blur: bool,
    use_brightness: bool,
    use_contrast: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply MIST's training-time augmentations to one CPU-side patch.

    Mirrors dali_loader.py's TrainPipeline.define_graph precedence exactly,
    including the zoom/DTM asymmetry described in the module docstring:
    zoom only ever runs on the DTM-free path.
    """
    if dtm is not None:
        if use_flips:
            image, label, dtm = _flip_fn(image, label, dtm, rng)
    else:
        if use_zoom:
            image, label = _zoom_fn(image, label, roi_size, rng)
        if use_flips:
            image, label, dtm = _flip_fn(image, label, None, rng)

    if use_noise:
        image = _noise_fn(image, rng)
    if use_blur:
        image = _blur_fn(image, rng)
    if use_brightness:
        image = _brightness_fn(image, rng)
    if use_contrast:
        image = _contrast_fn(image, rng)

    return image, label, dtm


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
        use_augmentation: bool = False,
        use_flips: bool = False,
        use_zoom: bool = False,
        use_noise: bool = False,
        use_blur: bool = False,
        use_brightness: bool = False,
        use_contrast: bool = False,
    ):
        self._image_paths = image_paths
        self._label_paths = label_paths
        self._dtm_paths = dtm_paths
        self._roi_size = tuple(roi_size)
        self._labels = labels
        self._oversampling = oversampling
        self._extract_patches = extract_patches

        # Matches dali_loader.py's TrainPipeline: the master use_augmentation
        # switch ANDs into every individual flag, rather than being checked
        # separately at call time.
        self._use_augmentation = use_augmentation
        self._use_flips = use_flips and use_augmentation
        self._use_zoom = use_zoom and use_augmentation
        self._use_noise = use_noise and use_augmentation
        self._use_blur = use_blur and use_augmentation
        self._use_brightness = use_brightness and use_augmentation
        self._use_contrast = use_contrast and use_augmentation

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = _load_case(self._image_paths[index])
        label = _load_case(self._label_paths[index])
        dtm = _load_case(self._dtm_paths[index]) if self._dtm_paths else None

        rng = np.random.default_rng()

        if self._extract_patches:
            image, label, dtm = _extract_patch(
                image,
                label,
                dtm,
                self._roi_size,
                self._labels,
                self._oversampling,
                rng,
            )

        if self._use_augmentation:
            image, label, dtm = _apply_augmentations(
                image,
                label,
                dtm,
                self._roi_size,
                use_flips=self._use_flips,
                use_zoom=self._use_zoom,
                use_noise=self._use_noise,
                use_blur=self._use_blur,
                use_brightness=self._use_brightness,
                use_contrast=self._use_contrast,
                rng=rng,
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
    the module docstring). Augmentations run per `data_loading_constants.py`
    (same probabilities/ranges as `dali_loader.py`); the module docstring's
    note on the zoom/DTM asymmetry applies -- zoom only ever runs when
    `dtm_paths` is None.

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
        use_augmentation: Master switch for all augmentations below; ANDed
            into each individual flag, matching dali_loader.py.
        use_flips: Whether to randomly flip each spatial axis.
        use_zoom: Whether to randomly zoom in on the patch. Only applied
            when dtm_paths is None -- see the module docstring.
        use_noise: Whether to add random Gaussian noise to the image.
        use_blur: Whether to apply random Gaussian blur to the image.
        use_brightness: Whether to randomly scale image brightness.
        use_contrast: Whether to randomly scale image contrast.

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
        use_augmentation=use_augmentation,
        use_flips=use_flips,
        use_zoom=use_zoom,
        use_noise=use_noise,
        use_blur=use_blur,
        use_brightness=use_brightness,
        use_contrast=use_contrast,
    )
    shuffle_generator = torch.Generator().manual_seed(seed + rank)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=shuffle_generator,
        drop_last=False,
        multiprocessing_context=_multiprocessing_context(num_workers),
        # Deliberately NOT persistent_workers=True: MIST builds a fresh
        # training DataLoader per fold, and worker pools from earlier folds
        # were never explicitly torn down before the next one existed --
        # see the module docstring for the "fork" note right above this,
        # found via the same Stage 4 hang.
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
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=_multiprocessing_context(num_workers),
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
    )
    return GenericIterator(data_loader, device=_target_device(rank))
