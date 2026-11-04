"""Training-time data augmentation for MIST.

Augmentation runs in PyTorch on each training batch after the data loader has
moved it to the training device, so the DALI and generic loaders share one
implementation. The loaders only extract patches: when rotation or scaling is
enabled, they crop a larger patch around the same center (see
`spatial_margin`), and the spatial transform resamples it down to the final
patch size, so rotated or zoomed-out patches have real image content at their
borders instead of padding.

The transforms, their order, and their probabilities and ranges follow nnU-Net
v2's default training pipeline (see data_loading_constants.py): rotation and
scaling, Gaussian noise, Gaussian blur, brightness, contrast, simulated low
resolution, inverted gamma, gamma, and flips. Spatial transforms apply to the
image, label, and DTM alike; intensity transforms apply to the image only.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mist.data_loading.data_loading_constants import DataLoadingConstants as constants

# Names of the configurable transforms, as they appear under
# training.augmentation.transforms in config.json.
TRANSFORM_NAMES = (
    "flips",
    "rotation",
    "scaling",
    "noise",
    "blur",
    "brightness",
    "contrast",
    "low_resolution",
    "gamma",
)


@dataclasses.dataclass(frozen=True)
class AugmentationSettings:
    """Which augmentations are enabled."""

    flips: bool = True
    rotation: bool = True
    scaling: bool = True
    noise: bool = True
    blur: bool = True
    brightness: bool = True
    contrast: bool = True
    low_resolution: bool = True
    gamma: bool = True

    @property
    def any_enabled(self) -> bool:
        """Whether at least one transform is enabled."""
        return any(getattr(self, name) for name in TRANSFORM_NAMES)

    @property
    def spatial(self) -> bool:
        """Whether rotation or scaling is enabled."""
        return self.rotation or self.scaling


def settings_from_config(augmentation_config: Mapping[str, Any]) -> AugmentationSettings:
    """Read augmentation settings from training.augmentation in config.json.

    Transforms missing from the config are enabled, so configs written before
    a transform existed get it. The pre-2.5.0 "zoom" key is read as "scaling".

    Args:
        augmentation_config: The training.augmentation section, with an
            "enabled" flag and a "transforms" mapping of names to booleans.

    Returns:
        The settings; every transform is disabled if "enabled" is false.
    """
    if not augmentation_config.get("enabled", True):
        return AugmentationSettings(**dict.fromkeys(TRANSFORM_NAMES, False))

    transforms = dict(augmentation_config.get("transforms", {}))
    if "scaling" not in transforms and "zoom" in transforms:
        transforms["scaling"] = transforms["zoom"]
    return AugmentationSettings(
        **{name: bool(transforms.get(name, True)) for name in TRANSFORM_NAMES}
    )


def in_plane_axis(patch_size: Sequence[int]) -> int | None:
    """Return the through-plane axis of a quasi-2D patch, or None.

    A patch is quasi-2D when its largest dimension is more than
    ANISOTROPIC_PATCH_RATIO times its smallest (e.g., 256 x 256 x 5). Such
    patches rotate and scale only in the plane of the two larger dimensions,
    and simulated low resolution leaves the through-plane axis alone.
    """
    if max(patch_size) / min(patch_size) > constants.ANISOTROPIC_PATCH_RATIO:
        return int(np.argmin(patch_size))
    return None


def _rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """Return the 3 x 3 rotation about one array axis by angle radians."""
    rotation = np.eye(3)
    i, j = [a for a in range(3) if a != axis]
    cos, sin = math.cos(angle), math.sin(angle)
    rotation[i, i], rotation[i, j] = cos, -sin
    rotation[j, i], rotation[j, j] = sin, cos
    return rotation


def spatial_margin(
    patch_size: Sequence[int], settings: AugmentationSettings
) -> tuple[int, int, int]:
    """Return how many voxels to extend each side of the patch before cropping.

    The larger patch is the final patch's bounding box rotated by the maximum
    angle about each rotation axis, divided by LARGER_PATCH_MIN_SCALE, as in
    nnU-Net's get_patch_size. The through-plane axis of a quasi-2D patch is not
    extended, since nothing rotates or scales along it.

    Args:
        patch_size: The final patch size.
        settings: The augmentation settings.

    Returns:
        The margin along each axis; all zero if rotation and scaling are off.
    """
    if not settings.spatial:
        return (0, 0, 0)

    through_plane = in_plane_axis(patch_size)
    max_degrees = (
        constants.ROTATION_MAX_DEGREES_3D
        if through_plane is None
        else constants.ROTATION_MAX_DEGREES_IN_PLANE
    )
    max_angle = math.radians(min(max_degrees, 90.0)) if settings.rotation else 0.0
    axes = range(3) if through_plane is None else [through_plane]

    size = np.asarray(patch_size, dtype=float)
    larger = size.copy()
    for axis in axes:
        rotated = np.abs(_rotation_matrix(axis, max_angle)) @ size
        larger = np.maximum(larger, rotated)
    larger /= constants.LARGER_PATCH_MIN_SCALE

    margin = np.ceil((larger.astype(int) - size) / 2).clip(min=0).astype(int)
    if through_plane is not None:
        margin[through_plane] = 0
    return tuple(int(m) for m in margin)


def _bg_contrast(rng: np.random.Generator, low: float, high: float) -> float:
    """Draw a factor below or above 1 with equal probability.

    Matches nnU-Net's BGContrast: half the time from [low, 1), otherwise from
    [1, high], so factors that darken and brighten are equally likely however
    asymmetric the range is.
    """
    if low < 1 and rng.random() < 0.5:
        return float(rng.uniform(low, 1.0))
    return float(rng.uniform(max(low, 1.0), high))


class BatchAugmenter:
    """Apply MIST's training augmentations to a batch on its device.

    Attributes:
        patch_size: The final patch size; inputs may be larger by the
            spatial margin.
        settings: Which transforms are enabled.
    """

    def __init__(
        self,
        patch_size: Sequence[int],
        settings: AugmentationSettings,
        seed: int,
        normalized_dtms: bool = True,
    ):
        """Initialize the augmenter.

        Args:
            patch_size: The final patch size.
            settings: Which transforms are enabled.
            seed: Seed for the random parameters (offset it per rank).
            normalized_dtms: Whether DTMs are normalized to [-1, 1]
                (preprocessing.normalize_dtms). Unnormalized DTMs hold
                distances in voxels, which scaling has to rescale.
        """
        self.patch_size = tuple(int(s) for s in patch_size)
        self.settings = settings
        self.normalized_dtms = normalized_dtms
        self._through_plane = in_plane_axis(self.patch_size)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._noise_generators: dict[torch.device, torch.Generator] = {}

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Augment a batch of (N, C, D, H, W) image, label, and DTM tensors.

        Args:
            batch: Dict with "image", "label", and optionally "dtm".

        Returns:
            The augmented batch, cropped or resampled to patch_size.
        """
        image = batch["image"]
        label = batch["label"]
        dtm = batch.get("dtm")

        images, labels, dtms = [], [], []
        for n in range(image.shape[0]):
            sample_image, sample_label, sample_dtm = self._spatial(
                image[n],
                label[n],
                dtm[n] if dtm is not None else None,
            )
            sample_image = self._intensity(sample_image)
            if self.settings.flips:
                sample_image, sample_label, sample_dtm = self._flip(
                    sample_image, sample_label, sample_dtm
                )
            images.append(sample_image)
            labels.append(sample_label)
            dtms.append(sample_dtm)

        augmented = {"image": torch.stack(images), "label": torch.stack(labels)}
        if dtm is not None:
            augmented["dtm"] = torch.stack(dtms)
        return augmented

    # Spatial transforms.

    def _spatial(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        dtm: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Rotate and scale one (C, D, H, W) sample, or center-crop it.

        DTMs hold signed distances to each label's boundary, which flips and
        rotations preserve. Scaling every axis by the same factor scales the
        distances by it too: normalized DTMs are unaffected, since their
        normalizing maximum distance scales alike, and unnormalized ones are
        divided by the factor. Scaling only in-plane, as for quasi-2D patches,
        would leave no valid distance map, so those patches skip scaling when
        DTMs are present.
        """
        rotate = self.settings.rotation and self._rng.random() < constants.ROTATION_PROBABILITY
        scale = self.settings.scaling and self._rng.random() < constants.SCALING_PROBABILITY
        if dtm is not None and self._through_plane is not None:
            scale = False

        if not (rotate or scale):
            return (
                self._center_crop(image),
                self._center_crop(label),
                self._center_crop(dtm) if dtm is not None else None,
            )

        matrix = np.eye(3)
        if rotate:
            matrix = self._random_rotation() @ matrix
        factor = 1.0
        if scale:
            factor = self._rng.uniform(*constants.SCALING_RANGE)
            scaling = np.full(3, factor)
            if self._through_plane is not None:
                scaling[self._through_plane] = 1.0
            matrix = matrix @ np.diag(scaling)

        grid = self._sampling_grid(matrix, image.shape[1:], image.device)
        image = self._resample(image, grid, mode="bilinear", padding_mode="zeros")
        label = self._resample_labels(label, grid)
        if dtm is not None:
            dtm = self._resample(dtm, grid, mode="bilinear", padding_mode="border")
            # One output voxel spans `factor` input voxels, so distances in
            # voxels shrink by that factor.
            if not self.normalized_dtms:
                dtm = dtm / factor
        return image, label, dtm

    def _random_rotation(self) -> np.ndarray:
        """Draw a random rotation (in-plane only for quasi-2D patches)."""
        if self._through_plane is not None:
            max_angle = math.radians(constants.ROTATION_MAX_DEGREES_IN_PLANE)
            angle = self._rng.uniform(-max_angle, max_angle)
            return _rotation_matrix(self._through_plane, angle)

        max_angle = math.radians(constants.ROTATION_MAX_DEGREES_3D)
        rotation = np.eye(3)
        for axis in range(3):
            angle = self._rng.uniform(-max_angle, max_angle)
            rotation = _rotation_matrix(axis, angle) @ rotation
        return rotation

    def _sampling_grid(
        self,
        matrix: np.ndarray,
        input_size: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build a grid_sample grid mapping output voxels to input voxels.

        `matrix` maps an output voxel's offset from the patch center to an
        input voxel's offset from the input center, in (D, H, W) voxel units.
        grid_sample works in normalized coordinates ordered (W, H, D), so the
        matrix is rescaled by each size and its axes reversed.
        """
        input_half = np.asarray(input_size, dtype=float) / 2.0
        output_half = np.asarray(self.patch_size, dtype=float) / 2.0
        normalized = np.diag(1.0 / input_half) @ matrix @ np.diag(output_half)
        normalized = normalized[::-1, ::-1]

        theta = np.zeros((1, 3, 4))
        theta[0, :, :3] = normalized
        theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
        return F.affine_grid(theta, size=(1, 1, *self.patch_size), align_corners=False)

    @staticmethod
    def _resample(
        volume: torch.Tensor,
        grid: torch.Tensor,
        mode: str,
        padding_mode: str,
    ) -> torch.Tensor:
        """Resample one (C, D, H, W) volume on a sampling grid."""
        resampled = F.grid_sample(
            volume.unsqueeze(0).float(),
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        return resampled.squeeze(0).to(volume.dtype)

    @staticmethod
    def _resample_labels(label: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Resample a (C, D, H, W) label volume by per-class linear interpolation.

        Each foreground class's indicator is resampled linearly and every voxel
        takes the class with the highest score, as in nnU-Net's SpatialTransform.
        Background needs no resampling of its own: the indicators sum to one and
        interpolation is linear, so its score is one minus the foreground scores.
        Outside the input, every foreground score falls to zero, so that region
        becomes background. Ties go to background, then to the lower class.
        """
        channels = []
        for channel in label:
            indicator = torch.empty_like(channel, dtype=torch.float32)
            result = torch.zeros(grid.shape[1:4], dtype=label.dtype, device=label.device)
            best = None
            total = None
            for value in channel.unique():
                if value == 0:
                    continue
                torch.eq(channel, value, out=indicator)
                score = F.grid_sample(
                    indicator[None, None],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0]
                if best is None:
                    best = score.clone()
                    total = score.clone()
                    result[:] = value
                else:
                    better = score > best
                    best[better] = score[better]
                    result[better] = value
                    total += score
            if best is not None:
                result[best <= 1.0 - total] = 0
            channels.append(result)
        return torch.stack(channels)

    def _center_crop(self, volume: torch.Tensor) -> torch.Tensor:
        """Crop the central patch_size region of one (C, D, H, W) volume."""
        starts = [
            (size - patch) // 2
            for size, patch in zip(volume.shape[1:], self.patch_size, strict=True)
        ]
        return volume[
            :,
            starts[0] : starts[0] + self.patch_size[0],
            starts[1] : starts[1] + self.patch_size[1],
            starts[2] : starts[2] + self.patch_size[2],
        ]

    def _flip(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        dtm: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Flip each spatial axis independently."""
        dims = [axis + 1 for axis in range(3) if self._rng.random() < constants.FLIP_PROBABILITY]
        if not dims:
            return image, label, dtm
        return (
            torch.flip(image, dims),
            torch.flip(label, dims),
            torch.flip(dtm, dims) if dtm is not None else None,
        )

    # Intensity transforms, applied to the image only.

    def _intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the enabled intensity transforms to one (C, D, H, W) image."""
        settings = self.settings
        image = image.float()
        if settings.noise and self._rng.random() < constants.NOISE_PROBABILITY:
            image = self._noise(image)
        if settings.blur and self._rng.random() < constants.BLUR_PROBABILITY:
            image = self._blur(image)
        if settings.brightness and self._rng.random() < constants.BRIGHTNESS_PROBABILITY:
            image = self._brightness(image)
        if settings.contrast and self._rng.random() < constants.CONTRAST_PROBABILITY:
            image = self._contrast(image)
        if settings.low_resolution and self._rng.random() < constants.LOW_RESOLUTION_PROBABILITY:
            image = self._low_resolution(image)
        if settings.gamma:
            if self._rng.random() < constants.GAMMA_INVERTED_PROBABILITY:
                image = self._gamma(image, invert=True)
            if self._rng.random() < constants.GAMMA_PROBABILITY:
                image = self._gamma(image, invert=False)
        return image

    def _noise_generator(self, device: torch.device) -> torch.Generator:
        """Return a seeded generator for per-voxel noise on this device."""
        if device not in self._noise_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._seed)
            self._noise_generators[device] = generator
        return self._noise_generators[device]

    def _noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise with one standard deviation per sample."""
        std = self._rng.uniform(*constants.NOISE_STD_RANGE)
        noise = torch.randn(
            image.shape,
            generator=self._noise_generator(image.device),
            device=image.device,
            dtype=image.dtype,
        )
        return image + std * noise

    def _blur(self, image: torch.Tensor) -> torch.Tensor:
        """Blur each channel with probability BLUR_PER_CHANNEL_PROBABILITY."""
        channels = []
        for channel in image:
            if self._rng.random() < constants.BLUR_PER_CHANNEL_PROBABILITY:
                sigmas = self._rng.uniform(*constants.BLUR_SIGMA_RANGE, size=3)
                channel = _gaussian_blur(channel, sigmas)
            channels.append(channel)
        return torch.stack(channels)

    def _brightness(self, image: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by its own factor."""
        factors = [_bg_contrast(self._rng, *constants.BRIGHTNESS_RANGE) for _ in image]
        factors = torch.tensor(factors, dtype=image.dtype, device=image.device)
        return image * factors.view(-1, 1, 1, 1)

    def _contrast(self, image: torch.Tensor) -> torch.Tensor:
        """Scale each channel about its mean, clipped to its original range."""
        channels = []
        for channel in image:
            factor = _bg_contrast(self._rng, *constants.CONTRAST_RANGE)
            mean = channel.mean()
            contrasted = (channel - mean) * factor + mean
            channels.append(contrasted.clamp(channel.min(), channel.max()))
        return torch.stack(channels)

    def _low_resolution(self, image: torch.Tensor) -> torch.Tensor:
        """Downsample, then upsample, each channel with some probability."""
        size = image.shape[1:]
        channels = []
        for channel in image:
            if self._rng.random() < constants.LOW_RESOLUTION_PER_CHANNEL_PROBABILITY:
                factor = self._rng.uniform(*constants.LOW_RESOLUTION_SCALE_RANGE)
                low_size = [max(1, round(s * factor)) for s in size]
                if self._through_plane is not None:
                    low_size[self._through_plane] = size[self._through_plane]
                low = F.interpolate(channel[None, None], size=low_size, mode="nearest-exact")
                channel = F.interpolate(
                    low, size=tuple(size), mode="trilinear", align_corners=False
                )[0, 0]
            channels.append(channel)
        return torch.stack(channels)

    def _gamma(self, image: torch.Tensor, invert: bool) -> torch.Tensor:
        """Apply gamma per channel, then restore its mean and std."""
        channels = []
        for channel in image:
            gamma = _bg_contrast(self._rng, *constants.GAMMA_RANGE)
            if invert:
                channel = -channel
            mean, std = channel.mean(), channel.std()
            minimum = channel.min()
            value_range = channel.max() - minimum
            channel = ((channel - minimum) / (value_range + 1e-7)) ** gamma * value_range + minimum
            channel = (channel - channel.mean()) / (channel.std() + 1e-8) * std + mean
            if invert:
                channel = -channel
            channels.append(channel)
        return torch.stack(channels)


def _gaussian_blur(volume: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
    """Blur a (D, H, W) volume with a separable Gaussian, one sigma per axis."""
    blurred = volume[None, None]
    for axis, sigma in enumerate(sigmas):
        radius = max(1, int(math.ceil(constants.BLUR_TRUNCATE * sigma)))
        offsets = torch.arange(-radius, radius + 1, dtype=volume.dtype, device=volume.device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        padding = [0, 0, 0, 0, 0, 0]
        # F.pad lists padding from the last axis backwards.
        padding[2 * (2 - axis)] = radius
        padding[2 * (2 - axis) + 1] = radius
        blurred = F.conv3d(F.pad(blurred, padding, mode="replicate"), kernel.view(shape))
    return blurred[0, 0]


class AugmentedIterator:
    """Wrap a training iterator so each batch is augmented on its device.

    Keeps the `.next()` / `.reset()` contract shared by the DALI and generic
    loaders: `.next()` returns a one-element list holding the batch dict.
    """

    def __init__(self, iterator: Any, augmenter: BatchAugmenter):
        self._iterator = iterator
        self._augmenter = augmenter

    def next(self) -> list[dict[str, torch.Tensor]]:
        """Return the next augmented batch."""
        return [self._augmenter(self._iterator.next()[0])]

    def reset(self) -> None:
        """Reset the wrapped iterator."""
        self._iterator.reset()
