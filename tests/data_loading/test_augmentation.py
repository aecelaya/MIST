"""Tests for the shared PyTorch training augmentation."""

import math

import numpy as np
import pytest
import torch

from mist.data_loading import augmentation as aug
from mist.data_loading.data_loading_constants import DataLoadingConstants as constants

ALL_OFF = aug.AugmentationSettings(**dict.fromkeys(aug.TRANSFORM_NAMES, False))


def _only(**enabled) -> aug.AugmentationSettings:
    """Settings with only the given transforms enabled."""
    return aug.AugmentationSettings(
        **{name: enabled.get(name, False) for name in aug.TRANSFORM_NAMES}
    )


def _batch(shape, n=2, channels=1, with_dtm=False, seed=0):
    """Random (N, C, D, H, W) image, integer label, and optional DTM."""
    generator = torch.Generator().manual_seed(seed)
    batch = {
        "image": torch.randn(n, channels, *shape, generator=generator),
        "label": torch.randint(0, 3, (n, 1, *shape), generator=generator).to(torch.uint8),
    }
    if with_dtm:
        batch["dtm"] = torch.rand(n, 1, *shape, generator=generator) * 2 - 1
    return batch


@pytest.fixture
def always(monkeypatch):
    """Set the named transform probabilities to 1."""

    def _always(*names):
        for name in names:
            monkeypatch.setattr(constants, name, 1.0)

    return _always


# Settings.


def test_settings_from_config_disabled_turns_everything_off():
    """enabled: false disables every transform."""
    settings = aug.settings_from_config({"enabled": False, "transforms": {"flips": True}})
    assert not settings.any_enabled


def test_settings_from_config_missing_transforms_are_enabled():
    """Transforms missing from an older config are enabled."""
    settings = aug.settings_from_config({"enabled": True, "transforms": {"flips": False}})
    assert settings.flips is False
    assert settings.rotation and settings.gamma and settings.low_resolution


def test_settings_from_config_reads_zoom_as_scaling():
    """The pre-2.5.0 zoom key sets scaling."""
    settings = aug.settings_from_config({"enabled": True, "transforms": {"zoom": False}})
    assert settings.scaling is False
    explicit = aug.settings_from_config(
        {"enabled": True, "transforms": {"zoom": False, "scaling": True}}
    )
    assert explicit.scaling is True


# Patch geometry.


@pytest.mark.parametrize(
    "patch_size, expected",
    [((128, 128, 128), None), ((256, 256, 5), 2), ((5, 192, 160), 0), ((96, 96, 32), None)],
)
def test_in_plane_axis(patch_size, expected):
    """Quasi-2D patches report their through-plane axis."""
    assert aug.in_plane_axis(patch_size) == expected


def test_spatial_margin_zero_without_spatial_transforms():
    """No margin when rotation and scaling are off."""
    assert aug.spatial_margin((128, 128, 128), _only(flips=True, gamma=True)) == (0, 0, 0)


def test_spatial_margin_matches_nnunet_formula_for_cube():
    """A 3D cube grows by the 30-degree rotation bounding box over 0.85."""
    margin = aug.spatial_margin((128, 128, 128), aug.AugmentationSettings())
    angle = math.radians(constants.ROTATION_MAX_DEGREES_3D)
    larger = int(128 * (math.cos(angle) + math.sin(angle)) / constants.LARGER_PATCH_MIN_SCALE)
    expected = math.ceil((larger - 128) / 2)
    assert margin == (expected, expected, expected)


def test_spatial_margin_scaling_only():
    """Scaling alone still enlarges the patch by 1/0.85."""
    margin = aug.spatial_margin((100, 100, 100), _only(scaling=True))
    assert margin == (9, 9, 9)  # ceil((int(100 / 0.85) - 100) / 2) = ceil(17 / 2)


def test_spatial_margin_skips_through_plane_axis():
    """Quasi-2D patches are not extended through-plane."""
    margin = aug.spatial_margin((256, 256, 5), aug.AugmentationSettings())
    assert margin[2] == 0
    assert margin[0] > 0 and margin[1] > 0


# Spatial transforms.


def test_no_transforms_center_crop_enlarged_patch():
    """With spatial transforms skipped, the output is the central crop."""
    batch = _batch((12, 14, 16))
    augmenter = aug.BatchAugmenter((8, 10, 12), _only(rotation=True), seed=0)
    augmenter.settings = ALL_OFF  # Never rotate; only crop.
    out = augmenter(batch)
    assert out["image"].shape == (2, 1, 8, 10, 12)
    torch.testing.assert_close(out["image"], batch["image"][:, :, 2:10, 2:12, 2:14])
    torch.testing.assert_close(out["label"], batch["label"][:, :, 2:10, 2:12, 2:14])


def test_identity_resampling_equals_center_crop():
    """The sampling grid maps output voxels exactly onto input voxels."""
    augmenter = aug.BatchAugmenter((8, 10, 12), aug.AugmentationSettings(), seed=0)
    image = torch.randn(1, 12, 14, 16)
    grid = augmenter._sampling_grid(np.eye(3), image.shape[1:], image.device)
    out = augmenter._resample(image, grid, mode="bilinear", padding_mode="zeros")
    torch.testing.assert_close(out, image[:, 2:10, 2:12, 2:14], atol=1e-5, rtol=0)


@pytest.mark.parametrize("axis, dims", [(0, (2, 3)), (1, (1, 3)), (2, (1, 2))])
def test_ninety_degree_rotation_matches_rot90(axis, dims):
    """A 90-degree rotation about an axis permutes voxels like rot90."""
    augmenter = aug.BatchAugmenter((8, 8, 8), aug.AugmentationSettings(), seed=0)
    volume = torch.arange(8**3, dtype=torch.float32).reshape(1, 8, 8, 8)
    grid = augmenter._sampling_grid(
        aug._rotation_matrix(axis, math.pi / 2), volume.shape[1:], volume.device
    )
    out = augmenter._resample(volume, grid, mode="nearest", padding_mode="zeros")
    candidates = [torch.rot90(volume, k, dims) for k in (1, 3)]
    assert any(torch.equal(out, c) for c in candidates)


def test_scaling_by_half_zooms_into_center():
    """A 0.5 scale samples the central half of the input."""
    augmenter = aug.BatchAugmenter((8, 8, 8), aug.AugmentationSettings(), seed=0)
    volume = torch.zeros(1, 8, 8, 8)
    volume[:, 2:6, 2:6, 2:6] = 1.0
    grid = augmenter._sampling_grid(np.eye(3) * 0.5, volume.shape[1:], volume.device)
    out = augmenter._resample(volume, grid, mode="nearest", padding_mode="zeros")
    assert torch.all(out == 1.0)


def test_rotation_keeps_labels_and_shapes(always):
    """Rotated labels stay integer class values from the input."""
    always("ROTATION_PROBABILITY", "SCALING_PROBABILITY")
    batch = _batch((20, 20, 20), with_dtm=True)
    augmenter = aug.BatchAugmenter((16, 16, 16), _only(rotation=True, scaling=True), seed=1)
    out = augmenter(batch)
    assert out["image"].shape == out["label"].shape == out["dtm"].shape == (2, 1, 16, 16, 16)
    assert out["label"].dtype == torch.uint8
    assert set(out["label"].unique().tolist()) <= {0, 1, 2}
    assert out["dtm"].min() >= -1 and out["dtm"].max() <= 1


def test_in_plane_rotation_leaves_through_plane_slices_aligned(always):
    """Quasi-2D rotation never mixes slices along the through-plane axis."""
    always("ROTATION_PROBABILITY")
    shape = (20, 20, 4)
    label = torch.zeros(1, 1, *shape, dtype=torch.uint8)
    label[..., 1] = 1  # One slice is foreground, the rest background.
    batch = {"image": torch.zeros(1, 1, *shape), "label": label}
    augmenter = aug.BatchAugmenter((16, 16, 4), _only(rotation=True), seed=3)
    assert augmenter._through_plane == 2
    out = augmenter(batch)
    assert out["label"][..., [0, 2, 3]].sum() == 0
    assert out["label"][..., 1].sum() > 0


# Flips.


def test_flips_apply_the_same_flip_to_image_label_and_dtm(always):
    """Image, label, and DTM are flipped together."""
    always("FLIP_PROBABILITY")
    batch = _batch((6, 6, 6), n=1, with_dtm=True)
    augmenter = aug.BatchAugmenter((6, 6, 6), _only(flips=True), seed=0)
    out = augmenter(batch)
    for key in ("image", "label", "dtm"):
        torch.testing.assert_close(out[key], torch.flip(batch[key], (2, 3, 4)))


# Intensity transforms.


def _augmenter(**enabled):
    return aug.BatchAugmenter((8, 8, 8), _only(**enabled), seed=0)


def test_noise_changes_image_only(always):
    """Noise perturbs the image and leaves the label alone."""
    always("NOISE_PROBABILITY")
    batch = _batch((8, 8, 8))
    out = _augmenter(noise=True)(batch)
    assert not torch.equal(out["image"], batch["image"])
    torch.testing.assert_close(out["label"], batch["label"])


def test_blur_smooths_the_image(always):
    """Blurring every channel lowers its variance."""
    always("BLUR_PROBABILITY", "BLUR_PER_CHANNEL_PROBABILITY")
    batch = _batch((12, 12, 12))
    out = _augmenter(blur=True)(batch)
    assert out["image"].var() < batch["image"].var()


def test_brightness_scales_each_channel_within_range(always):
    """Each channel is multiplied by a factor in the brightness range."""
    always("BRIGHTNESS_PROBABILITY")
    batch = {"image": torch.ones(1, 3, 4, 4, 4), "label": torch.zeros(1, 1, 4, 4, 4)}
    out = _augmenter(brightness=True)(batch)
    factors = out["image"].flatten(2).mean(-1).squeeze(0)
    low, high = constants.BRIGHTNESS_RANGE
    assert torch.all((factors >= low) & (factors <= high))


def test_contrast_preserves_range(always):
    """Contrast keeps every channel within its original range."""
    always("CONTRAST_PROBABILITY")
    batch = _batch((8, 8, 8), channels=2)
    out = _augmenter(contrast=True)(batch)
    for c in range(2):
        original = batch["image"][:, c]
        assert out["image"][:, c].min() >= original.min() - 1e-6
        assert out["image"][:, c].max() <= original.max() + 1e-6


def test_low_resolution_keeps_shape(always):
    """Simulated low resolution returns the original patch shape."""
    always("LOW_RESOLUTION_PROBABILITY", "LOW_RESOLUTION_PER_CHANNEL_PROBABILITY")
    batch = _batch((8, 10, 12))
    augmenter = aug.BatchAugmenter((8, 10, 12), _only(low_resolution=True), seed=0)
    out = augmenter(batch)
    assert out["image"].shape == batch["image"].shape


def test_low_resolution_leaves_through_plane_axis(always):
    """For quasi-2D patches, slices are never resampled through-plane."""
    always("LOW_RESOLUTION_PROBABILITY", "LOW_RESOLUTION_PER_CHANNEL_PROBABILITY")
    image = torch.zeros(1, 1, 16, 16, 4)
    image[..., 1] = 1.0
    augmenter = aug.BatchAugmenter((16, 16, 4), _only(low_resolution=True), seed=0)
    out = augmenter({"image": image, "label": torch.zeros_like(image)})
    torch.testing.assert_close(out["image"], image)


@pytest.mark.parametrize("invert", [False, True])
def test_gamma_retains_mean_and_std(invert):
    """Gamma restores each channel's mean and standard deviation."""
    image = torch.randn(2, 8, 8, 8)
    out = _augmenter(gamma=True)._gamma(image, invert=invert)
    torch.testing.assert_close(out.mean((1, 2, 3)), image.mean((1, 2, 3)), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std((1, 2, 3)), image.std((1, 2, 3)), atol=1e-4, rtol=0)
    assert not torch.allclose(out, image)


def test_bg_contrast_draws_both_sides_of_one():
    """Factors fall below and above 1 about equally often."""
    rng = np.random.default_rng(0)
    draws = np.array([aug._bg_contrast(rng, 0.7, 1.5) for _ in range(2000)])
    assert draws.min() >= 0.7 and draws.max() <= 1.5
    assert 0.4 < np.mean(draws < 1) < 0.6


# Determinism and the iterator wrapper.


def test_same_seed_gives_same_augmentation():
    """Two augmenters with the same seed produce the same batches."""
    batch = _batch((20, 20, 20))
    settings = aug.AugmentationSettings()
    first = aug.BatchAugmenter((16, 16, 16), settings, seed=7)
    second = aug.BatchAugmenter((16, 16, 16), settings, seed=7)
    for _ in range(3):
        a, b = first(batch), second(batch)
        torch.testing.assert_close(a["image"], b["image"])
        torch.testing.assert_close(a["label"], b["label"])


class _ListIterator:
    def __init__(self, batch):
        self.batch = batch
        self.resets = 0

    def next(self):
        return [self.batch]

    def reset(self):
        self.resets += 1


def test_augmented_iterator_wraps_next_and_reset():
    """The wrapper augments .next() and forwards .reset()."""
    inner = _ListIterator(_batch((10, 10, 10)))
    augmenter = aug.BatchAugmenter((8, 8, 8), ALL_OFF, seed=0)
    iterator = aug.AugmentedIterator(inner, augmenter)
    (batch,) = iterator.next()
    assert batch["image"].shape == (2, 1, 8, 8, 8)
    iterator.reset()
    assert inner.resets == 1


# DTMs under scaling.


def _ramp_dtm(shape):
    """A DTM whose value is the signed voxel distance along axis 0 from the center."""
    d = torch.arange(shape[0], dtype=torch.float32) - (shape[0] - 1) / 2
    return d.view(1, 1, -1, 1, 1).expand(1, 1, *shape).clone()


def _scaled_dtm(monkeypatch, normalized, factor):
    """Scale a ramp DTM by a fixed factor and return input and output."""
    monkeypatch.setattr(constants, "SCALING_PROBABILITY", 1.0)
    monkeypatch.setattr(constants, "SCALING_RANGE", (factor, factor))
    shape = (24, 24, 24)
    batch = {
        "image": torch.zeros(1, 1, *shape),
        "label": torch.zeros(1, 1, *shape, dtype=torch.uint8),
        "dtm": _ramp_dtm(shape),
    }
    augmenter = aug.BatchAugmenter(
        (16, 16, 16), _only(scaling=True), seed=0, normalized_dtms=normalized
    )
    return augmenter(batch)["dtm"][0, 0, :, 8, 8]


@pytest.mark.parametrize("factor", [0.7, 1.25])
def test_unnormalized_dtm_distances_follow_the_zoom(monkeypatch, factor):
    """Unnormalized DTMs are rescaled to distances in output voxels."""
    profile = _scaled_dtm(monkeypatch, normalized=False, factor=factor)
    slope = (profile[-2] - profile[1]) / (len(profile) - 3)
    assert slope.item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("factor", [0.7, 1.25])
def test_normalized_dtm_values_are_not_rescaled(monkeypatch, factor):
    """Normalized DTMs keep their values; only the grid is resampled."""
    profile = _scaled_dtm(monkeypatch, normalized=True, factor=factor)
    slope = (profile[-2] - profile[1]) / (len(profile) - 3)
    assert slope.item() == pytest.approx(factor, abs=1e-4)


def test_quasi_2d_patches_skip_scaling_with_dtms(monkeypatch):
    """In-plane scaling would distort DTMs, so it is skipped when they exist."""
    monkeypatch.setattr(constants, "SCALING_PROBABILITY", 1.0)
    monkeypatch.setattr(constants, "SCALING_RANGE", (0.5, 0.5))
    shape = (20, 20, 4)
    batch = _batch(shape, n=1, with_dtm=True)
    augmenter = aug.BatchAugmenter((16, 16, 4), _only(scaling=True), seed=0)
    out = augmenter(batch)
    torch.testing.assert_close(out["image"], batch["image"][:, :, 2:18, 2:18, :])
    torch.testing.assert_close(out["dtm"], batch["dtm"][:, :, 2:18, 2:18, :])

    # Without DTMs, the same patch is scaled in-plane.
    no_dtm = {"image": batch["image"], "label": batch["label"]}
    scaled = aug.BatchAugmenter((16, 16, 4), _only(scaling=True), seed=0)(no_dtm)
    assert not torch.allclose(scaled["image"], batch["image"][:, :, 2:18, 2:18, :])


# Per-class label resampling.


def _label_grid(augmenter, matrix, size):
    return augmenter._sampling_grid(matrix, size, torch.device("cpu"))


def test_resample_labels_identity_equals_center_crop():
    """With no transform, labels come back unchanged."""
    augmenter = aug.BatchAugmenter((8, 8, 8), aug.AugmentationSettings(), seed=0)
    label = torch.randint(0, 4, (1, 12, 12, 12)).to(torch.uint8)
    out = augmenter._resample_labels(label, _label_grid(augmenter, np.eye(3), (12, 12, 12)))
    assert out.dtype == torch.uint8
    assert torch.equal(out, label[:, 2:10, 2:10, 2:10])


def test_resample_labels_matches_argmax_over_all_classes():
    """The result equals an explicit argmax over every class, background included."""
    augmenter = aug.BatchAugmenter((12, 12, 12), aug.AugmentationSettings(), seed=0)
    label = torch.randint(0, 4, (1, 16, 16, 16)).to(torch.uint8)
    matrix = aug._rotation_matrix(0, 0.4) @ aug._rotation_matrix(2, 0.3) * 0.9
    grid = _label_grid(augmenter, matrix, (16, 16, 16))

    # Outside the input counts as background, so sample (indicator - 1) and add 1.
    scores = []
    for value in range(4):
        indicator = (label[0] == value).float()
        offset = 1.0 if value == 0 else 0.0
        scores.append(
            torch.nn.functional.grid_sample(
                (indicator - offset)[None, None], grid, mode="bilinear", align_corners=False
            )[0, 0]
            + offset
        )
    stacked = torch.stack(scores)
    out = augmenter._resample_labels(label, grid)

    # Compare by score, so a voxel with a tied top score may take either class.
    own = stacked.gather(0, out.long())[0]
    torch.testing.assert_close(own, stacked.max(0).values, atol=1e-5, rtol=0)


def test_resample_labels_never_invents_a_class():
    """Interpolation between classes 1 and 3 can't produce class 2."""
    augmenter = aug.BatchAugmenter((16, 16, 16), aug.AugmentationSettings(), seed=0)
    label = torch.ones(1, 16, 16, 16, dtype=torch.uint8)
    label[:, :, 8:] = 3
    out = augmenter._resample_labels(
        label, _label_grid(augmenter, aug._rotation_matrix(0, 0.5), (16, 16, 16))
    )
    assert set(out.unique().tolist()) <= {0, 1, 3}


def test_resample_labels_outside_the_input_is_background():
    """Zooming far out pads the label with background."""
    augmenter = aug.BatchAugmenter((8, 8, 8), aug.AugmentationSettings(), seed=0)
    label = torch.ones(1, 8, 8, 8, dtype=torch.uint8)
    out = augmenter._resample_labels(label, _label_grid(augmenter, np.eye(3) * 2.0, (8, 8, 8)))
    assert out[0, 0, 0, 0] == 0
    assert out[0, 4, 4, 4] == 1


def test_resample_labels_all_background_stays_background():
    """A patch with no foreground needs no interpolation."""
    augmenter = aug.BatchAugmenter((8, 8, 8), aug.AugmentationSettings(), seed=0)
    label = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    out = augmenter._resample_labels(
        label, _label_grid(augmenter, aug._rotation_matrix(1, 0.3), (8, 8, 8))
    )
    assert out.sum() == 0
