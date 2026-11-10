"""Dataclass for constants used in data loading."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoadingConstants:
    """Constants for training-time data augmentation.

    Probabilities and ranges follow nnU-Net v2's default training transforms
    (nnUNetTrainer.get_training_transforms). Probabilities are per sample
    unless named per channel.
    """

    # Rotation. Patches whose largest dimension is more than
    # ANISOTROPIC_PATCH_RATIO times their smallest are treated as quasi-2D:
    # they rotate (and scale) only in the plane of the two larger dimensions.
    ROTATION_PROBABILITY = 0.2
    ROTATION_MAX_DEGREES_3D = 30.0
    ROTATION_MAX_DEGREES_IN_PLANE = 180.0
    ANISOTROPIC_PATCH_RATIO = 3.0

    # Scaling, the same factor along every (in-plane) axis. Factors below 1
    # zoom in and factors above 1 zoom out.
    SCALING_PROBABILITY = 0.2
    SCALING_RANGE = (0.7, 1.4)

    # Rotation and scaling resample a larger patch cropped around the same
    # center, so the final patch has real image content at its borders. The
    # larger patch fits the final patch rotated by the maximum angle and
    # divided by this scale (nnU-Net's get_patch_size).
    LARGER_PATCH_MIN_SCALE = 0.85

    # Additive Gaussian noise; the standard deviation is drawn per sample.
    NOISE_PROBABILITY = 0.1
    NOISE_STD_RANGE = (0.0, 0.1)

    # Gaussian blur; sigma is drawn per channel and per axis.
    BLUR_PROBABILITY = 0.2
    BLUR_PER_CHANNEL_PROBABILITY = 0.5
    BLUR_SIGMA_RANGE = (0.5, 1.0)
    BLUR_TRUNCATE = 3.0

    # Multiplicative brightness, per channel.
    BRIGHTNESS_PROBABILITY = 0.15
    BRIGHTNESS_RANGE = (0.75, 1.25)

    # Contrast around the channel mean, clipped to the channel's range.
    CONTRAST_PROBABILITY = 0.15
    CONTRAST_RANGE = (0.75, 1.25)

    # Simulated low resolution: downsample by a factor drawn per channel
    # (nearest neighbor), then upsample back (linear).
    LOW_RESOLUTION_PROBABILITY = 0.25
    LOW_RESOLUTION_PER_CHANNEL_PROBABILITY = 0.5
    LOW_RESOLUTION_SCALE_RANGE = (0.5, 1.0)

    # Gamma correction with intensity statistics restored afterwards. The
    # inverted variant applies gamma to the negated image.
    GAMMA_INVERTED_PROBABILITY = 0.1
    GAMMA_PROBABILITY = 0.3
    GAMMA_RANGE = (0.7, 1.5)

    # Flips, independently along each axis.
    FLIP_PROBABILITY = 0.5
