"""Constants for the Analyzer class."""

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class AnalyzeConstants:
    """Dataclass for constants used in the analyze_data module."""

    # Maximum recommended memory in bytes for each example. We want to keep
    # the memory usage below this value to improve computational efficiency.
    MAX_RECOMMENDED_MEMORY_SIZE = 2e9

    # Minimum average volume reduction expressed as a fraction after cropping
    # each image to its foreground.
    MIN_AVERAGE_VOLUME_REDUCTION_FRACTION = 0.2

    # Minimum sparsity expressed as a fraction.
    MIN_SPARSITY_FRACTION = 0.2

    # A dataset is anisotropic if its coarsest axis has more than this many times
    # the spacing of the next coarsest axis, and fewer than one over this many
    # times the voxels of the smallest other axis.
    ANISOTROPY_THRESHOLD = 3.0

    # If a dataset is anisotropic, the target spacing of its coarsest axis is this
    # percentile of that axis's spacings across the dataset.
    ANISOTROPIC_LOW_RESOLUTION_AXIS_PERCENTILE = 10

    # If that percentile is finer than the next coarsest axis, the coarsest axis
    # is set to the next coarsest axis's spacing plus this, so it stays coarsest.
    ANISOTROPIC_SPACING_MARGIN = 1e-5

    # For CT images only. We gather every i-th voxel in the image where the
    # ground truth mask is non-zero. We use the resulting list of values
    # to compute the mean and standard deviation for z-score normalization.
    CT_GATHER_EVERY_ITH_VOXEL_VALUE = 10

    # For CT images only. We clip according to the 0.5 and 99.5 percentiles of
    # the voxels corresponding to the non-zero regions in the ground truth masks
    # over the entire dataset.
    CT_GLOBAL_CLIP_MIN_PERCENTILE = 0.5
    CT_GLOBAL_CLIP_MAX_PERCENTILE = 99.5

    # Histogram parameters for streaming CT percentile estimation. The range
    # covers the full clinical HU scale; with CT_HU_HIST_BINS bins the
    # resolution is 1 HU per bin, which is more than sufficient for windowing.
    CT_HU_HIST_MIN = -1024.0
    CT_HU_HIST_MAX = 3072.0
    CT_HU_HIST_BINS = 4096

    # How many digits are printed for floating point numbers.
    PRINT_FLOATING_POINT_PRECISION = 4

    # Create the base_config.json path.
    BASE_CONFIG_JSON_PATH = Path(__file__).parent / "base_config.json"
