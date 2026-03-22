"""Constants for the Analyzer class."""
import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class AnalyzeConstants:
    """Dataclass for constants used in the analyze_data module."""
    # RAI orientation direction for ANTs.
    RAI_ANTS_DIRECTION = np.eye(3)

    # Maximum recommended memory in bytes for each example. We want to keep
    # the memory usage below this value to improve computational efficiency.
    MAX_RECOMMENDED_MEMORY_SIZE = 2e9

    # Minimum average volume reduction expressed as a fraction after cropping
    # each image to its foreground.
    MIN_AVERAGE_VOLUME_REDUCTION_FRACTION = 0.2

    # Minimum sparsity expressed as a fraction.
    MIN_SPARSITY_FRACTION = 0.2

    # Maximum ratio of the maximum and minimum components of the target spacing
    # for an image to be considered anisotropic.
    MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD = 3.0

    # If the median target spacing of a dataset is anisotropic, then we
    # take this percentile of the coarsest axis and replace that value with
    # the resulting percentile value.
    ANISOTROPIC_LOW_RESOLUTION_AXIS_PERCENTILE = 10

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

    # Maximum number of voxel coordinates sampled per label per patient for
    # PCA-based shape descriptor computation.
    MAX_SHAPE_COORDS = 10_000

    # Labels with more voxels than this are skipped for skeletonization to
    # bound compute time. Structures this large are rarely thin/tubular.
    MAX_SKELETON_VOXELS = 500_000

    # Skeleton ratio threshold for flagging a label as tubular/branching.
    # Above this fraction of label voxels lie on the medial axis → thin
    # branching structure; clDice loss is recommended.
    TUBULAR_SKELETON_RATIO_THRESHOLD = 0.05

    # Two-tier image-fraction thresholds. Labels below VERY_SPARSE get the
    # "very sparse" observation; labels in [VERY_SPARSE, SPARSE) get "sparse".
    # Both are relative to the effective image volume (fg bbox when
    # crop_to_foreground is enabled, otherwise full original image).
    # Distinct from size_category, which is relative to foreground only.
    VERY_SPARSE_IMAGE_FRACTION_PCT_THRESHOLD = 1.0
    SPARSE_IMAGE_FRACTION_PCT_THRESHOLD = 5.0

    # Size categories considered "small" for the resampling warning. Labels
    # in these categories that are also geometrically thin may lose structural
    # detail when resampled to a heuristic target spacing derived from
    # whole-image statistics.
    SMALL_STRUCTURE_SIZE_CATEGORIES = frozenset({"tiny", "small"})

    # Minimum patch size along the low-resolution (anisotropic) axis in the
    # quasi-2D patch selection mode. With a 3×3×3 kernel, at least 5 voxels
    # are needed in the low-res axis to ensure the kernel has genuine
    # (non-padded) context from neighboring slices at every position.
    MIN_LOW_RES_AXIS_PATCH_SIZE = 5

    # Reference GPU memory (bytes) used to scale the patch voxel budget.
    # At this memory level the reference voxel budget is used as-is; smaller
    # GPUs get a proportionally smaller budget and larger GPUs a larger one.
    PATCH_BUDGET_REFERENCE_GPU_MEMORY_BYTES = 16 * (1024 ** 3)  # 16 GB

    # Per-patch voxel budget at the reference GPU memory and reference batch
    # size. This is a conservative heuristic chosen to leave headroom for
    # network activations, gradients, and optimizer state. It is not tuned to
    # a specific architecture — heavier models (e.g. MedNeXt-large, SwinUNETR)
    # may require a smaller patch size, which can be set manually in config.json
    # or via --patch-size.
    PATCH_BUDGET_REFERENCE_VOXELS = 128 ** 3

    # The batch size per GPU assumed when the reference voxel budget was set.
    # Matches the MIST default (training.batch_size_per_gpu = 2). The budget
    # scales inversely with actual batch size so that total memory per step
    # (batch_size × patch_voxels × overhead) stays constant.
    PATCH_BUDGET_REFERENCE_BATCH_SIZE = 2

    # Fallback voxel budget used when no CUDA device is available.
    PATCH_BUDGET_DEFAULT_VOXELS = 128 ** 3

    # Create the base_config.json path.
    BASE_CONFIG_JSON_PATH = Path(__file__).parent / "base_config.json"
