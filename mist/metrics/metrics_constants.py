"""Constants for metrics computation."""


class LesionWiseMetricsConstants:
    """Constants for lesion-wise metric computation."""
    # Minimum GT lesion volume in mm³ to include in analysis. Lesions smaller
    # than this are excluded from both GT counting and FP detection.
    MIN_LESION_VOLUME = 10.0

    # Tolerance in mm for the surface Dice metric. Boundary voxels within this
    # distance of the opposing surface are considered matching.
    SURFACE_DICE_TOLERANCE_MM = 1.0

    # Number of dilation iterations used when searching for predicted
    # components that overlap a GT lesion. Larger values allow more spatially
    # separated predictions to be credited as detections.
    DILATION_ITERS = 3

    # Number of dilation iterations for GT lesion consolidation. GT lesions
    # whose dilated footprints connect are merged into a single lesion before
    # analysis. Set to 0 to skip consolidation.
    GT_CONSOLIDATION_ITERS = 0
