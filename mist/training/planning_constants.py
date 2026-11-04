"""Constants for planning the patch and batch size."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class PlanningConstants:
    """Constants for planning the patch and batch size (patch_planning.py)."""

    # Peak memory per GPU is estimated as overhead + parameters * bytes per
    # parameter + activations * batch size * bytes per activation.

    # Weights, gradients, and the optimizer's two moments, 4 bytes each.
    BYTES_PER_PARAMETER = 16.0

    # CUDA context and allocator overhead.
    OVERHEAD_BYTES = 0.5 * (1024**3)

    # Fitted to one run: nnunet-plain, patch 128 x 128 x 64, batch size 2, FP32,
    # peak 10.1 GiB on a Quadro RTX 6000, including validation and prediction. It
    # covers the layers between the convolutions counted as activations.
    BYTES_PER_ACTIVATION = 17.2

    # BF16 roughly halves activation memory. This is inferred, not measured: it's
    # about what nnU-Net's half-precision reference (a 128^3 patch at batch size 2
    # in 8 GB) implies against the FP32 run above.
    BF16_ACTIVATION_FACTOR = 0.55

    # Share of GPU memory to plan for, leaving room for the estimate's error.
    MEMORY_FRACTION = 0.9

    # Assumed memory when no GPU is visible, as when training on a CPU.
    DEFAULT_GPU_MEMORY_GB = 16

    # Edge of the cube whose volume sets the patch that the search shrinks from.
    INITIAL_EDGE = 512

    # The batch size never goes below this, and one global batch sees at most this
    # share of the dataset's voxels (both from nnU-Net).
    MIN_BATCH_SIZE = 2
    MAX_DATASET_FRACTION = 0.05

    # How many times to double a patch too small for a network to build.
    MAX_GROW_STEPS = 6

    # Most patches to predict at once, in validation and inference.
    MAX_INFERENCE_BATCH_SIZE = 4

    # Inference keeps no gradients and frees most activations as it goes, so each
    # patch takes less than in training. This is a cautious guess, not a
    # measurement; running out of memory lowers the batch during inference anyway.
    INFERENCE_BYTES_PER_ACTIVATION = 6.0

    # Warn at training if the GPU has at least this many times the planned memory.
    UNDERUSED_RATIO = 1.5

    # Smallest patch along any axis. With a 3 x 3 x 3 kernel, at least 5 voxels
    # give the kernel genuine (non-padded) context at every position.
    MIN_AXIS_SIZE = 5
