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

    # Fitted to steady training memory on four GPUs (Tesla T4, L4, A100, RTX PRO
    # 6000): U-Net and MedNeXt networks take at most about 10 bytes in FP32. This
    # adds a margin for the first steps, which can take more. The calibration runs
    # used 5 classes. Known gap: primusv2 networks in FP32 grow faster than this on
    # the A100, L4, and RTX PRO 6000, though not on the T4, and can run out of memory.
    BYTES_PER_ACTIVATION = 12.0

    # BF16 activation memory measured 0.50 to 0.60 of FP32's on the same networks.
    BF16_ACTIVATION_FACTOR = 0.55

    # Networks whose memory differs from the values above, by name prefix: (bytes
    # per activation in FP32, BF16 factor). These come from swinunetr-small and
    # primus-medium only. Attention and other layers between the counted activations
    # take more, and much of Swin's memory doesn't shrink with BF16.
    ACTIVATION_MEMORY_BY_PREFIX = {
        "swinunetr": (17.0, 0.9),
        "primus-": (12.0, 0.7),
    }

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

    # FP32 replaces BF16 only if the speed test finds it this many times faster,
    # since the test leaves out data loading and a few steps are noisy.
    FP32_MIN_SPEEDUP = 1.10

    # Training steps the speed test runs for each precision: some to warm up, then
    # some to time, stopping early on a slow network once it has the minimum.
    SPEED_TEST_WARMUP_STEPS = 3
    SPEED_TEST_STEPS = 15
    SPEED_TEST_MIN_STEPS = 5
    SPEED_TEST_MAX_SECONDS = 60

    # Give up on a speed test that takes this long, and keep BF16.
    SPEED_TEST_TIMEOUT_SECONDS = 600

    # Smallest patch along any axis. With a 3 x 3 x 3 kernel, at least 5 voxels
    # give the kernel genuine (non-padded) context at every position.
    MIN_AXIS_SIZE = 5
