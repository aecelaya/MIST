"""Sliding window inferer implementation using MONAI."""
from typing import Callable, Tuple, Union, Optional
import torch
import monai

# MIST imports.
from mist.inference.inference_constants import InferenceConstants as ic
from mist.inference.inferers.base import AbstractInferer
from mist.inference.inferers.inferer_registry import register_inferer
from mist.inference.inference_utils import get_default_device


@register_inferer("sliding_window")
class SlidingWindowInferer(AbstractInferer):
    """Sliding window inference using MONAI's built-in API."""
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        patch_overlap: float=0.5,
        patch_blend_mode: str="gaussian",
        device: Optional[Union[str, torch.device]]=None,
    ):
        """Initialize the sliding window inferer.

        This is a 3D sliding window inferer that uses MONAI's built-in API for
        sliding window inference. It allows for flexible patch sizes, overlap,
        and blending modes. The inferer is designed to work with 3D images and
        models.

        Args:
            patch_size: Tuple representing the size of each patch (D, H, W).
            patch_overlap: Fractional overlap between patches must be in the
                range [0, 1). Default is 0.5.
            patch_blend_mode: Blending mode to use (i.e., "gaussian" or
                "constant"). Default is "gaussian".
            device: Device to run inference on. If None, defaults to "cuda" if
                available or "cpu" otherwise.
        """
        # Initialize the base class.
        super().__init__()

        # Validate input parameters.
        if len(patch_size) != 3:
            raise ValueError(
                f"patch_size must be a tuple of length 3, got: {patch_size}"
            )
        if not all(isinstance(dim, int) and dim > 0 for dim in patch_size):
            raise ValueError(
                "All patch dimensions must be positive integers, got: "
                f"{patch_size}"
            )
        if not 0 <= patch_overlap < 1:
            raise ValueError(
                "patch_overlap must be in the range [0, 1), got: "
                f"{patch_overlap}"
            )
        if patch_blend_mode not in ic.SLIDING_WINDOW_PATCH_BLEND_MODES:
            raise ValueError(
                f"Unsupported blend mode: '{patch_blend_mode}'. Supported "
                f"modes: {sorted(ic.SLIDING_WINDOW_PATCH_BLEND_MODES)}"
            )

        # Set the patch size, overlap, and blend mode.
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_blend_mode = patch_blend_mode

        # Set the device for inference.
        self.device = device or get_default_device()

    def infer(
        self,
        image: torch.Tensor,
        model: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply sliding window inference on an input image.

        Args:
            image: Input image tensor of shape (1, C, D, H, W).
            model: PyTorch model used for prediction.

        Returns:
            prediction: Softmax prediction tensor of shape (1, C, D, H, W).
        """
        # Run MONAI's sliding window inference.
        # MONAI stubs don't fully annotate sliding_window_inference.
        prediction = monai.inferers.sliding_window_inference(  # type: ignore[attr-defined]
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=ic.SLIDING_WINDOW_BATCH_SIZE,
            predictor=model,
            overlap=self.patch_overlap,
            mode=self.patch_blend_mode,
            device=self.device,
        )

        # Apply softmax to the prediction tensor.
        prediction = torch.nn.functional.softmax(
            prediction, dim=ic.SOFTMAX_AXIS
        )
        return prediction
