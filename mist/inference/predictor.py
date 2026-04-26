"""Predictor class to chain together inference, TTA, and ensembling."""
from collections.abc import Callable

import torch

from mist.inference.inferers.base import AbstractInferer
from mist.inference.ensemblers.base import AbstractEnsembler
from mist.inference.tta.transforms import AbstractTransform
from mist.inference.inference_utils import get_default_device


class Predictor:
    """Performs inference with test time augmentation and ensembling.

    This class orchestrates the entire inference pipeline, including applying
    TTA transforms, running inference with multiple models, and ensembling the
    predictions.

    Attributes:
        models: List of PyTorch models to use for inference.
        inferer: An instance of a subclass of AbstractInferer to run inference.
        ensembler: An instance of a subclass of AbstractEnsembler to aggregate
            outputs.
        tta_transforms: List of TTA transforms to apply. Predefined lists of
            transforms are available as TTA strategies in the
            mist.inference.tta.strategies module.
        device: Torch device to use for inference. If None, defaults to CUDA if
            available, otherwise CPU.
    """

    def __init__(
        self,
        models: list[Callable[[torch.Tensor], torch.Tensor]],
        inferer: AbstractInferer,
        ensembler: AbstractEnsembler,
        tta_transforms: list[AbstractTransform],
        device: str | torch.device | None = None,
    ):
        """Initialize the predictor.

        Args:
            models: List of PyTorch models.
            inferer: A subclass of AbstractInferer.
            ensembler: A subclass of AbstractEnsembler to aggregate outputs.
            tta_transforms: List of TTA transforms. Predefined lists of
                transforms are available as tta strategies in the
                mist.inference.tta.strategies module.
            device: Torch device to use for inference.
        """
        self.models = models
        self.inferer = inferer
        self.ensembler = ensembler
        self.tta_transforms = tta_transforms
        self.device = device or get_default_device()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Call the predictor like a function."""
        return self.predict(image)

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run prediction with TTA and ensembling.

        Args:
            image: Input tensor of shape (1, C, D, H, W).

        Returns:
            Combined prediction tensor of shape (1, C, D, H, W).
        """
        image = image.to(self.device)
        all_predictions = []

        for model in self.models:
            model_predictions = []

            # Apply TTA if defined.
            for transform in self.tta_transforms:
                augmented_image = transform(image)
                prediction = self.inferer(augmented_image, model)
                restored = transform.inverse(prediction)
                model_predictions.append(restored)

            # Ensembling across TTA variants for this model.
            combined_prediction = self.ensembler(model_predictions)
            all_predictions.append(combined_prediction)

        # Final ensemble across all models.
        return self.ensembler(all_predictions)
