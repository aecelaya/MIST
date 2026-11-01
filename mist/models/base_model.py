"""Base class for all MIST model wrappers."""

from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn


class MISTModel(nn.Module, ABC):
    """Abstract base class for all MIST model wrappers.

    All MIST model wrappers must inherit from this class and implement
    get_encoder_state_dict(). This interface is the contract between MIST
    training and encoder-based transfer learning (e.g., MISFIT pretraining).
    """

    @abstractmethod
    def get_encoder_state_dict(self) -> OrderedDict:
        """Return the encoder weights as a state dict.

        Keys must match those returned by self.state_dict() exactly. Only
        encoder weights are included — decoder, output heads, and deep
        supervision heads are excluded.

        Returns:
            OrderedDict of encoder parameter name → tensor.
        """

    def adapt_pretrained_state_dict(self, state_dict: dict) -> dict:
        """Adapt a pretrained state dict to this model before encoder loading.

        Override when pretrained tensors can be transformed to fit this model,
        for example resizing a learned position embedding pretrained at a
        different patch size. The default returns the state dict unchanged.

        Args:
            state_dict: Source state dict, already unwrapped and stripped of
                any DDP prefix.

        Returns:
            The (possibly modified) state dict.
        """
        return state_dict
