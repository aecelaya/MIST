"""Dataclass containing constants for MIST inference modules."""
from typing import FrozenSet
import dataclasses


@dataclasses.dataclass(frozen=True)
class InferenceConstants:
    """Dataclass containing constants for MIST inference modules."""
    # Axis to apply softmax to for the model output.
    SOFTMAX_AXIS: int = 1

    # Axis for applying argmax to for the model output.
    ARGMAX_AXIS: int = 1

    # Batch axis for the model output.
    BATCH_AXIS: int = 0

    # Batch size for sliding window inference.
    SLIDING_WINDOW_BATCH_SIZE: int = 1

    # Ignored columns in the patient data CSV file.
    PATIENT_DF_IGNORED_COLUMNS: FrozenSet[str] = frozenset(
        {"id", "fold", "mask"}
    )

    # Numpy to PyTorch transpose axes.
    NUMPY_TO_TORCH_TRANSPOSE_AXES: tuple[int, ...] = (3, 0, 1, 2)

    # Numpy to PyTorch expand dimensions axes. This adds a batch dimension to
    # the input tensor.
    NUMPY_TO_TORCH_EXPAND_DIMS_AXES: int = 0

    # Valid patch blend modes for sliding window inference.
    SLIDING_WINDOW_PATCH_BLEND_MODES: FrozenSet[str] = frozenset(
        {"gaussian", "constant"}
    )
