"""Tests for the Predictor class."""
import pytest
import torch
from mist.inference.predictor import Predictor


class DummyInferer:
    """Dummy inferer that returns the model's output."""

    def __call__(self, image: torch.Tensor, model):
        return model(image)


class DummyEnsembler:
    """Dummy ensembler that averages predictions."""

    def __call__(self, predictions):
        return sum(predictions) / len(predictions)


class DummyTransform:
    """Dummy TTA transform that adds and subtracts 1."""

    def __init__(self):
        self.name = "dummy"

    def __call__(self, image: torch.Tensor):
        return image + 1

    def inverse(self, prediction: torch.Tensor):
        return prediction - 1


class IdentityTransform:
    """TTA transform that passes images and predictions through unchanged."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_predictor(**overrides):
    """Build a Predictor with sensible defaults, allowing per-test overrides."""
    defaults = dict(
        models=[lambda x: x],
        inferer=DummyInferer(),
        ensembler=DummyEnsembler(),
        tta_transforms=[IdentityTransform()],
        device="cpu",
    )
    defaults.update(overrides)
    return Predictor(**defaults)


# =====================
# Existing / basic tests
# =====================

def test_predictor_basic_flow():
    """One model, one transform: result equals transform ∘ model ∘ inverse."""
    # image=1 → +1 → *2 → -1 = 2 (forward:+1, model:*2, inverse:-1 → net +2)
    image = torch.ones(1, 1, 4, 4, 4)
    predictor = _make_predictor(
        models=[lambda x: x * 2],
        tta_transforms=[DummyTransform()],
    )
    prediction = predictor(image)
    assert torch.allclose(prediction, image + 2)


def test_predictor_multiple_models_and_tta():
    """Two models and two transforms are all ensembled correctly."""
    image = torch.ones(1, 1, 4, 4, 4)

    class AddOne:
        def __call__(self, x): return x + 1
        def inverse(self, x): return x - 1

    predictor = _make_predictor(
        models=[lambda x: x + 2, lambda x: x * 3],
        tta_transforms=[IdentityTransform(), AddOne()],
    )
    # model1 on identity: 1+2=3; model1 on AddOne: (1+1+2)-1=3 → ensemble 3
    # model2 on identity: 1*3=3; model2 on AddOne: (1+1)*3-1=5 → ensemble 4
    # final ensemble: (3+4)/2 = 3.5
    prediction = predictor(image)
    assert torch.allclose(prediction, torch.full_like(image, 3.5))


# =====================
# New behavioral tests
# =====================

def test_predictor_moves_image_to_device():
    """Predictor calls .to(device) on the input before inference."""
    received_devices = []

    class DeviceTrackingInferer:
        def __call__(self, image, model):
            received_devices.append(str(image.device))
            return model(image)

    predictor = _make_predictor(
        inferer=DeviceTrackingInferer(),
        device="cpu",
    )
    predictor(torch.ones(1, 1, 4, 4, 4))
    assert received_devices == ["cpu"]


def test_predictor_transform_and_inverse_both_called():
    """Each TTA transform's forward and inverse are called once per model."""
    forward_calls, inverse_calls = [], []

    class TrackingTransform:
        def __call__(self, x):
            forward_calls.append(1)
            return x

        def inverse(self, x):
            inverse_calls.append(1)
            return x

    predictor = _make_predictor(tta_transforms=[TrackingTransform()])
    predictor(torch.ones(1, 1, 4, 4, 4))

    assert len(forward_calls) == 1
    assert len(inverse_calls) == 1


def test_predictor_inferer_receives_transformed_image():
    """The inferer sees the augmented image, not the original."""
    received = []

    class RecordingInferer:
        def __call__(self, image, model):
            received.append(image.clone())
            return model(image)

    class AddTwo:
        def __call__(self, x): return x + 2
        def inverse(self, x): return x - 2

    image = torch.zeros(1, 1, 4, 4, 4)
    predictor = _make_predictor(
        inferer=RecordingInferer(),
        tta_transforms=[AddTwo()],
    )
    predictor(image)

    assert torch.allclose(received[0], torch.full_like(image, 2.0))


def test_predictor_multiple_transforms_calls_inferer_once_each():
    """With N transforms, the inferer is called N times per model."""
    call_count = [0]

    class CountingInferer:
        def __call__(self, image, model):
            call_count[0] += 1
            return model(image)

    n_transforms = 3
    predictor = _make_predictor(
        inferer=CountingInferer(),
        tta_transforms=[IdentityTransform() for _ in range(n_transforms)],
    )
    predictor(torch.ones(1, 1, 4, 4, 4))

    assert call_count[0] == n_transforms


def test_predictor_multiple_models_all_contribute_to_ensemble():
    """With M models, the ensembler is called M+1 times (once per model + final)."""
    ensembler_calls = [0]

    class CountingEnsembler:
        def __call__(self, predictions):
            ensembler_calls[0] += 1
            return sum(predictions) / len(predictions)

    n_models = 3
    predictor = _make_predictor(
        models=[lambda x: x for _ in range(n_models)],
        ensembler=CountingEnsembler(),
    )
    predictor(torch.ones(1, 1, 4, 4, 4))

    # M calls (one per model's TTA ensemble) + 1 final ensemble call.
    assert ensembler_calls[0] == n_models + 1


def test_predictor_model_exception_propagates():
    """A RuntimeError raised inside a model is not swallowed."""
    def bad_model(_):
        raise RuntimeError("model exploded")

    predictor = _make_predictor(models=[bad_model])

    with pytest.raises(RuntimeError, match="model exploded"):
        predictor(torch.ones(1, 1, 4, 4, 4))


def test_predictor_default_device_is_cpu_without_cuda(monkeypatch):
    """When no device is given and CUDA is unavailable, device defaults to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    predictor = Predictor(
        models=[lambda x: x],
        inferer=DummyInferer(),
        ensembler=DummyEnsembler(),
        tta_transforms=[IdentityTransform()],
    )
    assert predictor.device == "cpu"
