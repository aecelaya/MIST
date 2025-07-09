# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the SegmentationLoss base class."""
import pytest
import torch

# MIST imports.
from mist.loss_functions.base import SegmentationLoss


class DummyLoss(SegmentationLoss):
    """Minimal subclass to enable testing of preprocess."""
    def forward(self, y_true, y_pred, *args, **kwargs):
        return torch.tensor(0.0)


def make_inputs(n_classes=3, exclude_background=False):
    """Generate synthetic data for loss input testing."""
    y_true = torch.randint(0, n_classes, size=(1, 1, 4, 4, 4)) # Integer labels.
    y_pred = torch.randn((1, n_classes, 4, 4, 4)) # Logits.
    return y_true, y_pred


def test_preprocess_with_background():
    """Test preprocessing with background included."""
    loss = DummyLoss(exclude_background=False)
    y_true, y_pred = make_inputs()
    y_true_onehot, y_pred_softmax = loss.preprocess(y_true, y_pred)

    assert y_true_onehot.shape == y_pred_softmax.shape
    assert y_true_onehot.shape[1] == y_pred.shape[1]  # num_classes
    assert torch.allclose(
        y_pred_softmax.sum(dim=1), torch.ones_like(y_pred_softmax[:, 0]),
        atol=1e-4
    )


def test_preprocess_excludes_background():
    """Test that background channel is removed when specified."""
    loss = DummyLoss(exclude_background=True)
    y_true, y_pred = make_inputs()
    y_true_onehot, y_pred_softmax = loss.preprocess(y_true, y_pred)

    assert y_true_onehot.shape[1] == y_pred.shape[1] - 1
    assert y_pred_softmax.shape[1] == y_pred.shape[1] - 1


def test_preprocess_invalid_inputs_raise():
    """Test that invalid input shapes raise errors."""
    loss = DummyLoss()
    y_pred = torch.randn((1, 3, 4, 4, 4))

    bad_shapes = [
        torch.randn((1, 2, 4, 4)),      # Not 5D.
        torch.randn((1, 2, 4, 4, 4)),   # Wrong channel count.
    ]
    for y_true in bad_shapes:
        with pytest.raises(ValueError):
            _ = loss.preprocess(y_true, y_pred)
