"""Configuration for pytest."""
import sys
import types
from unittest import mock


class DummyPipeline:
    def __init__(self, *args, **kwargs):
        pass


# Create the fake dali structure.
fake_dali = types.SimpleNamespace(
    fn=mock.MagicMock(),
    types=mock.MagicMock(),
    tensors=mock.MagicMock(),
    math=mock.MagicMock(),
    ops=mock.MagicMock(),
    pipeline=types.SimpleNamespace(Pipeline=DummyPipeline),
    plugin=types.SimpleNamespace(pytorch=mock.MagicMock())
)

# Register modules.
sys.modules["nvidia"] = types.SimpleNamespace(dali=fake_dali)
sys.modules["nvidia.dali"] = fake_dali
sys.modules["nvidia.dali.fn"] = fake_dali.fn
sys.modules["nvidia.dali.types"] = fake_dali.types
sys.modules["nvidia.dali.tensors"] = fake_dali.tensors
sys.modules["nvidia.dali.math"] = fake_dali.math
sys.modules["nvidia.dali.ops"] = fake_dali.ops
sys.modules["nvidia.dali.pipeline"] = fake_dali.pipeline
sys.modules["nvidia.dali.plugin"] = fake_dali.plugin
sys.modules["nvidia.dali.plugin.pytorch"] = fake_dali.plugin.pytorch
