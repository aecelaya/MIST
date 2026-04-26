"""Tests for mist.preprocessing.preprocess."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import pytest

# MIST imports.
from mist.preprocessing import preprocess as pp
from mist.utils import console as console_mod


class _DummyAntsImage:
    """Minimal ANTs-like image used in tests."""

    def __init__(
        self,
        arr: np.ndarray | None = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, ...] = (1.0,) * 9,
    ) -> None:
        self._arr = (
            np.zeros((2, 2, 2), dtype=np.float32)
            if arr is None
            else np.asarray(arr)
        )
        self._spacing = spacing
        self._origin = origin
        self._direction = direction

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return image shape."""
        return tuple(self._arr.shape)

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Return image spacing."""
        return self._spacing

    @property
    def origin(self) -> tuple[float, float, float]:
        """Return image origin."""
        return self._origin

    @property
    def direction(self) -> tuple[float, ...]:
        """Return image direction."""
        return self._direction

    def set_spacing(self, s: tuple[float, float, float]) -> None:
        """Set image spacing."""
        self._spacing = s

    def set_origin(self, o: tuple[float, float, float]) -> None:
        """Set image origin."""
        self._origin = o

    def set_direction(self, d: tuple[float, ...]) -> None:
        """Set image direction."""
        self._direction = d

    def numpy(self) -> np.ndarray:
        """Return image as a NumPy array."""
        return np.asarray(self._arr)


class _DummySitkImage:
    """Minimal SimpleITK-like image used in tests.

    SimpleITK-like API (CamelCase preserved intentionally).
    """

    def __init__(
        self,
        size=(2, 2, 2),
        spacing=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0,) * 9,
        pixel_id=1,
    ):
        self._size = size
        self._spacing = spacing
        self._origin = origin
        self._direction = direction
        self._pixel_id = pixel_id

    def GetSize(self):
        """Return the size of the image."""
        return self._size

    def GetSpacing(self):
        """Return the spacing of the image."""
        return self._spacing

    def GetOrigin(self):
        """Return the origin of the image."""
        return self._origin

    def GetDirection(self):
        """Return the direction of the image."""
        return self._direction

    def GetDepth(self):
        """Return the depth of the image."""
        return self._size[0]

    def GetWidth(self):
        """Return the width of the image."""
        return self._size[2]

    def GetHeight(self):
        """Return the height of the image."""
        return self._size[1]

    def GetPixelID(self):
        """Return the pixel ID of the image."""
        return self._pixel_id

    def SetSpacing(self, s):
        """Set the spacing of the image."""
        self._spacing = tuple(s)

    def SetOrigin(self, o):
        """Set the origin of the image."""
        self._origin = tuple(o)

    def SetDirection(self, d):
        """Set the direction of the image."""
        self._direction = tuple(d)

    def __lt__(self, other):
        """Less than comparison for sorting."""
        return _DummySitkImage(
            self._size,
            self._spacing,
            self._origin,
            self._direction,
            self._pixel_id
        )

    def __gt__(self, other):
        """Greater than comparison for sorting."""
        return _DummySitkImage(
            self._size,
            self._spacing,
            self._origin,
            self._direction,
            self._pixel_id
        )

    def __imul__(self, other):
        """In-place multiplication for scaling."""
        return self

    def __mul__(self, other):
        """Multiplication operator for scaling."""
        return _DummySitkImage(
            self._size,
            self._spacing,
            self._origin,
            self._direction,
            self._pixel_id
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        """True division operator for scaling."""
        return _DummySitkImage(
            self._size,
            self._spacing,
            self._origin,
            self._direction,
            self._pixel_id
        )

    def __sub__(self, other):
        """Subtraction operator for images."""
        return _DummySitkImage(
            self._size,
            self._spacing,
            self._origin,
            self._direction,
            self._pixel_id
        )


class _DummyDTM:
    """SITK-like DTM image that records divisors for zero-guard tests."""

    def __init__(self, tag, divlog):
        self.tag = tag  # 'dtm', 'int', 'ext', etc.
        self._divlog = divlog
        self._spacing = (1.0, 1.0, 1.0)
        self._origin = (0.0, 0.0, 0.0)
        self._direction = (1.0,) * 9

    def __lt__(self, other):
        """Less than comparison for sorting."""
        return _DummyDTM("int_mask", self._divlog)

    def __gt__(self, other):
        """Greater than comparison for sorting."""
        return _DummyDTM("ext_mask", self._divlog)

    def __imul__(self, other):
        """In-place multiplication for scaling."""
        if "int_mask" in self.tag:
            self.tag = "int"
        elif "ext_mask" in self.tag:
            self.tag = "ext"
        return self

    def __truediv__(self, other):
        """True division operator for scaling."""
        self._divlog.append((self.tag, other))
        return _DummyDTM(f"{self.tag}_div", self._divlog)

    def __sub__(self, other):
        """Subtraction operator for images."""
        return _DummyDTM("result", self._divlog)

    def SetSpacing(self, s):
        """Set the spacing of the image."""
        self._spacing = s

    def SetOrigin(self, o):
        """Set the origin of the image."""
        self._origin = o

    def SetDirection(self, d):
        """Set the direction of the image."""
        self._direction = d


class _PB:
    """Very small progress-bar stub used by tests."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def track(self, iterable, **kwargs):
        """Track progress of an iterable."""
        return iterable


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a CSV to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Return a minimal base config for preprocessing."""
    return {
        "dataset_info": {
            "modality": "ct",
            "labels": [0, 1],
            "images": ["image"],
        },
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": True,
            "compute_dtms": False,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }


def test_window_and_normalize_ct_uses_config_values():
    """CT path uses configured window and z-score parameters."""
    img = np.array([-200.0, -100.0, 0.0, 50.0, 200.0], dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "ct"},
        "preprocessing": {
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 10.0,
            },
        },
    }
    out = pp.window_and_normalize(img, cfg)
    expected = np.array([-10.0, -10.0, 0.0, 5.0, 10.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32


def test_window_and_normalize_nonct_with_nonzero_mask():
    """Non-CT path with nonzero-mask normalization behavior."""
    img = np.array([0.0, 0.0, 1.0, 3.0], dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": True},
    }
    out = pp.window_and_normalize(img, cfg)

    mask = np.array([0, 0, 1, 1], dtype=np.float32)
    clip_low = np.percentile(img[mask > 0], 0.5)
    clip_high = np.percentile(img[mask > 0], 99.5)
    mean = img[mask > 0].mean()
    std = img[mask > 0].std()
    expected = np.clip(img, clip_low, clip_high) * mask
    expected = ((expected - mean) / std) * mask

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32


def test_window_and_normalize_nonct_full_image():
    """Non-CT path with full-image normalization."""
    img = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": False},
    }
    out = pp.window_and_normalize(img, cfg)
    assert out.dtype == np.float32
    assert np.isclose(out.mean(), 0.0, atol=1e-5)


def test_window_and_normalize_nonzero_mask_all_zeros_falls_back_to_full_image():
    """All-zero image with nonzero-mask flag falls back to full-image normalization."""
    img = np.zeros(5, dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": True},
    }
    out = pp.window_and_normalize(img, cfg)
    assert out.dtype == np.float32
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_window_and_normalize_zero_std_returns_zeros():
    """Constant (non-zero) image with std=0 returns all-zeros without warning."""
    img = np.full(5, 3.0, dtype=np.float32)
    cfg = {
        "dataset_info": {"modality": "mri"},
        "preprocessing": {"normalize_with_nonzero_mask": False},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes an error
        out = pp.window_and_normalize(img, cfg)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_resample_image_calls_utils_and_resample(monkeypatch):
    """Resample path converts to/from SITK and calls sitk.Resample."""
    dummy_sitk = _DummySitkImage(size=(2, 2, 2), spacing=(2.0, 2.0, 2.0))
    out_ants = _DummyAntsImage(np.ones((4, 4, 4), dtype=np.float32))

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "ants_to_sitk",
        lambda _img: dummy_sitk,
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_to_ants",
        lambda _s: out_ants,
        raising=True,
    )
    monkeypatch.setattr(
        pp.analyzer_utils,
        "get_resampled_image_dimensions",
        lambda size, sp, tgt: (4, 4, 4),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _s: {"is_anisotropic": False},
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "aniso_intermediate_resample",
        lambda *_a, **_k: pytest.fail("Unexpected."),
    )

    monkeypatch.setattr(
        pp.sitk, "Resample", lambda *_a, **_k: object(), raising=True
    )

    img_in = _DummyAntsImage(np.zeros((2, 2, 2), dtype=np.float32))
    out = pp.resample_image(img_in, target_spacing=(1.0, 1.0, 1.0))
    assert isinstance(out, _DummyAntsImage)
    np.testing.assert_array_equal(out.numpy(), out_ants.numpy())


def test_resample_image_aniso_axis_type_error(monkeypatch):
    """Anisotropic path with non-int axis raises ValueError."""
    dummy_sitk = _DummySitkImage()
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "ants_to_sitk",
        lambda _img: dummy_sitk,
        raising=True,
    )
    monkeypatch.setattr(
        pp.analyzer_utils,
        "get_resampled_image_dimensions",
        lambda size, sp, tgt: (4, 4, 4),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _s: {"is_anisotropic": True, "low_resolution_axis": "z"},
        raising=True,
    )

    with pytest.raises(ValueError, match="must be an integer"):
        pp.resample_image(_DummyAntsImage(), (1.0, 1.0, 1.0))


def test_resample_image_aniso_intermediate_called(monkeypatch):
    """Anisotropic path with int axis calls intermediate resampler."""
    dummy_before = _DummySitkImage()
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "ants_to_sitk",
        lambda _x: dummy_before,
        raising=True,
    )
    monkeypatch.setattr(
        pp.analyzer_utils,
        "get_resampled_image_dimensions",
        lambda size, spacing, tgt: (10, 12, 14),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _img: {"is_anisotropic": True, "low_resolution_axis": 2},
        raising=True,
    )

    def _aniso_intermediate(img, new_size, tgt_spacing, low_axis):
        seen["args"] = (img, new_size, tgt_spacing, low_axis)
        return dummy_before

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "aniso_intermediate_resample",
        _aniso_intermediate,
        raising=True,
    )
    monkeypatch.setattr(pp.sitk, "Transform", lambda: object(), raising=True)
    monkeypatch.setattr(
        pp.sitk, "Resample", lambda *_a, **_k: object(), raising=True
    )

    final_ants = object()
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_to_ants",
        lambda _img: final_ants,
        raising=True,
    )

    out = pp.resample_image(img_ants=object(), target_spacing=(1.0, 1.0, 1.0))

    assert seen["args"][0] is dummy_before
    assert seen["args"][1] == (10, 12, 14)
    assert seen["args"][2] == (1.0, 1.0, 1.0)
    assert seen["args"][3] == 2
    assert out is final_ants


def test_resample_mask_happy_path(monkeypatch):
    """Resample mask happy path with label series and join."""
    labels = [0, 1, 2]
    onehots = [_DummySitkImage(size=(3, 3, 3)) for _ in labels]
    src_ants = _DummyAntsImage(
        np.zeros((2, 2, 2), dtype=np.float32),
        spacing=(2, 2, 2),
        origin=(1, 2, 3),
        direction=(1,) * 9,
    )

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "make_onehot",
        lambda _m, _lbls: onehots,
        raising=True,
    )
    monkeypatch.setattr(
        pp.analyzer_utils,
        "get_resampled_image_dimensions",
        lambda size, sp, tgt: (3, 3, 3),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _s: {"is_anisotropic": False},
        raising=True,
    )
    monkeypatch.setattr(
        pp.sitk, "Resample", lambda *a, **k: a[0], raising=True
    )
    monkeypatch.setattr(
        pp.sitk, "JoinSeries", lambda seq: object(), raising=True
    )

    def _sitk_to_ants(_):
        arr = np.zeros((3, 3, 3, len(labels)), dtype=np.float32)
        return _DummyAntsImage(arr)

    monkeypatch.setattr(
        pp.preprocessing_utils, "sitk_to_ants", _sitk_to_ants, raising=True
    )
    monkeypatch.setattr(
        pp.ants,
        "from_numpy",
        lambda data: _DummyAntsImage(np.asarray(data)),
        raising=True,
    )

    out = pp.resample_mask(
        src_ants, labels=labels, target_spacing=(1.0, 1.0, 1.0)
    )
    assert isinstance(out, _DummyAntsImage)
    assert out.spacing == (1.0, 1.0, 1.0)
    assert out.origin == src_ants.origin
    assert out.direction == src_ants.direction
    assert out.numpy().shape == (3, 3, 3)
    assert out.numpy().dtype == np.float32


def test_resample_mask_aniso_axis_not_int_raises(monkeypatch):
    """Anisotropic resample with non-int axis raises ValueError."""
    masks = [_DummySitkImage()]
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "make_onehot",
        lambda _m, _l: masks,
        raising=True,
    )
    monkeypatch.setattr(
        pp.analyzer_utils,
        "get_resampled_image_dimensions",
        lambda size, spacing, tgt: (8, 8, 8),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _img: {"is_anisotropic": True, "low_resolution_axis": "z"},
        raising=True,
    )

    with pytest.raises(
        ValueError, match="low resolution axis must be an integer"
    ):
        pp.resample_mask(
            mask_ants=SimpleNamespace(origin=(1, 2, 3), direction=(1.0,) * 9),
            labels=[0, 1],
            target_spacing=(1.0, 1.0, 1.0),
        )


def test_resample_mask_aniso_intermediate_called_for_each_label(monkeypatch):
    """Anisotropic mask calls intermediate resample for each label."""
    m0, m1 = _DummySitkImage(), _DummySitkImage()
    masks = [m0, m1]
    labels = [0, 1]
    new_size = (10, 12, 14)
    target_spacing = (1.0, 1.0, 1.0)

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "make_onehot",
        lambda _m, _lbls: masks,
        raising=True,
    )
    monkeypatch.setattr(
        pp.analyzer_utils,
        "get_resampled_image_dimensions",
        lambda size, spacing, tgt: new_size,
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "check_anisotropic",
        lambda _img: {"is_anisotropic": True, "low_resolution_axis": 2},
        raising=True,
    )

    calls: list[tuple[Any, Any, Any, Any]] = []

    def _aniso(img, ns, tgt, axis):
        calls.append((img, ns, tgt, axis))
        return img

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "aniso_intermediate_resample",
        _aniso,
        raising=True,
    )
    monkeypatch.setattr(pp.sitk, "Resample", lambda *a, **k: a[0], raising=True)
    monkeypatch.setattr(pp.sitk, "JoinSeries", lambda seq: seq, raising=True)

    def _sitk_to_ants(seq):
        arr = np.stack(
            [
                np.zeros((5, 6, 7), dtype=np.float32),
                np.ones((5, 6, 7), dtype=np.float32),
            ],
            axis=-1,
        )
        return SimpleNamespace(numpy=lambda: arr)

    monkeypatch.setattr(
        pp.preprocessing_utils, "sitk_to_ants", _sitk_to_ants, raising=True
    )

    class _Out:
        """Simple ANTs-like output holder."""

        def __init__(self) -> None:
            self.spacing = None
            self.origin = None
            self.direction = None

        def set_spacing(self, s):
            """Set the spacing of the output image."""
            self.spacing = s

        def set_origin(self, o):
            """Set the origin of the output image."""
            self.origin = o

        def set_direction(self, d):
            """Set the direction of the output image."""
            self.direction = d

    out_img = _Out()
    monkeypatch.setattr(
        pp.ants, "from_numpy", lambda data: out_img, raising=True
    )

    src_mask = SimpleNamespace(origin=(9, 9, 9), direction=(1.0,) * 9)
    result = pp.resample_mask(
        mask_ants=src_mask,
        labels=labels,
        target_spacing=target_spacing,
    )

    assert len(calls) == len(labels)
    assert calls[0] == (m0, new_size, target_spacing, 2)
    assert calls[1] == (m1, new_size, target_spacing, 2)
    assert result is out_img
    assert out_img.spacing == target_spacing
    assert out_img.origin == src_mask.origin
    assert out_img.direction == src_mask.direction


def test_compute_dtm_shapes_and_types(monkeypatch):
    """compute_dtm returns expected shape/dtype with one empty class."""
    labels = [0, 1]
    src_mask = _DummyAntsImage(np.zeros((5, 6, 7), dtype=np.float32))

    onehots = [
        _DummySitkImage(size=(5, 6, 7)), _DummySitkImage(size=(5, 6, 7))
    ]
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "make_onehot",
        lambda _m, _lbls: onehots,
        raising=True,
    )

    sums = iter([1, 0])  # First non-empty, second empty.
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_get_sum",
        lambda _m: next(sums),
        raising=True,
    )
    monkeypatch.setattr(
        pp.sitk,
        "SignedMaurerDistanceMap",
        lambda *_a, **_k: _DummySitkImage(size=(5, 6, 7)),
        raising=True,
    )
    monkeypatch.setattr(pp.sitk, "Cast", lambda img, _t: img, raising=True)
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_get_min_max",
        lambda _img: (-2.0, 3.0),
        raising=True,
    )
    monkeypatch.setattr(
        pp.sitk,
        "GetImageFromArray",
        lambda arr: _DummySitkImage(
            size=(arr.shape[0], arr.shape[1], arr.shape[2])
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pp.sitk, "JoinSeries", lambda seq: object(), raising=True
    )

    def _sitk_to_ants(_x):
        arr = np.zeros((5, 6, 7, len(labels)), dtype=np.float32)
        return _DummyAntsImage(arr)

    monkeypatch.setattr(
        pp.preprocessing_utils, "sitk_to_ants", _sitk_to_ants, raising=True
    )

    out = pp.compute_dtm(mask_ants=src_mask, labels=labels, normalize_dtm=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 6, 7, 2)
    assert out.dtype == np.float32


def test_compute_dtm_zero_guards_combined(monkeypatch):
    """ext_max==0 → 1 and int_min==0 → -1 are guarded correctly."""
    labels = [0]
    masks = [_DummySitkImage()]

    monkeypatch.setattr(
        pp.preprocessing_utils, "make_onehot", lambda *_: masks, raising=True
    )
    monkeypatch.setattr(
        pp.preprocessing_utils, "sitk_get_sum", lambda _m: 1, raising=True
    )
    monkeypatch.setattr(
        pp.sitk, "Cast", lambda img, *_a, **_k: img, raising=True
    )
    monkeypatch.setattr(pp.sitk, "JoinSeries", lambda seq: seq, raising=True)

    class _AntsArray:
        def numpy(self):
            return np.zeros((2, 2, 2, len(labels)), dtype=np.float32)

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_to_ants",
        lambda s: _AntsArray(),
        raising=True,
    )

    # Scenario A: ext_max == 0 → guard to 1.
    recorder: list[tuple[str, float]] = []
    monkeypatch.setattr(
        pp.sitk,
        "SignedMaurerDistanceMap",
        lambda *_a, **_k: _DummyDTM("dtm", recorder),
        raising=False,
    )

    def _minmax_ext_zero(img):
        if isinstance(img, _DummyDTM):
            if img.tag == "int":
                return (-2.0, -0.1)
            if img.tag == "ext":
                return (0.0, 0.0)
        return (0.0, 1.0)

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_get_min_max",
        _minmax_ext_zero,
        raising=True,
    )

    out = pp.compute_dtm(
        mask_ants=SimpleNamespace(), labels=labels, normalize_dtm=True
    )
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    ext_divs = [d for tag, d in recorder if tag == "ext"]
    int_divs = [d for tag, d in recorder if tag == "int"]
    assert ext_divs and ext_divs[0] == 1
    assert int_divs and int_divs[0] == -2.0

    # Scenario B: int_min == 0 -> guard to -1.
    recorder.clear()

    def _minmax_int_zero(img):
        if isinstance(img, _DummyDTM):
            if img.tag == "int":
                return (0.0, 0.0)
            if img.tag == "ext":
                return (0.1, 5.0)
        return (0.0, 1.0)

    monkeypatch.setattr(
        pp.preprocessing_utils,
        "sitk_get_min_max",
        _minmax_int_zero,
        raising=True,
    )

    out = pp.compute_dtm(
        mask_ants=SimpleNamespace(), labels=labels, normalize_dtm=True
    )
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    ext_divs = [d for tag, d in recorder if tag == "ext"]
    int_divs = [d for tag, d in recorder if tag == "int"]
    assert int_divs and int_divs[0] == -1
    assert ext_divs and ext_divs[0] == 5.0


def test_compute_dtm_empty_mask_diagonal_distance(monkeypatch):
    """Empty mask with normalize_dtm=False uses diagonal distance."""
    d, h, w = 2, 2, 2
    expected = np.sqrt(d**2 + w**2 + h**2).astype(np.float32)

    class _ArrayImage:
        """Wrapper returned by sitk.GetImageFromArray."""

        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)

        def SetSpacing(self, *_a, **_k):
            """Set spacing of the image."""
            return None

        def SetOrigin(self, *_a, **_k):
            """Set origin of the image."""
            return None

        def SetDirection(self, *_a, **_k):
            """Set direction of the image."""
            return None

    labels = [0, 1]
    masks = [_DummySitkImage() for _ in labels]
    monkeypatch.setattr(
        pp.preprocessing_utils, "make_onehot", lambda *_: masks, raising=True
    )
    monkeypatch.setattr(
        pp.preprocessing_utils, "sitk_get_sum", lambda _m: 0, raising=True
    )
    monkeypatch.setattr(
        pp.sitk, "GetImageFromArray", lambda arr: _ArrayImage(arr), raising=True
    )
    monkeypatch.setattr(
        pp.sitk, "Cast", lambda img, *_a, **_k: img, raising=True
    )
    monkeypatch.setattr(pp.sitk, "JoinSeries", lambda seq: seq, raising=True)

    def _sitk_to_ants(seq):
        return SimpleNamespace(
            numpy=lambda: np.stack([im._arr for im in seq], axis=-1)
        )

    monkeypatch.setattr(
        pp.preprocessing_utils, "sitk_to_ants", _sitk_to_ants, raising=True
    )

    out = pp.compute_dtm(
        mask_ants=SimpleNamespace(), labels=labels, normalize_dtm=False
    )
    assert out.shape == (d, h, w, len(labels))
    assert out.dtype == np.float32
    assert np.allclose(out[..., 0], expected)
    assert np.allclose(out[..., 1], expected)


def test_preprocess_example_full_flow_no_skip_with_crop_and_dtm(monkeypatch):
    """Full flow: crop, resample, normalize, and compute DTM."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": True,
            "compute_dtms": True,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    img0 = _DummyAntsImage(
        np.ones((2, 2, 2), dtype=np.float32), spacing=(2.0, 2.0, 2.0)
    )
    img1 = _DummyAntsImage(
        2 * np.ones((2, 2, 2), dtype=np.float32), spacing=(2.0, 2.0, 2.0)
    )
    mask_img = _DummyAntsImage(
        np.zeros((2, 2, 2), dtype=np.float32), spacing=(2.0, 2.0, 2.0)
    )

    seq = iter([img0, img1, mask_img])
    monkeypatch.setattr(
        pp.ants, "image_read", lambda _p: next(seq), raising=True
    )
    fg_bbox = {"x0": 0, "x1": 2, "y0": 0, "y1": 2, "z0": 0, "z1": 2}
    monkeypatch.setattr(
        pp.ants, "reorient_image2", lambda im, _ori: im, raising=True
    )
    monkeypatch.setattr(pp.pc, "RAI_ANTS_DIRECTION", (1.0,) * 9, raising=True)

    calls = {"crop_calls": 0}

    def _crop(im, _bb):
        calls["crop_calls"] += 1
        return im

    monkeypatch.setattr(
        pp.preprocessing_utils, "crop_to_fg", _crop, raising=True
    )
    monkeypatch.setattr(
        pp, "resample_image", lambda im, target_spacing: im, raising=True
    )
    monkeypatch.setattr(
        pp, "resample_mask", lambda m, labels, target_spacing: m, raising=True
    )
    monkeypatch.setattr(
        pp, "window_and_normalize", lambda arr, cfg_: arr, raising=True
    )

    def _compute_dtm(_m, labels, normalize_dtm):
        return np.ones((2, 2, 2, len(labels)), dtype=np.float32)

    monkeypatch.setattr(pp, "compute_dtm", _compute_dtm, raising=True)

    out = pp.preprocess_example(
        cfg,
        image_paths_list=["i0", "i1"],
        mask_path="m.nii.gz",
        fg_bbox=fg_bbox,
    )

    assert out["image"].shape == (2, 2, 2, 2)
    assert out["image"].dtype == np.float32
    assert out["mask"].shape == (2, 2, 2, 1)
    assert out["mask"].dtype == np.uint8
    assert out["dtm"].shape == (2, 2, 2, 2)
    assert out["dtm"].dtype == np.float32
    assert calls["crop_calls"] == 3  # Two images + one mask.


def test_preprocess_example_skip_true_no_resample_no_normalize(monkeypatch):
    """skip=True is a pure pass-through: no reorient, resample, or normalize."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "mri"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": True,
            "crop_to_foreground": False,
            "compute_dtms": False,
            "normalize_dtms": False,
            "normalize_with_nonzero_mask": False,
        },
    }

    img0 = _DummyAntsImage(
        np.ones((2, 2, 2), dtype=np.float32), spacing=(1.5, 1.5, 1.5)
    )
    mask_img = _DummyAntsImage(np.zeros((2, 2, 2), dtype=np.float32))

    seq = iter([img0, mask_img])
    monkeypatch.setattr(
        pp.ants, "image_read", lambda _p: next(seq), raising=True
    )
    monkeypatch.setattr(
        pp.ants,
        "reorient_image2",
        lambda *_a, **_k: pytest.fail("reorient_image2 must not be called when skip=True."),
        raising=True,
    )
    monkeypatch.setattr(
        pp, "resample_image", lambda *_a, **_k: pytest.fail("resample_image must not be called when skip=True.")
    )
    monkeypatch.setattr(
        pp, "window_and_normalize", lambda *_a, **_k: pytest.fail("window_and_normalize must not be called when skip=True.")
    )

    out = pp.preprocess_example(
        cfg, image_paths_list=["i0"], mask_path="m.nii.gz", fg_bbox=None
    )
    assert out["image"].shape == (2, 2, 2, 1)
    assert out["mask"].shape == (2, 2, 2, 1)
    assert out["dtm"] is None
    assert out["fg_bbox"] is None


def test_preprocess_example_crop_requires_bbox_error(monkeypatch):
    """skip=False + crop=True + get_fg_mask_bbox returns None raises ValueError."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1, 1, 1),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": True,
            "compute_dtms": False,
            "normalize_dtms": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    monkeypatch.setattr(
        pp.ants, "image_read", lambda _p: _DummyAntsImage(), raising=True
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "get_fg_mask_bbox",
        lambda _im: None,
        raising=True,
    )

    with pytest.raises(ValueError, match="Foreground bounding box is required"):
        pp.preprocess_example(
            cfg, image_paths_list=["img.nii.gz"], mask_path=None, fg_bbox=None
        )


def test_preprocess_example_skip_true_crop_flag_ignored(monkeypatch):
    """skip=True ignores crop_to_foreground: no crop, no reorient, fg_bbox=None."""
    cfg = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": True,
            "crop_to_foreground": True,
            "compute_dtms": False,
            "normalize_dtms": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100,
                "window_max": 100,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    monkeypatch.setattr(
        pp.ants, "image_read", lambda _p: _DummyAntsImage(), raising=True
    )
    monkeypatch.setattr(
        pp.ants,
        "reorient_image2",
        lambda *_a, **_k: pytest.fail("reorient_image2 must not be called when skip=True."),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "get_fg_mask_bbox",
        lambda *_a, **_k: pytest.fail("get_fg_mask_bbox must not be called when skip=True."),
        raising=True,
    )
    monkeypatch.setattr(
        pp.preprocessing_utils,
        "crop_to_fg",
        lambda *_a, **_k: pytest.fail("crop_to_fg must not be called when skip=True."),
        raising=True,
    )

    out = pp.preprocess_example(
        cfg, image_paths_list=["img.nii.gz"], mask_path=None, fg_bbox=None
    )
    assert out["fg_bbox"] is None
    assert out["image"].shape == (2, 2, 2, 1)


def test_preprocess_example_inference_mode_sets_mask_and_dtm_none(monkeypatch):
    """Inference mode sets mask=None and dtm=None and does not compute DTM."""
    monkeypatch.setattr(
        pp.ants,
        "image_read",
        lambda _p: _DummyAntsImage(np.zeros((4, 5, 6))),
        raising=True,
    )
    monkeypatch.setattr(
        pp.ants,
        "reorient_image2",
        lambda *_a, **_k: pytest.fail("reorient_image2 must not be called when skip=True."),
        raising=True,
    )

    def _no_compute_dtm(*_a, **_k):
        pytest.fail("compute_dtm should not be called in inference mode.")

    monkeypatch.setattr(pp, "compute_dtm", _no_compute_dtm, raising=True)

    config = {
        "dataset_info": {"labels": [0, 1], "modality": "ct"},
        "spatial_config": {
            "target_spacing": (1.0, 1.0, 1.0),
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": True,
            "crop_to_foreground": False,
            "compute_dtms": True,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }

    out = pp.preprocess_example(
        config=config,
        image_paths_list=["/fake/image.nii.gz"],
        mask_path=None,
        fg_bbox=None,
    )

    assert out["mask"] is None
    assert out["dtm"] is None
    assert isinstance(out["image"], np.ndarray)
    assert out["image"].shape == (4, 5, 6, 1)
    assert out["image"].dtype == np.float32
    assert out["fg_bbox"] is None


def test_preprocess_dataset_end_to_end_saves_arrays_and_updates_config(
    tmp_path, monkeypatch, base_config
):
    """End-to-end preprocess_dataset writes arrays and updates config."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results / "models").mkdir(parents=True, exist_ok=True)
    _write_json(results / "config.json", base_config)

    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame(
            [
                {
                    "id": "p1",
                    "fold": 0,
                    "image": "/tmp/p1.nii.gz",
                    "mask": "/tmp/p1_mask.nii.gz",
                }
            ]
        ),
    )
    _write_csv(
        results / "fg_bboxes.csv",
        pd.DataFrame(
            [{"id": "p1", "x0": 0, "x1": 2, "y0": 0, "y1": 2, "z0": 0, "z1": 2}]
        ),
    )

    monkeypatch.setattr(
        pp.progress_bar, "get_progress_bar", lambda *_: _PB(), raising=True
    )
    monkeypatch.setattr(
        pp.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor
    )

    def _pe(config, image_paths_list, mask_path, fg_bbox):
        img = np.ones((2, 2, 2, 1), dtype=np.float32)
        mask = np.zeros((2, 2, 2, 1), dtype=np.uint8)
        dtm = np.full((2, 2, 2, 2), 2.0, dtype=np.float32)
        return {"image": img, "mask": mask, "dtm": dtm, "fg_bbox": fg_bbox}

    monkeypatch.setattr(pp, "preprocess_example", _pe, raising=True)

    ns = argparse.Namespace(
        results=str(results),
        numpy=str(numpy_dir),
        compute_dtms=True,
        no_preprocess=True,
        num_workers_preprocess=None,
    )
    pp.preprocess_dataset(ns)

    img_npy = numpy_dir / "images" / "p1.npy"
    lab_npy = numpy_dir / "labels" / "p1.npy"
    dtm_npy = numpy_dir / "dtms" / "p1.npy"
    assert img_npy.exists()
    assert lab_npy.exists()
    assert dtm_npy.exists()
    assert np.load(img_npy).dtype == np.float32
    assert np.load(lab_npy).dtype == np.uint8
    assert np.load(dtm_npy).dtype == np.float32

    cfg = json.loads((results / "config.json").read_text(encoding="utf-8"))
    assert cfg["preprocessing"]["compute_dtms"] is True
    assert cfg["preprocessing"]["skip"] is True


def test_preprocess_dataset_missing_files_raise(tmp_path, base_config):
    """Missing config/train_paths/fg_bboxes should raise FileNotFoundError."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        pp.preprocess_dataset(
            argparse.Namespace(
                results=str(results),
                numpy=str(numpy_dir),
                compute_dtms=False,
                no_preprocess=False,
            )
        )

    _write_json(results / "config.json", base_config)
    with pytest.raises(FileNotFoundError):
        pp.preprocess_dataset(
            argparse.Namespace(
                results=str(results),
                numpy=str(numpy_dir),
                compute_dtms=False,
                no_preprocess=False,
            )
        )

    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame(
            [
                {
                    "id": "p1",
                    "fold": 0,
                    "image": "/tmp/p1.nii.gz",
                    "mask": "/tmp/p1_mask.nii.gz",
                }
            ]
        ),
    )
    with pytest.raises(FileNotFoundError):
        pp.preprocess_dataset(
            argparse.Namespace(
                results=str(results),
                numpy=str(numpy_dir),
                compute_dtms=False,
                no_preprocess=False,
            )
        )


def test_preprocess_dataset_sets_fg_bbox_none_when_crop_disabled(
    tmp_path, monkeypatch
):
    """When cropping disabled, pass fg_bbox=None to preprocess_example."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results.mkdir(parents=True, exist_ok=True)
    numpy_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset_info": {"images": ["image"], "labels": [0, 1]},
        "spatial_config": {
            "target_spacing": [1.0, 1.0, 1.0],
            "patch_size": [64, 64, 64],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": False,
            "compute_dtms": False,
            "normalize_dtms": True,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
    }
    (results / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame([{
            "id": "p1",
            "image": "/tmp/p1_image.nii.gz",
            "mask": "/tmp/p1_mask.nii.gz"
        }]),
    )
    _write_csv(
        results / "fg_bboxes.csv",
        pd.DataFrame([{
            "id": "p1", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1
        }]),
    )

    monkeypatch.setattr(
        pp.progress_bar,
        "get_progress_bar",
        lambda *_a, **_k: _PB(),
        raising=True,
    )
    monkeypatch.setattr(
        pp.io,
        "read_json_file",
        lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
        raising=True,
    )
    monkeypatch.setattr(
        pp.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor
    )

    observed: dict[str, Any] = {}

    def _fake_preprocess_example(**kwargs):
        observed["fg_bbox"] = kwargs.get("fg_bbox", "MISSING")
        return {
            "image": np.zeros((2, 2, 2, 1), dtype=np.float32),
            "mask": np.zeros((2, 2, 2, 1), dtype=np.uint8),
            "dtm": np.zeros((2, 2, 2, 1), dtype=np.float32),
        }

    monkeypatch.setattr(
        pp, "preprocess_example", _fake_preprocess_example, raising=True
    )

    ns = SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        no_preprocess=False,
        compute_dtms=False,
        num_workers_preprocess=None,
    )
    pp.preprocess_dataset(ns)

    assert "fg_bbox" in observed
    assert observed["fg_bbox"] is None
    assert (numpy_dir / "images" / "p1.npy").exists()
    assert (numpy_dir / "labels" / "p1.npy").exists()


# ---------------------------------------------------------------------------
# _preprocess_single_patient
# ---------------------------------------------------------------------------

def test_preprocess_single_patient_happy_path_saves_arrays(
    tmp_path, monkeypatch
):
    """_preprocess_single_patient saves image, mask, and dtm and returns None."""
    output_dirs = {
        "images": tmp_path / "images",
        "labels": tmp_path / "labels",
        "dtms": tmp_path / "dtms",
    }
    for d in output_dirs.values():
        d.mkdir()

    def _pe(config, image_paths_list, mask_path, fg_bbox):
        return {
            "image": np.ones((2, 2, 2, 1), dtype=np.float32),
            "mask": np.zeros((2, 2, 2, 1), dtype=np.uint8),
            "dtm": np.full((2, 2, 2, 2), 3.0, dtype=np.float32),
            "fg_bbox": fg_bbox,
        }

    monkeypatch.setattr(pp, "preprocess_example", _pe, raising=True)

    config = {
        "dataset_info": {"labels": [0, 1]},
        "preprocessing": {"skip": False},
    }
    patient = {"id": "pat_01", "image": "/fake/img.nii.gz", "mask": "/fake/msk.nii.gz"}

    err = pp._preprocess_single_patient(
        config=config,
        patient=patient,
        image_columns=["image"],
        fg_bbox=None,
        output_dirs=output_dirs,
        compute_dtms=True,
    )

    assert err is None
    assert (output_dirs["images"] / "pat_01.npy").exists()
    assert (output_dirs["labels"] / "pat_01.npy").exists()
    assert (output_dirs["dtms"] / "pat_01.npy").exists()
    np.testing.assert_array_equal(
        np.load(output_dirs["images"] / "pat_01.npy"),
        np.ones((2, 2, 2, 1), dtype=np.float32),
    )


def test_preprocess_single_patient_returns_error_on_exception(
    tmp_path, monkeypatch
):
    """_preprocess_single_patient returns an error string when processing fails."""
    output_dirs = {
        "images": tmp_path / "images",
        "labels": tmp_path / "labels",
        "dtms": tmp_path / "dtms",
    }
    for d in output_dirs.values():
        d.mkdir()

    def _pe_raises(**kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pp, "preprocess_example", _pe_raises, raising=True)

    patient = {"id": "bad_pat", "image": "/fake/img.nii.gz", "mask": "/fake/msk.nii.gz"}

    err = pp._preprocess_single_patient(
        config={},
        patient=patient,
        image_columns=["image"],
        fg_bbox=None,
        output_dirs=output_dirs,
        compute_dtms=False,
    )

    assert err is not None
    assert "bad_pat" in err
    assert "simulated failure" in err


def test_preprocess_single_patient_skips_dtm_when_disabled(
    tmp_path, monkeypatch
):
    """_preprocess_single_patient does not write dtm file when compute_dtms=False."""
    output_dirs = {
        "images": tmp_path / "images",
        "labels": tmp_path / "labels",
        "dtms": tmp_path / "dtms",
    }
    for d in output_dirs.values():
        d.mkdir()

    def _pe(**kwargs):
        return {
            "image": np.zeros((2, 2, 2, 1), dtype=np.float32),
            "mask": np.zeros((2, 2, 2, 1), dtype=np.uint8),
            "dtm": np.zeros((2, 2, 2, 2), dtype=np.float32),
            "fg_bbox": None,
        }

    monkeypatch.setattr(pp, "preprocess_example", _pe, raising=True)

    patient = {"id": "p1", "image": "/img.nii.gz", "mask": "/msk.nii.gz"}
    pp._preprocess_single_patient(
        config={},
        patient=patient,
        image_columns=["image"],
        fg_bbox=None,
        output_dirs=output_dirs,
        compute_dtms=False,
    )

    assert (output_dirs["images"] / "p1.npy").exists()
    assert (output_dirs["labels"] / "p1.npy").exists()
    assert not (output_dirs["dtms"] / "p1.npy").exists()


def test_preprocess_dataset_prints_error_summary_on_failures(
    tmp_path, monkeypatch, base_config
):
    """preprocess_dataset prints 'N of M patients had errors' on failure."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results.mkdir(parents=True)
    _write_json(results / "config.json", base_config)
    _write_csv(
        results / "train_paths.csv",
        pd.DataFrame([
            {"id": "p1", "image": "/a.nii.gz", "mask": "/a_msk.nii.gz"},
            {"id": "p2", "image": "/b.nii.gz", "mask": "/b_msk.nii.gz"},
        ]),
    )
    _write_csv(
        results / "fg_bboxes.csv",
        pd.DataFrame([
            {"id": "p1", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1},
            {"id": "p2", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1},
        ]),
    )

    monkeypatch.setattr(
        pp.progress_bar, "get_progress_bar", lambda *_: _PB(), raising=True
    )
    monkeypatch.setattr(
        pp.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor
    )

    def _pe_always_fail(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pp, "preprocess_example", _pe_always_fail, raising=True)

    printed = []
    monkeypatch.setattr(
        console_mod.console, "print", lambda msg, **k: printed.append(str(msg))
    )

    ns = argparse.Namespace(
        results=str(results),
        numpy=str(numpy_dir),
        compute_dtms=False,
        no_preprocess=False,
        num_workers_preprocess=None,
    )
    pp.preprocess_dataset(ns)

    assert any("2 of 2" in msg for msg in printed)
