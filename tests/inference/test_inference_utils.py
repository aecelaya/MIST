"""Tests for MIST inference utilities."""
from typing import Tuple, Dict
from unittest.mock import patch, MagicMock
from pathlib import Path
import ants
import numpy as np
import pandas as pd
import pytest

# MIST imports.
from mist.inference import inference_utils as iu


@pytest.fixture()
def mock_mist_config():
    """Fixture to provide a mock MIST configuration."""
    return {
        "dataset_info": {
            "modality": "ct",
        },
        "spatial_config": {
            "patch_size": [64, 64, 64],
            "target_spacing": [1.0, 1.0, 1.0],
        },
        "preprocessing": {
            "skip": False,
            "crop_to_foreground": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
        },
        "model": {
            "architecture": "nnunet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
            }
        },
        "training": {
            "seed": 42,
            "hardware": {
                "num_gpus": 2,
                "num_cpu_workers": 8
            }
        },
        "inference": {
            "inferer": {
                "name": "sliding_window",
                "params": {
                    "patch_blend_mode": "gaussian",
                    "patch_overlap": 0.5
                }
            },
            "ensemble": {
                "strategy": "mean"
            },
            "tta": {
                "enabled": True,
                "strategy": "all_flips"
            },
        },
    }


@pytest.mark.parametrize("bbox", [
    None,
    {
        "x_start": 0, "y_start": 0, "z_start": 0,
        "x_end": 31, "y_end": 31, "z_end": 31,
        "x_og_size": 64, "y_og_size": 64, "z_og_size": 64,
    }
])
@patch("mist.inference.inference_utils.decrop_from_fg")
@patch("mist.preprocessing.preprocess.resample_mask")
@patch("ants.get_orientation")
@patch("ants.reorient_image2")
@patch("ants.from_numpy")
def test_back_to_original_space(
    mock_from_numpy,
    mock_reorient_image2,
    mock_get_orientation,
    mock_resample_mask,
    mock_decrop_from_fg,
    bbox,
):
    """Test back_to_original_space function with and without bounding box."""
    # Input data.
    raw_prediction = np.ones((32, 32, 32))
    target_spacing = (1.0, 1.0, 1.0)
    training_labels = [0, 1, 2]

    # Mock original ANTs image.
    original_image = MagicMock(name="original_image")
    original_image.shape = (64, 64, 64)
    original_image.spacing = (1.2, 1.2, 1.2)
    original_image.direction = "FAKE_DIRECTION"

    # Mock prediction image steps.
    mock_ants_img = MagicMock(name="ants_image")
    mock_ants_img.numpy.return_value = "mock_numpy_data"
    mock_from_numpy.return_value = mock_ants_img
    mock_reorient_image2.return_value = mock_ants_img
    mock_get_orientation.return_value = "RAI"
    mock_resample_mask.return_value = mock_ants_img
    mock_decrop_from_fg.return_value = mock_ants_img
    final_ants_image = MagicMock(name="final_image")
    original_image.new_image_like.return_value = final_ants_image

    # Call back_to_original_space function.
    result = iu.back_to_original_space(
        raw_prediction,
        original_ants_image=original_image,
        target_spacing=target_spacing,
        training_labels=training_labels,
        foreground_bounding_box=bbox
    )

    # Assertions.
    assert result == final_ants_image

    mock_from_numpy.assert_called_once_with(
        data=raw_prediction, spacing=target_spacing
    )
    mock_get_orientation.assert_called_once_with(original_image)
    mock_reorient_image2.assert_called_once_with(mock_ants_img, "RAI")
    mock_resample_mask.assert_called_once()

    if bbox is not None:
        mock_decrop_from_fg.assert_called_once_with(mock_ants_img, bbox)
    else:
        mock_decrop_from_fg.assert_not_called()

    original_image.new_image_like.assert_called_once_with("mock_numpy_data")


class _DummyModel:
    """Minimal stub for a torch.nn.Module, supporting .to(...).eval()."""

    def __init__(self):
        self.device_arg = None

    def to(self, device):
        """Simulate moving model to a device."""
        self.device_arg = device
        return self

    def eval(self):
        """Simulate setting model to evaluation mode."""
        return self


def test_load_test_time_models_missing_dir(tmp_path: Path, mock_mist_config):
    """Raise FileNotFoundError when models_dir is missing."""
    missing_dir = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        iu.load_test_time_models(
            str(missing_dir), mist_config=mock_mist_config, device="cpu"
        )


def test_load_test_time_models_no_checkpoints(tmp_path: Path, mock_mist_config):
    """Raise ValueError when no fold_*.pt files are found."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No model checkpoints"):
        iu.load_test_time_models(
            str(models_dir), mist_config=mock_mist_config, device="cpu"
        )


@patch("mist.models.model_loader.load_model_from_config")
@patch("torch.cuda.is_available", return_value=False)
def test_load_test_time_models_loads_all_on_cpu_when_no_cuda(
    mock_cuda_available,
    mock_load_model,
    tmp_path: Path,
    mock_mist_config,
):
    """Load all weights and place models on CPU when CUDA is unavailable."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fold_0.pt").write_bytes(b"")  # Real file path for discovery.
    (models_dir / "fold_1.pt").write_bytes(b"")
    (models_dir / ".fold_hidden.pt").write_bytes(b"")  # Should be ignored.

    dummy = _DummyModel()
    mock_load_model.return_value = dummy

    models = iu.load_test_time_models(
        str(models_dir),
        mist_config=mock_mist_config,
        device=None,  # Trigger internal device resolution -> CPU.
    )

    assert len(models) == 2
    assert all(m.device_arg == "cpu" for m in models)

    called_paths = [c.args[0] for c in mock_load_model.call_args_list]
    assert called_paths == [
        str(models_dir / "fold_0.pt"),
        str(models_dir / "fold_1.pt"),
    ]


@patch("mist.models.model_loader.load_model_from_config")
@patch("torch.cuda.is_available")
def test_load_test_time_models_uses_provided_device(
    mock_cuda_available,
    mock_load_model,
    tmp_path: Path,
    mock_mist_config,
):
    """Use provided device directly without querying CUDA availability."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fold_0.pt").write_bytes(b"")

    dummy = _DummyModel()
    mock_load_model.return_value = dummy

    models = iu.load_test_time_models(
        str(models_dir),
        mist_config=mock_mist_config,
        device="cuda:2",  # Explicitly set device.
    )

    assert len(models) == 1
    assert models[0].device_arg == "cuda:2"
    mock_cuda_available.assert_not_called()

    # Optional: confirm correct call into loader.
    mock_load_model.assert_called_once_with(
        str(models_dir / "fold_0.pt"),
        mock_mist_config,
    )


@pytest.mark.parametrize("input_mask,original_labels,expected_output", [
    # Identity mapping.
    (np.array([[0, 1], [2, 0]]), [0, 1, 2], np.array([[0, 1], [2, 0]])),

    # Non-contiguous label mapping.
    (
        np.array([[0, 1, 1], [2, 2, 0]]),
        [4, 8, 16],
        np.array([[4, 8, 8], [16, 16, 4]])
    ),

    # Single-class mask.
    (np.zeros((2, 2), dtype=int), [42], np.full((2, 2), 42)),

    # Empty mask.
    (
        np.array([], dtype=int).reshape(0, 0),
        [0],
        np.array([], dtype=int).reshape(0, 0)
    ),
])
def test_remap_mask_labels(input_mask, original_labels, expected_output):
    """Test remap_mask_labels with various mappings."""
    result = iu.remap_mask_labels(input_mask, original_labels)
    assert np.array_equal(result, expected_output)
    assert result.shape == input_mask.shape
    assert result.dtype == input_mask.dtype


@patch("mist.analyze_data.analyzer_utils.compare_headers")
@patch("mist.analyze_data.analyzer_utils.is_image_3d")
@patch("ants.image_read")
@patch("ants.image_header_info")
@patch("pathlib.Path.is_file")
def test_validate_inference_images_success(
    mock_isfile,
    mock_image_header_info,
    mock_image_read,
    mock_is_image_3d,
    mock_compare_headers,
):
    """Test successful validation of compatible 3D images."""
    patient_dict = {
        "id": "abc",
        "img1": "/some/path/img1.nii.gz",
        "img2": "/some/path/img2.nii.gz"
    }

    # Mock all files exist.
    mock_isfile.return_value = True

    # Mock headers and 3D checks.
    header1 = {"dimensions": [64, 64, 64]}
    header2 = {"dimensions": [64, 64, 64]}
    mock_image_header_info.side_effect = [header1, header2]
    mock_is_image_3d.return_value = True
    mock_compare_headers.return_value = True

    # Mock ANTs image.
    mock_img = MagicMock()
    mock_image_read.return_value = mock_img

    anchor_image, image_paths = iu.validate_inference_images(patient_dict)

    assert anchor_image == mock_img
    assert image_paths == ["/some/path/img1.nii.gz", "/some/path/img2.nii.gz"]


def test_validate_inference_images_missing_id():
    """Test that missing 'id' field raises ValueError."""
    with pytest.raises(ValueError, match="must contain an 'id' field"):
        iu.validate_inference_images({"img": "/some/path/img.nii.gz"})


@patch("pathlib.Path.is_file")
def test_validate_inference_images_missing_file(mock_isfile):
    """Test that missing image file raises FileNotFoundError."""
    patient_dict = {"id": "abc", "img1": "/missing/path.nii.gz"}
    mock_isfile.return_value = False
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        iu.validate_inference_images(patient_dict)


@patch("mist.analyze_data.analyzer_utils.is_image_3d")
@patch("ants.image_header_info")
@patch("pathlib.Path.is_file")
def test_validate_inference_images_anchor_not_3d(
    mock_isfile,
    mock_image_header_info,
    mock_is_image_3d,
):
    """Test that non-3D anchor image raises ValueError."""
    patient_dict = {"id": "abc", "img1": "/some/path/img.nii.gz"}
    mock_isfile.return_value = True
    mock_image_header_info.return_value = {"dim": [2]}
    mock_is_image_3d.return_value = False
    with pytest.raises(ValueError, match="Anchor image is not 3D"):
        iu.validate_inference_images(patient_dict)


@patch("mist.analyze_data.analyzer_utils.compare_headers")
@patch("mist.analyze_data.analyzer_utils.is_image_3d")
@patch("ants.image_read")
@patch("ants.image_header_info")
@patch("pathlib.Path.is_file")
def test_validate_inference_images_header_mismatch(
    mock_isfile,
    mock_image_header_info,
    mock_image_read,
    mock_is_image_3d,
    mock_compare_headers,
):
    """Test that header mismatch raises ValueError."""
    patient_dict = {
        "id": "abc",
        "img1": "/some/path/img1.nii.gz",
        "img2": "/some/path/img2.nii.gz"
    }

    mock_isfile.return_value = True
    mock_image_header_info.side_effect = [{"dim": [3]}, {"dim": [3]}]
    mock_is_image_3d.return_value = True
    mock_compare_headers.return_value = False
    mock_image_read.return_value = MagicMock()

    with pytest.raises(ValueError, match="Image headers do not match"):
        iu.validate_inference_images(patient_dict)


def test_validate_inference_images_no_image_columns():
    """Test that patient dict with only ignored columns raises ValueError."""
    patient_dict = {
        "id": "123",        # Ignored.
        "mask": "abc.nii",  # Ignored.
        "fold": "val"       # Ignored.
    }
    with pytest.raises(ValueError, match="No image paths found for patient"):
        iu.validate_inference_images(patient_dict)


@patch("mist.analyze_data.analyzer_utils.compare_headers")
@patch("mist.analyze_data.analyzer_utils.is_image_3d")
@patch("ants.image_read")
@patch("ants.image_header_info")
@patch("pathlib.Path.is_file")
def test_validate_inference_images_secondary_image_not_3d(
    mock_isfile,
    mock_image_header_info,
    mock_image_read,
    mock_is_image_3d,
    mock_compare_headers,
):
    """Test that non-3D secondary image raises ValueError."""
    patient_dict = {
        "id": "abc",
        "img1": "/path/img1.nii.gz",
        "img2": "/path/img2.nii.gz"
    }

    # Simulate file existence.
    mock_isfile.return_value = True

    # Simulate 3D anchor and non-3D second image.
    mock_image_header_info.side_effect = [{"dim": [3]}, {"dim": [2]}]
    mock_is_image_3d.side_effect = [True, False]  # Second image fails.
    mock_compare_headers.return_value = True
    mock_image_read.return_value = MagicMock()

    with pytest.raises(ValueError, match="Image is not 3D: img2.nii.gz"):
        iu.validate_inference_images(patient_dict)


@pytest.mark.parametrize("df,should_raise,match", [
    # Valid: One .nii.gz and one .nii.
    (
        pd.DataFrame({
            "id": ["001"],
            "T1": ["subject_001_T1.nii.gz"],
            "T2": ["subject_001_T2.nii"]
        }),
        False,
        None
    ),

    # Invalid: Missing 'id' column.
    (
        pd.DataFrame({
            "T1": ["subj_T1.nii.gz"]
        }),
        True,
        "must contain an 'id' column"
    ),

    # Invalid: No valid NIfTI columns.
    (
        pd.DataFrame({
            "id": ["001"],
            "notes": ["text"],
            "fold": ["val"]
        }),
        True,
        "must contain at least one column with valid NIfTI"
    ),

    # Invalid: Mixed valid and invalid extensions.
    (
        pd.DataFrame({
            "id": ["001", "002"],
            "T1": ["img1.nii.gz", "img2.png"]
        }),
        True,
        "must contain at least one column with valid NIfTI"
    ),

    # Valid: Multiple valid NIfTI columns.
    (
        pd.DataFrame({
            "id": ["001"],
            "T1": ["img1.nii.gz"],
            "T2": ["img2.nii"]
        }),
        False,
        None
    ),
])
def test_validate_paths_dataframe_parametrized(df, should_raise, match):
    """Test validate_paths_dataframe with multiple edge cases."""
    if should_raise:
        with pytest.raises(ValueError, match=match):
            iu.validate_paths_dataframe(df)
    else:
        iu.validate_paths_dataframe(df)  # Should not raise.


def _make_ants_image(
    arr_xyz: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Create an ANTs image with basic metadata."""
    img = ants.from_numpy(arr_xyz.astype(np.float32))
    img.set_spacing(spacing)
    img.set_origin(origin)
    img.set_direction(np.eye(3))
    return img


def _crop_with_bbox(arr: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
    """Numpy crop using the same inclusive end convention as crop_to_fg."""
    xs, xe = bbox["x_start"], bbox["x_end"]
    ys, ye = bbox["y_start"], bbox["y_end"]
    zs, ze = bbox["z_start"], bbox["z_end"]
    return arr[xs:xe + 1, ys:ye + 1, zs:ze + 1]


def _expected_decrop(
    cropped: np.ndarray,
    og_shape: Tuple[int, int, int],
    bbox: Dict[str, int],
) -> np.ndarray:
    """Construct the expected zero-padded array according to bbox placement."""
    out = np.zeros(og_shape, dtype=cropped.dtype)
    xs, xe = bbox["x_start"], bbox["x_end"]
    ys, ye = bbox["y_start"], bbox["y_end"]
    zs, ze = bbox["z_start"], bbox["z_end"]
    out[xs:xe + 1, ys:ye + 1, zs:ze + 1] = cropped
    return out


def test_decrop_from_fg_restores_original_region():
    """Decropping pads the cropped block back to its original placement."""
    og_shape = (12, 10, 8)
    rng = np.random.default_rng(0)
    vol = rng.normal(size=og_shape).astype(np.float32)

    # Define a nontrivial bbox strictly inside the volume.
    bbox = {
        "x_start": 2, "x_end": 8,
        "y_start": 1, "y_end": 7,
        "z_start": 3, "z_end": 6,
        "x_og_size": og_shape[0],
        "y_og_size": og_shape[1],
        "z_og_size": og_shape[2],
    }

    cropped_np = _crop_with_bbox(vol, bbox)
    cropped_img = _make_ants_image(cropped_np)

    decropped_img = iu.decrop_from_fg(cropped_img, bbox)
    decropped_np = decropped_img.numpy()

    assert decropped_np.shape == og_shape
    expected = _expected_decrop(cropped_np, og_shape, bbox)
    assert np.allclose(decropped_np, expected)


@pytest.mark.parametrize(
    "bbox",
    [
        # Right padding only: starts at 0, ends before last index.
        {"x_start": 0, "x_end": 7, "y_start": 0, "y_end": 8,
         "z_start": 1, "z_end": 5},
        # Left padding only: ends at last index.
        {"x_start": 3, "x_end": 11, "y_start": 2, "y_end": 9,
         "z_start": 0, "z_end": 7},
        # Both sides padding on all axes.
        {"x_start": 2, "x_end": 9, "y_start": 1, "y_end": 7,
         "z_start": 1, "z_end": 6},
    ],
)
def test_decrop_from_fg_padding_patterns(bbox):
    """Decropping handles left-only, right-only, and both-side padding."""
    og_shape = (12, 10, 8)
    # Fill in *_og_size keys.
    bbox = dict(bbox, **{
        "x_og_size": og_shape[0],
        "y_og_size": og_shape[1],
        "z_og_size": og_shape[2],
    })

    # Use a deterministic pattern so misalignment is obvious.
    vol = np.zeros(og_shape, dtype=np.float32)
    xs, xe = bbox["x_start"], bbox["x_end"]
    ys, ye = bbox["y_start"], bbox["y_end"]
    zs, ze = bbox["z_start"], bbox["z_end"]
    vol[xs:xe + 1, ys:ye + 1, zs:ze + 1] = 7.0

    cropped_np = _crop_with_bbox(vol, bbox)
    cropped_img = _make_ants_image(cropped_np)

    decropped_img = iu.decrop_from_fg(cropped_img, bbox)
    decropped_np = decropped_img.numpy()

    assert decropped_np.shape == og_shape
    expected = _expected_decrop(cropped_np, og_shape, bbox)
    assert np.array_equal(decropped_np, expected)


def test_decrop_from_fg_bbox_full_image_is_noop():
    """When bbox covers full image, decropping returns the same image."""
    og_shape = (8, 7, 6)
    rng = np.random.default_rng(123)
    vol = rng.uniform(size=og_shape).astype(np.float32)

    bbox = {
        "x_start": 0, "x_end": og_shape[0] - 1,
        "y_start": 0, "y_end": og_shape[1] - 1,
        "z_start": 0, "z_end": og_shape[2] - 1,
        "x_og_size": og_shape[0],
        "y_og_size": og_shape[1],
        "z_og_size": og_shape[2],
    }

    # Cropped equals original region since bbox spans entire volume.
    cropped_img = _make_ants_image(vol.copy())

    decropped_img = iu.decrop_from_fg(cropped_img, bbox)
    decropped_np = decropped_img.numpy()

    assert decropped_np.shape == og_shape
    assert np.allclose(decropped_np, vol)
