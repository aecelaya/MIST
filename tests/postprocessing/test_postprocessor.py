"""Tests for the MIST Postprocessor class."""
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import ants
import pytest

# MIST imports.
from mist.postprocessing import postprocessor as pp_mod
from mist.postprocessing.postprocessor import Postprocessor
from mist.utils import console as console_mod


# ---------------------------------------------------------------------------
# Strategy fixtures
# ---------------------------------------------------------------------------

VALID_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": [1, 2],
        "per_label": False,
        "kwargs": {"small_object_threshold": 64}
    },
    {
        "transform": "fill_holes_with_label",
        "apply_to_labels": [4],
        "per_label": True
    }
]

MISSING_KEY_STRATEGY = [
    {
        "apply_to_labels": [1, 2],
        "per_label": False
    }
]

MISSING_LABELS_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "per_label": True
    }
]

MISSING_PER_LABEL_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": [1],
    }
]

INVALID_LABELS_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": "not_a_list",
        "per_label": True
    }
]

INVALID_PER_LABEL_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": [1],
        "per_label": "yes"
    }
]

INVALID_TOP_LEVEL_TYPE = {
    "transform": "remove_small_objects",
    "apply_to_labels": [1],
    "per_label": True
}

INVALID_TRANSFORM = [
    {
        "transform": "non_existent_transform",
        "apply_to_labels": [1],
        "per_label": True
    }
]

REPLACE_SMALL_OBJECTS_GROUPED = [
    {
        "transform": "replace_small_objects_with_label",
        "apply_to_labels": [1],
        "per_label": False,
    }
]


# ---------------------------------------------------------------------------
# _load_strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strategy_data, should_raise",
    [
        (VALID_STRATEGY, False),
        (MISSING_KEY_STRATEGY, True),
        (MISSING_LABELS_STRATEGY, True),
        (MISSING_PER_LABEL_STRATEGY, True),
        (INVALID_LABELS_STRATEGY, True),
        (INVALID_PER_LABEL_STRATEGY, True),
        (INVALID_TOP_LEVEL_TYPE, True),
        (INVALID_TRANSFORM, True),
        (REPLACE_SMALL_OBJECTS_GROUPED, True),
    ]
)
def test_load_strategy_validation(strategy_data, should_raise):
    """Test the _load_strategy method of the Postprocessor class."""
    with patch("mist.utils.io.read_json_file", return_value=strategy_data):
        if should_raise:
            with pytest.raises(ValueError):
                Postprocessor("_mock_path")
        else:
            processor = Postprocessor("_mock_path")
            assert len(processor.transforms) == len(strategy_data)
            assert isinstance(processor.transforms[0], str)


# ---------------------------------------------------------------------------
# _gather_base_filepaths
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_strategy():
    """Minimal valid strategy for Postprocessor instantiation."""
    return [{
        "transform": "remove_small_objects",
        "apply_to_labels": [1],
        "per_label": True,
        "kwargs": {}
    }]


def test_gather_base_filepaths_returns_valid_files(tmp_path, minimal_strategy):
    """Valid .nii.gz files are returned; non-files and other extensions skipped."""
    (tmp_path / "valid1.nii.gz").touch()
    (tmp_path / "valid2.nii.gz").touch()
    (tmp_path / "notes.txt").touch()      # wrong extension — ignored

    with patch("mist.utils.io.read_json_file", return_value=minimal_strategy):
        post = Postprocessor(strategy_path="fake/path/strategy.json")

    valid = post._gather_base_filepaths(tmp_path)
    assert sorted(p.name for p in valid) == ["valid1.nii.gz", "valid2.nii.gz"]


def test_gather_base_filepaths_skips_non_files(tmp_path, minimal_strategy):
    """Directories named .nii.gz are skipped with a warning."""
    (tmp_path / "real.nii.gz").touch()
    (tmp_path / "dir.nii.gz").mkdir()     # directory, not a file

    printed = []
    with patch("mist.utils.io.read_json_file", return_value=minimal_strategy):
        post = Postprocessor(strategy_path="fake/path/strategy.json")

    with patch.object(console_mod.console, "print", side_effect=lambda *a, **k: printed.append(str(a[0]))):
        valid = post._gather_base_filepaths(tmp_path)

    assert len(valid) == 1
    assert valid[0].name == "real.nii.gz"
    assert any("dir.nii.gz" in msg for msg in printed)


def test_gather_base_filepaths_empty_dir(tmp_path, minimal_strategy):
    """Empty directory returns an empty list without error."""
    with patch("mist.utils.io.read_json_file", return_value=minimal_strategy):
        post = Postprocessor(strategy_path="fake/path/strategy.json")

    assert post._gather_base_filepaths(tmp_path) == []


# ---------------------------------------------------------------------------
# _print_strategy
# ---------------------------------------------------------------------------

@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.Table")
def test_print_strategy(mock_table_class, mock_read_json):
    """_print_strategy constructs the correct table and prints it."""
    mock_read_json.return_value = [
        {
            "transform": "remove_small_objects",
            "apply_to_labels": [1, 2],
            "per_label": True,
            "kwargs": {"small_object_threshold": 32},
        },
        {
            "transform": "fill_holes_with_label",
            "apply_to_labels": [3],
            "per_label": False,
            "kwargs": {"fill_label": 0},
        },
    ]

    mock_table = MagicMock()
    mock_table_class.return_value = mock_table

    postprocessor = Postprocessor(strategy_path="fake_path.json")

    printed = []
    with patch.object(console_mod.console, "print", side_effect=lambda *a, **k: printed.append(a[0])):
        postprocessor._print_strategy()

    mock_table.add_column.assert_any_call("Transform", style="bold")
    mock_table.add_column.assert_any_call("Per Label", justify="center")
    mock_table.add_column.assert_any_call("Target Labels", justify="center")
    assert mock_table.add_row.call_count == 2
    mock_table.add_row.assert_any_call("remove_small_objects", "True", "1, 2")
    mock_table.add_row.assert_any_call("fill_holes_with_label", "False", "3")
    assert mock_table in printed


# ---------------------------------------------------------------------------
# apply_strategy_to_single_example
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_ants_image():
    """Create a dummy ANTsImage for testing."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[1:3, 1:3] = 1
    return ants.from_numpy(arr)


@pytest.mark.parametrize("simulate_error", [False, True])
@patch("mist.postprocessing.postprocessor.get_transform")
@patch("mist.utils.io.read_json_file")
def test_apply_strategy_to_single_example(
    mock_read_json, mock_get_transform, simulate_error, dummy_ants_image
):
    """Test both successful and failing transform scenarios."""
    transform_name = "remove_small_objects" if not simulate_error else "fill_holes_with_label"
    mock_read_json.return_value = [{
        "transform": transform_name,
        "apply_to_labels": [1],
        "per_label": True,
        "kwargs": {}
    }]

    if simulate_error:
        def transform_fn(*args, **kwargs):
            raise ValueError("Simulated failure")
    else:
        def transform_fn(arr, **kwargs):
            return arr + 1

    mock_get_transform.return_value = transform_fn

    post = Postprocessor(strategy_path="fake_strategy.json")
    result_image, messages = post.apply_strategy_to_single_example(
        patient_id="test123", mask=dummy_ants_image
    )

    if simulate_error:
        assert len(messages) == 1
        assert "Error applying fill_holes_with_label to test123" in messages[0]
    else:
        np.testing.assert_array_equal(
            result_image.numpy(), dummy_ants_image.numpy() + 1
        )
        assert messages == []


# ---------------------------------------------------------------------------
# _postprocess_single_file
# ---------------------------------------------------------------------------

def test_postprocess_single_file_happy_path(tmp_path, monkeypatch):
    """Worker copies file, applies transform, writes result, returns []."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:4, 2:4] = 1
    img = ants.from_numpy(arr.astype(np.float32))
    input_path = tmp_path / "p1.nii.gz"
    output_path = tmp_path / "out" / "p1.nii.gz"
    output_path.parent.mkdir()
    ants.image_write(img, str(input_path))

    def _identity(arr, **kwargs):
        return arr

    monkeypatch.setattr(pp_mod, "get_transform", lambda _: _identity)

    messages = pp_mod._postprocess_single_file(
        input_path=input_path,
        output_path=output_path,
        transforms=["identity"],
        apply_to_labels=[[1]],
        per_label=[False],
        transform_kwargs=[{}],
    )

    assert messages == []
    assert output_path.exists()


def test_postprocess_single_file_transform_error_returns_message(
    tmp_path, monkeypatch
):
    """Worker returns an error message when a transform raises ValueError."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    img = ants.from_numpy(arr.astype(np.float32))
    input_path = tmp_path / "p1.nii.gz"
    output_path = tmp_path / "out" / "p1.nii.gz"
    output_path.parent.mkdir()
    ants.image_write(img, str(input_path))

    def _failing(arr, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pp_mod, "get_transform", lambda _: _failing)

    messages = pp_mod._postprocess_single_file(
        input_path=input_path,
        output_path=output_path,
        transforms=["bad_transform"],
        apply_to_labels=[[1]],
        per_label=[False],
        transform_kwargs=[{}],
    )

    assert len(messages) == 1
    assert "bad_transform" in messages[0]
    assert "boom" in messages[0]
    # Copy is still written even when transform fails.
    assert output_path.exists()


def test_postprocess_single_file_per_label_and_labels_not_swapped(tmp_path):
    """Regression: per_label and apply_to_labels must not be swapped in the
    zip call inside _postprocess_single_file.

    If they are swapped, the transform receives per_label=[1,2,3] (truthy)
    and labels_list=False, causing 'bool object is not iterable'.
    """
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 1
    arr[6:8, 6:8, 6:8] = 2
    img = ants.from_numpy(arr.astype(np.float32))
    input_path = tmp_path / "p1.nii.gz"
    output_path = tmp_path / "out" / "p1.nii.gz"
    output_path.parent.mkdir()
    ants.image_write(img, str(input_path))

    # Two-step strategy: grouped (per_label=False) then per-label (per_label=True).
    # If the zip order in _postprocess_single_file is wrong, the first step will
    # receive labels_list=False and raise 'bool object is not iterable'.
    messages = pp_mod._postprocess_single_file(
        input_path=input_path,
        output_path=output_path,
        transforms=["get_top_k_connected_components", "remove_small_objects"],
        apply_to_labels=[[1, 2], [1, 2]],
        per_label=[False, True],
        transform_kwargs=[
            {"top_k_connected_components": 1, "apply_morphological_cleaning": False},
            {"small_object_threshold": 1},
        ],
    )

    assert messages == [], f"Unexpected errors: {messages}"
    assert output_path.exists()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_strategy():
    """Minimal strategy fixture for run tests."""
    return [{
        "transform": "remove_small_objects",
        "apply_to_labels": [1],
        "per_label": True,
        "kwargs": {}
    }]


@pytest.fixture
def temp_dirs_with_nii():
    """Temporary directories with a single dummy NIfTI file."""
    base_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:4, 2:4] = 1
    ants.image_write(
        ants.from_numpy(arr.astype(np.float32)),
        str(Path(base_dir) / "example1.nii.gz"),
    )
    return Path(base_dir), Path(output_dir)


@pytest.mark.parametrize(
    "transform_behavior,expect_error",
    [
        ("success", False),
        ("fail", True),
    ]
)
@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.get_transform")
@patch("mist.utils.progress_bar.get_progress_bar")
def test_run_postprocessor(
    mock_get_progress_bar,
    mock_get_transform,
    mock_read_json_file,
    transform_behavior,
    expect_error,
    dummy_strategy,
    temp_dirs_with_nii,
):
    """Test the run method of the Postprocessor class."""
    class _DummyPB:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def track(self, items, total=None): return items

    base_dir, output_dir = temp_dirs_with_nii
    mock_get_progress_bar.return_value = _DummyPB()
    mock_read_json_file.return_value = dummy_strategy

    if transform_behavior == "success":
        def transform_fn(arr, **kwargs):
            result = arr.copy()
            result[arr == 1] += 1
            return result
    else:
        def transform_fn(*args, **kwargs):
            raise ValueError("test error")

    mock_get_transform.return_value = transform_fn

    postprocessor = Postprocessor(strategy_path="fake_path.json")

    printed = []
    with patch.object(console_mod.console, "print", side_effect=lambda *a, **k: printed.append(str(a[0]))):
        with patch(
            "concurrent.futures.ProcessPoolExecutor", ThreadPoolExecutor
        ):
            postprocessor.run(base_dir, output_dir)

    if expect_error:
        assert any("Error applying" in msg for msg in printed)
        assert any(
            "Postprocessing completed with the following messages:" in msg
            for msg in printed
        )
    else:
        output_file = output_dir / "example1.nii.gz"
        assert output_file.exists()
        result = ants.image_read(str(output_file)).numpy()
        assert np.all(result[2:4, 2:4] == 2)
        assert np.all(result[:2, :2] == 0)
        assert any(
            "Postprocessing completed successfully" in msg for msg in printed
        )


@patch("mist.utils.io.read_json_file")
def test_run_empty_base_dir_warns_and_returns(
    mock_read_json_file, dummy_strategy, tmp_path
):
    """run() prints a warning and returns early when no .nii.gz files exist."""
    mock_read_json_file.return_value = dummy_strategy
    output_dir = tmp_path / "out"

    postprocessor = Postprocessor(strategy_path="fake_path.json")

    printed = []
    with patch.object(
        console_mod.console, "print",
        side_effect=lambda *a, **k: printed.append(str(a[0]))
    ):
        postprocessor.run(tmp_path, output_dir)

    assert any("No .nii.gz files" in msg for msg in printed)
    # Strategy table and completion message must NOT appear.
    assert not any("Strategy Summary" in msg for msg in printed)
    assert not any("completed" in msg for msg in printed)


@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.get_transform")
@patch("mist.utils.progress_bar.get_progress_bar")
def test_run_unexpected_worker_exception_is_caught(
    mock_get_progress_bar,
    mock_get_transform,
    mock_read_json_file,
    dummy_strategy,
    temp_dirs_with_nii,
):
    """Non-ValueError exceptions from workers produce a graceful error message."""
    class _DummyPB:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def track(self, items, total=None): return items

    base_dir, output_dir = temp_dirs_with_nii
    mock_get_progress_bar.return_value = _DummyPB()
    mock_read_json_file.return_value = dummy_strategy
    mock_get_transform.side_effect = RuntimeError("unexpected crash")

    postprocessor = Postprocessor(strategy_path="fake_path.json")

    printed = []
    with patch.object(
        console_mod.console, "print",
        side_effect=lambda *a, **k: printed.append(str(a[0]))
    ):
        with patch("concurrent.futures.ProcessPoolExecutor", ThreadPoolExecutor):
            postprocessor.run(base_dir, output_dir)

    assert any("Unexpected error" in msg for msg in printed)
    assert any(
        "Postprocessing completed with the following messages:" in msg
        for msg in printed
    )


@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.get_transform")
@patch("mist.utils.progress_bar.get_progress_bar")
def test_run_patient_id_preserves_dots(
    mock_get_progress_bar,
    mock_get_transform,
    mock_read_json_file,
    dummy_strategy,
    tmp_path,
):
    """patient_id extracted from filename strips .nii.gz, preserving inner dots."""
    class _DummyPB:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def track(self, items, total=None): return items

    arr = np.zeros((4, 4), dtype=np.uint8)
    input_path = tmp_path / "patient.001.nii.gz"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    ants.image_write(ants.from_numpy(arr.astype(np.float32)), str(input_path))

    mock_get_progress_bar.return_value = _DummyPB()
    mock_read_json_file.return_value = dummy_strategy

    def _failing(*args, **kwargs):
        raise ValueError("boom")
    mock_get_transform.return_value = _failing

    postprocessor = Postprocessor(strategy_path="fake_path.json")

    printed = []
    with patch.object(
        console_mod.console, "print",
        side_effect=lambda *a, **k: printed.append(str(a[0]))
    ):
        with patch("concurrent.futures.ProcessPoolExecutor", ThreadPoolExecutor):
            postprocessor.run(tmp_path, output_dir)

    # patient_id should be "patient.001", not "patient".
    assert any("patient.001" in msg for msg in printed)
    assert not any(msg for msg in printed if "patient" in msg and "001" not in msg and "Error" in msg)
