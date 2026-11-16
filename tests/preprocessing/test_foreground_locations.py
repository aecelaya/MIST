"""Tests for mist.preprocessing.foreground_locations."""

from pathlib import Path

import numpy as np

from mist.preprocessing import foreground_locations as fl


def test_path_for_puts_locations_next_to_the_labels_folder():
    """A label's locations live in a sibling foreground folder, as .npz."""
    path = fl.path_for("/data/numpy/labels/case_7.npy")
    assert path == Path("/data/numpy") / fl.FOLDER / "case_7.npz"


def test_saved_locations_load_unchanged(tmp_path):
    """Every class, including the any-voxel fallback, round-trips."""
    locations = {
        1: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        2: np.array([[7, 8, 9]], dtype=np.uint16),
    }
    fl.save(tmp_path / "a.npz", locations, [1, 2])
    loaded = fl.load(tmp_path / "a.npz", [1, 2])
    assert loaded.keys() == locations.keys()
    for key, value in locations.items():
        np.testing.assert_array_equal(loaded[key], value)
        assert loaded[key].dtype == np.uint16

    fallback = {None: np.array([[0, 0, 1]], dtype=np.uint16)}
    fl.save(tmp_path / "b.npz", fallback, [1, 2])
    assert fl.load(tmp_path / "b.npz", [1, 2])[None].tolist() == [[0, 0, 1]]


def test_empty_locations_load_as_empty(tmp_path):
    """A case with no foreground saves and loads as no locations, not a scan."""
    fl.save(tmp_path / "a.npz", {}, [1])
    assert fl.load(tmp_path / "a.npz", [1]) == {}


def test_load_returns_none_when_it_cant_be_used(tmp_path):
    """Missing, unreadable, or other-class files mean the label gets scanned."""
    assert fl.load(tmp_path / "missing.npz", [1]) is None

    (tmp_path / "broken.npz").write_bytes(b"not a zip file")
    assert fl.load(tmp_path / "broken.npz", [1]) is None

    fl.save(tmp_path / "other.npz", {1: np.zeros((1, 3), dtype=np.uint16)}, [1])
    assert fl.load(tmp_path / "other.npz", [1, 2]) is None


def test_find_matches_what_the_loader_needs():
    """find returns per-class voxel coordinates in the label's own axes."""
    label = np.zeros((4, 5, 6, 1), dtype=np.uint8)
    label[1, 2, 3, 0] = 2
    locations = fl.find(label, [1, 2], np.random.default_rng(0))
    assert list(locations) == [2]
    assert locations[2].tolist() == [[1, 2, 3]]
