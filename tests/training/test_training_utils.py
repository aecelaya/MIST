import re
import pytest
from pathlib import Path

# MIST imports.
from mist.training import training_utils as tu


def test_running_mean_initial_state():
    """Test that RunningMean initializes correctly."""
    rm = tu.RunningMean()
    assert rm.count == 0
    assert rm.total == 0
    assert rm.result() == pytest.approx(0.0)


def test_running_mean_updates_return_current_mean():
    """Test that RunningMean updates and returns the current mean."""
    rm = tu.RunningMean()
    vals = [1.0, 3.0, 5.0]
    expected_means = [1.0, 2.0, 3.0]
    for v, m in zip(vals, expected_means):
        assert rm(v) == pytest.approx(m)
    assert rm.count == 3
    assert rm.total == pytest.approx(sum(vals))
    assert rm.result() == pytest.approx(3.0)


def test_running_mean_reset_states():
    """Test that RunningMean resets its state correctly."""
    rm = tu.RunningMean()
    rm(2.0)
    rm(4.0)
    assert rm.result() == pytest.approx(3.0)

    rm.reset_states()
    assert rm.count == 0
    assert rm.total == 0
    assert rm.result() == pytest.approx(0.0)

    # After reset, should start fresh
    assert rm(10.0) == pytest.approx(10.0)
    assert rm.count == 1
    assert rm.total == pytest.approx(10.0)
    assert rm.result() == pytest.approx(10.0)


def test_running_mean_handles_ints_and_negatives():
    """Test that RunningMean handles integers and negatives correctly."""
    rm = tu.RunningMean()
    for v in [2, -2, 4]:   # mix ints, negatives
        rm(v)
    # mean = (2 + (-2) + 4) / 3 = 4/3
    assert rm.count == 3
    assert rm.total == pytest.approx(4.0)
    assert rm.result() == pytest.approx(4.0 / 3.0)


def _make_paths(tmp_path: Path, subdir: str, ids):
    """Create dummy paths in a subdirectory and return them as a list."""
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in ids:
        p = d / f"{i}.npy"
        p.touch()
        out.append(p)
    return out


def test_happy_path_no_dtms(tmp_path):
    """Test that a valid fold with no DTMs does not raise."""
    train_ids = ["p1", "p2", "p3"]
    val_ids = ["p4", "p5"]
    tr_img = _make_paths(tmp_path, "images/train", train_ids)
    tr_lbl = _make_paths(tmp_path, "labels/train", train_ids)
    va_img = _make_paths(tmp_path, "images/val", val_ids)
    va_lbl = _make_paths(tmp_path, "labels/val", val_ids)

    # Should not raise
    tu.sanity_check_fold_data(
        fold=0,
        train_images=tr_img,
        train_labels=tr_lbl,
        val_images=va_img,
        val_labels=va_lbl,
        train_dtms=None,
    )


def test_happy_path_with_dtms(tmp_path):
    """Test that a valid fold with DTMs does not raise."""
    train_ids = ["p1", "p2"]
    val_ids = ["p3"]
    tr_img = _make_paths(tmp_path, "images/train", train_ids)
    tr_lbl = _make_paths(tmp_path, "labels/train", train_ids)
    va_img = _make_paths(tmp_path, "images/val", val_ids)
    va_lbl = _make_paths(tmp_path, "labels/val", val_ids)
    dtms = _make_paths(tmp_path, "dtms/train", train_ids)

    # Should not raise
    tu.sanity_check_fold_data(
        fold=1,
        train_images=tr_img,
        train_labels=tr_lbl,
        val_images=va_img,
        val_labels=va_lbl,
        train_dtms=dtms,
    )


def test_raises_on_empty_split(tmp_path):
    """Test that empty train or val data raises error."""
    # Empty validation set.
    train_ids = ["p1"]
    tr_img = _make_paths(tmp_path, "images/train", train_ids)
    tr_lbl = _make_paths(tmp_path, "labels/train", train_ids)
    va_img = []  # Empty.
    va_lbl = []
    dtms = None

    with pytest.raises(ValueError, match="empty data"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_count_mismatch_train(tmp_path):
    """Test that train_images and train_labels count mismatch raises error."""
    tr_img = _make_paths(tmp_path, "images/train", ["p1", "p2"])
    tr_lbl = _make_paths(tmp_path, "labels/train", ["p1"])  # Fewer labels.
    va_img = _make_paths(tmp_path, "images/val", ["p3"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p3"])
    dtms = None

    with pytest.raises(ValueError, match="train_images .* != train_labels"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_duplicates_in_train_images(tmp_path):
    """Test that duplicate entries in training images raises error."""
    base = tmp_path / "images/train"
    base.mkdir(parents=True, exist_ok=True)
    p1 = base / "p1.npy"
    p2 = base / "p2.npy"
    p1.touch(); p2.touch()

    tr_img = [p1, p2, p1]  # Duplicate p1.
    tr_lbl = _make_paths(tmp_path, "labels/train", ["p1", "p2", "p1"])
    va_img = _make_paths(tmp_path, "images/val", ["p3"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p3"])
    dtms = None

    with pytest.raises(ValueError, match="duplicate entries in train_images"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_train_val_image_leakage(tmp_path):
    """Test that train/val image leakage raises error."""
    # Same file appears in train and val images.
    p_shared = (_make_paths(tmp_path, "images/train", ["p1"])[0])
    tr_img = [p_shared]
    tr_lbl = _make_paths(tmp_path, "labels/train", ["p1"])
    va_img = [p_shared]  # Intentional leak.
    va_lbl = _make_paths(tmp_path, "labels/val", ["p1"])
    dtms = None

    with pytest.raises(ValueError, match="overlap in images"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_label_stem_mismatch(tmp_path):
    """Test that image/label stem mismatch raises error."""
    tr_img = _make_paths(tmp_path, "images/train", ["p1", "p2"])
    # Label stems differ ("q1", "q2").
    tr_lbl = _make_paths(tmp_path, "labels/train", ["q1", "q2"])
    va_img = _make_paths(tmp_path, "images/val", ["p3"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p3"])
    dtms = None

    with pytest.raises(ValueError, match="image/label stem mismatch.*training"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_none_dtms_skips_dtm_checks(tmp_path):
    """Test that train_dtms=None skips DTM checks entirely."""
    tr_img = _make_paths(tmp_path, "images/train", ["p1"])
    tr_lbl = _make_paths(tmp_path, "labels/train", ["p1"])
    va_img = _make_paths(tmp_path, "images/val", ["p2"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p2"])

    # Should not raise — DTM checks are skipped when train_dtms is None.
    tu.sanity_check_fold_data(
        fold=0,
        train_images=tr_img,
        train_labels=tr_lbl,
        val_images=va_img,
        val_labels=va_lbl,
        train_dtms=None,
    )


def test_raises_on_dtm_count_mismatch(tmp_path):
    """Test that train_dtms count mismatch with train_images raises error."""
    tr_ids = ["p1", "p2"]
    tr_img = _make_paths(tmp_path, "images/train", tr_ids)
    tr_lbl = _make_paths(tmp_path, "labels/train", tr_ids)
    va_img = _make_paths(tmp_path, "images/val", ["p3"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p3"])
    # Only one DTM for two images.
    dtms = _make_paths(tmp_path, "dtms/train", ["p1"])

    with pytest.raises(ValueError, match="train_dtms .* != train_images"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_dtm_duplicates(tmp_path):
    """Test that duplicate entries in DTMs raises error."""
    # Two training samples.
    tr_ids = ["p1", "p2"]
    tr_img = _make_paths(tmp_path, "images/train", tr_ids)
    tr_lbl = _make_paths(tmp_path, "labels/train", tr_ids)
    va_img = _make_paths(tmp_path, "images/val", ["p3"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p3"])

    # Create two DTM files but intentionally use the same file twice.
    d_base = tmp_path / "dtms/train"
    d_base.mkdir(parents=True, exist_ok=True)
    d1 = d_base / "p1.npy"
    d2 = d_base / "p2.npy"
    d1.touch(); d2.touch()

    # Length matches train_images (2) but contains a duplicate.
    dtms = [d1, d1]

    with pytest.raises(ValueError, match="duplicate entries in DTMs"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_dtm_stem_mismatch(tmp_path):
    """Test that DTM stems mismatch with training images raises error."""
    tr_img = _make_paths(tmp_path, "images/train", ["p1", "p2"])
    tr_lbl = _make_paths(tmp_path, "labels/train", ["p1", "p2"])
    va_img = _make_paths(tmp_path, "images/val", ["p3"])
    va_lbl = _make_paths(tmp_path, "labels/val", ["p3"])
    # DTMs with different stems ("q1", "q2").
    dtms = _make_paths(tmp_path, "dtms/train", ["q1", "q2"])

    with pytest.raises(ValueError, match="image/DTM stem mismatch"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=dtms,
        )


def test_raises_on_val_image_label_count_mismatch(tmp_path):
    """Test that validation images and labels count mismatch raises error."""
    # Train is valid and paired (so we reach the *val* check).
    train_ids = ["p1", "p2"]
    tr_img = _make_paths(tmp_path, "images/train", train_ids)
    tr_lbl = _make_paths(tmp_path, "labels/train", train_ids)
    # Val is non-empty but mismatched: 2 images vs 1 label.
    val_img_ids = ["p3", "p4"]
    val_lbl_ids = ["p3"]  # One missing on purpose.
    va_img = _make_paths(tmp_path, "images/val", val_img_ids)
    va_lbl = _make_paths(tmp_path, "labels/val", val_lbl_ids)

    with pytest.raises(
        ValueError,
        match=r"mismatch: val_images \(\d+\) != val_labels \(\d+\)"
    ):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=[],
        )


def test_raises_on_duplicate_train_labels(tmp_path):
    """Test that duplicate entries in training labels raises error."""
    # Train: 2 imgs, 2 labels but labels contain a duplicate.
    tr_img = _make_paths(tmp_path, "train/images", ["p1", "p2"])
    tr_lbl = _make_paths(tmp_path, "train/labels", ["p1", "p1"])  # Duplicate.
    # Val: clean and paired.
    va_img = _make_paths(tmp_path, "val/images", ["q1", "q2"])
    va_lbl = _make_paths(tmp_path, "val/labels", ["q1", "q2"])

    with pytest.raises(ValueError, match="duplicate entries in train_labels"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=[],
        )


def test_raises_on_duplicate_val_images(tmp_path):
    """Test that duplicate entries in validation images raises error."""
    # Train: clean and paired.
    tr_img = _make_paths(tmp_path, "train/images", ["p1", "p2"])
    tr_lbl = _make_paths(tmp_path, "train/labels", ["p1", "p2"])
    # Val: 2 imgs (duplicate), 2 labels (distinct) to keep counts equal.
    va_img = _make_paths(tmp_path, "val/images", ["q1", "q1"])  # Duplicate.
    va_lbl = _make_paths(tmp_path, "val/labels", ["q1", "q2"])

    with pytest.raises(ValueError, match="duplicate entries in val_images"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=[],
        )


def test_raises_on_duplicate_val_labels(tmp_path: Path):
    """Test that duplicate entries in validation labels raises error."""
    # train: clean and paired
    tr_img = _make_paths(tmp_path, "train/images", ["p1", "p2"])
    tr_lbl = _make_paths(tmp_path, "train/labels", ["p1", "p2"])
    # val: 2 imgs (distinct), 2 labels (duplicate)
    va_img = _make_paths(tmp_path, "val/images", ["q1", "q2"])
    va_lbl = _make_paths(tmp_path, "val/labels", ["q1", "q1"])  # duplicate

    with pytest.raises(ValueError, match="duplicate entries in val_labels"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=[],
        )


def test_raises_on_label_overlap(tmp_path):
    """Test that train/val label overlap raises error."""
    # Train split: paired and clean.
    tr_img = _make_paths(tmp_path, "train/images", ["p1", "p2"])
    tr_lbl = _make_paths(tmp_path, "train/labels", ["p1", "p2"])

    # Val split: images clean and distinct from train images.
    va_img = _make_paths(tmp_path, "val/images", ["q1", "q2"])

    # Create label overlap by reusing one TRAIN label file path in VAL labels.
    va_lbl = [
        _make_paths(tmp_path, "val/labels", ["q1"])[0],  # q1 label in val.
        tr_lbl[1],  # <-- overlap: same path as train/labels/p2.npy.
    ]

    with pytest.raises(ValueError, match=r"train/val overlap in labels"):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=[],
        )


def test_raises_on_validation_stem_mismatch(tmp_path):
    """Test that validation images and labels stem mismatch raises error."""
    # Train split: paired and clean.
    tr_img = _make_paths(tmp_path, "train/images", ["t1", "t2"])
    tr_lbl = _make_paths(tmp_path, "train/labels", ["t1", "t2"])

    # Val split: images are clean and distinct from train.
    va_img = _make_paths(tmp_path, "val/images", ["v1", "v2"])

    # Labels count matches images, but stems do NOT align
    # (v1 matches, w2 != v2).
    va_lbl = _make_paths(tmp_path, "val/labels", ["v1", "w2"])

    with pytest.raises(
        ValueError, match=r"image/label stem mismatch in validation set"
    ):
        tu.sanity_check_fold_data(
            fold=0,

            train_images=tr_img,
            train_labels=tr_lbl,
            val_images=va_img,
            val_labels=va_lbl,
            train_dtms=[],
        )


def _touch_ids(base: Path, ids, suffix=".npy"):
    """Create empty files with given IDs in the specified base directory."""
    base.mkdir(parents=True, exist_ok=True)
    for i in ids:
        (base / f"{i}{suffix}").touch()


def test_get_npy_paths_happy_default_suffix(tmp_path: Path):
    """Test that get_npy_paths returns correct paths with default suffix."""
    base = tmp_path / "data"
    ids = ["p1", "p2"]
    _touch_ids(base, ids, suffix=".npy")

    paths = tu.get_npy_paths(base, ids)  # Default suffix, must_exist=True.
    expected = [str((base / f"{i}.npy").resolve()) for i in ids]

    assert paths == expected  # Order preserved, absolute paths returned.
    for p in paths:
        assert p.endswith(".npy")
        assert Path(p).exists()


def test_get_npy_paths_must_exist_false_does_not_raise(tmp_path):
    """Test that must_exist=False does not raise when some files are missing."""
    base = tmp_path / "data"
    # Only one of the requested files exists.
    _touch_ids(base, ["p1"], suffix=".npy")

    ids = ["p1", "p2"]
    # Should not raise even though p2.npy is missing.
    paths = tu.get_npy_paths(base, ids, must_exist=False)

    assert len(paths) == 2
    assert paths[0].endswith("p1.npy")
    assert paths[1].endswith("p2.npy")
    # The function returns absolute paths regardless of existence.
    assert str(base.resolve()) in paths[0]
    assert str(base.resolve()) in paths[1]


def test_get_npy_paths_raises_with_preview_and_more(tmp_path):
    """Test that FileNotFoundError includes preview and "(+N more)"."""
    base = tmp_path / "data"
    # Make p0 exist; others will be missing to trigger preview + "more".
    _touch_ids(base, ["p0"], suffix=".npy")

    ids = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]  # Six missing.
    with pytest.raises(FileNotFoundError) as exc:
        tu.get_npy_paths(base, ids, must_exist=True)

    msg = str(exc.value)
    # Count and base path are included.
    assert "Missing 6 expected files under" in msg
    assert str(base.resolve()) in msg
    # Preview truncates to first five and appends "(+1 more)".
    assert "(+1 more)" in msg
    # Sanity check: one of the missing filenames appears in the message.
    assert "p1.npy" in msg


def test_get_npy_paths_custom_suffix_and_path_ids(tmp_path):
    """Test that custom suffix and Path IDs work correctly."""
    base = tmp_path / "nii"
    ids = [Path("A"), Path("B")]  # Path objects are supported.
    _touch_ids(base, ["A", "B"], suffix=".nii.gz")

    paths = tu.get_npy_paths(base, ids, suffix=".nii.gz", must_exist=True)
    expected = [str((base / f"{i}.nii.gz").resolve()) for i in ["A", "B"]]

    assert paths == expected
    for p in paths:
        assert p.endswith(".nii.gz")
        assert Path(p).exists()


def test_get_npy_paths_empty_ids_returns_empty_list(tmp_path):
    """Test that empty patient IDs returns an empty list."""
    base = tmp_path / "data"
    paths = tu.get_npy_paths(base, [], must_exist=True)
    assert paths == []


def test_get_npy_paths_raises_without_more_tail_when_leq_five_missing(tmp_path):
    """Test that no "(+N more)" tail is added when <= 5 files are missing."""
    base = tmp_path / "data"
    # No files exist; request 3 -> missing <= 5, no "(+N more)" tail.
    ids = ["x1", "x2", "x3"]
    with pytest.raises(FileNotFoundError) as exc:
        tu.get_npy_paths(base, ids, must_exist=True)

    msg = str(exc.value)
    assert "Missing 3 expected files under" in msg
    # Ensure there is no "(+N more)" when <= 5 missing.
    assert re.search(r"\(\+\d+ more\)", msg) is None
    # Preview should list some of the missing filenames.
    assert "x1.npy" in msg
