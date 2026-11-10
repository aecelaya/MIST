"""Tests for the DALI data loading utilities module in MIST."""

from unittest import mock

import pytest

import mist.data_loading.data_loading_utils as utils


class TestIsValidGenericPipelineInput:
    """Tests for utils.is_valid_generic_pipeline_input."""

    def test_valid_npy_file(self, tmp_path):
        """Returns True for a list containing a single existing .npy file."""
        valid_file = tmp_path / "valid.npy"
        valid_file.write_text("test")
        assert utils.is_valid_generic_pipeline_input([str(valid_file)])

    @pytest.mark.parametrize(
        "bad_input",
        [
            "string.npy",
            42,
            [],
            ["file.txt"],
            ["nonexistent.npy"],
        ],
    )
    def test_invalid_inputs(self, bad_input):
        """Returns False for non-list, wrong extension, or non-existent files."""
        assert not utils.is_valid_generic_pipeline_input(bad_input)


class TestValidateTrainAndEvalInputs:
    """Tests for utils.validate_train_and_eval_inputs."""

    @pytest.mark.parametrize(
        "imgs,lbls,dtms,err_msg",
        [
            ([], ["lbl"], None, "No images found!"),
            (["img"], [], None, "No labels found!"),
            (
                ["img1"],
                ["lbl1", "lbl2"],
                None,
                "Number of images and labels do not match!",
            ),
            (
                ["img"],
                ["lbl"],
                ["dtm1", "dtm2"],
                "Number of images and DTMs do not match!",
            ),
            (["img"], ["lbl"], [], "No DTM data found!"),
        ],
    )
    def test_raises_for_invalid_inputs(self, imgs, lbls, dtms, err_msg):
        """Raises ValueError with the expected message for invalid inputs."""
        with pytest.raises(ValueError, match=err_msg):
            utils.validate_train_and_eval_inputs(imgs, lbls, dtms)

    @pytest.mark.parametrize("dtms", [None, ["dtm"]])
    def test_valid_inputs_do_not_raise(self, dtms):
        """Does not raise for matching counts and valid (or absent) DTMs."""
        utils.validate_train_and_eval_inputs(["img"], ["lbl"], dtms)


class TestGetNumpyReader:
    """Tests for utils.get_numpy_reader."""

    @mock.patch("mist.data_loading.data_loading_utils.ops.readers.Numpy")
    def test_calls_numpy_reader_and_returns_result(self, mock_numpy_reader):
        """Calls ops.readers.Numpy and returns its result."""
        result = utils.get_numpy_reader(
            files=["f.npy"], shard_id=0, num_shards=1, seed=42, shuffle=True
        )
        mock_numpy_reader.assert_called_once()
        assert result == mock_numpy_reader.return_value
