"""Tests for the DALI data loading utilities module in MIST."""

from unittest import mock

import pytest

import mist.data_loading.data_loading_utils as utils
from mist.data_loading.data_loading_constants import DataLoadingConstants as constants


class TestIsValidGenericPipelineInput:
    """Tests for utils.is_valid_generic_pipeline_input."""

    def test_valid_npy_file(self, tmp_path):
        """Returns True for a list containing a single existing .npy file."""
        valid_file = tmp_path / "valid.npy"
        valid_file.write_text("test")
        assert utils.is_valid_generic_pipeline_input([str(valid_file)])

    @pytest.mark.parametrize("bad_input", [
        "string.npy",
        42,
        [],
        ["file.txt"],
        ["nonexistent.npy"],
    ])
    def test_invalid_inputs(self, bad_input):
        """Returns False for non-list, wrong extension, or non-existent files."""
        assert not utils.is_valid_generic_pipeline_input(bad_input)


class TestValidateTrainAndEvalInputs:
    """Tests for utils.validate_train_and_eval_inputs."""

    @pytest.mark.parametrize("imgs,lbls,dtms,err_msg", [
        ([], ["lbl"], None, "No images found!"),
        (["img"], [], None, "No labels found!"),
        (["img1"], ["lbl1", "lbl2"], None, "Number of images and labels do not match!"),
        (["img"], ["lbl"], ["dtm1", "dtm2"], "Number of images and DTMs do not match!"),
        (["img"], ["lbl"], [], "No DTM data found!"),
    ])
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


class TestRandomAugmentation:
    """Tests for utils.random_augmentation."""

    @mock.patch("mist.data_loading.data_loading_utils.fn.random.coin_flip")
    @mock.patch("mist.data_loading.data_loading_utils.fn.cast")
    def test_blends_augmented_and_original_based_on_coin_flip(
        self, mock_cast, mock_coin_flip
    ):
        """Returns condition * augmented + neg_condition * original."""
        condition = mock.MagicMock(name="condition")
        neg_condition = mock.MagicMock(name="neg_condition")
        condition.__xor__.return_value = neg_condition
        condition.__mul__.return_value = "aug_part"
        neg_condition.__mul__.return_value = "orig_part"
        mock_cast.return_value = condition

        result = utils.random_augmentation(0.15, "aug", "orig")
        assert result == "aug_part" + "orig_part"


class TestNoiseFn:
    """Tests for utils.noise_fn."""

    @mock.patch("mist.data_loading.data_loading_utils.random_augmentation", return_value="augmented")
    @mock.patch("mist.data_loading.data_loading_utils.math.clamp", return_value="clamped")
    @mock.patch("mist.data_loading.data_loading_utils.fn.reductions.min", return_value="img_min")
    @mock.patch("mist.data_loading.data_loading_utils.fn.reductions.max", return_value="img_max")
    @mock.patch("mist.data_loading.data_loading_utils.fn.random.normal", return_value="noise")
    def test_clamps_to_image_range_and_applies_probability(
        self, mock_normal, mock_max, mock_min, mock_clamp, mock_aug
    ):
        """Adds noise, clamps result to original image range, then applies probability gate."""
        result = utils.noise_fn("img")

        mock_clamp.assert_called_once_with("imgnoise", "img_min", "img_max")
        mock_aug.assert_called_once_with(constants.NOISE_FN_PROBABILITY, "clamped", "img")
        assert result == "augmented"


class TestBlurFn:
    """Tests for utils.blur_fn."""

    @mock.patch("mist.data_loading.data_loading_utils.random_augmentation", return_value="augmented")
    @mock.patch("mist.data_loading.data_loading_utils.math.clamp", return_value="clamped")
    @mock.patch("mist.data_loading.data_loading_utils.fn.reductions.min", return_value="img_min")
    @mock.patch("mist.data_loading.data_loading_utils.fn.reductions.max", return_value="img_max")
    @mock.patch("mist.data_loading.data_loading_utils.fn.gaussian_blur", return_value="blurred")
    def test_clamps_to_image_range_and_applies_probability(
        self, mock_blur, mock_max, mock_min, mock_clamp, mock_aug
    ):
        """Blurs image, clamps result to original image range, then applies probability gate."""
        result = utils.blur_fn("img")

        mock_clamp.assert_called_once_with("blurred", "img_min", "img_max")
        mock_aug.assert_called_once_with(constants.BLUR_FN_PROBABILITY, "clamped", "img")
        assert result == "augmented"


class TestBrightnessFn:
    """Tests for utils.brightness_fn."""

    @mock.patch("mist.data_loading.data_loading_utils.random_augmentation")
    def test_scales_image_by_random_brightness(self, mock_aug):
        """Multiplies image by a randomly selected brightness scale."""
        scale = mock.MagicMock(name="scale")
        scale.__rmul__.return_value = "scaled_image"
        mock_aug.return_value = scale

        result = utils.brightness_fn("img")
        assert result == "scaled_image"


class TestContrastFn:
    """Tests for utils.contrast_fn."""

    @mock.patch("mist.data_loading.data_loading_utils.random_augmentation")
    @mock.patch("mist.data_loading.data_loading_utils.math.clamp", return_value="clamped")
    @mock.patch("mist.data_loading.data_loading_utils.fn.reductions.min", return_value="min")
    @mock.patch("mist.data_loading.data_loading_utils.fn.reductions.max", return_value="max")
    def test_clamps_contrast_scaled_image(self, mock_max, mock_min, mock_clamp, mock_aug):
        """Scales by contrast factor and clamps to original image intensity range."""
        scale = mock.MagicMock(name="scale")
        scale.__rmul__.return_value = "scaled"
        mock_aug.return_value = scale

        result = utils.contrast_fn("img")
        assert result == "clamped"
