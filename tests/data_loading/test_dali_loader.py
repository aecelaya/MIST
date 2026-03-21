"""Tests for the DALI data loading module in MIST."""

import pytest
from unittest import mock

from mist.data_loading import dali_loader
import mist.data_loading.data_loading_utils as utils


@pytest.fixture(autouse=True)
def _mock_dali_io(monkeypatch):
    """Patch DALI IO helpers so pipelines can instantiate without real files."""
    monkeypatch.setattr(utils, "is_valid_generic_pipeline_input", lambda _: True)
    monkeypatch.setattr(utils, "get_numpy_reader", mock.MagicMock())


@pytest.fixture
def base_pipeline_args():
    return {
        "image_paths": ["image.npy"],
        "label_paths": ["label.npy"],
        "dtm_paths": ["dtm.npy"],
        "roi_size": (64, 64, 64),
        "labels": [0, 1],
        "oversampling": 0.25,
        "extract_patches": True,
        "use_augmentation": True,
        "use_flips": True,
        "use_zoom": True,
        "use_noise": True,
        "use_blur": True,
        "use_brightness": True,
        "use_contrast": True,
        "batch_size": 1,
        "num_threads": 1,
        "device_id": 0,
        "shard_id": 0,
        "seed": 42,
        "num_gpus": 1,
    }


class TestTrainPipelineInit:
    """Tests for TrainPipeline instantiation and attribute defaults."""

    @pytest.mark.parametrize("dtm_present", [False, True])
    def test_instantiates_with_and_without_dtm(self, base_pipeline_args, dtm_present):
        """Pipeline can be created with or without DTM paths."""
        args = {**base_pipeline_args, "dtm_paths": ["dtm.npy"] if dtm_present else None}
        pipeline = dali_loader.TrainPipeline(**args)
        assert pipeline.extract_patches is True
        assert pipeline.use_augmentation is True

    def test_disables_all_augmentations_when_use_augmentation_false(
        self, base_pipeline_args
    ):
        """All augmentation flags are forced off when use_augmentation=False."""
        pipeline = dali_loader.TrainPipeline(**{**base_pipeline_args, "use_augmentation": False})
        for flag in ("use_flips", "use_zoom", "use_noise", "use_blur",
                     "use_brightness", "use_contrast"):
            assert not getattr(pipeline, flag)

    @pytest.mark.parametrize("bad_arg,expected_message", [
        ("image", "Input images are not valid"),
        ("label", "Input labels are not valid"),
        ("dtm", "Input DTMs are not valid"),
    ])
    def test_invalid_inputs_raise(
        self, monkeypatch, bad_arg, expected_message, base_pipeline_args
    ):
        """Raises ValueError when any input file list fails validation."""
        monkeypatch.setattr(utils, "is_valid_generic_pipeline_input", lambda _: False)
        args = base_pipeline_args.copy()
        args["image_paths"] = ["fake_image.npy"] if bad_arg == "image" else None
        args["label_paths"] = ["fake_label.npy"] if bad_arg == "label" else None
        args["dtm_paths"] = ["fake_dtm.npy"] if bad_arg == "dtm" else None
        with pytest.raises(ValueError, match=expected_message):
            dali_loader.TrainPipeline(**args)


class TestLoadData:
    """Tests for TrainPipeline.load_data."""

    @pytest.mark.parametrize("use_dtm", [False, True])
    @mock.patch("mist.data_loading.dali_loader.fn.reshape")
    def test_returns_correct_tuple(self, mock_reshape, base_pipeline_args, use_dtm):
        """Returns (image, label) or (image, label, dtm) based on has_dtms."""
        args = {**base_pipeline_args, "dtm_paths": ["dtm.npy"] if use_dtm else None}
        pipeline = dali_loader.TrainPipeline(**args)
        pipeline.input_images = mock.Mock(return_value="mock_image")
        pipeline.input_labels = mock.Mock(return_value="mock_label")
        if use_dtm:
            pipeline.input_dtms = mock.Mock(return_value="mock_dtm")

        output = pipeline.load_data()

        if use_dtm:
            assert output == (
                mock_reshape.return_value,
                mock_reshape.return_value,
                mock_reshape.return_value,
            )
            mock_reshape.assert_called_with("mock_dtm", layout="DHWC")
        else:
            assert output == (mock_reshape.return_value, mock_reshape.return_value)
            mock_reshape.assert_called_with("mock_label", layout="DHWC")


class TestBiasedCropFn:
    """Tests for TrainPipeline.biased_crop_fn."""

    @pytest.mark.parametrize("use_dtm", [False, True])
    @mock.patch("mist.data_loading.dali_loader.fn.slice")
    @mock.patch("mist.data_loading.dali_loader.fn.roi_random_crop")
    @mock.patch("mist.data_loading.dali_loader.fn.segmentation.random_object_bbox")
    @mock.patch("mist.data_loading.dali_loader.fn.pad")
    def test_crops_with_and_without_dtm(
        self, mock_pad, mock_bbox, mock_crop, mock_slice, base_pipeline_args, use_dtm,
    ):
        """Biased crop returns GPU tensors for each input (image, label[, dtm])."""
        args = {**base_pipeline_args, "dtm_paths": ["dtm.npy"] if use_dtm else None}
        pipeline = dali_loader.TrainPipeline(**args)

        dummy = mock.MagicMock()
        dummy.gpu.return_value = "gpu_output"
        mock_pad.return_value = dummy
        mock_bbox.return_value = ("roi_start", "roi_end")
        mock_crop.return_value = "anchor"
        mock_slice.return_value = (dummy, dummy, dummy) if use_dtm else (dummy, dummy)

        out = pipeline.biased_crop_fn("image", "label", "dtm" if use_dtm else None)
        expected = (
            ("gpu_output", "gpu_output", "gpu_output") if use_dtm
            else ("gpu_output", "gpu_output")
        )
        assert out == expected


class TestFlipsFn:
    """Tests for TrainPipeline.flips_fn."""

    @pytest.mark.parametrize("use_dtm", [False, True])
    @mock.patch("mist.data_loading.dali_loader.fn.flip")
    @mock.patch("mist.data_loading.dali_loader.fn.random.coin_flip")
    def test_flips_all_inputs(
        self, mock_coin_flip, mock_flip, base_pipeline_args, use_dtm,
    ):
        """Flips are applied to every input tensor (image, label[, dtm])."""
        mock_coin_flip.side_effect = ["flip_h", "flip_v", "flip_d"]
        mock_flip.side_effect = lambda x, **kwargs: f"flipped_{x}"
        args = {
            **base_pipeline_args,
            "dtm_paths": ["dtm.npy"] if use_dtm else None,
            "use_zoom": False, "use_noise": False,
            "use_blur": False, "use_brightness": False, "use_contrast": False,
        }
        pipeline = dali_loader.TrainPipeline(**args)

        inputs = ("img", "lbl", "dtm") if use_dtm else ("img", "lbl")
        result = pipeline.flips_fn(*inputs)
        assert result == tuple(f"flipped_{x}" for x in inputs)


class TestZoomFn:
    """Tests for TrainPipeline.zoom_fn."""

    @mock.patch("mist.data_loading.dali_loader.fn.resize")
    @mock.patch("mist.data_loading.dali_loader.fn.crop")
    @mock.patch("mist.data_loading.dali_loader.fn.random.uniform")
    @mock.patch("mist.data_loading.data_loading_utils.random_augmentation")
    def test_crops_and_resizes_to_roi(
        self, mock_aug, mock_uniform, mock_crop, mock_resize, base_pipeline_args,
    ):
        """Zoom scales down, crops to scaled dimensions, then resizes back to roi_size."""
        mock_aug.return_value = 0.8
        mock_crop.side_effect = lambda x, crop_h, crop_w, crop_d: f"cropped_{x}"
        mock_resize.side_effect = lambda x, **kwargs: f"resized_{x}"
        args = {
            **base_pipeline_args,
            "dtm_paths": None, "labels": [1],
            "use_flips": False, "use_noise": False,
            "use_blur": False, "use_brightness": False, "use_contrast": False,
        }
        pipeline = dali_loader.TrainPipeline(**args)

        result = pipeline.zoom_fn("image", "label")

        assert result == ("resized_cropped_image", "resized_cropped_label")
        mock_crop.assert_any_call("image", crop_h=51.2, crop_w=51.2, crop_d=51.2)
        mock_crop.assert_any_call("label", crop_h=51.2, crop_w=51.2, crop_d=51.2)
        assert mock_resize.call_count == 2


class TestDefineGraph:
    """Tests for TrainPipeline.define_graph."""

    @pytest.mark.parametrize("use_dtm", [False, True])
    @mock.patch("mist.data_loading.dali_loader.fn.transpose", side_effect=lambda x, perm: f"transposed_{x}")
    @mock.patch.object(dali_loader.TrainPipeline, "flips_fn")
    @mock.patch.object(dali_loader.TrainPipeline, "zoom_fn", return_value=("zoomed_image", "zoomed_label"))
    @mock.patch.object(dali_loader.TrainPipeline, "biased_crop_fn")
    @mock.patch.object(dali_loader.TrainPipeline, "load_data")
    @mock.patch("mist.data_loading.data_loading_utils.contrast_fn", return_value="contrast_image")
    @mock.patch("mist.data_loading.data_loading_utils.brightness_fn", return_value="bright_image")
    @mock.patch("mist.data_loading.data_loading_utils.blur_fn", return_value="blurred_image")
    @mock.patch("mist.data_loading.data_loading_utils.noise_fn", return_value="noisy_image")
    def test_pipeline_output_with_and_without_dtm(
        self,
        mock_noise, mock_blur, mock_brightness, mock_contrast,
        mock_load_data, mock_biased_crop_fn, mock_zoom_fn, mock_flips_fn,
        mock_transpose,
        base_pipeline_args, use_dtm,
    ):
        """define_graph returns the correct output tuple with and without DTM."""
        args = {**base_pipeline_args, "dtm_paths": ["dtm.npy"] if use_dtm else None}
        if use_dtm:
            mock_load_data.return_value = ("image_raw", "label_raw", "dtm_raw")
            mock_biased_crop_fn.return_value = ("cropped_image", "cropped_label", "cropped_dtm")
            mock_flips_fn.return_value = ("flipped_image", "flipped_label", "flipped_dtm")
        else:
            mock_load_data.return_value = ("image_raw", "label_raw")
            mock_biased_crop_fn.return_value = ("cropped_image", "cropped_label")
            mock_flips_fn.return_value = ("flipped_image", "flipped_label")

        pipeline = dali_loader.TrainPipeline(**args)
        result = pipeline.define_graph()

        if use_dtm:
            assert result == (
                "transposed_contrast_image",
                "transposed_flipped_label",
                "transposed_flipped_dtm",
            )
        else:
            assert result == (
                "transposed_contrast_image",
                "transposed_flipped_label",
            )


class TestTestPipelineDefineGraph:
    """Tests for TestPipeline.define_graph."""

    @mock.patch("mist.data_loading.dali_loader.fn.transpose", return_value="transposed_image")
    @mock.patch("mist.data_loading.dali_loader.fn.reshape", return_value="reshaped_image")
    def test_returns_transposed_image(self, mock_reshape, mock_transpose):
        """Loads, reshapes, and transposes the image tensor."""
        pipeline = dali_loader.TestPipeline(
            image_paths=["test_image.npy"],
            batch_size=1, num_threads=1, device_id=0,
            shard_id=0, seed=42, num_gpus=1,
        )
        mock_gpu = mock.MagicMock()
        mock_gpu.gpu.return_value = "gpu_image"
        pipeline.input_images = mock.MagicMock(return_value=mock_gpu)

        output = pipeline.define_graph()

        assert output == "transposed_image"
        mock_reshape.assert_called_once_with("gpu_image", layout="DHWC")
        mock_transpose.assert_called_once_with("reshaped_image", perm=[3, 0, 1, 2])


class TestEvalPipelineDefineGraph:
    """Tests for EvalPipeline.define_graph."""

    @mock.patch("mist.data_loading.dali_loader.fn.transpose", side_effect=["transposed_image", "transposed_label"])
    @mock.patch("mist.data_loading.dali_loader.fn.reshape", side_effect=["reshaped_image", "reshaped_label"])
    def test_returns_transposed_image_and_label(self, mock_reshape, mock_transpose):
        """Loads, reshapes, and transposes both image and label tensors."""
        pipeline = dali_loader.EvalPipeline(
            image_paths=["image1.npy"], label_paths=["label1.npy"],
            batch_size=1, num_threads=1, device_id=0,
            shard_id=0, seed=42, num_gpus=1,
        )
        mock_image_tensor = mock.MagicMock()
        mock_image_tensor.gpu.return_value = "gpu_image"
        pipeline.input_images = mock.MagicMock(return_value=mock_image_tensor)

        mock_label_tensor = mock.MagicMock()
        mock_label_tensor.gpu.return_value = "gpu_label"
        pipeline.input_labels = mock.MagicMock(return_value=mock_label_tensor)

        image, label = pipeline.define_graph()

        assert image == "transposed_image"
        assert label == "transposed_label"
        mock_reshape.assert_has_calls([
            mock.call("gpu_image", layout="DHWC"),
            mock.call("gpu_label", layout="DHWC"),
        ])
        mock_transpose.assert_has_calls([
            mock.call("reshaped_image", perm=[3, 0, 1, 2]),
            mock.call("reshaped_label", perm=[3, 0, 1, 2]),
        ])


class TestGetTrainingDataset:
    """Tests for dali_loader.get_training_dataset."""

    @pytest.mark.parametrize("dtm_paths,expected_keys", [
        (["dtm.npy"], ["image", "label", "dtm"]),
        (None, ["image", "label"]),
    ])
    @mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
    @mock.patch("mist.data_loading.dali_loader.TrainPipeline")
    @mock.patch("mist.data_loading.data_loading_utils.validate_train_and_eval_inputs")
    def test_returns_dali_iterator(
        self, mock_validate, mock_train_pipeline, mock_dali_iter,
        dtm_paths, expected_keys,
    ):
        """Returns a DALIGenericIterator with correct output keys."""
        result = dali_loader.get_training_dataset(
            image_paths=["image.npy"],
            label_paths=["label.npy"],
            dtm_paths=dtm_paths,
            batch_size=1,
            roi_size=(64, 64, 64),
            labels=[0, 1],
            oversampling=0.25,
            seed=42,
            num_workers=1,
            rank=0,
            world_size=1,
        )
        mock_validate.assert_called_once_with(["image.npy"], ["label.npy"], dtm_paths)
        mock_train_pipeline.assert_called_once()
        mock_dali_iter.assert_called_once_with(mock_train_pipeline.return_value, expected_keys)
        assert result == mock_dali_iter.return_value


class TestGetValidationDataset:
    """Tests for dali_loader.get_validation_dataset."""

    @mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
    @mock.patch("mist.data_loading.dali_loader.EvalPipeline")
    @mock.patch("mist.data_loading.data_loading_utils.validate_train_and_eval_inputs")
    def test_returns_dali_iterator(self, mock_validate, mock_pipeline, mock_dali_iter):
        """Returns a DALIGenericIterator with image and label keys."""
        result = dali_loader.get_validation_dataset(
            image_paths=["image.npy"],
            label_paths=["label.npy"],
            seed=123,
            num_workers=2,
            rank=0,
            world_size=1,
        )
        mock_validate.assert_called_once_with(["image.npy"], ["label.npy"])
        mock_pipeline.assert_called_once()
        mock_dali_iter.assert_called_once_with(mock_pipeline.return_value, ["image", "label"])
        assert result == mock_dali_iter.return_value


class TestGetTestDataset:
    """Tests for dali_loader.get_test_dataset."""

    @mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
    @mock.patch("mist.data_loading.dali_loader.TestPipeline")
    def test_returns_dali_iterator(self, mock_pipeline, mock_dali_iter):
        """Returns a DALIGenericIterator with image key."""
        result = dali_loader.get_test_dataset(
            image_paths=["image.npy"],
            seed=42, num_workers=1, rank=0, world_size=1,
        )
        mock_pipeline.assert_called_once()
        mock_dali_iter.assert_called_once_with(mock_pipeline.return_value, ["image"])
        assert result == mock_dali_iter.return_value

    def test_raises_for_empty_image_paths(self):
        """Raises ValueError when no images are provided."""
        with pytest.raises(ValueError, match="No images found!"):
            dali_loader.get_test_dataset(
                image_paths=[], seed=42, num_workers=1, rank=0, world_size=1,
            )
