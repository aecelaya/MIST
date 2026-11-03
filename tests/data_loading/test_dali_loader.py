"""Tests for the DALI data loading module in MIST."""

from unittest import mock

import pytest

import mist.data_loading.data_loading_utils as utils
from mist.data_loading import dali_loader


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
        assert pipeline.patch_margin == (0, 0, 0)

    @mock.patch("mist.data_loading.dali_loader.types.Constant")
    def test_crop_shape_includes_margin(self, mock_constant, base_pipeline_args):
        """The crop shape is the patch size plus the margin on both sides."""
        dali_loader.TrainPipeline(**{**base_pipeline_args, "patch_margin": (5, 0, 3)})
        crop_shape = mock_constant.call_args_list[0].args[0]
        margin = mock_constant.call_args_list[1].args[0]
        assert crop_shape.tolist() == [74, 64, 70]
        assert margin.tolist() == [5, 0, 3]

    @pytest.mark.parametrize(
        "bad_arg,expected_message",
        [
            ("image", "Input images are not valid"),
            ("label", "Input labels are not valid"),
            ("dtm", "Input DTMs are not valid"),
        ],
    )
    def test_invalid_inputs_raise(self, monkeypatch, bad_arg, expected_message, base_pipeline_args):
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
        self,
        mock_pad,
        mock_bbox,
        mock_crop,
        mock_slice,
        base_pipeline_args,
        use_dtm,
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
            ("gpu_output", "gpu_output", "gpu_output") if use_dtm else ("gpu_output", "gpu_output")
        )
        assert out == expected

    @mock.patch("mist.data_loading.dali_loader.fn.slice")
    @mock.patch("mist.data_loading.dali_loader.fn.roi_random_crop")
    @mock.patch("mist.data_loading.dali_loader.fn.segmentation.random_object_bbox")
    @mock.patch("mist.data_loading.dali_loader.fn.pad")
    def test_margin_shifts_anchor_and_enlarges_crop(
        self,
        mock_pad,
        mock_bbox,
        mock_crop,
        mock_slice,
        base_pipeline_args,
    ):
        """With a margin, the anchor moves back by it and the crop grows."""
        args = {**base_pipeline_args, "dtm_paths": None, "patch_margin": (4, 4, 4)}
        pipeline = dali_loader.TrainPipeline(**args)

        dummy = mock.MagicMock()
        anchor = mock.MagicMock()
        mock_pad.return_value = dummy
        mock_bbox.return_value = ("roi_start", "roi_end")
        mock_slice.side_effect = [anchor, (dummy, dummy)]

        pipeline.biased_crop_fn("image", "label")

        anchor.__sub__.assert_called_once_with(pipeline.margin)
        crop_call = mock_slice.call_args_list[1]
        assert crop_call.args[1] is anchor.__sub__.return_value
        assert crop_call.args[2] is pipeline.crop_shape


class TestDefineGraph:
    """Tests for TrainPipeline.define_graph."""

    @pytest.mark.parametrize("use_dtm", [False, True])
    @mock.patch(
        "mist.data_loading.dali_loader.fn.transpose",
        side_effect=lambda x, perm: f"transposed_{x}",
    )
    @mock.patch.object(dali_loader.TrainPipeline, "biased_crop_fn")
    @mock.patch.object(dali_loader.TrainPipeline, "load_data")
    def test_pipeline_output_with_and_without_dtm(
        self,
        mock_load_data,
        mock_biased_crop_fn,
        mock_transpose,
        base_pipeline_args,
        use_dtm,
    ):
        """define_graph crops and transposes, with no augmentation in DALI."""
        args = {**base_pipeline_args, "dtm_paths": ["dtm.npy"] if use_dtm else None}
        if use_dtm:
            mock_load_data.return_value = ("image_raw", "label_raw", "dtm_raw")
            mock_biased_crop_fn.return_value = ("cropped_image", "cropped_label", "cropped_dtm")
        else:
            mock_load_data.return_value = ("image_raw", "label_raw")
            mock_biased_crop_fn.return_value = ("cropped_image", "cropped_label")

        pipeline = dali_loader.TrainPipeline(**args)
        result = pipeline.define_graph()

        if use_dtm:
            assert result == (
                "transposed_cropped_image",
                "transposed_cropped_label",
                "transposed_cropped_dtm",
            )
        else:
            assert result == ("transposed_cropped_image", "transposed_cropped_label")


class TestTestPipelineDefineGraph:
    """Tests for TestPipeline.define_graph."""

    @mock.patch("mist.data_loading.dali_loader.fn.transpose", return_value="transposed_image")
    @mock.patch("mist.data_loading.dali_loader.fn.reshape", return_value="reshaped_image")
    def test_returns_transposed_image(self, mock_reshape, mock_transpose):
        """Loads, reshapes, and transposes the image tensor."""
        pipeline = dali_loader.TestPipeline(
            image_paths=["test_image.npy"],
            batch_size=1,
            num_threads=1,
            device_id=0,
            shard_id=0,
            seed=42,
            num_gpus=1,
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

    @mock.patch(
        "mist.data_loading.dali_loader.fn.transpose",
        side_effect=["transposed_image", "transposed_label"],
    )
    @mock.patch(
        "mist.data_loading.dali_loader.fn.reshape",
        side_effect=["reshaped_image", "reshaped_label"],
    )
    def test_returns_transposed_image_and_label(self, mock_reshape, mock_transpose):
        """Loads, reshapes, and transposes both image and label tensors."""
        pipeline = dali_loader.EvalPipeline(
            image_paths=["image1.npy"],
            label_paths=["label1.npy"],
            batch_size=1,
            num_threads=1,
            device_id=0,
            shard_id=0,
            seed=42,
            num_gpus=1,
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
        mock_reshape.assert_has_calls(
            [
                mock.call("gpu_image", layout="DHWC"),
                mock.call("gpu_label", layout="DHWC"),
            ]
        )
        mock_transpose.assert_has_calls(
            [
                mock.call("reshaped_image", perm=[3, 0, 1, 2]),
                mock.call("reshaped_label", perm=[3, 0, 1, 2]),
            ]
        )


class TestGetTrainingDataset:
    """Tests for dali_loader.get_training_dataset."""

    @pytest.mark.parametrize(
        "dtm_paths,expected_keys",
        [
            (["dtm.npy"], ["image", "label", "dtm"]),
            (None, ["image", "label"]),
        ],
    )
    @mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
    @mock.patch("mist.data_loading.dali_loader.TrainPipeline")
    @mock.patch("mist.data_loading.data_loading_utils.validate_train_and_eval_inputs")
    def test_returns_dali_iterator(
        self,
        mock_validate,
        mock_train_pipeline,
        mock_dali_iter,
        dtm_paths,
        expected_keys,
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
            patch_margin=(2, 2, 2),
        )
        mock_validate.assert_called_once_with(["image.npy"], ["label.npy"], dtm_paths)
        mock_train_pipeline.assert_called_once()
        _, call_kwargs = mock_train_pipeline.call_args
        assert call_kwargs["patch_margin"] == (2, 2, 2)
        assert "dimension" not in call_kwargs, (
            "dimension was removed from TrainPipeline; must not be passed at call site"
        )
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
            seed=42,
            num_workers=1,
            rank=0,
            world_size=1,
        )
        mock_pipeline.assert_called_once()
        mock_dali_iter.assert_called_once_with(mock_pipeline.return_value, ["image"])
        assert result == mock_dali_iter.return_value

    def test_raises_for_empty_image_paths(self):
        """Raises ValueError when no images are provided."""
        with pytest.raises(ValueError, match="No images found!"):
            dali_loader.get_test_dataset(
                image_paths=[],
                seed=42,
                num_workers=1,
                rank=0,
                world_size=1,
            )
