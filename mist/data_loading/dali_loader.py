"""DALI loaders for loading data into models during training."""

from collections.abc import Sequence
from typing import List, Optional, Tuple

import numpy as np


from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.tensors import TensorCPU, TensorGPU
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from mist.data_loading.data_loading_constants import DataLoadingConstants as constants
import mist.data_loading.data_loading_utils as utils


class GenericPipeline(Pipeline):
    """Generic pipeline for loading images and labels using DALI.

    This pipeline is used for loading images and labels using DALI. It provides
    a base class for creating custom pipelines for training, validation, and
    testing.

    Attributes:
        input_image_files: DALI Numpy reader operator for reading images.
        input_label_files: DALI Numpy reader operator for reading labels.
        input_dtm: DALI Numpy reader operator for reading DTM data.
    """

    def __init__(
        self,
        batch_size: int,
        num_threads: int,
        device_id: int,
        shard_id: int,
        seed: int,
        num_gpus: int,
        shuffle_input: bool = True,
        input_image_files: Optional[List[str]] = None,
        input_label_files: Optional[List[str]] = None,
        input_dtm_files: Optional[List[str]] = None,
    ):
        """Initialize the pipeline with the given parameters.

        Args:
            batch_size: The batch size for data loading.
            num_threads: The number of threads for data loading.
            device_id: The ID of the current device (GPU).
            shard_id: The ID of the current shard, used for distributed data
                loading.
            seed: Random seed for shuffling or any other randomness in the
                reader.
            num_gpus: Total number of GPUs used for training.
            shuffle_input: Whether to shuffle the input data.
            input_x_files: List of file paths to the image data.
            input_y_files: List of file paths to the label data.
            input_dtm_files: List of file paths to the DTM data.
        """
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        # Initialize the input readers for images, labels, and DTM data.
        if input_image_files:
            if utils.is_valid_generic_pipeline_input(input_image_files):
                self.input_images = utils.get_numpy_reader(
                    files=input_image_files,
                    shard_id=shard_id,
                    seed=seed,
                    num_shards=num_gpus,
                    shuffle=shuffle_input,
                )
            else:
                raise ValueError(
                    "Input images are not valid. Please check the input paths."
                )
        if input_label_files:
            if utils.is_valid_generic_pipeline_input(input_label_files):
                self.input_labels = utils.get_numpy_reader(
                    files=input_label_files,
                    shard_id=shard_id,
                    seed=seed,
                    num_shards=num_gpus,
                    shuffle=shuffle_input,
                )
            else:
                raise ValueError(
                    "Input labels are not valid. Please check the input paths."
                )
        if input_dtm_files:
            if utils.is_valid_generic_pipeline_input(input_dtm_files):
                self.input_dtms = utils.get_numpy_reader(
                    files=input_dtm_files,
                    shard_id=shard_id,
                    seed=seed,
                    num_shards=num_gpus,
                    shuffle=shuffle_input,
                )
            else:
                raise ValueError(
                    "Input DTMs are not valid. Please check the input paths."
                )


class TrainPipeline(GenericPipeline):
    """Training pipeline for loading images, labels, and DTM data using DALI.

    Unlike the TrainPipeline, this pipeline includes DTM data in addition to the
    image and label data. The DTM data is used for training models that use
    boundary-based loss functions. Because we precompute the DTMs, this pipeline
    does not include as many augmentations as the TrainPipeline. The only
    augmentations applied are flips and augmentations that do not affect the
    label. These include noise, blur, brightness, and contrast.

    Attributes:
        labels: List of labels in the dataset.
        label_weights: Weighting for each label. This is a list with entries
            1/len(labels) for each label, giving equal weight to each label.
        oversampling: The oversampling factor for the training data.
        roi_size: The size of the region of interest (ROI) used for training.
        crop_shape: The shape of the cropped image data.
        crop_shape_float: The shape of the cropped image data as a float.
        has_dtms: Whether the pipeline includes DTM data.
        extract_patches: Whether to extract patches from the input data. If
            True, the pipeline extracts random patches from the input data for
            training. If False, the entire image is output for training. In
            this case, the images are expected to be the same size and the
            ROI size must be set to the size of the images.
        use_augmentations: Whether to apply any augmentations to the input data.
        use_flips: Whether to apply flips to the input data. More fine-grained
            control over augmentations can be achieved by setting this to True.
        use_zoom: Whether to apply zoom to the input data during augmentation.
        use_noise: Whether to apply noise to the input data during augmentation.
        use_blur: Whether to apply blur to the input data during augmentation.
        use_brightness: Whether to adjust brightness in the input data
            during augmentation.
        use_contrast: Whether to adjust contrast in the input data during
            augmentation.
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        dtm_paths: Optional[List[str]],
        roi_size: Tuple[int, int, int],
        labels: Optional[List[int]],
        oversampling: Optional[float],
        extract_patches: bool = True,
        use_augmentation: bool = True,
        use_flips: bool = True,
        use_zoom: bool = True,
        use_noise: bool = True,
        use_blur: bool = True,
        use_brightness: bool = True,
        use_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_image_files=image_paths,
            input_label_files=label_paths,
            input_dtm_files=dtm_paths,
            shuffle_input=True,
            **kwargs
        )
        self.has_dtms = dtm_paths is not None

        # If we are not extracting patches, then the entire image is output for
        # training. This is useful for models that do not require patch-based
        # training.
        self.extract_patches = extract_patches

        # Set ROI size. If we are not extracting patches, the ROI size should
        # match the image size. Otherwise, it should match the patch size.
        self.roi_size = roi_size

        # If we are extracting patches, we need to define the labels, label
        # weights, oversampling factor, and patch size. These are used for
        # biased cropping to extract patches from the input data and labels.
        if self.extract_patches:
            if labels:
                self.labels = labels
                self.label_weights = [
                    1./len(self.labels) for _ in range(len(self.labels))
                ]
            self.oversampling = oversampling
            self.crop_shape = types.Constant(
                np.array(self.roi_size), dtype=types.INT64
            )
            self.crop_shape_float = types.Constant(
                np.array(self.roi_size), dtype=types.FLOAT
            )

        # Define the augmentations to apply to the input data. If we are not
        # applying any augmentations, then the input data is returned
        # unmodified. Otherwise, we can control the augmentations applied using
        # the input flags.
        self.use_augmentation = use_augmentation
        if not self.use_augmentation:
            self.use_flips = False
            self.use_zoom = False
            self.use_noise = False
            self.use_blur = False
            self.use_brightness = False
            self.use_contrast = False
        else:
            self.use_flips = use_flips
            self.use_zoom = use_zoom
            self.use_noise = use_noise
            self.use_blur = use_blur
            self.use_brightness = use_brightness
            self.use_contrast = use_contrast

    def load_data(self):
        """Load the image, label, and DTM data from the input readers."""
        image = self.input_images(name="image_reader")
        image = fn.reshape(image, layout="DHWC")

        label = self.input_labels(name="label_reader")
        label = fn.reshape(label, layout="DHWC")

        if self.has_dtms:
            dtm = self.input_dtms(name="dtm_reader")
            dtm = fn.reshape(dtm, layout="DHWC")
            return image, label, dtm

        return image, label

    def biased_crop_fn(
            self,
            image: TensorCPU,
            label: TensorCPU,
            dtm: Optional[TensorCPU] = None,
    ) -> Sequence[TensorGPU]:
        """Extract a random patch from the image, label, and DTM.

        Perform a biased crop of the input image, label, and DTM, focusing on
        regions containing foreground objects. The image and label are padded to
        ensure that the patch size fits, and the crop is centered around regions
        of interest in the label data, typically objects in a segmentation task.

        Args:
            image: The input image data to be cropped.
            label: The input label data to be cropped, typically corresponding to
                segmentation masks.
            dtm: The input DTM data to be cropped.

        Returns:
            The cropped image, label and DTM, all transferred to the GPU for
                further processing.
        """
        # Pad the data to ensure their dimensions are at least the size of the
        # patch.
        image = fn.pad(image, axes=(0, 1, 2), shape=self.roi_size)
        label = fn.pad(label, axes=(0, 1, 2), shape=self.roi_size)
        if self.has_dtms:
            dtm = fn.pad(dtm, axes=(0, 1, 2), shape=self.roi_size)

        # Generate a region of interest (ROI) by identifying bounding boxes
        # around the foreground objects in the label. 'foreground_prob' controls
        # how often the crop is centered on objects rather than the background.
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            label,
            format="start_end",  # ROI format as (start, end) coordinates.
            background=0,        # Background pixel value to ignore.
            classes=self.labels,  # List of labels in the dataset.
            class_weights=self.label_weights,  # Class weights.
            foreground_prob=self.oversampling,  # Probability of foreground.
            device="cpu",        # Perform the operation on the CPU.
            cache_objects=True,  # Cache object locations for efficiency.
        )

        # Randomly select an anchor point within the identified ROI for
        # cropping. The crop shape is the patch size plus one channel. Here the
        # the image and label are still in DHWC format. We will change this
        # later to CDHW for PyTorch compatibility.
        anchor = fn.roi_random_crop(
            label,
            roi_start=roi_start,
            roi_end=roi_end,
            crop_shape=[*self.roi_size, 1],
        )

        # Slice the anchor to drop the channel dimension
        # (keeping only spatial dimensions).
        anchor = fn.slice(anchor, 0, 3, axes=[0])

        # Crop the image, label, and optionally the DTM based on the selected
        # anchor point and the patch size. The 'out_of_bounds_policy' ensures
        # the crop is padded if it exceeds the image bounds.
        if self.has_dtms:
            image, label, dtm = fn.slice(
                [image, label, dtm],
                anchor,
                self.crop_shape,  # Crop size matches the desired patch size.
                axis_names="DHW",  # Perform cropping along DWH axes.
                out_of_bounds_policy="pad",  # Pad out-of-bounds regions.
                device="cpu",  # Perform this on the CPU before moving to GPU.
            )

            # Return the cropped image, label, and dtm transferred to the GPU
            # for further processing.
            return image.gpu(), label.gpu(), dtm.gpu()

        # If no DTM data is provided, only crop the image and label.
        image, label = fn.slice(
            [image, label],
            anchor,
            self.crop_shape,  # Crop size matches the desired patch size.
            axis_names="DHW",  # Perform cropping along DWH axes.
            out_of_bounds_policy="pad",  # Pad out-of-bounds regions.
            device="cpu",  # Perform this on the CPU before moving to GPU.
        )

        # Return the cropped image and label, transferred to the GPU for
        # further processing.
        return image.gpu(), label.gpu()

    def flips_fn(
        self,
        image: TensorGPU,
        label: TensorGPU,
        dtm: Optional[TensorGPU]=None,
    ) -> Sequence[TensorGPU]:
        """Apply random flips to the input image, labels, and DTMs.

        Apply random flips to the input data. The flips can be applied
        horizontally, vertically, or depthwise with a 0.5 probability.

        Args:
            image: The input image data to apply flips to.
            label: The input label data to apply the same flips as the image.
            dtm: The input DTM data to apply the same flips as the image.

        Returns:
            The flipped image, label, and DTM data.
        """
        # Define the flip options for horizontal, vertical, and depthwise flips.
        kwargs = {
            "horizontal": (
                fn.random.coin_flip(
                    probability=constants.HORIZONTAL_FLIP_PROBABILITY
                )
            ),
            "vertical": (
                fn.random.coin_flip(
                    probability=constants.VERTICAL_FLIP_PROBABILITY
                )
            ),
            "depthwise": (
                fn.random.coin_flip(
                    probability=constants.DEPTH_FLIP_PROBABILITY
                )
            ),
        }

        # Apply the flips to the image, label, and DTM data and return the
        # results.
        flipped_image = fn.flip(image, **kwargs)
        flipped_label = fn.flip(label, **kwargs)
        if self.has_dtms:
            flipped_dtm = fn.flip(dtm, **kwargs)
            return flipped_image, flipped_label, flipped_dtm
        return flipped_image, flipped_label

    def zoom_fn(
        self,
        image: TensorGPU,
        label: TensorGPU,
    ) -> Tuple[TensorGPU, TensorGPU]:
        """Apply a random zoom to the input image and labels.

        Apply a random zoom to the input image and label by scaling them down
        and then resizing them back to the original patch size. The zoom factor
        is randomly selected between 0.7 and 1.0 with a probability of 0.15.
        This augmentation simulates zooming into different parts of the image
        while maintaining the overall image size.

        Args:
            image: The input image tensor to apply zoom to.
            label: The input label tensor, typically segmentation labels, to apply
                the same zoom as the image.

        Returns:
            The zoomed and resized image and label.
        """
        # Randomly choose a scaling factor between 0.7 and 1.0 with a 0.15
        # probability of applying the augmentation. If not applied, the scale
        # remains 1.0.
        scale = utils.random_augmentation(
            constants.ZOOM_FN_PROBABILITY,
            fn.random.uniform(
                range=(
                    constants.ZOOM_FN_RANGE_MIN,
                    constants.ZOOM_FN_RANGE_MAX,
                )
            ),
            1.0,
        )

        # Compute the new dimensions (depth, height, width) based on the scaling
        # factor.
        d, h, w = [scale * x for x in self.roi_size]

        # Crop both the image and label using the new scaled dimensions.
        image = fn.crop(image, crop_h=h, crop_w=w, crop_d=d)
        label = fn.crop(label, crop_h=h, crop_w=w, crop_d=d)

        # Resize the cropped image and label back to the original patch size.
        # Use cubic interpolation for the image and nearest neighbor for the
        # label to maintain the segmentation mask.
        image = fn.resize(
            image,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            size=self.crop_shape_float,
        )
        label = fn.resize(
            label,
            interp_type=types.DALIInterpType.INTERP_NN,
            size=self.crop_shape_float,
        )

        # Return the resized image and label.
        return image, label

    def define_graph(self):
        """Define the training pipeline graph for data loading.

        This function defines the training pipeline graph for loading and
        augmenting the image, label, and DTM data. The pipeline starts by
        loading the data from the input readers. It then applies biased
        cropping to extract a random patch from the image and label data. The
        bias cropping function also transfers the extracted patches to the GPU.
        Next, the pipeline applies a series of random augmentations to the image
        including zooming, flips, adding noise, blurring, adjusting brightness,
        and changing contrast. The final image, label, and DTM data are then
        transposed to CDHW format for PyTorch compatibility.
        """
        # Load images, labels, and possibly DTMs. Apply biased cropping to the
        # image, label, and DTM data. Transfer the cropped patches to the GPU.
        # Apply flips and zooming for additional augmentation.
        if self.has_dtms:
            image, label, dtm = self.load_data()
            if self.extract_patches:
                image, label, dtm = self.biased_crop_fn(image, label, dtm)

            if self.use_augmentation and self.use_flips:
                image, label, dtm = self.flips_fn(image, label, dtm)
        else:
            image, label = self.load_data()
            if self.extract_patches:
                image, label = self.biased_crop_fn(image, label)

            if self.use_augmentation:
                if self.use_zoom:
                    image, label = self.zoom_fn(image, label)

                if self.use_flips:
                    image, label = self.flips_fn(image, label)

        # Apply random augmentations to the image data only.
        if self.use_augmentation:
            if self.use_noise:
                image = utils.noise_fn(image)
            if self.use_blur:
                image = utils.blur_fn(image)
            if self.use_brightness:
                image = utils.brightness_fn(image)
            if self.use_contrast:
                image = utils.contrast_fn(image)

        # Change format to CDWH for PyTorch compatibility.
        image = fn.transpose(image, perm=[3, 0, 1, 2])
        label = fn.transpose(label, perm=[3, 0, 1, 2])
        if self.has_dtms:
            dtm = fn.transpose(dtm, perm=[3, 0, 1, 2])
            return image, label, dtm
        return image, label


class TestPipeline(GenericPipeline):
    """Test pipeline for loading images using DALI.

    This pipeline is used for loading images during testing. It does not include
    any patch extraction or augmentation. This pipeline simply streams the
    images from the input readers and transposes them to CDHW format for PyTorch
    compatibility.
    """

    def __init__(
        self,
        image_paths: List[str],
        **kwargs,
    ):
        super().__init__(
            input_image_files=image_paths,
            input_label_files=None,
            shuffle_input=False,  # Do not shuffle the input data.
            **kwargs
        )

    def define_graph(self):
        """Define the test pipeline graph for data loading."""
        # Load the image data from the input reader and transfer it to the GPU.
        image = self.input_images(name="image_reader").gpu()

        # Reshape the image data to DHWC format.
        image = fn.reshape(image, layout="DHWC")

        # Change format to CDHW for pytorch compatibility
        image = fn.transpose(image, perm=[3, 0, 1, 2])

        return image


class EvalPipeline(GenericPipeline):
    """Evaluation pipeline for loading images and labels using DALI.

    This pipeline is used for loading images and labels during evaluation. It
    does not include any patch extraction or augmentation. This pipeline simply
    streams the images and labels from the input readers and transposes them to
    CDHW format for PyTorch compatibility.
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        **kwargs,
    ):
        super().__init__(
            input_image_files=image_paths,
            input_label_files=label_paths,
            shuffle_input=False,
            **kwargs
        )

    def define_graph(self):
        """Define the evaluation pipeline graph for data loading."""
        # Load the image and label data from the input readers and transfer them
        # to the GPU.
        image = self.input_images(name="image_reader").gpu()
        image = fn.reshape(image, layout="DHWC")

        label = self.input_labels(name="label_reader").gpu()
        label = fn.reshape(label, layout="DHWC")

        # Change format to CDHW for pytorch compatibility.
        image = fn.transpose(image, perm=[3, 0, 1, 2])
        label = fn.transpose(label, perm=[3, 0, 1, 2])

        return image, label


def get_training_dataset(
    image_paths: List[str],
    label_paths: List[str],
    dtm_paths: Optional[List[str]],
    batch_size: int,
    roi_size: Tuple[int, int, int],
    labels: Optional[List[int]],
    oversampling: Optional[float],
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
    extract_patches: bool = True,
    use_augmentation: bool = True,
    use_flips: bool = True,
    use_zoom: bool = True,
    use_noise: bool = True,
    use_blur: bool = True,
    use_brightness: bool = True,
    use_contrast: bool = True,
) -> DALIGenericIterator:
    """Retrieve the appropriate training pipeline based on the input data.

    This function returns a DALI training pipeline for loading images, labels,
    and DTM data during training. The pipeline includes patch extraction and
    augmentation to prepare the data for training. The pipeline is configured
    based on the input parameters, including the data files, batch size,
    oversampling factor, patch size, random seed, number of workers, and the
    current rank and total number of GPUs.

    Args:
        image_paths: List of file paths to the image data.
        label_paths: List of file paths to the label data.
        dtm_paths: List of file paths to the DTM data.
        batch_size: The batch size for training.
        roi_size: The size of the region of interest (ROI) used for training.
        extract_patches: Whether to extract patches from the input data. If
            True, the pipeline extracts random patches from the input data for
            training. If False, the entire image is output for training. In this
            case, the images are expected to be the same size and the ROI size
            must be set to the size of the images.
        labels: List of labels in the dataset.
        oversampling: The oversampling factor for the training data when
            extracting patches.
        use_augmentations: Whether to apply any augmentations to the input data.
        use_flips: Whether to apply flips to the input data. More fine-grained
            control over augmentations can be achieved by setting this to True.
        use_zoom: Whether to apply zoom to the input data during augmentation.
        use_noise: Whether to apply noise to the input data during augmentation.
        use_blur: Whether to apply blur to the input data during augmentation.
        use_brightness: Whether to adjust brightness in the input data during
            augmentation.
        use_contrast: Whether to adjust contrast in the input data during
            augmentation.
        seed: Random seed for shuffling or any other randomness in the reader.
        num_workers: The number of workers for data loading.
        rank: The rank of the current process (GPU).
        world_size: The total number of GPUs used for training.

    Returns:
        dali_iter: A DALI iterator for training data loading.

    Raises:
        AssertionError: If the input data is invalid or missing.
    """
    # Check that inputs are valid.
    utils.validate_train_and_eval_inputs(image_paths, label_paths, dtm_paths)

    # Configure the DALI pipeline based on the input parameters.
    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": batch_size,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank,
    }

    # Create the training pipeline with the specified parameters.
    pipeline = TrainPipeline(
        image_paths=image_paths,
        label_paths=label_paths,
        dtm_paths=dtm_paths,
        roi_size=roi_size,
        labels=labels,
        oversampling=oversampling,
        extract_patches=extract_patches,
        use_augmentation=use_augmentation,
        use_flips=use_flips if use_augmentation else False,
        use_zoom=use_zoom if use_augmentation else False,
        use_noise=use_noise if use_augmentation else False,
        use_blur=use_blur if use_augmentation else False,
        use_brightness=use_brightness if use_augmentation else False,
        use_contrast=use_contrast if use_augmentation else False,
        **pipe_kwargs
    )

    # Return a DALI iterator for the training data. If DTM data is provided,
    # include it in the iterator. Otherwise, return an iterator without DTMs.
    if dtm_paths:
        return DALIGenericIterator(pipeline, ["image", "label", "dtm"])
    return DALIGenericIterator(pipeline, ["image", "label"])


def get_validation_dataset(
    image_paths: List[str],
    label_paths: List[str],
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> DALIGenericIterator:
    """Build a DALI validation pipeline for loading images and labels.

    This function returns a DALI validation pipeline for loading images and
    labels during validation. The pipeline does not include patch extraction or
    augmentation. It simply streams the images and labels from the input readers
    to the GPU to prepare the data for evaluation. The pipeline is configured
    based on the input parameters, including the data files, random seed, number
    of workers, and the current rank and total number of GPUs.

    Args:
        image_paths: List of file paths to the image data.
        label_paths: List of file paths to the label data.
        seed: Random seed for shuffling or any other randomness in the reader.
        num_workers: The number of workers for data loading.
        rank: The rank of the current process (GPU).
        world_size: The total number of GPUs used for training.

    Returns:
        dali_iter: A DALI iterator for validation data loading.
    """
    # Check that inputs are valid.
    utils.validate_train_and_eval_inputs(image_paths, label_paths)

    # Configure the DALI pipeline based on the input parameters.
    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": 1,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank
    }

    pipeline = EvalPipeline(image_paths, label_paths, **pipe_kwargs)
    dali_iter = DALIGenericIterator(pipeline, ["image", "label"])
    return dali_iter


def get_test_dataset(
    image_paths: List[str],
    seed: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
) -> DALIGenericIterator:
    """"Build a DALI test pipeline for loading images.

    This function returns a DALI test pipeline for loading images during
    testing. The pipeline does not include patch extraction or augmentation. It
    simply streams the images from the input readers to the GPU. This pipeline
    is configured based on the input parameters, including the data files,
    random seed, number of workers, and the current rank and total number of
    GPUs. Note that this pipeline will only use one GPU, hence the default rank
    and world size values are set to 0 and 1, respectively.

    Args:
        image_paths: List of file paths to the image data.
        seed: Random seed for shuffling or any other randomness in the reader.
        num_workers: The number of workers for data loading.
        rank: The rank of the current process (GPU). Defaults to 0.

    Returns:
        dali_iter: A DALI iterator for test data loading.

    Raises:
        ValueError: If no images are found in the input data.
    """
    # Check that inputs are valid.
    if len(image_paths) == 0:
        raise ValueError("No images found!")

    # Configure the DALI pipeline based on the input parameters.
    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": 1,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank
    }

    pipeline = TestPipeline(image_paths, **pipe_kwargs)
    dali_iter = DALIGenericIterator(pipeline, ["image"])
    return dali_iter
