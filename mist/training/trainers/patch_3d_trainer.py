"""3D patch trainer for MIST built on top of BaseTrainer."""

from typing import Tuple, Any, Dict

import torch
from monai.inferers import sliding_window_inference

from mist.data_loading import dali_loader
from mist.training.trainers.base_trainer import BaseTrainer
from mist.training.trainers.trainer_constants import TrainerConstants as constants


class Patch3DTrainer(BaseTrainer):
    """Trainer for 3D patch-based training in MIST."""

    def build_dataloaders(
        self,
        fold_data: Dict[str, Any],
        rank: int,
        world_size: int,
    ) -> Tuple[Any, Any]:
        """Build DALI dataloaders for training and validation.

        This method constructs the DALI-based training and validation data
        loaders for MIST's 3D patch training. It uses the provided fold
        information to access the correct image and label paths, and applies
        the necessary transformations and augmentations as specified in the
        configuration.

        Args:
            fold_data: Dictionary containing the following key-value pairs:
                - "train_images": List of paths to training images.
                - "train_labels": List of paths to training labels.
                - "train_dtms": List of paths to DTM images (optional).
                - "val_images": List of paths to validation images.
                - "val_labels": List of paths to validation labels.
            rank: The rank of the current process in distributed training.
            world_size: The total number of processes in distributed training.

        Returns:
            Tuple containing:
                - train_loader: DALI training data loader. This loader will
                    yield batches of randomly sampled patches from the training
                    images, applying the specified augmentations.
                - val_loader: DALI validation data loader. This loader will
                    yield full images for validation, without random sampling.
        """
        # Build training loader.
        training = self.config["training"]
        train_labels = self.config["dataset_info"]["labels"][1:]
        train_loader = dali_loader.get_training_dataset(
            extract_patches=True,
            image_paths=fold_data["train_images"],
            label_paths=fold_data["train_labels"],
            dtm_paths=fold_data["train_dtms"],
            batch_size=training["batch_size_per_gpu"],
            oversampling=training["dali_foreground_prob"],
            labels=train_labels,
            roi_size=self.config["spatial_config"]["patch_size"],
            seed=training["seed"],
            num_workers=training["hardware"]["num_cpu_workers"],
            use_augmentation=training["augmentation"]["enabled"],
            use_flips=training["augmentation"]["transforms"]["flips"],
            use_blur=training["augmentation"]["transforms"]["blur"],
            use_noise=training["augmentation"]["transforms"]["noise"],
            use_brightness=training["augmentation"]["transforms"]["brightness"],
            use_contrast=training["augmentation"]["transforms"]["contrast"],
            use_zoom=training["augmentation"]["transforms"]["zoom"],
            rank=rank,
            world_size=world_size,
        )

        # Build validation loader.
        val_loader = dali_loader.get_validation_dataset(
            image_paths=fold_data["val_images"],
            label_paths=fold_data["val_labels"],
            seed=training["seed"],
            num_workers=training["hardware"]["num_cpu_workers"],
            rank=rank,
            world_size=world_size,
        )
        return train_loader, val_loader

    def training_step(self, **kwargs) -> torch.Tensor:
        """Perform a single training step.

        This method executes a training step by performing a forward pass
        through the model, computing the loss, and returning the loss value.

        Args:
            **kwargs: Keyword arguments for the training step. For this specific
                implementation, it expects the following keys:
                - "state": Dictionary containing the current training state,
                    including the model and optimizer.
                - "data": Batch of data containing the input images and labels.

        Returns:
            The training loss.
        """
        # Unpack the state and batch from kwargs.
        state = kwargs["state"]
        batch = kwargs["data"]

        # Unpack the state. This includes the model, optimizer, scaler, loss
        # function (i.e., criterion), and the alpha weight for composite losses
        # (pre-computed once per epoch by the training loop).
        model = state["model"]
        optimizer = state["optimizer"]
        scaler = state["scaler"]
        criterion = state["loss_function"]
        alpha = state["alpha"]

        image = batch["image"]
        label = batch["label"]
        dtm = batch.get("dtm", None)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(image)

                y_deep_supervision = output.get("deep_supervision", None)
                loss = criterion(
                    y_true=label,
                    y_pred=output["prediction"],
                    y_supervision=y_deep_supervision,
                    alpha=alpha,
                    dtm=dtm,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=constants.GRAD_CLIP_VALUE
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(image)

            y_deep_supervision = output.get("deep_supervision", None)
            loss = criterion(
                y_true=label,
                y_pred=output["prediction"],
                y_supervision=y_deep_supervision,
                alpha=alpha,
                dtm=dtm,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=constants.GRAD_CLIP_VALUE
            )
            optimizer.step()
        return loss

    def validation_step(self, **kwargs) -> torch.Tensor:
        """Perform a single validation step.

        This method executes a validation step by performing a sliding window
        inference on the validation images, computing the loss, and returning
        the loss value.

        Args:
            **kwargs: Keyword arguments for the validation step. For this
                specific implementation, it expects the following keys:
                - "state": Dictionary containing the current training state,
                    including the model.
                - "data": Batch of data containing the input images and labels.

        Returns:
            The validation loss.
        """
        # Unpack the state and batch from kwargs.
        state = kwargs["state"]
        batch = kwargs["data"]

        # Unpack the model from the state.
        model = state["model"]
        patch_size = self.config["spatial_config"]["patch_size"]

        # Unpack overlap parameters from the configuration.
        overlap = self.config["inference"]["inferer"]["params"]["patch_overlap"]

        # Unpack the batch. This includes the input image and label.
        # Note that the input image is expected to be a full image, not patches.
        # The label is also expected to be the full label image.
        image = batch["image"]
        label = batch["label"]

        # Perform sliding window inference for validation.
        pred = sliding_window_inference(
            inputs=image,
            roi_size=patch_size,
            overlap=overlap,
            sw_batch_size=1,
            predictor=model,
            device=image.device,
        )

        # Compute the loss using the criterion.
        return self.validation_loss(label, pred)
