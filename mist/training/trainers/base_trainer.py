"""Base trainer class for MIST."""

from typing import Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
import contextlib
import copy
import os
import math
import random

import numpy as np
import pandas as pd
import rich
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from mist.utils import io, progress_bar
from mist.utils.console import (
    print_section_header,
    print_info,
    print_warning,
    print_error,
    print_success,
)
from mist.models.model_registry import get_model_from_registry
from mist.models.model_loader import (
    load_pretrained_encoder,
    validate_encoder_compatibility,
)
from mist.loss_functions.loss_registry import get_loss
from mist.loss_functions.deep_supervision_wrapper import DeepSupervisionLoss
from mist.loss_functions.losses.dice import DiceLoss
from mist.loss_functions.alpha_schedulers import (
    get_alpha_scheduler,
    get_default_scheduler_config,
)
from mist.training.lr_schedulers.lr_scheduler_registry import get_lr_scheduler
from mist.training.optimizers.optimizer_registry import get_optimizer
from mist.training import training_utils
from mist.training.trainers.trainer_constants import TrainerConstants


class BaseTrainer(ABC):
    """Base trainer that provides common functionality for training models."""

    def __init__(self, mist_args):
        """Initialize the trainer class."""
        self.mist_args = mist_args
        self.results_dir = Path(self.mist_args.results).expanduser().resolve()
        self.numpy_dir = Path(self.mist_args.numpy).expanduser().resolve()
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Required files.
        paths_csv = self.results_dir / "train_paths.csv"
        self.paths = pd.read_csv(paths_csv)

        self.config_json = self.results_dir / "config.json"
        self.config = io.read_json_file(self.config_json)

        # Snapshot config before CLI overrides so resume checks can compare.
        _original_config = copy.deepcopy(self.config)

        # Merge command line overrides into the config.
        self._overwrite_config_from_args()

        # Set up console early so _check_resume_overrides can print.
        self.console = rich.console.Console()

        # Warn or raise if --resume is combined with config-altering overrides.
        if getattr(self.mist_args, "resume", False):
            self._check_resume_overrides(_original_config)

        # Resolve hardware configuration.
        self._update_num_gpus_in_config()

        # Set up batch size based on the number of GPUs.
        self.batch_size = (
            self.config["training"]["batch_size_per_gpu"] *
            max(1, self.config["training"]["hardware"]["num_gpus"])
        )

        # Prepare folds.
        self._setup_folds()

        # Set up validation loss.
        self.validation_loss = DiceLoss(exclude_background=True)

        # Validate pretrained encoder config if provided. This runs once in
        # the main process before any spawned training processes start, so
        # incompatible configs fail fast.
        self._validate_pretrained_config()

    def __getstate__(self):
        """Customize pickling so mp.spawn can serialize this object."""
        state = self.__dict__.copy()
        # Drop unpicklable attributes (contain thread locks, file handles, etc.)
        state["console"] = None
        return state

    def __setstate__(self, state):
        """Recreate transient/unpicklable attributes after unpickling."""
        self.__dict__.update(state)
        # Recreate the console on the child process.
        if self.__dict__.get("console") is None:
            self.console = rich.console.Console()

    @abstractmethod
    def build_dataloaders(self, fold_data, rank, world_size):
        """Abstract method to build data loaders for training and validation."""
        raise NotImplementedError(  # pragma: no cover
            "build_dataloaders method must be implemented in the subclass."
        )

    @abstractmethod
    def training_step(self, **kwargs):
        """Abstract method for training step."""
        raise NotImplementedError(  # pragma: no cover
            "training_step method must be implemented in the subclass."
        )

    @abstractmethod
    def validation_step(self, **kwargs):
        """Abstract method for validation step."""
        raise NotImplementedError(  # pragma: no cover
            "validation_step method must be implemented in the subclass."
        )

    def _set_seed(self, seed: int) -> None:
        """Seed Python, NumPy, and PyTorch RNGs (DDP-aware).

        Uses the provided base `seed`, offset by the process rank so that each 
        rank gets a distinct but reproducible RNG stream. Also sets
        PYTHONHASHSEED for consistent hashing across runs.
        """
        # Determine rank (works before/after dist.init_process_group).
        rank = int(os.environ.get("RANK", 0))
        try:
            if dist.is_initialized():
                rank = dist.get_rank()
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        final_seed = int(seed) + int(rank)

        # Python and env.
        os.environ["PYTHONHASHSEED"] = str(final_seed)
        random.seed(final_seed)

        # NumPy.
        np.random.seed(final_seed)

        # PyTorch (CPU & CUDA).
        torch.manual_seed(final_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(final_seed)
            torch.cuda.manual_seed_all(final_seed)

    def _overwrite_config_from_args(self):
        """Overwrite certain config parameters from command line arguments."""
        # Overwrite model settings from command line arguments.
        # Overwrite the architecture if if is specified in command line
        # arguments.
        if self.mist_args.model is not None:
            self.config["model"]["architecture"] = self.mist_args.model

        # Overwrite the patch size if it is specified in command line arguments.
        if self.mist_args.patch_size is not None:
            self.config["spatial_config"]["patch_size"] = [
                int(dim) for dim in self.mist_args.patch_size
            ]

        # Overwrite training parameters from command line arguments.
        # If the user specified to run only on certain folds, then update the
        # configuration with which folds to use.
        if self.mist_args.folds is not None:
            # Check that the folds are a subset of the enumeration of the
            # total number of folds.
            if not set(self.mist_args.folds).issubset(
                set(range(self.config["training"]["nfolds"]))
            ):
                raise ValueError(
                    "Folds specified in command line arguments must be a "
                    "subset of [0, 1, ..., nfolds-1]. Found folds: "
                    f"{self.mist_args.folds}."
                )
            self.config["training"]["folds"] = [
                int(fold) for fold in self.mist_args.folds
            ]

        # If the user specifies a number of epochs, then update the
        # configuration with the number of epochs.
        if self.mist_args.epochs is not None:
            self.config["training"]["epochs"] = int(self.mist_args.epochs)

        # If the user specifies a different batch size to use on each GPU,
        # then update the configuration with the new batch size.
        if self.mist_args.batch_size_per_gpu is not None:
            self.config["training"]["batch_size_per_gpu"] = int(
                self.mist_args.batch_size_per_gpu
            )

        # Overwrite the loss function and its parameters if specified in
        # command line arguments.
        # Overwrite the loss function name.
        if self.mist_args.loss is not None:
            self.config["training"]["loss"]["name"] = self.mist_args.loss

        # Overwrite the composite loss weighting if specified. Populate with
        # scheduler defaults so users can see and tune all available params.
        if self.mist_args.composite_loss_weighting is not None:
            self.config["training"]["loss"]["composite_loss_weighting"] = (
                get_default_scheduler_config(self.mist_args.composite_loss_weighting)
            )

        # Overwrite the optimizer and its parameters if specified in command
        # line arguments.
        if self.mist_args.optimizer is not None:
            self.config["training"]["optimizer"] = self.mist_args.optimizer

        if self.mist_args.l2_penalty is not None:
            self.config["training"]["l2_penalty"] = float(
                self.mist_args.l2_penalty
            )

        # Overwrite the learning rate scheduler and its parameters if specified
        # in command line arguments.
        if self.mist_args.learning_rate is not None:
            self.config["training"]["learning_rate"] = float(
                self.mist_args.learning_rate
            )

        if self.mist_args.lr_scheduler is not None:
            self.config["training"]["lr_scheduler"] = (
                self.mist_args.lr_scheduler
            )

        if getattr(self.mist_args, "warmup_epochs", None) is not None:
            self.config["training"]["warmup_epochs"] = int(
                self.mist_args.warmup_epochs
            )

        # Overwrite the validation percentage if specified in command line.
        if self.mist_args.val_percent is not None:
            self.config["training"]["val_percent"] = float(
                self.mist_args.val_percent
            )

        # Write the updated configuration to the config.json file.
        io.write_json_file(self.config_json, self.config)

    def _check_resume_overrides(self, original_config: Dict[str, Any]) -> None:
        """Warn or raise when --resume is combined with config-altering flags.

        Architecture and patch-size changes are incompatible with a saved
        checkpoint (weights won't load) and raise immediately. Changes to
        loss, optimizer, or learning-rate schedule alter training dynamics
        mid-run and emit a yellow warning so the user can make an informed
        decision.

        Args:
            original_config: Deep copy of the config as it was on disk before
                any CLI overrides were applied.
        """
        new = self.config
        old = original_config

        # --- Hard errors: checkpoint weights will not load ----------------
        incompatible = []
        old_arch = old["model"]["architecture"]
        new_arch = new["model"]["architecture"]
        if old_arch != new_arch:
            incompatible.append(
                f"  --model: '{old_arch}' → '{new_arch}' "
                f"(checkpoint weights are incompatible)"
            )

        old_patch = old["spatial_config"]["patch_size"]
        new_patch = new["spatial_config"]["patch_size"]
        if old_patch != new_patch:
            incompatible.append(
                f"  --patch-size: {old_patch} → {new_patch} "
                f"(checkpoint weights are incompatible)"
            )

        if incompatible:
            raise ValueError(
                "--resume is set but the following overrides are incompatible "
                "with the saved checkpoint:\n"
                + "\n".join(incompatible)
                + "\nRemove these flags or start a fresh run with --overwrite."
            )

        # --- Warnings: training dynamics change mid-run -------------------
        warnings = []
        tr_old = old["training"]
        tr_new = new["training"]

        if tr_old["loss"]["name"] != tr_new["loss"]["name"]:
            warnings.append(
                f"  --loss: '{tr_old['loss']['name']}' → "
                f"'{tr_new['loss']['name']}'"
            )

        old_clw = tr_old["loss"]["composite_loss_weighting"]
        new_clw = tr_new["loss"]["composite_loss_weighting"]
        if old_clw != new_clw:
            warnings.append(
                f"  --composite-loss-weighting: {old_clw} → {new_clw}"
            )

        if tr_old["optimizer"] != tr_new["optimizer"]:
            warnings.append(
                f"  --optimizer: '{tr_old['optimizer']}' → "
                f"'{tr_new['optimizer']}'"
            )

        if tr_old["learning_rate"] != tr_new["learning_rate"]:
            warnings.append(
                f"  --learning-rate: {tr_old['learning_rate']} → "
                f"{tr_new['learning_rate']}"
            )

        if tr_old["lr_scheduler"] != tr_new["lr_scheduler"]:
            warnings.append(
                f"  --lr-scheduler: '{tr_old['lr_scheduler']}' → "
                f"'{tr_new['lr_scheduler']}'"
            )

        old_warmup = tr_old.get("warmup_epochs", 0)
        new_warmup = tr_new.get("warmup_epochs", 0)
        if old_warmup != new_warmup:
            warnings.append(
                f"  --warmup-epochs: {old_warmup} → {new_warmup}"
            )

        if tr_old["l2_penalty"] != tr_new["l2_penalty"]:
            warnings.append(
                f"  --l2-penalty: {tr_old['l2_penalty']} → "
                f"{tr_new['l2_penalty']}"
            )

        if warnings:
            print_warning(
                "--resume is set but the following overrides change training "
                "configuration from the saved run. Training dynamics may be "
                "inconsistent from the resumed epoch onwards."
            )
            for w in warnings:
                print_info(w)
            print_info("")

    def _update_num_gpus_in_config(self) -> None:
        """Get the number of GPUs and add it to the configuration."""
        # Fast, explicit CUDA checks.
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available. Ensure the CUDA toolkit/driver matches "
                "your PyTorch build, and that you're running on a GPU host."
            )

        # If CUDA is available, check the number of GPUs. If the number of GPUs
        # is zero, raise an error.
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise ValueError(
                "torch.cuda.is_available() is True but device_count() == 0. "
                "Check CUDA_VISIBLE_DEVICES or container runtime GPU flags"
            )

        # If there are GPUs available, update the configuration with the number
        # of GPUs.
        self.config["training"]["hardware"]["num_gpus"] = num_gpus
        io.write_json_file(self.config_json, self.config)

    def _validate_pretrained_config(self) -> None:
        """Validate pretrained encoder compatibility if --pretrained-weights set.

        Runs once in the main process before training starts so that config
        mismatches fail fast rather than inside a spawned worker.
        """
        pretrained_weights = getattr(self.mist_args, "pretrained_weights", None)
        pretrained_config_path = getattr(
            self.mist_args, "pretrained_config", None
        )

        if not pretrained_weights:
            return

        if not pretrained_config_path:
            import warnings
            warnings.warn(
                "--pretrained-weights is set but --pretrained-config was not "
                "provided. Skipping encoder compatibility validation."
            )
            return

        source_config = io.read_json_file(pretrained_config_path)
        validate_encoder_compatibility(source_config, self.config)

    def _use_dtms(self) -> bool:
        """Return True when the selected loss requires distance transform maps."""
        return (
            self.config["training"]["loss"]["name"]
            in TrainerConstants.DTM_AWARE_LOSSES
        )

    def _setup_folds(self) -> None:
        """Setup data paths and parameters for a specific fold.

        Args:
            fold: The fold number to set up data for.
        """
        # Get fold setup from the paths dataframe. For each fold, we will
        # create a dictionary with the training and validation images and
        # labels. If we are using distance transform maps, we will also
        # create a dictionary with the training distance transform maps. We will
        # also determine the number of training steps per epoch, which is
        # max(250, len(train_images) // batch_size).
        training = self.config["training"]
        self.folds = {}
        for fold in range(training["nfolds"]):
            self.folds[fold] = {}
            train_ids = list(self.paths.loc[self.paths["fold"] != fold]["id"])
            test_ids = list(self.paths.loc[self.paths["fold"] == fold]["id"])

            # Get list of training images and labels.
            train_images = training_utils.get_npy_paths(
                data_dir=self.numpy_dir / "images",
                patient_ids=train_ids,
            )
            train_labels = training_utils.get_npy_paths(
                data_dir=self.numpy_dir / "labels",
                patient_ids=train_ids,
            )

            # Get list of validation images and labels.
            val_images = training_utils.get_npy_paths(
                data_dir=self.numpy_dir / "images",
                patient_ids=test_ids,
            )
            val_labels = training_utils.get_npy_paths(
                data_dir=self.numpy_dir / "labels",
                patient_ids=test_ids,
            )

            # If we are using distance transform maps, get the list of training
            # distance transform maps.
            train_dtms = None
            if self._use_dtms():
                train_dtms = training_utils.get_npy_paths(
                    data_dir=self.numpy_dir / "dtms",
                    patient_ids=train_ids,
                )

            # Split training data into training and validation sets if the
            # validation percentage is greater than zero. The idea here is to
            # leave the original validation set as an unseen test set and use
            # the smaller partition of the training dataset as the validation
            # set to pick the best model.
            if training["val_percent"] > 0.0:
                split_inputs = [train_images, train_labels]
                if self._use_dtms():
                    split_inputs.append(train_dtms)

                splits = train_test_split(
                    *split_inputs,
                    test_size=training["val_percent"],
                    random_state=training["seed"],
                )

                # Unpack while handling optional DTMs.
                (
                    train_images, val_images, train_labels,
                    val_labels, *maybe_dtms
                ) = splits
                if self._use_dtms():
                    train_dtms, _ = maybe_dtms

            # Sanity check the fold data.
            training_utils.sanity_check_fold_data(
                fold=fold,
                train_images=train_images,
                train_labels=train_labels,
                val_images=val_images,
                val_labels=val_labels,
                train_dtms=train_dtms,
            )

            # Save the fold configuration.
            self.folds[fold]["train_images"] = train_images
            self.folds[fold]["train_labels"] = train_labels
            self.folds[fold]["val_images"] = val_images
            self.folds[fold]["val_labels"] = val_labels
            self.folds[fold]["train_dtms"] = train_dtms
            self.folds[fold]["steps_per_epoch"] = max(
                training["min_steps_per_epoch"],
                math.ceil(len(train_images) / max(1, self.batch_size)),
            )

    def build_components(self, rank: int, world_size: int) -> Dict[str, Any]:
        """Build the model, loss function, and optimizer components."""
        training = self.config["training"]

        # Build the model based on the architecture and parameters specified in
        # the configuration file. Merge spatial_config so adaptive architectures
        # receive patch_size and target_spacing alongside model-specific params.
        model_kwargs = {
            **self.config["model"]["params"],
            **self.config["spatial_config"],
        }
        model = get_model_from_registry(
            self.config["model"]["architecture"],
            **model_kwargs
        )

        # Load pretrained encoder weights if requested. This runs on every rank
        # so each process starts from the same initialization.
        pretrained_weights = getattr(self.mist_args, "pretrained_weights", None)
        if pretrained_weights:
            strategy = getattr(
                self.mist_args, "input_channel_strategy", "average"
            )
            model, transfer_summary = load_pretrained_encoder(
                model, pretrained_weights, strategy
            )
            if rank == 0:
                print_info(
                    f"Pretrained encoder loaded from {pretrained_weights}\n"
                    f"  Loaded:                   {len(transfer_summary['loaded'])} params\n"
                    f"  Channel strategy applied: "
                    f"{len(transfer_summary['channel_strategy_applied'])} params\n"
                    f"  Skipped:                  {len(transfer_summary['skipped'])} params"
                )

        use_ddp = world_size > 1
        # Make batch normalization compatible with DDP. This only matters if
        # the model uses batch normalization layers. None of the current MIST
        # models use batch normalization, but this is a good practice to
        # ensure compatibility with DDP if a new model is added in the future.
        if use_ddp:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Send model to device.
        model.to(torch.device(f"cuda:{rank}"))

        # Set up model for distributed data parallel training.
        if use_ddp:
            model = DDP(model, device_ids=[rank])

        # Get loss function. First get the loss function from the registry.
        # We then wrap it in a DeepSupervisionLoss, which will handle
        # deep supervision if the model supports it.
        loss_name = training["loss"]["name"]
        loss_cls = get_loss(loss_name)
        loss_params = {}
        if loss_name in TrainerConstants.SPACING_AWARE_LOSSES:
            loss_params["sddl_spacing_xyz"] = (
                self.config["spatial_config"]["target_spacing"]
            )
        loss_function = loss_cls(**loss_params)
        loss_function = DeepSupervisionLoss(loss_function)

        # Build the alpha scheduler for composite losses. Only applies when the
        # selected loss blends two terms via alpha (e.g. bl, gsl, cldice).
        clw_cfg = training["loss"]["composite_loss_weighting"]
        if loss_name in TrainerConstants.COMPOSITE_LOSSES and clw_cfg is not None:
            composite_loss_weighting = get_alpha_scheduler(
                clw_cfg["name"],
                num_epochs=training["epochs"],
                **clw_cfg.get("params", {}),
            )
        else:
            composite_loss_weighting = None

        # Get the optimizer.
        eps = (
            TrainerConstants.AMP_EPS if training["amp"]
            else TrainerConstants.NO_AMP_EPS
        )
        optimizer = get_optimizer(
            name=training["optimizer"],
            params=model.parameters(),
            learning_rate=training["learning_rate"],
            weight_decay=training["l2_penalty"],
            eps=eps,
        )

        # Get learning rate scheduler.
        lr_scheduler = get_lr_scheduler(
            name=training["lr_scheduler"],
            optimizer=optimizer,
            epochs=training["epochs"],
            warmup_epochs=training.get("warmup_epochs", 0),
        )

        # Get gradient scaler if AMP is enabled.
        scaler = torch.amp.GradScaler(
            "cuda") if training["amp"] else None  # type: ignore

        return {
            "model": model,
            "loss_function": loss_function,
            "composite_loss_weighting": composite_loss_weighting,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "scaler": scaler,
            "epoch": 0,
            "global_step": 0,
            "best_val_loss": np.inf,
            "alpha": 0.5,
        }

    def _checkpoint_path(self, fold: int) -> Path:
        """Return the checkpoint path for a given fold."""
        return self.checkpoints_dir / f"fold_{fold}_checkpoint.pt"

    def save_checkpoint(self, fold: int, state: Dict[str, Any]) -> None:
        """Save a training checkpoint for the given fold (rank 0 only).

        Saves the current model weights, optimizer, LR scheduler, and scaler
        states alongside training bookkeeping (epoch, global step, best
        validation loss) so that training can be resumed exactly from this
        point.

        Args:
            fold: The fold index being trained.
            state: The current training state dictionary.
        """
        model = state["model"]
        unwrapped = model.module if hasattr(model, "module") else model
        scaler = state["scaler"]
        checkpoint = {
            "fold": fold,
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "best_val_loss": state["best_val_loss"],
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": state["optimizer"].state_dict(),
            "lr_scheduler_state_dict": state["lr_scheduler"].state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        }
        # Write to a temp file then atomically rename so a mid-write SIGKILL
        # never leaves a corrupted checkpoint on disk.
        path = self._checkpoint_path(fold)
        tmp = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp)
        tmp.rename(path)

    def load_checkpoint(self, fold: int, state: Dict[str, Any]) -> bool:
        """Load a training checkpoint into state (all ranks).

        Restores model weights, optimizer, LR scheduler, and scaler states
        from the latest checkpoint for the given fold. If no checkpoint exists,
        returns False and leaves state unchanged.

        Args:
            fold: The fold index being trained.
            state: The training state dictionary to restore into.

        Returns:
            True if a checkpoint was loaded, False if none was found.
        """
        path = self._checkpoint_path(fold)
        if not path.exists():
            return False

        checkpoint = torch.load(path, weights_only=False)

        # Restore model weights (unwrap DDP before loading).
        model = state["model"]
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer and scheduler states.
        state["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
        state["lr_scheduler"].load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # Restore scaler state if AMP is enabled.
        if state["scaler"] is not None and checkpoint["scaler_state_dict"] is not None:
            state["scaler"].load_state_dict(checkpoint["scaler_state_dict"])

        # Restore bookkeeping — epoch is incremented so the loop starts on the
        # next unfinished epoch.
        state["epoch"] = checkpoint["epoch"] + 1
        state["global_step"] = checkpoint["global_step"]
        state["best_val_loss"] = checkpoint["best_val_loss"]

        return True

    def setup(self, rank: int, world_size: int) -> None:
        """Initialize the process group for distributed training."""
        if dist.is_initialized() or world_size == 1:
            return

        # Set the environment variables for distributed training.
        hw = self.config["training"]["hardware"]
        os.environ["MASTER_ADDR"] = hw["master_addr"]
        os.environ["MASTER_PORT"] = str(hw["master_port"])
        dist.init_process_group(
            hw["communication_backend"], rank=rank, world_size=world_size
        )

    # Clean up processes after distributed training
    def cleanup(self) -> None:
        """Clean up processes after distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def train_fold(self, fold: int, rank: int, world_size: int) -> None:
        """Generic training loop for a single fold."""
        # Set up for distributed training.
        use_ddp = world_size > 1
        torch.cuda.set_device(rank)
        self.setup(rank, world_size)

        # Set random seed for reproducibility.
        self._set_seed(self.config["training"]["seed"])

        # Get the fold data.
        fold_data = self.folds[fold]
        if use_ddp and len(fold_data["val_images"]) < world_size:
            raise ValueError(
                "Not enough validation data for the number of GPUs. "
                "Ensure that the validation set is large enough for the number "
                "of GPUs being used or reduce the number of GPUs."
            )
        val_steps = math.ceil(
            len(fold_data["val_images"]) / max(1, world_size))

        # Build components for the fold.
        state = self.build_components(rank=rank, world_size=world_size)

        # Resume from checkpoint if requested.
        if getattr(self.mist_args, "resume", False):
            loaded = self.load_checkpoint(fold, state)
            if rank == 0:
                if loaded:
                    print_info(
                        f"Resuming fold {fold} from epoch {state['epoch']}"
                    )
                else:
                    print_warning(
                        f"No checkpoint found for fold {fold}, "
                        "starting from scratch."
                    )

        # Build data loaders for the fold.
        train_loader, val_loader = self.build_dataloaders(
            fold_data=self.folds[fold], rank=rank, world_size=world_size,
        )

        # Only log metrics on first process (i.e., rank 0).
        if rank == 0:
            # Compute running averages for training and validation losses.
            running_train_loss = training_utils.RunningMean()
            running_val_loss = training_utils.RunningMean()

            # Set up tensorboard summary writer.
            logs_writer = SummaryWriter(str(self.logs_dir / f"fold_{fold}"))

            # Path and name for best model for this fold.
            model_name = str(self.models_dir / f"fold_{fold}.pt")

        # Stop training flag if we encounter nan or inf losses.
        stop_training = torch.tensor(
            [0], dtype=torch.int, device=f"cuda:{rank}"
        )

        # Start training for the specified number of epochs.
        for epoch in range(state["epoch"], self.config["training"]["epochs"]):
            # Update the epoch in the state.
            state["epoch"] = epoch

            # Compute alpha once per epoch so training_step doesn't repeat it.
            clw = state["composite_loss_weighting"]
            state["alpha"] = clw(epoch) if clw is not None else 0.5

            # Set up model for training.
            state["model"].train()

            # Run the training steps for this epoch. Rank 0 shows a progress
            # bar; other ranks run the same steps silently.
            pb_ctx = (
                progress_bar.TrainProgressBar(
                    current_epoch=epoch + 1,
                    fold=fold,
                    epochs=self.config["training"]["epochs"],
                    train_steps=fold_data["steps_per_epoch"],
                )
                if rank == 0
                else contextlib.nullcontext()
            )
            with pb_ctx as pb:
                for _ in range(fold_data["steps_per_epoch"]):
                    # Get a batch of training data.
                    batch = train_loader.next()[0]

                    # Compute loss and perform training step.
                    loss = self.training_step(state=state, data=batch)

                    # Update the global step in the state.
                    state["global_step"] += 1

                    # Detach and aggregate across all ranks (mean).
                    with torch.no_grad():
                        loss_det = loss.detach()
                        if use_ddp:
                            dist.all_reduce(loss_det, op=dist.ReduceOp.SUM)
                            mean_loss = (loss_det / world_size).item()
                        else:
                            mean_loss = loss_det.item()

                    # Check for NaN/Inf and flag for early exit.
                    if not np.isfinite(mean_loss):
                        if rank == 0:
                            print_error(
                                "Stopping training: Detected NaN or inf "
                                "loss value!"
                            )
                        stop_training[0] = 1

                    # Update running average and progress bar (rank 0 only).
                    if rank == 0:
                        train_mean_loss = running_train_loss(mean_loss)
                        pb.update(
                            loss=train_mean_loss,
                            lr=state["optimizer"].param_groups[0]["lr"]
                        )

            # Broadcast stop flag so all ranks exit together if needed.
            if use_ddp:
                dist.broadcast(stop_training, src=0)
            if stop_training.item() == 1:
                if rank == 0:
                    logs_writer.close()
                self.cleanup()
                return  # Exit training early.

            # Update learning rate scheduler.
            state["lr_scheduler"].step()

            # Wait for all processes to finish the training part of the epoch.
            if use_ddp:
                dist.barrier()

            # Run the validation steps for this epoch. Rank 0 shows a progress
            # bar and tracks best loss; other ranks run silently.
            state["model"].eval()
            with torch.no_grad():
                pbv_ctx = (
                    progress_bar.ValidationProgressBar(val_steps)
                    if rank == 0
                    else contextlib.nullcontext()
                )
                with pbv_ctx as pbv:
                    for _ in range(val_steps):
                        # Get a batch of validation data.
                        batch = val_loader.next()[0]

                        # Compute validation loss.
                        val_loss = self.validation_step(
                            state=state, data=batch
                        )

                        # Aggregate mean across ranks.
                        val_det = val_loss.detach()
                        if use_ddp:
                            dist.all_reduce(val_det, op=dist.ReduceOp.SUM)
                            mean_val = (val_det / world_size).item()
                        else:
                            mean_val = val_det.item()

                        # Update running average and progress bar (rank 0 only).
                        if rank == 0:
                            val_mean_loss = running_val_loss(mean_val)
                            pbv.update(loss=val_mean_loss)

                # Check if validation loss improved and save model (rank 0 only).
                if rank == 0:
                    if val_mean_loss < state["best_val_loss"]:
                        print_success(
                            f"Validation loss improved from "
                            f"{state['best_val_loss']:.4f} to "
                            f"{val_mean_loss:.4f}"
                        )

                        # Update the best validation loss.
                        state["best_val_loss"] = val_mean_loss

                        # Save the model. Unwrap DDP so it loads cleanly in
                        # non-DDP contexts.
                        to_save = (
                            state["model"].module
                            if hasattr(state["model"], "module")
                            else state["model"]
                        )
                        torch.save(to_save.state_dict(), model_name)
                    else:
                        print_info("Validation loss did not improve.")

            # Reset the dataloaders for the next epoch.
            train_loader.reset()
            val_loader.reset()

            # Log the running losses to tensorboard.
            if rank == 0:
                summary_data = {
                    "train": float(train_mean_loss),
                    "validation": float(val_mean_loss),
                }
                logs_writer.add_scalars("losses", summary_data, epoch + 1)
                logs_writer.flush()

                # Reset states of running losses for the next epoch.
                running_train_loss.reset_states()
                running_val_loss.reset_states()

            # Save checkpoint after every epoch (rank 0 only).
            if rank == 0:
                self.save_checkpoint(fold, state)

            # Wait for all processes to finish before starting the next epoch.
            if use_ddp:
                dist.barrier()

        # Close the tensorboard writer.
        if rank == 0:
            logs_writer.close()

        # Clean up distributed processes.
        self.cleanup()

    def run_cross_validation(self, rank: int, world_size: int) -> None:
        """Run cross-validation for selected folds."""
        # Display the start of training message.
        if rank == 0:
            print_section_header("Starting training")

        for fold in self.config["training"]["folds"]:
            # Skip folds that are already complete when resuming.
            if getattr(self.mist_args, "resume", False):
                path = self._checkpoint_path(fold)
                if path.exists():
                    checkpoint = torch.load(path, weights_only=False)
                    if checkpoint["epoch"] >= self.config["training"]["epochs"] - 1:
                        if rank == 0:
                            print_info(
                                f"Fold {fold} already complete, skipping."
                            )
                        continue

            # Train the model for the current fold.
            self.train_fold(fold=fold, rank=rank, world_size=world_size)

    def fit(self):
        """Fit the model using multiprocessing.

        This function uses multiprocessing to train the model on multiple GPUs.
        It uses the `torch.multiprocessing.spawn` function to create multiple
        instances of the training function, each on a separate GPU.
        """
        # Enable some performance optimizations.
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cudnn.benchmark = True

        # Train model.
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn(  # type: ignore
                self.run_cross_validation,
                args=(world_size,),
                nprocs=world_size,
                join=True,
            )
        # To enable pdb do not spawn multiprocessing for world_size = 1.
        else:
            self.run_cross_validation(0, world_size)

        print_success("Training complete.")
