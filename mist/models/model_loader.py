"""Unified interface for validating, building, and loading MIST models."""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import warnings
import torch

# MIST imports.
from mist.models.model_registry import get_model_from_registry


def average_fold_weights(
    weights_paths: List[str],
    output_path: Optional[str] = None,
) -> OrderedDict:
    """Average weights from multiple fold checkpoints.

    Produces a single state dict by element-wise averaging across all
    checkpoints. This is the recommended way to prepare a pretrained
    initialization from a cross-validation run — the averaged weights
    represent the consensus of all folds and generalize better than any
    single fold model.

    Args:
        weights_paths: List of paths to fold checkpoint files (.pt or .pth).
        output_path: If provided, the averaged state dict is saved here.

    Returns:
        Averaged state dict with float32 weights.

    Raises:
        ValueError: If checkpoints have mismatched keys, indicating
            incompatible architectures.
    """
    state_dicts = [torch.load(p, weights_only=True) for p in weights_paths]

    # Strip DDP module. prefix if present.
    cleaned = []
    for sd in state_dicts:
        if any(k.startswith("module.") for k in sd):
            sd = OrderedDict(
                {(k[7:] if k.startswith("module.") else k): v
                 for k, v in sd.items()}
            )
        cleaned.append(sd)

    # Validate all checkpoints have identical keys.
    reference_keys = set(cleaned[0].keys())
    for i, sd in enumerate(cleaned[1:], start=1):
        if set(sd.keys()) != reference_keys:
            raise ValueError(
                f"Checkpoint at index {i} has different keys than checkpoint "
                "0. All checkpoints must share the same architecture."
            )

    # Average element-wise in float32.
    avg = OrderedDict()
    for key in cleaned[0]:
        avg[key] = torch.stack([sd[key].float() for sd in cleaned]).mean(0)

    if output_path:
        torch.save(avg, output_path)

    return avg


# Architectures whose encoder structure is computed from patch_size and
# target_spacing. Both must match for encoder weights to be compatible.
_ADAPTIVE_ARCHITECTURES = {"nnunet", "fmgnet", "wnet"}


def validate_encoder_compatibility(
    source_config: Dict,
    target_config: Dict,
    force: bool = False,
) -> None:
    """Validate that source and target configs have compatible encoder structure.

    Args:
        source_config: MIST config dict for the pretrained source model.
        target_config: MIST config dict for the target model being trained.
        force: If True, emit warnings instead of raising on mismatches.

    Raises:
        ValueError: If architectures are incompatible and force=False.
    """
    def _fail(msg: str) -> None:
        if force:
            warnings.warn(msg, stacklevel=3)
        else:
            raise ValueError(msg)

    source_arch = source_config["model"]["architecture"]
    target_arch = target_config["model"]["architecture"]

    if source_arch != target_arch:
        _fail(
            f"Architecture mismatch: source '{source_arch}' vs target "
            f"'{target_arch}'. Encoder weights cannot transfer between "
            "different architectures."
        )
        return  # No point checking further if architectures differ.

    # For adaptive architectures the encoder depth and filter counts are
    # derived from patch_size and target_spacing; both must match.
    if source_arch in _ADAPTIVE_ARCHITECTURES:
        source_params = source_config["model"]["params"]
        target_params = target_config["model"]["params"]

        if source_params["patch_size"] != target_params["patch_size"]:
            _fail(
                f"patch_size mismatch: source {source_params['patch_size']} "
                f"vs target {target_params['patch_size']}. For adaptive "
                "architectures the encoder structure is determined by "
                "patch_size."
            )

        if source_params["target_spacing"] != target_params["target_spacing"]:
            _fail(
                "target_spacing mismatch: source "
                f"{source_params['target_spacing']} vs target "
                f"{target_params['target_spacing']}. For adaptive "
                "architectures the encoder structure is determined by "
                "target_spacing."
            )


def load_pretrained_encoder(
    model: torch.nn.Module,
    weights_path: str,
    in_channel_strategy: str = "average",
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    """Load pretrained encoder weights into a model.

    Only encoder weights (as defined by model.get_encoder_state_dict()) are
    loaded. The decoder and output heads always retain their randomly
    initialized weights from model construction.

    When in_channels differs between source and target, the in_channel_strategy
    controls how the first conv weight is handled:
        - "average": average source channels to 1, then tile to target count.
        - "first":   take the first source channel, then tile to target count.
        - "skip":    leave the input conv at its random initialization.

    Args:
        model: Target MIST model (must implement get_encoder_state_dict()).
        weights_path: Path to the source checkpoint (.pt or .pth). This can be
            a single fold checkpoint or the output of average_fold_weights().
        in_channel_strategy: How to handle in_channels mismatch. One of
            "average" (default), "first", or "skip".

    Returns:
        Tuple of (model with encoder weights loaded, summary dict). The summary
        dict has keys "loaded", "channel_strategy_applied", and "skipped",
        each containing a list of parameter names.

    Raises:
        ValueError: If in_channel_strategy is not a valid option.
        AttributeError: If the model does not implement get_encoder_state_dict.
    """
    valid_strategies = ("average", "first", "skip")
    if in_channel_strategy not in valid_strategies:
        raise ValueError(
            f"in_channel_strategy must be one of {valid_strategies}, "
            f"got '{in_channel_strategy}'."
        )

    if not hasattr(model, "get_encoder_state_dict"):
        raise AttributeError(
            f"{type(model).__name__} does not implement get_encoder_state_dict. "
            "All MIST models must inherit from MISTModel."
        )

    # Load source weights and strip DDP module. prefix if present.
    source_sd = torch.load(weights_path, weights_only=True)
    if any(k.startswith("module.") for k in source_sd):
        source_sd = OrderedDict(
            {(k[7:] if k.startswith("module.") else k): v
             for k, v in source_sd.items()}
        )

    target_encoder_sd = model.get_encoder_state_dict()
    partial_sd = OrderedDict()
    summary: Dict[str, List[str]] = {
        "loaded": [],
        "channel_strategy_applied": [],
        "skipped": [],
    }

    for key, target_weight in target_encoder_sd.items():
        if key not in source_sd:
            summary["skipped"].append(key)
            continue

        source_weight = source_sd[key]

        if source_weight.shape == target_weight.shape:
            # Shapes match: load directly.
            partial_sd[key] = source_weight
            summary["loaded"].append(key)

        elif (
            source_weight.dim() == target_weight.dim()
            and source_weight.shape[0] == target_weight.shape[0]
            and source_weight.shape[1] != target_weight.shape[1]
            and source_weight.shape[2:] == target_weight.shape[2:]
        ):
            # Only dim[1] (in_channels) differs: apply channel strategy.
            if in_channel_strategy == "skip":
                summary["skipped"].append(key)
                continue

            target_channels = target_weight.shape[1]
            if in_channel_strategy == "average":
                base = source_weight.float().mean(dim=1, keepdim=True)
            else:  # "first"
                base = source_weight.float()[:, :1, ...]

            # Tile base channels to match target channel count.
            repeat_dims = [1, target_channels] + [1] * (source_weight.dim() - 2)
            new_weight = base.repeat(repeat_dims).to(source_weight.dtype)
            partial_sd[key] = new_weight
            summary["channel_strategy_applied"].append(key)

        else:
            # Shape mismatch beyond in_channels (e.g., filter count change
            # from use_pocket_model mismatch): skip and retain random init.
            summary["skipped"].append(key)

    # strict=False preserves decoder random initialization.
    model.load_state_dict(partial_sd, strict=False)
    return model, summary


def validate_mist_config_for_model_loading(config: Dict) -> None:
    """Validate structure of the MIST configuration.

    Args:
        config: MIST configuration dictionary.

    Raises:
        ValueError: If required keys are missing or have incorrect types.
    """
    if "model" not in config:
        raise ValueError("Missing required key 'model' in configuration.")

    required_model_keys = ["architecture", "params"]
    for key in required_model_keys:
        if key not in config["model"]:
            raise ValueError(f"Missing required key '{key}' in model section.")

    required_params_keys = [
        "in_channels", "out_channels", "patch_size", "target_spacing",
        "use_deep_supervision", "use_residual_blocks", "use_pocket_model"
    ]
    for key in required_params_keys:
        if key not in config["model"]["params"]:
            raise ValueError(
                f"Missing required key '{key}' in model parameters."
            )


def convert_legacy_model_config(model_config: Dict) -> Dict:
    """Convert legacy model config to new format.

    Args:
        model_config_path: Path to the legacy model config file. This file will
            contain the following structure:
            {
                "model_name": "model_name",
                "n_channels": 1,
                "n_classes": 2,
                "deep_supervision": false,
                "pocket": false,
                "patch_size": [128, 128, 128],
                "target_spacing": [1.0, 1.0, 1.0],
                "use_res_block": false
            }

    Returns:
        A configuration dictionary in the new format:
        {
            "model": {
                "architecture": "model_name",
                "params": {
                    "in_channels": 1,
                    "out_channels": 2,
                    "patch_size": [128, 128, 128],
                    "target_spacing": [1.0, 1.0, 1.0],
                    "use_deep_supervision": false,
                    "use_residual_blocks": false,
                    "use_pocket_model": false
                }
            }
        }

    Raises:
        ValueError: If the config does not contain the expected keys.
    """
    required_keys = [
        "model_name", "n_channels", "n_classes", "patch_size", "target_spacing",
        "deep_supervision", "pocket", "use_res_block"
    ]
    for key in required_keys:
        if key not in model_config:
            raise ValueError(
                f"Missing required key '{key}' in legacy model config."
            )

    # Convert the legacy config to the new format.
    new_config = {
        "model": {
            "architecture": model_config["model_name"],
            "params": {
                "in_channels": model_config["n_channels"],
                "out_channels": model_config["n_classes"],
                "patch_size": model_config["patch_size"],
                "target_spacing": model_config["target_spacing"],
                "use_deep_supervision": model_config["deep_supervision"],
                "use_residual_blocks": model_config["use_res_block"],
                "use_pocket_model": model_config["pocket"]
            }
        }
    }
    return new_config


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Build a model instance using the registered model name and parameters.

    Args:
        model_name: Name of the registered model.
        kwargs: Model-specific arguments.

    Returns:
        Instantiated PyTorch model.
    """
    return get_model_from_registry(model_name, **kwargs)


def load_model_from_config(
    weights_path: str,
    config: Dict,
) -> torch.nn.Module:
    """Load a model and its weights from a config dictionary and checkpoint.

    Args:
        weights_path: Path to the PyTorch checkpoint file (.pt or .pth).
        config: MIST configuration dictionary.

    Returns:
        PyTorch model with weights loaded.

    Raises:
        FileNotFoundError: If the config or weights file does not exist.
        ValueError: If the model name is invalid or required config keys are
            missing.
    """
    # Load and validate the config file.
    validate_mist_config_for_model_loading(config)

    # Build model from registry.
    model = get_model(
        config["model"]["architecture"], **config["model"]["params"]
    )

    # Load checkpoint weights.
    state_dict = torch.load(weights_path, weights_only=True)

    # If keys come from DDP, strip 'module.' prefix.
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model
