"""Utilities for the DataDumper module."""
from typing import Dict, Any, List
import os

import numpy as np
import ants
import pandas as pd

from mist.utils import progress_bar as progress_bar_utils


def get_dataset_size_gb(paths_df: pd.DataFrame) -> float:
    """Get total size of all image and mask files in GB.

    Args:
        paths_df: DataFrame with file paths for each patient.

    Returns:
        Total size of all files in GB.
    """
    total_bytes = 0
    for _, row in paths_df.iterrows():
        for col in row.index:
            if col in ("id", "fold"):
                continue
            val = row[col]
            if isinstance(val, str) and os.path.exists(val):
                total_bytes += os.path.getsize(val)
    return round(total_bytes / 1e9, 4)


# Max voxels to sample per label per patient for PCA.
_MAX_SHAPE_COORDS = 10_000


def compute_shape_descriptors(coords_mm: np.ndarray) -> Dict[str, Any]:
    """Compute PCA-based shape descriptors from a set of 3D coordinates.

    Given the mm-space coordinates of all voxels belonging to a label region,
    the eigenvalues of the covariance matrix (λ1 ≥ λ2 ≥ λ3) decompose the
    shape into three orthogonal components that sum to 1:

        linearity  = (λ1 - λ2) / λ1  -- one dominant axis → tubular/vessel-like
        planarity  = (λ2 - λ3) / λ1  -- two dominant axes → sheet/surface-like
        sphericity = λ3 / λ1         -- all axes comparable → blob-like

    The shape_class is whichever component is largest.

    Args:
        coords_mm: (N, 3) array of voxel coordinates in millimeters.

    Returns:
        Dictionary with linearity, planarity, sphericity (all floats in [0, 1])
        and shape_class (one of "tubular", "planar", "blob").
        Returns None if there are fewer than 4 points (degenerate case).
    """
    if len(coords_mm) < 4:
        return None

    # Subsample to keep PCA fast on large label regions.
    if len(coords_mm) > _MAX_SHAPE_COORDS:
        idx = np.random.choice(
            len(coords_mm), _MAX_SHAPE_COORDS, replace=False
        )
        coords_mm = coords_mm[idx]

    cov = np.cov(coords_mm.T)
    if cov.shape != (3, 3):
        return None

    # eigvalsh returns eigenvalues in ascending order; reverse to descending.
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)  # numerical stability
    lam1, lam2, lam3 = eigenvalues

    if lam1 < 1e-8:
        return None

    linearity = float((lam1 - lam2) / lam1)
    planarity = float((lam2 - lam3) / lam1)
    sphericity = float(lam3 / lam1)

    components = {
        "tubular": linearity,
        "planar": planarity,
        "blob": sphericity,
    }
    shape_class = max(components, key=lambda k: components[k])

    return {
        "linearity": round(linearity, 4),
        "planarity": round(planarity, 4),
        "sphericity": round(sphericity, 4),
        "shape_class": shape_class,
    }


def collect_per_patient_stats(
    paths_df: pd.DataFrame,
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Single-pass collection of per-patient statistics for the data dump.

    Iterates over each patient once to collect spacings, original dimensions,
    per-channel foreground intensity samples, per-label voxel counts, label
    presence flags, non-zero fractions, and per-label shape descriptors.

    Args:
        paths_df: DataFrame with file paths for each patient.
        dataset_info: Dataset description from the JSON file.

    Returns:
        Dictionary with the following keys:
            - spacings: (n_patients, 3) array of voxel spacings.
            - original_dims: (n_patients, 3) array of original dimensions.
            - nonzero_fractions: (n_patients,) array of non-zero fractions.
            - total_fg_voxels: list of foreground voxel counts per patient.
            - channel_intensities: dict mapping channel name to list of sampled
                foreground intensity values (pooled across all patients).
            - label_voxel_counts: dict mapping label int to list of voxel
                counts per patient.
            - label_presence: dict mapping label int to list of 0/1 per patient.
            - label_shape_descriptors: dict mapping label int to list of
                shape descriptor dicts (one per patient where label is present).
    """
    n_patients = len(paths_df)
    image_cols = list(dataset_info["images"].keys())
    non_bg_labels = [lbl for lbl in dataset_info["labels"] if lbl != 0]

    spacings = np.zeros((n_patients, 3))
    original_dims = np.zeros((n_patients, 3))
    nonzero_fractions = np.zeros(n_patients)
    total_fg_voxels: List[int] = []

    # Per-channel: sampled foreground intensities pooled across all patients.
    channel_intensities: Dict[str, List[float]] = {
        col: [] for col in image_cols
    }

    # Per-label: voxel count, presence, and shape descriptors per patient.
    label_voxel_counts: Dict[int, List[int]] = {
        lbl: [] for lbl in non_bg_labels
    }
    label_presence: Dict[int, List[int]] = {
        lbl: [] for lbl in non_bg_labels
    }
    label_shape_descriptors: Dict[int, List[Dict[str, Any]]] = {
        lbl: [] for lbl in non_bg_labels
    }

    progress = progress_bar_utils.get_progress_bar("Building data dump")
    with progress as pb:
        for i in pb.track(range(n_patients)):
            patient = paths_df.iloc[i].to_dict()

            mask = ants.image_read(patient["mask"])
            mask_arr = mask.numpy()

            spacings[i, :] = mask.spacing
            original_dims[i, :] = mask.shape

            fg_mask = mask_arr != 0
            fg_count = int(np.sum(fg_mask))
            total_fg_voxels.append(fg_count)
            nonzero_fractions[i] = fg_count / max(
            np.prod(mask.shape), 1
        )

            # Per-label statistics and shape descriptors.
            spacing_arr = np.array(mask.spacing)
            for lbl in non_bg_labels:
                lbl_mask = mask_arr == lbl
                voxel_count = int(np.sum(lbl_mask))
                label_voxel_counts[lbl].append(voxel_count)
                label_presence[lbl].append(1 if voxel_count > 0 else 0)

                if voxel_count > 0:
                    coords = np.argwhere(lbl_mask).astype(np.float32)
                    coords_mm = coords * spacing_arr
                    shape = compute_shape_descriptors(coords_mm)
                    if shape is not None:
                        label_shape_descriptors[lbl].append(shape)

            # Per-channel intensity statistics sampled from foreground.
            if fg_count > 0:
                for col in image_cols:
                    if col not in patient or not isinstance(
                        patient[col], str
                    ):
                        continue
                    img = ants.image_read(patient[col])
                    fg_vals = img.numpy()[fg_mask]
                    # Sample up to ~5000 voxels per patient to keep memory low.
                    step = max(1, len(fg_vals) // 5000)
                    channel_intensities[col].extend(fg_vals[::step].tolist())

    return {
        "spacings": spacings,
        "original_dims": original_dims,
        "nonzero_fractions": nonzero_fractions,
        "total_fg_voxels": total_fg_voxels,
        "channel_intensities": channel_intensities,
        "label_voxel_counts": label_voxel_counts,
        "label_presence": label_presence,
        "label_shape_descriptors": label_shape_descriptors,
    }


def _axis_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for a 1D array.

    Args:
        values: 1D numpy array of values.

    Returns:
        Dictionary with mean, std, min, p25, median, p75, max.
    """
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "min": round(float(np.min(values)), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "median": round(float(np.median(values)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "max": round(float(np.max(values)), 4),
    }


def build_image_statistics(
    raw_stats: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the image_statistics section of the data dump.

    Args:
        raw_stats: Output of collect_per_patient_stats.
        config: MIST configuration dict already computed by the Analyzer.

    Returns:
        image_statistics dict with spacing, dimension, and intensity
        sub-sections.
    """
    spacings = raw_stats["spacings"]
    dims = raw_stats["original_dims"]
    nz_fracs = raw_stats["nonzero_fractions"]
    channel_intensities = raw_stats["channel_intensities"]

    spacing_stats = {
        f"axis_{ax}": _axis_stats(spacings[:, ax]) for ax in range(3)
    }
    median_spacing = [float(np.median(spacings[:, ax])) for ax in range(3)]
    anisotropy_ratio = round(
        max(median_spacing) / max(min(median_spacing), 1e-8),
        4,
    )

    dim_stats = {
        f"axis_{ax}": _axis_stats(dims[:, ax]) for ax in range(3)
    }

    intensity_stats: Dict[str, Any] = {}
    for col, vals in channel_intensities.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float32)
        intensity_stats[col] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p01": round(float(np.percentile(arr, 1)), 4),
            "p05": round(float(np.percentile(arr, 5)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
        }

    return {
        "spacing": {
            "per_axis": spacing_stats,
            "median_spacing_mm": median_spacing,
            "anisotropy_ratio": anisotropy_ratio,
            "is_anisotropic": anisotropy_ratio > 3.0,
        },
        "dimensions": {
            "original": {"per_axis": dim_stats},
            "resampled_median": (
                config["preprocessing"]["median_resampled_image_size"]
            ),
        },
        "intensity": {
            "per_channel": intensity_stats,
            "nonzero_fraction": _axis_stats(nz_fracs),
        },
    }


def _size_category(volume_fraction_pct: float) -> str:
    """Categorize label size based on its mean volume fraction of foreground.

    Args:
        volume_fraction_pct: Mean volume fraction as a percentage.

    Returns:
        One of "tiny", "small", "medium", or "large".
    """
    if volume_fraction_pct < 0.1:
        return "tiny"
    if volume_fraction_pct < 1.0:
        return "small"
    if volume_fraction_pct < 5.0:
        return "medium"
    return "large"


def build_label_statistics(
    raw_stats: Dict[str, Any],
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the label_statistics section of the data dump.

    Args:
        raw_stats: Output of collect_per_patient_stats.
        dataset_info: Dataset description from the JSON file.

    Returns:
        label_statistics dict with per_label, final_classes, and
        class_imbalance sub-sections.
    """
    non_bg_labels = [lbl for lbl in dataset_info["labels"] if lbl != 0]
    final_classes = dataset_info["final_classes"]
    total_fg = raw_stats["total_fg_voxels"]
    mean_total_fg = float(np.mean(total_fg)) if total_fg else 1.0

    per_label: Dict[str, Any] = {}
    presence_rates: Dict[int, float] = {}
    mean_vol_fractions: Dict[int, float] = {}

    for lbl in non_bg_labels:
        counts = np.array(raw_stats["label_voxel_counts"][lbl])
        presence = np.array(raw_stats["label_presence"][lbl])
        presence_rate = round(float(np.mean(presence)) * 100, 2)
        mean_vol_frac_pct = round(
            float(np.mean(counts)) / max(mean_total_fg, 1e-8) * 100,
            4,
        )

        presence_rates[lbl] = presence_rate
        mean_vol_fractions[lbl] = mean_vol_frac_pct

        # Aggregate shape descriptors across patients where label is present.
        shape_descs = raw_stats["label_shape_descriptors"][lbl]
        if shape_descs:
            mean_lin = float(
                np.mean([d["linearity"] for d in shape_descs])
            )
            mean_plan = float(
                np.mean([d["planarity"] for d in shape_descs])
            )
            mean_sph = float(
                np.mean([d["sphericity"] for d in shape_descs])
            )
            components = {
                "tubular": mean_lin,
                "planar": mean_plan,
                "blob": mean_sph,
            }
            shape_class = max(components, key=lambda k: components[k])
            shape_info: Dict[str, Any] = {
                "linearity": round(mean_lin, 4),
                "planarity": round(mean_plan, 4),
                "sphericity": round(mean_sph, 4),
                "shape_class": shape_class,
            }
        else:
            shape_info = {
                "linearity": None,
                "planarity": None,
                "sphericity": None,
                "shape_class": "unknown",
            }

        per_label[str(lbl)] = {
            "voxel_count": {
                "mean": round(float(np.mean(counts)), 2),
                "std": round(float(np.std(counts)), 2),
                "min": int(np.min(counts)),
                "median": round(float(np.median(counts)), 2),
                "max": int(np.max(counts)),
            },
            "mean_volume_fraction_of_foreground_pct": mean_vol_frac_pct,
            "presence_rate_pct": presence_rate,
            "size_category": _size_category(mean_vol_frac_pct),
            "shape": shape_info,
        }

    # Aggregate statistics per final class (combining constituent labels).
    final_class_stats: Dict[str, Any] = {}
    for class_name, class_labels in final_classes.items():
        combined_counts = np.zeros(len(total_fg))
        combined_presence = np.zeros(len(total_fg))
        for lbl in class_labels:
            if lbl != 0 and lbl in raw_stats["label_voxel_counts"]:
                combined_counts += np.array(
                    raw_stats["label_voxel_counts"][lbl])
                combined_presence = np.maximum(
                    combined_presence,
                    np.array(raw_stats["label_presence"][lbl]),
                )
        mean_combined = float(np.mean(combined_counts))
        vol_frac_pct = round(
            mean_combined / max(mean_total_fg, 1e-8) * 100, 4
        )
        final_class_stats[class_name] = {
            "constituent_labels": class_labels,
            "mean_volume_fraction_of_foreground_pct": vol_frac_pct,
            "presence_rate_pct": round(
                float(np.mean(combined_presence)) * 100, 2
            ),
            "size_category": _size_category(vol_frac_pct),
        }

    # Class imbalance ratio across non-background labels with nonzero presence.
    active_fractions = {
        lbl: mean_vol_fractions[lbl]
        for lbl in non_bg_labels
        if presence_rates[lbl] > 0 and mean_vol_fractions[lbl] > 0
    }
    if len(active_fractions) >= 2:
        dominant = max(
            active_fractions, key=lambda l: active_fractions[l]
        )
        minority = min(
            active_fractions, key=lambda l: active_fractions[l]
        )
        imbalance_ratio = round(
            active_fractions[dominant]
            / max(active_fractions[minority], 1e-8),
            2,
        )
    else:
        dominant = non_bg_labels[0] if non_bg_labels else None
        minority = None
        imbalance_ratio = 1.0

    return {
        "per_label": per_label,
        "final_classes": final_class_stats,
        "class_imbalance": {
            "imbalance_ratio": imbalance_ratio,
            "dominant_label": dominant,
            "minority_label": minority,
        },
    }


def generate_observations(
    image_stats: Dict[str, Any],
    label_stats: Dict[str, Any],
    dataset_summary: Dict[str, Any],
) -> List[str]:
    """Generate plain-language observations from computed statistics.

    These are rule-based findings intended to surface actionable insights
    for LLM-driven configuration reasoning.

    Args:
        image_stats: Output of build_image_statistics.
        label_stats: Output of build_label_statistics.
        dataset_summary: Dataset summary section of the dump.

    Returns:
        List of plain-language observation strings.
    """
    observations: List[str] = []
    modality = dataset_summary["modality"]
    n_patients = dataset_summary["num_patients"]
    n_channels = dataset_summary["num_channels"]

    # Dataset size.
    if n_patients < 50:
        observations.append(
            f"Small dataset ({n_patients} patients). Consider aggressive "
            "data augmentation and evaluate with cross-validation."
        )
    elif n_patients > 500:
        observations.append(
            f"Large dataset ({n_patients} patients). Larger batch sizes "
            "and longer training schedules may improve convergence."
        )

    # Multi-channel input.
    if n_channels > 1:
        observations.append(
            f"Multi-channel input ({n_channels} channels: "
            f"{', '.join(dataset_summary['channel_names'])}). Ensure all "
            "channels are registered and consistently available across all "
            "patients."
        )

    # Anisotropy.
    aniso_ratio = image_stats["spacing"]["anisotropy_ratio"]
    if image_stats["spacing"]["is_anisotropic"]:
        observations.append(
            "Anisotropic spacing detected "
            f"(max/min ratio = {aniso_ratio:.2f}). "
            "MIST adjusts the target spacing to mitigate this. Consider "
            "whether the low-resolution axis warrants architecture-level "
            "treatment."
        )

    # Sparse images.
    mean_nz = image_stats["intensity"]["nonzero_fraction"]["mean"]
    if mean_nz < 0.2:
        observations.append(
            f"Sparse images detected "
            f"(mean non-zero fraction = {mean_nz:.1%}). "
            "MIST will apply normalization only to non-zero voxels."
        )

    # CT-specific intensity summary.
    if modality == "ct":
        for ch, stats in image_stats["intensity"]["per_channel"].items():
            observations.append(
                f"CT channel '{ch}': foreground HU range approximately "
                f"[{stats['p01']:.0f}, {stats['p99']:.0f}] HU, "
                f"mean = {stats['mean']:.1f} "
                f"\u00b1 {stats['std']:.1f} HU."
            )

    # Class imbalance.
    ci = label_stats["class_imbalance"]
    if ci["minority_label"] is not None:
        if ci["imbalance_ratio"] > 10:
            observations.append(
                f"Severe class imbalance: label {ci['dominant_label']} is "
                f"{ci['imbalance_ratio']:.1f}x larger than label "
                f"{ci['minority_label']} by foreground volume. Consider a "
                "weighted or boundary-aware loss function."
            )
        elif ci["imbalance_ratio"] > 3:
            observations.append(
                "Moderate class imbalance "
                f"(volume ratio = {ci['imbalance_ratio']:.1f}x). "
                "Monitor per-class Dice scores during training."
            )

    # Tiny or low-presence labels.
    for lbl_str, lbl_data in label_stats["per_label"].items():
        vol_frac = lbl_data["mean_volume_fraction_of_foreground_pct"]
        presence = lbl_data["presence_rate_pct"]
        if lbl_data["size_category"] == "tiny":
            observations.append(
                f"Label {lbl_str} is very small "
                f"({vol_frac:.4f}% of foreground voxels on average, "
                f"present in {presence:.1f}% of patients). "
                "Accurate segmentation of this label will be challenging."
            )
        elif presence < 50:
            observations.append(
                f"Label {lbl_str} is absent in "
                f"{100 - presence:.1f}% of patients. "
                "Ensure per-label evaluation excludes patients where it "
                "is not annotated."
            )

    # Shape-based observations.
    _SHAPE_ADVICE = {
        "tubular": (
            "tubular/vessel-like (high linearity). Standard volumetric "
            "losses like Dice may under-penalize topology errors on thin "
            "structures. Consider a topology-aware or clDice loss."
        ),
        "planar": (
            "planar/sheet-like (high planarity). Surface-based metrics "
            "and boundary-aware losses may improve thin-structure "
            "delineation."
        ),
        "blob": (
            "compact/blob-like (high sphericity). Standard Dice and "
            "cross-entropy losses are generally well-suited for this shape."
        ),
    }
    for lbl_str, lbl_data in label_stats["per_label"].items():
        shape = lbl_data["shape"]
        if shape["shape_class"] == "unknown":
            continue
        lin = shape["linearity"]
        plan = shape["planarity"]
        sph = shape["sphericity"]
        advice = _SHAPE_ADVICE.get(shape["shape_class"], "")
        observations.append(
            f"Label {lbl_str} appears {advice} "
            f"(linearity={lin:.2f}, planarity={plan:.2f}, "
            f"sphericity={sph:.2f})."
        )

    return observations
