"""DataDumper class for MIST.

This module produces a rich statistical summary of the dataset that goes
beyond MIST's heuristic configuration analysis. The summary is intended to
give LLM-based agents the detailed context they need to reason about model
architecture, loss function, and training configuration choices.

Two output files are saved to the results directory:
    - data_dump.json: Full structured statistics (machine-readable).
    - data_dump.md: Narrativized summary optimized for LLM consumption.
"""
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import rich

from mist.utils import io as io_utils
from mist.analyze_data import data_dump_utils


class DataDumper:
    """Compute and save a rich dataset statistics dump alongside config.json.

    Attributes:
        paths_df: DataFrame with file paths per patient.
        dataset_info: Dataset description from the JSON file.
        config: MIST configuration dict computed by the Analyzer.
        results_dir: Directory where outputs are saved.
        console: Rich console for printing messages.
    """

    def __init__(
        self,
        paths_df: pd.DataFrame,
        dataset_info: Dict[str, Any],
        config: Dict[str, Any],
        results_dir: str,
        cropped_dims: Optional[np.ndarray] = None,
    ):
        self.paths_df = paths_df
        self.dataset_info = dataset_info
        self.config = config
        self.results_dir = results_dir
        self.cropped_dims = cropped_dims
        self.console = rich.console.Console()

    def _build_dataset_summary(self) -> Dict[str, Any]:
        """Build the dataset_summary section of the data dump."""
        return {
            "task": self.dataset_info["task"],
            "modality": self.dataset_info["modality"].lower(),
            "num_patients": len(self.paths_df),
            "num_channels": len(self.dataset_info["images"]),
            "channel_names": list(self.dataset_info["images"].keys()),
            "num_labels": len(self.dataset_info["labels"]),
            "labels": self.dataset_info["labels"],
            "final_classes": dict(self.dataset_info["final_classes"]),
            "dataset_size_gb": data_dump_utils.get_dataset_size_gb(
            self.paths_df
        ),
        }

    def build_data_dump(self) -> Dict[str, Any]:
        """Compute all statistics and assemble the data dump dictionary.

        Returns:
            Dictionary with the following top-level keys:
                - dataset_summary
                - image_statistics
                - label_statistics
                - observations
                - mist_config_path
        """
        dataset_summary = self._build_dataset_summary()

        # When crop_to_foreground is enabled, use fg bounding box dims as the
        # image size denominator so vol-fraction-of-image reflects the actual
        # region the model sees rather than the full uncropped volume.
        crop_to_fg = bool(
            self.config.get("preprocessing", {}).get("crop_to_foreground", False)
        )
        effective_dims = (
            self.cropped_dims
            if (crop_to_fg and self.cropped_dims is not None)
            else None
        )

        # Single pass over all patients to collect raw statistics.
        raw_stats = data_dump_utils.collect_per_patient_stats(
            self.paths_df, self.dataset_info, effective_dims=effective_dims
        )

        image_stats = data_dump_utils.build_image_statistics(
            raw_stats, self.config
        )
        label_stats = data_dump_utils.build_label_statistics(
            raw_stats, self.dataset_info
        )
        observations = data_dump_utils.generate_observations(
            image_stats, label_stats, dataset_summary
        )

        return {
            "dataset_summary": dataset_summary,
            "image_statistics": image_stats,
            "label_statistics": label_stats,
            "observations": observations,
            "mist_config_path": os.path.join(self.results_dir, "config.json"),
        }

    def generate_markdown_summary(self, dump: Dict[str, Any]) -> str:
        """Generate a markdown narrative from the data dump dictionary.

        Args:
            dump: Output of build_data_dump.

        Returns:
            Markdown-formatted string.
        """
        ds = dump["dataset_summary"]
        img = dump["image_statistics"]
        lbl = dump["label_statistics"]
        obs = dump["observations"]

        lines = [
            f"# MIST Data Dump: {ds['task']}",
            "",
            (
                "> **How to use this document:** This file is an "
                "auto-generated draft produced by MIST. It is intended to "
                "be edited by the user before being passed to an LLM. Fill "
                "in the *Dataset Description and Goals* section below with "
                "clinical context, known data quality issues, and task "
                "priorities. Review and amend the *Observations* section "
                "at the bottom. The richer the context you provide, the "
                "better an LLM can reason about architecture, training, and "
                "evaluation choices."
            ),
            "",
            "## Dataset Description and Goals",
            "",
            (
                "*Describe the dataset and your segmentation goals here. "
                "Include any relevant clinical or scientific background, "
                "known data quality issues, annotation conventions, and "
                "what success looks like for this task. This section is "
                "intentionally left blank for you to complete.*"
            ),
            "",
            "<!-- User notes",
            "Example prompts to guide your writing:",
            "  - What anatomical structures or pathologies are being segmented?",
            "  - What is the intended clinical application or research question?",
            "  - Are there known annotation inconsistencies or quality issues?",
            "  - Which labels or classes are most important to get right?",
            "  - Are there patient subgroups, acquisition protocols, or sites",
            "    that differ meaningfully from the rest of the cohort?",
            "-->",
            "",
            "## Dataset Summary",
            f"- **Task:** {ds['task']}",
            f"- **Modality:** {ds['modality'].upper()}",
            f"- **Patients:** {ds['num_patients']}",
            (
                f"- **Channels ({ds['num_channels']}):** "
                f"{', '.join(ds['channel_names'])}"
            ),
            f"- **Labels:** {ds['labels']}",
            (
                "- **Final classes:** "
                + ", ".join(
                    f"{name} {labels}"
                    for name, labels in ds["final_classes"].items()
                )
            ),
            f"- **Dataset size:** {ds['dataset_size_gb']:.3f} GB",
            "",
            "## Image Statistics",
            "",
            "### Spacing (mm)",
            "| Axis | Mean | Std | Min | Median | Max |",
            "|------|------|-----|-----|--------|-----|",
        ]

        for ax in range(3):
            s = img["spacing"]["per_axis"][f"axis_{ax}"]
            lines.append(
                f"| {ax} | {s['mean']} | {s['std']} | {s['min']} "
                f"| {s['median']} | {s['max']} |"
            )

        aniso = img["spacing"]["anisotropy_ratio"]
        aniso_label = (
            "anisotropic"
            if img["spacing"]["is_anisotropic"]
            else "isotropic"
        )
        lines += [
            "",
            f"**Anisotropy ratio:** {aniso:.2f} ({aniso_label})",
            "",
            "### Original Dimensions (voxels)",
            "| Axis | Mean | Std | Min | Median | Max |",
            "|------|------|-----|-----|--------|-----|",
        ]

        for ax in range(3):
            d = img["dimensions"]["original"]["per_axis"][f"axis_{ax}"]
            lines.append(
                f"| {ax} | {d['mean']:.1f} | {d['std']:.1f} | {d['min']:.0f} "
                f"| {d['median']:.1f} | {d['max']:.0f} |"
            )

        med = img["dimensions"]["resampled_median"]
        lines += [
            "",
            (
                f"**Median resampled dimensions:** "
                f"{med[0]} \u00d7 {med[1]} \u00d7 {med[2]} voxels"
            ),
            "",
            "### Intensity Distributions (foreground voxels)",
        ]

        for ch, stats in img["intensity"]["per_channel"].items():
            lines += [
                "",
                f"**Channel: {ch}**",
                (
                    f"- Mean \u00b1 Std: "
                    f"{stats['mean']:.2f} \u00b1 {stats['std']:.2f}"
                ),
                (
                    f"- Percentiles: p01={stats['p01']:.2f}, "
                    f"p05={stats['p05']:.2f}, p25={stats['p25']:.2f}, "
                    f"p50={stats['p50']:.2f}, p75={stats['p75']:.2f}, "
                    f"p95={stats['p95']:.2f}, p99={stats['p99']:.2f}"
                ),
            ]

        nz = img["intensity"]["foreground_fraction"]
        lines += [
            "",
            (
            f"**Foreground density:** mean={nz['mean']:.3f}, "
            f"std={nz['std']:.3f}, "
            f"min={nz['min']:.3f}, max={nz['max']:.3f}"
        ),
            "",
            "## Label Statistics",
            "",
            "### Per-Label Summary",
            (
                "| Label | Mean Voxels \u00b1 Std | Presence Rate | "
                "Vol. Fraction of FG | Vol. Fraction of Img | Size | Shape "
                "| Lin. | Plan. | Sph. | IQ | Skel. |"
            ),
            (
                "|-------|---------------------|--------------|"
                "--------------------|----------------------|------|-------"
                "|------|-------|------|----|----|"
            ),
        ]

        for lbl_str, lbl_data in lbl["per_label"].items():
            vc = lbl_data["voxel_count"]
            sh = lbl_data["shape"]
            lin = (
                f"{sh['linearity']:.2f}"
                if sh["linearity"] is not None
                else "\u2014"
            )
            plan = (
                f"{sh['planarity']:.2f}"
                if sh["planarity"] is not None
                else "\u2014"
            )
            sph = (
                f"{sh['sphericity']:.2f}"
                if sh["sphericity"] is not None
                else "\u2014"
            )
            iq = (
                f"{sh['compactness']:.3f}"
                if sh.get("compactness") is not None
                else "\u2014"
            )
            skel = (
                f"{sh['skeleton_ratio']:.3f}"
                if sh.get("skeleton_ratio") is not None
                else "\u2014"
            )
            vol_frac_fg = (
                lbl_data['mean_volume_fraction_of_foreground_pct']
            )
            vol_frac_img = (
                lbl_data['mean_volume_fraction_of_image_pct']
            )
            lines.append(
                f"| {lbl_str} | {vc['mean']:.0f} \u00b1 {vc['std']:.0f} "
                f"| {lbl_data['presence_rate_pct']:.1f}% | "
                f"{vol_frac_fg:.4f}% | "
                f"{vol_frac_img:.4f}% | "
                f"{lbl_data['size_category']} | {sh['shape_class']} | "
                f"{lin} | {plan} | {sph} | {iq} | {skel} |"
            )

        lines += [
            "",
            "### Final Classes",
            "| Class | Labels | Vol. Fraction of FG | Presence Rate | Size |",
            "|-------|--------|---------------------|---------------|------|",
        ]

        for class_name, class_data in lbl["final_classes"].items():
            vol_frac_fc = (
                class_data['mean_volume_fraction_of_foreground_pct']
            )
            lines.append(
                f"| {class_name} | "
                f"{class_data['constituent_labels']} | "
                f"{vol_frac_fc:.4f}% | "
                f"{class_data['presence_rate_pct']:.1f}% | "
                f"{class_data['size_category']} |"
            )

        ci = lbl["class_imbalance"]
        lines += [
            "",
            (
                f"**Class imbalance ratio:** {ci['imbalance_ratio']:.1f}x "
                f"(label {ci['dominant_label']} vs label "
                f"{ci['minority_label']})"
            ),
            "",
            "## Metric Definitions",
            "",
            "The following definitions explain every metric reported above.",
            "",
            "### Spacing and Anisotropy",
            "- **Spacing (mm):** Physical size of each voxel along each axis "
            "(row, column, slice). Affects how the image is resampled before "
            "training.",
            "- **Anisotropy ratio:** max(spacing) / min(spacing) across all "
            "axes and patients. A ratio > 3 indicates the dataset is "
            "anisotropic — voxels are substantially thicker in one direction "
            "— which may require axis-specific handling during resampling and "
            "patch sampling.",
            "",
            "### Image Dimensions",
            "- **Original dimensions:** Voxel counts along each axis before "
            "resampling.",
            "- **Median resampled dimensions:** Estimated image size after "
            "resampling to the target spacing, derived from the MIST config. "
            "Used to inform patch size and memory budget.",
            "",
            "### Intensity",
            "- **Foreground voxels:** Voxels inside the ground-truth "
            "segmentation mask (non-background). Intensity statistics are "
            "computed over foreground only to avoid background bias.",
            "- **Percentiles (p01–p99):** Robust summary of the intensity "
            "distribution. Wide ranges suggest high inter-patient variability "
            "or the presence of outliers.",
            "- **Foreground density:** Proportion of voxels in the full image "
            "volume that belong to any non-background label. Low values are "
            "expected for tasks targeting small or sparse structures "
            "(e.g., vessels, small lesions) and do not indicate a data "
            "quality issue.",
            "",
            "### Label / Class Statistics",
            "- **Voxel count:** Number of voxels assigned to a label for each "
            "patient. Mean ± std summarises cross-patient variability.",
            "- **Presence rate:** Percentage of patients in which the label "
            "appears at all. A low presence rate means the label is absent in "
            "many scans.",
            "- **Vol. fraction of foreground (%):** Mean label voxel count "
            "divided by mean total foreground voxel count, expressed as a "
            "percentage. Measures how much of the foreground each label "
            "occupies. A label can be 100% of the foreground while still "
            "being a tiny fraction of the overall image.",
            "- **Vol. fraction of image (%):** Mean label voxel count divided "
            "by mean effective image voxel count, expressed as a percentage. "
            "When `crop_to_foreground` is enabled the denominator is the "
            "foreground bounding box volume (the region the model actually "
            "sees); otherwise it is the full original image volume. "
            "Captures how sparse the label is relative to the image region "
            "the model operates on. Low values are expected for small "
            "structures like vessels or lesions even when they dominate "
            "the foreground.",
            "- **Size category:** Qualitative bucket based on vol. fraction of "
            "foreground: tiny (< 0.1%), small (0.1–1%), medium (1–5%), "
            "large (≥ 5%).",
            "- **Class imbalance ratio:** dominant label vol. fraction / "
            "minority label vol. fraction. A ratio > 10 indicates severe "
            "imbalance that may require loss weighting or oversampling.",
            "",
            "### Shape Descriptors",
            "Shape descriptors are computed per label per patient and then "
            "averaged across patients. They characterise the geometry of the "
            "label region and inform loss function and architecture choices.",
            "",
            "**PCA-based descriptors** decompose the covariance matrix of the "
            "label voxel coordinates (in mm-space) into three eigenvalues "
            "\u03bb\u2081 \u2265 \u03bb\u2082 \u2265 \u03bb\u2083. The "
            "normalised values sum to 1:",
            "- **Linearity:** (\u03bb\u2081 \u2212 \u03bb\u2082) / "
            "(\u03bb\u2081 + \u03bb\u2082 + \u03bb\u2083). Dominant when the "
            "label extends strongly along one axis (e.g., a straight vessel "
            "segment or spine).",
            "- **Planarity:** (\u03bb\u2082 \u2212 \u03bb\u2083) / "
            "(\u03bb\u2081 + \u03bb\u2082 + \u03bb\u2083). Dominant when the "
            "label lies primarily in a plane (e.g., a thin cortical sheet).",
            "- **Sphericity:** \u03bb\u2083 / "
            "(\u03bb\u2081 + \u03bb\u2082 + \u03bb\u2083). Dominant when "
            "spread is roughly equal in all directions (e.g., a round tumor).",
            "- **Shape class:** The descriptor with the highest value "
            "determines the class: tubular (linearity), planar (planarity), "
            "or blob (sphericity). Note: PCA operates on the global bounding "
            "ellipsoid of the label. A branching vessel tree may appear "
            "blob-like by PCA even though it is locally thin and tubular — "
            "use skeleton ratio as the primary tubular signal.",
            "",
            "**Compactness (Isoperimetric Quotient, IQ):** "
            "36\u03c0 \u00b7 V\u00b2 / SA\u00b3, where V is label volume "
            "(mm\u00b3) and SA is surface area (mm\u00b2). A perfect sphere "
            "scores 1.0; thin, branching, or irregular structures score near "
            "0 because they have disproportionately large surface area "
            "relative to their volume.",
            "",
            "**Skeleton ratio:** skeleton voxels / total label voxels, where "
            "the skeleton is the morphological medial axis (skimage "
            "skeletonize). High values indicate that most label voxels lie "
            "close to the centerline — the hallmark of thin, branching "
            "structures such as vessels or airways. Skeletonization is skipped "
            "for labels exceeding 500,000 voxels (reported as \u2014); "
            "structures that large are rarely thin/tubular.",
            "",
            "## Observations",
            "",
            (
                "*The following observations are auto-generated by MIST "
                "from the statistics above. Review each item, remove any "
                "that are incorrect or not relevant, and add your own "
                "observations below before passing this document to an LLM.*"
            ),
            "",
        ]

        for obs_item in obs:
            lines.append(f"- {obs_item}")

        lines += [
            "",
            "<!-- User notes (review and extend the observations above)",
            "Example prompts to guide your annotations:",
            "  - Are any auto-generated observations incorrect or misleading?",
            "  - Are there labels that are clinically more important than others?",
            "  - Are there known annotation artefacts, protocol changes, or",
            "    site-specific biases that the statistics do not capture?",
            "  - Is the class imbalance a real biological phenomenon or an",
            "    artefact of the annotation protocol?",
            "  - Are there patient subgroups (e.g., age, disease stage,",
            "    scanner vendor) that should be treated differently?",
            "  - What evaluation metric matters most for this clinical task?",
            "-->",
        ]

        lines += [
            "",
            "---",
            f"*MIST config saved at: {dump['mist_config_path']}*",
        ]

        return "\n".join(lines)

    def run(self) -> None:
        """Build data dump and save data_dump.json and data_dump.md."""
        dump = self.build_data_dump()

        data_dump_json = os.path.join(self.results_dir, "data_dump.json")
        io_utils.write_json_file(data_dump_json, dump)

        data_dump_md = os.path.join(self.results_dir, "data_dump.md")
        with open(data_dump_md, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_summary(dump))

        self.console.print(
            f"[green]Data dump saved to {data_dump_json} "
            f"and {data_dump_md}[/green]"
        )
