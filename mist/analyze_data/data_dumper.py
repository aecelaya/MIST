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
from typing import Dict, Any

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
    ):
        self.paths_df = paths_df
        self.dataset_info = dataset_info
        self.config = config
        self.results_dir = results_dir
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

        # Single pass over all patients to collect raw statistics.
        raw_stats = data_dump_utils.collect_per_patient_stats(
            self.paths_df, self.dataset_info
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
                f"{', '.join(ds['final_classes'].keys())}"
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

        nz = img["intensity"]["nonzero_fraction"]
        lines += [
            "",
            (
            f"**Non-zero fraction:** mean={nz['mean']:.3f}, "
            f"std={nz['std']:.3f}, "
            f"min={nz['min']:.3f}, max={nz['max']:.3f}"
        ),
            "",
            "## Label Statistics",
            "",
            "### Per-Label Summary",
            (
                "| Label | Mean Voxels \u00b1 Std | Presence Rate | "
                "Vol. Fraction of FG | Size | Shape | Lin. | Plan. | Sph. |"
            ),
            (
                "|-------|---------------------|--------------|"
                "--------------------|------|-------|------|-------|------|"
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
            vol_frac = (
                lbl_data['mean_volume_fraction_of_foreground_pct']
            )
            lines.append(
                f"| {lbl_str} | {vc['mean']:.0f} \u00b1 {vc['std']:.0f} "
                f"| {lbl_data['presence_rate_pct']:.1f}% | "
                f"{vol_frac:.4f}% | "
                f"{lbl_data['size_category']} | {sh['shape_class']} | "
                f"{lin} | {plan} | {sph} |"
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
            "## Observations",
            "",
        ]

        for obs_item in obs:
            lines.append(f"- {obs_item}")

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
