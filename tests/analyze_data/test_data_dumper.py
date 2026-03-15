"""Tests for mist.analyze_data.data_dumper."""
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from mist.analyze_data.data_dumper import DataDumper
from mist.analyze_data import data_dump_utils as ddu
from mist.utils import io as io_mod


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _make_paths_df(n=3) -> pd.DataFrame:
    """Minimal paths DataFrame."""
    return pd.DataFrame({
        "id": list(range(n)),
        "fold": list(range(n)),
        "mask": [f"p{i}_mask.nii.gz" for i in range(n)],
        "t1": [f"p{i}_t1.nii.gz" for i in range(n)],
    })


def _make_dataset_info() -> Dict[str, Any]:
    """Minimal dataset_info dictionary."""
    return {
        "task": "test-task",
        "modality": "MR",
        "images": {"t1": ["t1"]},
        "labels": [0, 1, 2],
        "final_classes": {"tumor": [1, 2]},
    }


def _make_config() -> Dict[str, Any]:
    """Minimal MIST config dictionary."""
    return {
        "preprocessing": {"median_resampled_image_size": [64, 64, 32]}
    }


def _make_controlled_dump(results_dir: str) -> Dict[str, Any]:
    """Build a fully specified dump dict for markdown/run tests."""
    return {
        "dataset_summary": {
            "task": "test-task",
            "modality": "mr",
            "num_patients": 10,
            "num_channels": 2,
            "channel_names": ["t1", "t2"],
            "num_labels": 3,
            "labels": [0, 1, 2],
            "final_classes": {"tumor": [1, 2]},
            "dataset_size_gb": 0.5,
        },
        "image_statistics": {
            "spacing": {
                "per_axis": {
                    "axis_0": {
                        "mean": 1.0, "std": 0.0, "min": 1.0,
                        "p25": 1.0, "median": 1.0, "p75": 1.0,
                        "max": 1.0,
                    },
                    "axis_1": {
                        "mean": 1.0, "std": 0.0, "min": 1.0,
                        "p25": 1.0, "median": 1.0, "p75": 1.0,
                        "max": 1.0,
                    },
                    "axis_2": {
                        "mean": 2.5, "std": 0.0, "min": 2.5,
                        "p25": 2.5, "median": 2.5, "p75": 2.5,
                        "max": 2.5,
                    },
                },
                "median_spacing_mm": [1.0, 1.0, 2.5],
                "anisotropy_ratio": 2.5,
                "is_anisotropic": False,
            },
            "dimensions": {
                "original": {
                    "per_axis": {
                        "axis_0": {
                            "mean": 64.0, "std": 0.0, "min": 64.0,
                            "p25": 64.0, "median": 64.0, "p75": 64.0,
                            "max": 64.0,
                        },
                        "axis_1": {
                            "mean": 64.0, "std": 0.0, "min": 64.0,
                            "p25": 64.0, "median": 64.0, "p75": 64.0,
                            "max": 64.0,
                        },
                        "axis_2": {
                            "mean": 32.0, "std": 0.0, "min": 32.0,
                            "p25": 32.0, "median": 32.0, "p75": 32.0,
                            "max": 32.0,
                        },
                    }
                },
                "resampled_median": [64, 64, 32],
            },
            "intensity": {
                "per_channel": {
                    "t1": {
                        "mean": 100.0, "std": 50.0, "p01": -10.0,
                        "p05": 0.0, "p25": 60.0, "p50": 100.0,
                        "p75": 140.0, "p95": 190.0, "p99": 210.0,
                    },
                },
                "foreground_fraction": {
                    "mean": 0.85, "std": 0.05, "min": 0.8,
                    "p25": 0.83, "median": 0.85, "p75": 0.87,
                    "max": 0.9,
                },
            },
        },
        "label_statistics": {
            "per_label": {
                "1": {
                    "voxel_count": {
                        "mean": 500.0, "std": 50.0, "min": 450,
                        "median": 500.0, "max": 550,
                    },
                    "mean_volume_fraction_of_foreground_pct": 50.0,
                    "mean_volume_fraction_of_image_pct": 2.0,
                    "presence_rate_pct": 100.0,
                    "size_category": "large",
                    "shape": {
                        "shape_class": "tubular",
                        "linearity": 0.65,
                        "planarity": 0.25,
                        "sphericity": 0.10,
                        "compactness": 0.05,
                        "skeleton_ratio": 0.08,
                    },
                },
                "2": {
                    "voxel_count": {
                        "mean": 50.0, "std": 10.0, "min": 40,
                        "median": 50.0, "max": 60,
                    },
                    "mean_volume_fraction_of_foreground_pct": 5.0,
                    "mean_volume_fraction_of_image_pct": 0.2,
                    "presence_rate_pct": 80.0,
                    "size_category": "medium",
                    "shape": {
                        "shape_class": "blob",
                        "linearity": None,
                        "planarity": None,
                        "sphericity": None,
                        "compactness": None,
                        "skeleton_ratio": None,
                    },
                },
            },
            "final_classes": {
                "tumor": {
                    "constituent_labels": [1, 2],
                    "mean_volume_fraction_of_foreground_pct": 55.0,
                    "presence_rate_pct": 100.0,
                    "size_category": "large",
                },
            },
            "class_imbalance": {
                "imbalance_ratio": 10.0,
                "dominant_label": 1,
                "minority_label": 2,
            },
        },
        "observations": [
            "Observation one.",
            "Observation two.",
        ],
        "mist_config_path": os.path.join(results_dir, "config.json"),
    }


@pytest.fixture
def dumper(tmp_path):
    """Create a DataDumper instance with controlled inputs."""
    return DataDumper(
        paths_df=_make_paths_df(),
        dataset_info=_make_dataset_info(),
        config=_make_config(),
        results_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestDataDumperInit:
    """Tests for DataDumper.__init__."""

    def test_attributes_are_stored(self, tmp_path):
        """All constructor arguments are stored as instance attributes."""
        df = _make_paths_df()
        info = _make_dataset_info()
        cfg = _make_config()
        d = DataDumper(df, info, cfg, str(tmp_path))
        assert d.paths_df is df
        assert d.dataset_info is info
        assert d.config is cfg
        assert d.results_dir == str(tmp_path)
        assert d.cropped_dims is None

    def test_cropped_dims_stored(self, tmp_path):
        """cropped_dims argument is stored as an instance attribute."""
        dims = np.ones((2, 3))
        d = DataDumper(
            _make_paths_df(), _make_dataset_info(), _make_config(),
            str(tmp_path), cropped_dims=dims,
        )
        assert d.cropped_dims is dims


# ---------------------------------------------------------------------------
# _build_dataset_summary
# ---------------------------------------------------------------------------

class TestBuildDatasetSummary:
    """Tests for DataDumper._build_dataset_summary."""

    def test_all_fields_present_and_correct(self, dumper):
        """Summary contains all expected keys with correct values."""
        summary = dumper._build_dataset_summary()
        assert summary["task"] == "test-task"
        assert summary["modality"] == "mr"
        assert summary["num_patients"] == len(dumper.paths_df)
        assert summary["num_channels"] == 1
        assert summary["channel_names"] == ["t1"]
        assert summary["num_labels"] == 3
        assert summary["labels"] == [0, 1, 2]
        assert summary["final_classes"] == {"tumor": [1, 2]}
        assert isinstance(summary["dataset_size_gb"], float)

    def test_modality_is_lowercased(self, tmp_path):
        """Modality is normalised to lower-case regardless of input."""
        info = _make_dataset_info()
        info["modality"] = "CT"
        d = DataDumper(_make_paths_df(), info, _make_config(), str(tmp_path))
        assert d._build_dataset_summary()["modality"] == "ct"


# ---------------------------------------------------------------------------
# build_data_dump
# ---------------------------------------------------------------------------

class TestBuildDataDump:
    """Tests for DataDumper.build_data_dump."""

    def test_all_top_level_keys_present(self, dumper, monkeypatch):
        """Returned dict contains all five top-level keys."""
        # Stub heavy helpers to return minimal dicts.
        monkeypatch.setattr(
            ddu, "collect_per_patient_stats", lambda *_, **__: {}, raising=True
        )
        monkeypatch.setattr(
            ddu,
            "build_image_statistics",
            lambda *_: {
                "spacing": {}, "dimensions": {}, "intensity": {}
            },
            raising=True,
        )
        monkeypatch.setattr(
            ddu,
            "build_label_statistics",
            lambda *_: {
                "per_label": {}, "final_classes": {}, "class_imbalance": {}
            },
            raising=True,
        )
        monkeypatch.setattr(
            ddu, "generate_observations", lambda *_: [], raising=True
        )

        result = dumper.build_data_dump()
        assert set(result.keys()) == {
            "dataset_summary",
            "image_statistics",
            "label_statistics",
            "observations",
            "mist_config_path",
        }

    def test_mist_config_path_points_to_results_dir(self, dumper, monkeypatch):
        """mist_config_path is <results_dir>/config.json."""
        monkeypatch.setattr(
            ddu, "collect_per_patient_stats", lambda *_, **__: {}, raising=True
        )
        monkeypatch.setattr(
            ddu, "build_image_statistics", lambda *_: {}, raising=True
        )
        monkeypatch.setattr(
            ddu, "build_label_statistics", lambda *_: {}, raising=True
        )
        monkeypatch.setattr(
            ddu, "generate_observations", lambda *_: [], raising=True
        )

        result = dumper.build_data_dump()
        assert result["mist_config_path"] == os.path.join(
            dumper.results_dir, "config.json"
        )

    def test_collect_called_with_paths_df_and_dataset_info(
        self, dumper, monkeypatch
    ):
        """collect_per_patient_stats receives the right arguments."""
        calls = []
        monkeypatch.setattr(
            ddu, "collect_per_patient_stats",
            lambda df, info, **kw: (calls.append((df, info)) or {}),
            raising=True,
        )
        monkeypatch.setattr(
            ddu, "build_image_statistics", lambda *_: {}, raising=True
        )
        monkeypatch.setattr(
            ddu, "build_label_statistics", lambda *_: {}, raising=True
        )
        monkeypatch.setattr(
            ddu, "generate_observations", lambda *_: [], raising=True
        )

        dumper.build_data_dump()
        assert len(calls) == 1
        called_df, called_info = calls[0]
        assert called_df is dumper.paths_df
        assert called_info is dumper.dataset_info


# ---------------------------------------------------------------------------
# generate_markdown_summary
# ---------------------------------------------------------------------------

class TestGenerateMarkdownSummary:
    """Tests for DataDumper.generate_markdown_summary."""

    @pytest.fixture
    def dump(self, tmp_path):
        """Provide a controlled dump dict for markdown tests."""
        return _make_controlled_dump(str(tmp_path))

    def test_contains_task_name(self, dumper, dump):
        """Markdown title includes the task name."""
        md = dumper.generate_markdown_summary(dump)
        assert "test-task" in md

    def test_contains_all_section_headers(self, dumper, dump):
        """All major sections are present in the markdown output."""
        md = dumper.generate_markdown_summary(dump)
        for header in (
            "## Dataset Summary",
            "## Image Statistics",
            "## Label Statistics",
            "## Observations",
        ):
            assert header in md

    def test_per_label_table_present(self, dumper, dump):
        """Per-label rows appear in the label statistics table."""
        md = dumper.generate_markdown_summary(dump)
        assert "| 1 |" in md
        assert "| 2 |" in md

    def test_none_shape_values_render_as_emdash(self, dumper, dump):
        """Shape values that are None are shown as an em-dash in the table."""
        md = dumper.generate_markdown_summary(dump)
        # Label 2 has all-None shape values.
        assert "\u2014" in md

    def test_numeric_shape_values_rendered(self, dumper, dump):
        """Non-None shape values are formatted as floats."""
        md = dumper.generate_markdown_summary(dump)
        # Label 1 has linearity=0.65 → "0.65" should appear.
        assert "0.65" in md

    def test_observations_section_contains_obs(self, dumper, dump):
        """Each observation string appears in the markdown."""
        md = dumper.generate_markdown_summary(dump)
        assert "Observation one." in md
        assert "Observation two." in md

    def test_anisotropy_label_shown(self, dumper, dump):
        """Anisotropy label appears in markdown."""
        md = dumper.generate_markdown_summary(dump)
        assert "isotropic" in md

    def test_resampled_dims_shown(self, dumper, dump):
        """Median resampled dimensions appear in markdown."""
        md = dumper.generate_markdown_summary(dump)
        assert "64" in md and "32" in md

    def test_config_path_appears_at_end(self, dumper, dump):
        """The MIST config path appears in the closing line."""
        md = dumper.generate_markdown_summary(dump)
        assert dump["mist_config_path"] in md


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestDataDumperRun:
    """Tests for DataDumper.run — verifies file I/O and console output."""

    @pytest.fixture
    def _stub_build(self, monkeypatch, tmp_path):
        """Replace build_data_dump with a fast stub for run() tests."""
        dump = _make_controlled_dump(str(tmp_path))
        monkeypatch.setattr(
            DataDumper, "build_data_dump", lambda self: dump, raising=True
        )
        # Stub io.write_json_file to write the file (fast, no numpy issues).
        monkeypatch.setattr(
            io_mod, "write_json_file",
            lambda path, data: Path(path).write_text(
                json.dumps(data), encoding="utf-8"
            ),
            raising=True,
        )
        return dump

    def test_json_file_is_created(self, dumper, _stub_build, tmp_path):
        """run() creates data_dump.json in results_dir."""
        dumper.run()
        assert (tmp_path / "data_dump.json").exists()

    def test_markdown_file_is_created(self, dumper, _stub_build, tmp_path):
        """run() creates data_dump.md in results_dir."""
        dumper.run()
        assert (tmp_path / "data_dump.md").exists()

    def test_json_file_is_valid_json(self, dumper, _stub_build, tmp_path):
        """data_dump.json is parseable JSON."""
        dumper.run()
        content = (tmp_path / "data_dump.json").read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_markdown_file_has_content(self, dumper, _stub_build, tmp_path):
        """data_dump.md is non-empty and contains section headers."""
        dumper.run()
        md = (tmp_path / "data_dump.md").read_text(encoding="utf-8")
        assert len(md) > 0
        assert "## Dataset Summary" in md

    def test_console_print_called(self, dumper, _stub_build, monkeypatch):
        """run() prints a confirmation message to the Rich console."""
        printed = []
        def _print(self, *a, **k):
            """Capture printed messages."""
            printed.append(" ".join(map(str, a)))

        monkeypatch.setattr("rich.console.Console.print", _print)
        dumper.run()
        assert any("data_dump" in msg for msg in printed)
