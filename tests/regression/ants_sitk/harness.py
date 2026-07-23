"""Stage 0 regression harness: run the current pipeline and diff its artifacts.

This module drives ``mist_analyze`` -> ``mist_preprocess`` over the fixed
fixture set (see :mod:`fixtures`) and captures every intermediate artifact that
later migration stages are expected to reproduce exactly:

    results/config.json         (analyze)
    results/train_paths.csv     (analyze)
    results/fg_bboxes.csv       (analyze)  -- Stage 3 / Stage 4 diff target
    numpy/images/<id>.npy       (preprocess) -- Stage 4 diff target
    numpy/labels/<id>.npy       (preprocess) -- Stage 4 diff target
    numpy/dtms/<id>.npy         (preprocess) -- Stage 4 diff target

Prediction artifacts (Stage 5) require a trained model, which the migration
plan defers to that stage; :func:`run_prediction` is a thin, optional hook for
when a model/config is available rather than something baked into the Stage 0
gate.

Two ways to use it:

* Programmatically, from the pytest gate in ``test_stage0_selfdiff.py``.
* As a CLI, to capture a golden set once and diff future runs against it::

      python -m tests.regression.ants_sitk.harness capture --golden-dir GOLDEN
      python -m tests.regression.ants_sitk.harness diff    --golden-dir GOLDEN
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tests.regression.ants_sitk import fixtures

# Placeholder that absolute paths are collapsed to before comparison, so an
# artifact captured under one working directory diffs cleanly against a run
# under a different one.
_ROOT_PLACEHOLDER = "<ROOT>"

# Artifacts whose contents are pure paths/ids we still compare (after path
# normalization) because their *structure* must not drift silently.
_ANALYZE_ARTIFACTS = ("config.json", "train_paths.csv", "fg_bboxes.csv")
_NPY_SUBDIRS = ("images", "labels", "dtms")


# --------------------------------------------------------------------------- #
# Running the pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineOutputs:
    """Locations produced by :func:`run_pipeline`."""

    results_dir: Path
    numpy_dir: Path


def run_pipeline(
    dataset_json: Path,
    results_dir: Path,
    numpy_dir: Path,
    *,
    compute_dtms: bool = True,
    nfolds: int = 2,
) -> PipelineOutputs:
    """Run analyze -> preprocess over ``dataset_json`` into the given dirs.

    Uses a single worker throughout so the output is deterministic.
    """
    # Import entrypoints lazily so importing this module stays cheap and does
    # not pull heavy optional deps until the pipeline is actually run.
    from mist.cli.analyze_entrypoint import analyze_entry
    from mist.cli.preprocess_entrypoint import preprocess_entry

    results_dir = Path(results_dir)
    numpy_dir = Path(numpy_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    analyze_entry(
        [
            "--data", str(dataset_json),
            "--results", str(results_dir),
            "--nfolds", str(nfolds),
            "--num-workers-analyze", "1",
            "--overwrite",
        ]
    )

    preprocess_argv = [
        "--results", str(results_dir),
        "--numpy", str(numpy_dir),
        "--num-workers-preprocess", "1",
        "--overwrite",
    ]
    if compute_dtms:
        preprocess_argv.append("--compute-dtms")
    preprocess_entry(preprocess_argv)

    return PipelineOutputs(results_dir=results_dir, numpy_dir=numpy_dir)


def run_prediction(*_args: Any, **_kwargs: Any) -> None:
    """Optional Stage 5 hook (predictions require a trained model).

    Intentionally not implemented for the Stage 0 gate: the migration plan
    defers trained-model prediction diffs to Stage 5. Wire a model/config in
    here when that stage begins.
    """
    raise NotImplementedError(
        "Prediction capture is a Stage 5 concern and needs a trained model. "
        "The Stage 0 gate covers analyze + preprocess artifacts only."
    )


# --------------------------------------------------------------------------- #
# Collecting artifacts
# --------------------------------------------------------------------------- #
def collect_artifacts(outputs: PipelineOutputs) -> dict[str, Path]:
    """Map a stable relative key -> file path for every captured artifact."""
    artifacts: dict[str, Path] = {}

    for name in _ANALYZE_ARTIFACTS:
        path = outputs.results_dir / name
        if path.is_file():
            artifacts[name] = path

    for subdir in _NPY_SUBDIRS:
        directory = outputs.numpy_dir / subdir
        if not directory.is_dir():
            continue
        for npy in sorted(directory.glob("*.npy")):
            artifacts[f"{subdir}/{npy.name}"] = npy

    return artifacts


def copy_artifacts(artifacts: dict[str, Path], dest_root: Path) -> None:
    """Copy collected artifacts under ``dest_root`` preserving their keys."""
    dest_root = Path(dest_root)
    for key, src in artifacts.items():
        dest = dest_root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _map_from_root(root: Path) -> dict[str, Path]:
    """Reconstruct an artifact key -> path map from a captured golden dir."""
    root = Path(root)
    artifacts: dict[str, Path] = {}
    for name in _ANALYZE_ARTIFACTS:
        path = root / name
        if path.is_file():
            artifacts[name] = path
    for subdir in _NPY_SUBDIRS:
        directory = root / subdir
        if directory.is_dir():
            for npy in sorted(directory.glob("*.npy")):
                artifacts[f"{subdir}/{npy.name}"] = npy
    return artifacts


# --------------------------------------------------------------------------- #
# Diffing
# --------------------------------------------------------------------------- #
@dataclass
class DiffReport:
    """Result of comparing two artifact sets."""

    differences: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.differences

    def __str__(self) -> str:
        if self.identical:
            return "IDENTICAL: no differences found."
        lines = [f"{len(self.differences)} difference(s) found:"]
        lines.extend(f"  - {d}" for d in self.differences)
        return "\n".join(lines)


def _normalize_string(value: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old:
            value = value.replace(old, new)
    return value


def _normalize(obj: Any, replacements: list[tuple[str, str]]) -> Any:
    """Recursively normalize path-like strings inside a JSON-like object."""
    if isinstance(obj, str):
        return _normalize_string(obj, replacements)
    if isinstance(obj, dict):
        return {k: _normalize(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v, replacements) for v in obj]
    return obj


def _numbers_close(a: float, b: float, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * abs(b)


def _compare_json(
    key: str,
    golden: Any,
    candidate: Any,
    atol: float,
    rtol: float,
    path: str = "",
) -> list[str]:
    """Deep-compare two JSON-like objects with float tolerance."""
    diffs: list[str] = []
    loc = f"{key}{path}"

    if isinstance(golden, dict) and isinstance(candidate, dict):
        for missing in sorted(set(golden) - set(candidate)):
            diffs.append(f"{loc}.{missing}: present in golden, missing in candidate")
        for added in sorted(set(candidate) - set(golden)):
            diffs.append(f"{loc}.{added}: present in candidate, missing in golden")
        for shared in sorted(set(golden) & set(candidate)):
            diffs.extend(
                _compare_json(
                    key, golden[shared], candidate[shared], atol, rtol,
                    f"{path}.{shared}",
                )
            )
        return diffs

    if isinstance(golden, list) and isinstance(candidate, list):
        if len(golden) != len(candidate):
            diffs.append(
                f"{loc}: list length {len(golden)} (golden) != "
                f"{len(candidate)} (candidate)"
            )
            return diffs
        for i, (g, c) in enumerate(zip(golden, candidate)):
            diffs.extend(_compare_json(key, g, c, atol, rtol, f"{path}[{i}]"))
        return diffs

    if isinstance(golden, bool) or isinstance(candidate, bool):
        if golden != candidate:
            diffs.append(f"{loc}: {golden!r} (golden) != {candidate!r} (candidate)")
        return diffs

    if isinstance(golden, (int, float)) and isinstance(candidate, (int, float)):
        if not _numbers_close(float(golden), float(candidate), atol, rtol):
            diffs.append(
                f"{loc}: {golden} (golden) != {candidate} (candidate) "
                f"(atol={atol}, rtol={rtol})"
            )
        return diffs

    if golden != candidate:
        diffs.append(f"{loc}: {golden!r} (golden) != {candidate!r} (candidate)")
    return diffs


def _compare_csv(
    key: str,
    golden_path: Path,
    candidate_path: Path,
    replacements: list[tuple[str, str]],
    atol: float,
    rtol: float,
) -> list[str]:
    """Compare two CSV files cell-by-cell (numeric cells use float tolerance)."""
    diffs: list[str] = []
    with open(golden_path, newline="") as f:
        golden_rows = list(csv.reader(f))
    with open(candidate_path, newline="") as f:
        candidate_rows = list(csv.reader(f))

    if len(golden_rows) != len(candidate_rows):
        diffs.append(
            f"{key}: row count {len(golden_rows)} (golden) != "
            f"{len(candidate_rows)} (candidate)"
        )
        return diffs

    for r, (grow, crow) in enumerate(zip(golden_rows, candidate_rows)):
        if len(grow) != len(crow):
            diffs.append(f"{key}[row {r}]: column count differs")
            continue
        for c, (gcell, ccell) in enumerate(zip(grow, crow)):
            gcell_n = _normalize_string(gcell, replacements)
            ccell_n = _normalize_string(ccell, replacements)
            if gcell_n == ccell_n:
                continue
            try:
                if _numbers_close(float(gcell_n), float(ccell_n), atol, rtol):
                    continue
            except ValueError:
                pass
            diffs.append(
                f"{key}[row {r}, col {c}]: {gcell_n!r} (golden) != "
                f"{ccell_n!r} (candidate)"
            )
    return diffs


def _compare_npy(
    key: str,
    golden_path: Path,
    candidate_path: Path,
    atol: float,
    rtol: float,
) -> list[str]:
    """Compare two ``.npy`` arrays.

    Integer arrays (labels) must match exactly; floating arrays (images, dtms)
    use the given tolerance. Shape mismatches are reported first because a
    transposed axis is exactly the silent bug this harness exists to catch.
    """
    golden = np.load(golden_path)
    candidate = np.load(candidate_path)

    if golden.shape != candidate.shape:
        return [
            f"{key}: shape {golden.shape} (golden) != {candidate.shape} "
            f"(candidate) -- possible axis transpose"
        ]

    if np.issubdtype(golden.dtype, np.integer) and np.issubdtype(
        candidate.dtype, np.integer
    ):
        if not np.array_equal(golden, candidate):
            mismatched = int(np.count_nonzero(golden != candidate))
            return [
                f"{key}: {mismatched} voxel(s) differ (exact integer compare)"
            ]
        return []

    if not np.allclose(golden, candidate, atol=atol, rtol=rtol, equal_nan=True):
        abs_diff = np.abs(golden.astype(np.float64) - candidate.astype(np.float64))
        n_bad = int(np.count_nonzero(abs_diff > (atol + rtol * np.abs(candidate))))
        return [
            f"{key}: {n_bad} voxel(s) exceed tolerance "
            f"(max abs diff {abs_diff.max():.3e}, atol={atol}, rtol={rtol})"
        ]
    return []


def diff_artifacts(
    golden: dict[str, Path],
    candidate: dict[str, Path],
    *,
    replacements: list[tuple[str, str]] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> DiffReport:
    """Compare two artifact maps and return a :class:`DiffReport`.

    ``replacements`` maps working-directory roots to a placeholder so path
    strings inside JSON/CSV artifacts compare across different run locations.
    Defaults (``atol=rtol=0``) require exact equality, which is the correct
    gate for the Stage 0 self-diff sanity check; later migration stages loosen
    the tolerance for floating-point artifacts.
    """
    replacements = replacements or []
    report = DiffReport()

    for missing in sorted(set(golden) - set(candidate)):
        report.differences.append(f"{missing}: present in golden, missing in candidate")
    for added in sorted(set(candidate) - set(golden)):
        report.differences.append(f"{added}: present in candidate, missing in golden")

    for key in sorted(set(golden) & set(candidate)):
        gpath, cpath = golden[key], candidate[key]
        if key.endswith(".json"):
            g = _normalize(json.loads(gpath.read_text()), replacements)
            c = _normalize(json.loads(cpath.read_text()), replacements)
            report.differences.extend(_compare_json(key, g, c, atol, rtol))
        elif key.endswith(".csv"):
            report.differences.extend(
                _compare_csv(key, gpath, cpath, replacements, atol, rtol)
            )
        elif key.endswith(".npy"):
            report.differences.extend(_compare_npy(key, gpath, cpath, atol, rtol))
        else:  # pragma: no cover - defensive; no other artifact types today.
            report.differences.append(f"{key}: unknown artifact type, not compared")

    return report


# --------------------------------------------------------------------------- #
# CLI: capture / diff
# --------------------------------------------------------------------------- #
_MANIFEST_NAME = "manifest.json"


def capture(golden_dir: Path) -> dict[str, Path]:
    """Generate fixtures, run the pipeline, and store golden artifacts.

    A ``manifest.json`` records the capture-time working root so that a later
    ``diff`` (run from a different directory) can normalize path strings.
    """
    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ants_sitk_capture_") as work:
        work_path = Path(work)
        dataset = fixtures.generate_dataset(work_path / "dataset")
        outputs = run_pipeline(
            dataset.dataset_json,
            work_path / "results",
            work_path / "numpy",
        )
        artifacts = collect_artifacts(outputs)
        copy_artifacts(artifacts, golden_dir)
        (golden_dir / _MANIFEST_NAME).write_text(
            json.dumps(
                {"capture_root": str(work_path), "keys": sorted(artifacts)},
                indent=2,
            )
        )
    return _map_from_root(golden_dir)


def diff_against_golden(
    golden_dir: Path, *, atol: float = 0.0, rtol: float = 0.0
) -> DiffReport:
    """Regenerate a candidate run and diff it against a stored golden set."""
    golden_dir = Path(golden_dir)
    manifest_path = golden_dir / _MANIFEST_NAME
    capture_root = None
    if manifest_path.is_file():
        capture_root = json.loads(manifest_path.read_text()).get("capture_root")

    golden_map = _map_from_root(golden_dir)

    with tempfile.TemporaryDirectory(prefix="ants_sitk_diff_") as work:
        work_path = Path(work)
        dataset = fixtures.generate_dataset(work_path / "dataset")
        outputs = run_pipeline(
            dataset.dataset_json,
            work_path / "results",
            work_path / "numpy",
        )
        candidate_map = collect_artifacts(outputs)

        replacements = [(str(work_path), _ROOT_PLACEHOLDER)]
        if capture_root:
            replacements.append((str(capture_root), _ROOT_PLACEHOLDER))

        return diff_artifacts(
            golden_map, candidate_map, replacements=replacements, atol=atol, rtol=rtol
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 0 ANTs->SimpleITK regression harness."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Capture golden artifacts.")
    cap.add_argument("--golden-dir", required=True, type=Path)

    dif = sub.add_parser("diff", help="Diff a fresh run against golden.")
    dif.add_argument("--golden-dir", required=True, type=Path)
    dif.add_argument("--atol", type=float, default=0.0)
    dif.add_argument("--rtol", type=float, default=0.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "capture":
        artifacts = capture(args.golden_dir)
        print(f"Captured {len(artifacts)} artifact(s) to {args.golden_dir}")
        return 0
    if args.command == "diff":
        report = diff_against_golden(args.golden_dir, atol=args.atol, rtol=args.rtol)
        print(report)
        return 0 if report.identical else 1
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
