"""Deterministic edge-case fixtures for the ANTs -> SimpleITK regression harness.

The fixtures are intentionally built with SimpleITK only, so their geometry
(spacing, origin, direction cosines) is defined independently of the
ANTs-vs-SimpleITK code path that the migration is changing. That keeps the
fixtures a neutral ground truth: neither implementation "owns" how they were
written.

Each patient stresses a specific class of bug the migration plan calls out:

    iso_small       Tiny isotropic identity-direction volume. Multi-label mask.
    anisotropic     Strongly anisotropic spacing (1, 1, 3). Exercises the
                    resample/target-spacing path where axis order matters most.
    oblique         Non-identity / oblique direction cosines. This is the case
                    most likely to expose a silent transpose/flip if a reorient
                    replacement uses a different orientation-string convention.
    sparse_labels   Multi-label mask where one label is present on only a few
                    slices (absent from most). Exercises label-aware resampling
                    and the "label absent from some slices" edge case.

Arrays are constructed in SimpleITK's native ``(z, y, x)`` index order to avoid
any ambiguity; MIST's own ``(x, y, z)`` business logic is exactly what the
golden diff is there to protect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# CT-like intensities (Hounsfield units) so the "ct" modality path in analyze
# exercises global-percentile normalization deterministically.
_AIR_HU = -1024.0
_SOFT_TISSUE_HU = 40.0
_LABEL1_HU = 120.0
_LABEL2_HU = 220.0


@dataclass(frozen=True)
class FixtureSpec:
    """Geometry + content description for a single synthetic patient."""

    patient_id: str
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Row-major 3x3 direction cosine matrix, flattened (SimpleITK convention).
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # If True, label 2 appears on only a couple of z-slices (sparse label).
    sparse_second_label: bool = False


def _rotation_direction(angle_deg: float) -> tuple[float, ...]:
    """Return an orthonormal 3x3 direction (row-major) rotated about z then y.

    Produces a genuinely oblique (non-axis-aligned) frame while staying a valid
    rotation matrix, which SimpleITK requires for a direction.
    """
    a = np.deg2rad(angle_deg)
    rot_z = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    b = np.deg2rad(angle_deg * 0.5)
    rot_y = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
    direction = rot_z @ rot_y
    return tuple(float(v) for v in direction.flatten())


# The fixed fixture set. Sizes are kept small so the whole harness runs in
# seconds while still covering the edge cases from the migration plan.
FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        patient_id="iso_small",
        size_xyz=(24, 24, 20),
        spacing_xyz=(1.0, 1.0, 1.0),
    ),
    FixtureSpec(
        patient_id="anisotropic",
        size_xyz=(32, 32, 12),
        spacing_xyz=(1.0, 1.0, 3.0),
    ),
    FixtureSpec(
        patient_id="oblique",
        size_xyz=(28, 26, 22),
        spacing_xyz=(1.0, 1.2, 1.5),
        origin_xyz=(10.0, -5.0, 3.0),
        direction=_rotation_direction(20.0),
    ),
    FixtureSpec(
        patient_id="sparse_labels",
        size_xyz=(30, 30, 24),
        spacing_xyz=(1.0, 1.0, 1.0),
        sparse_second_label=True,
    ),
)


def _make_image_and_mask(
    spec: FixtureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(z, y, x)`` image and mask arrays for a fixture.

    The image is a CT-like volume: air background with a soft-tissue block in
    the interior, plus two labelled structures embedded with distinct
    intensities. The mask carries labels {0, 1, 2}. Content is fully
    deterministic (seeded RNG for a little texture, keyed on the patient id).
    """
    size_x, size_y, size_z = spec.size_xyz
    shape_zyx = (size_z, size_y, size_x)

    # Seed from a stable hash of the patient id (NOT Python's built-in hash(),
    # which is salted per process by PYTHONHASHSEED) so the fixtures are
    # byte-identical across processes -- required for capture/diff to run in
    # separate processes and still compare equal.
    digest = hashlib.sha256(spec.patient_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)

    image = np.full(shape_zyx, _AIR_HU, dtype=np.float32)
    mask = np.zeros(shape_zyx, dtype=np.uint8)

    # Interior soft-tissue block (guarantees a detectable foreground for the
    # Otsu-based bounding box, and headroom so fg-cropping is well defined).
    z0, z1 = size_z // 6, size_z - size_z // 6
    y0, y1 = size_y // 6, size_y - size_y // 6
    x0, x1 = size_x // 6, size_x - size_x // 6
    interior = image[z0:z1, y0:y1, x0:x1]
    image[z0:z1, y0:y1, x0:x1] = _SOFT_TISSUE_HU
    image[z0:z1, y0:y1, x0:x1] += rng.normal(0.0, 5.0, size=interior.shape).astype(np.float32)

    # Label 1: a solid block near the centre, present on most interior slices.
    lz0, lz1 = size_z // 3, size_z - size_z // 3
    ly0, ly1 = size_y // 3, size_y - size_y // 3
    lx0, lx1 = size_x // 3, size_x - size_x // 3
    mask[lz0:lz1, ly0:ly1, lx0:lx1] = 1
    image[lz0:lz1, ly0:ly1, lx0:lx1] = _LABEL1_HU

    # Label 2: a smaller structure. For the sparse case it lives on only two
    # z-slices (absent from every other slice); otherwise it spans a small run.
    if spec.sparse_second_label:
        sparse_slices = [lz0, lz1 - 1]
    else:
        sparse_slices = list(range(lz0, min(lz0 + 3, lz1)))
    sy0, sy1 = ly0, ly0 + max(2, (ly1 - ly0) // 3)
    sx0, sx1 = lx0, lx0 + max(2, (lx1 - lx0) // 3)
    for z in sparse_slices:
        mask[z, sy0:sy1, sx0:sx1] = 2
        image[z, sy0:sy1, sx0:sx1] = _LABEL2_HU

    return image, mask


def _write_image(
    array_zyx: np.ndarray,
    spec: FixtureSpec,
    path: Path,
    is_mask: bool,
) -> None:
    """Write a ``(z, y, x)`` array to NIfTI with the fixture's geometry."""
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(tuple(float(s) for s in spec.spacing_xyz))
    image.SetOrigin(tuple(float(o) for o in spec.origin_xyz))
    image.SetDirection(spec.direction)
    if is_mask:
        image = sitk.Cast(image, sitk.sitkUInt8)
    else:
        image = sitk.Cast(image, sitk.sitkFloat32)
    sitk.WriteImage(image, str(path))


@dataclass
class GeneratedDataset:
    """Paths produced by :func:`generate_dataset`."""

    root: Path
    dataset_json: Path
    train_data: Path
    patient_ids: list[str] = field(default_factory=list)


def generate_dataset(root: Path) -> GeneratedDataset:
    """Materialise the fixed fixture set as a MIST dataset under ``root``.

    Layout::

        root/
          dataset.json
          train-data/
            <patient_id>/
              image.nii.gz
              mask.nii.gz

    Returns the paths needed to drive ``mist_analyze`` / ``mist_preprocess``.
    """
    root = Path(root)
    train_data = root / "train-data"
    train_data.mkdir(parents=True, exist_ok=True)

    patient_ids: list[str] = []
    for spec in FIXTURES:
        patient_dir = train_data / spec.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        image, mask = _make_image_and_mask(spec)
        _write_image(image, spec, patient_dir / "image.nii.gz", is_mask=False)
        _write_image(mask, spec, patient_dir / "mask.nii.gz", is_mask=True)
        patient_ids.append(spec.patient_id)

    dataset_info = {
        "task": "ants-sitk-regression",
        "modality": "ct",
        "train-data": "train-data",
        "mask": ["mask.nii.gz"],
        "images": {"ct": ["image.nii.gz"]},
        "labels": [0, 1, 2],
        "final_classes": {"whole": [1, 2], "core": [2]},
    }
    dataset_json = root / "dataset.json"
    dataset_json.write_text(json.dumps(dataset_info, indent=2))

    return GeneratedDataset(
        root=root,
        dataset_json=dataset_json,
        train_data=train_data,
        patient_ids=patient_ids,
    )
