# ANTs → SimpleITK Migration Plan

## Goal

Retire the `antspyx` dependency from MIST, replacing its functionality with
SimpleITK (which MIST already depends on and already performs all the
numerically sensitive work: resampling, STAPLE, distance transforms,
statistics filters). This is **not** a functional or performance-driven
change — it is dependency-surface reduction and removal of the
`ants_to_sitk`/`sitk_to_ants` conversion glue in
`mist/preprocessing/preprocessing_utils.py`. Treat it as low-priority
cleanup, not urgent work.

## Why staged releases

The riskiest part of this migration is silent, non-crashing bugs: ANTs
arrays are indexed `(x, y, z)` while `sitk.GetArrayFromImage()` returns
`(z, y, x)` (axes reversed relative to `GetSize()`/`GetSpacing()`). MIST's
own business logic (foreground bounding box keys, `target_spacing` tuples,
crop/pad logic) is written entirely in the `(x, y, z)` convention inherited
from ANTs. A missed transpose at any conversion point produces spatially
transposed — not crashing, just *wrong* — output. Shipping this in small,
independently verifiable, independently releasable stages, with a bake
period between each, is how we catch that class of bug before it compounds
across stages.

## Stage 0 — Build a regression harness

**Scope:** No code migration. Assemble a small fixed set of reference
volumes that stress known edge cases:
- At least one anisotropic image.
- At least one image with non-identity/oblique direction cosines.
- A multi-label mask with a label absent from some slices.
- A very small/edge-case volume.

Run the *current* ANTs+SimpleITK pipeline (`mist_analyze` → `mist_preprocess`
→ `mist_predict`) on these fixtures and save every intermediate artifact
(`fg_bboxes.csv`, preprocessed `.npy` files, predictions) as the golden
reference for diffing in later stages.

**Release gate:** Harness exists, golden outputs are committed/stored, and
the diffing script correctly reports zero differences when diffed against
itself as a sanity check.

## Stage 1 — Build the SimpleITK-only helper module

**Scope:** New module (e.g. `mist/utils/sitk_io.py`) providing SimpleITK
replacements for every ANTs function MIST currently uses:

| Current (ANTs) | Replacement (SimpleITK) |
|---|---|
| `ants.image_read` / `ants.image_write` | `sitk.ReadImage` / `sitk.WriteImage` |
| `ants.image_header_info` | `sitk.ImageFileReader().ReadImageInformation()` (lazy header-only read) |
| `ants.from_numpy` / `img.numpy()` | `sitk.GetImageFromArray` / `sitk.GetArrayFromImage`, with the `(x,y,z)` ↔ `(z,y,x)` transpose applied in one place |
| `ants.reorient_image2("RAI")` | `sitk.DICOMOrient` |
| `ants.crop_indices` | `sitk.RegionOfInterest` |
| `ants.pad_image` | `sitk.ConstantPad` |
| `ants.merge_channels` | `sitk.Compose` |

This isolates the riskiest primitive — array axis conversion — into one
well-tested chokepoint before anything in the rest of the codebase depends
on it, rather than re-deriving `.T` logic ad hoc at each call site later.

**Release gate:** Unit tests comparing this module's output against the
current ANTs-based output byte-for-byte on the Stage 0 fixtures. This module
ships as unused/dead code at this stage — nothing else in `mist/` changes
yet, so risk is effectively zero.

## Stage 2 — Migrate postprocessing + evaluation

**Scope:** `mist/postprocessing/postprocessor.py`,
`mist/evaluation/evaluation_utils.py`, `mist/evaluation/evaluator.py`. These
modules only read/write/validate images — no resampling entanglement — and
have existing test coverage.

**Release gate:** Rewrite the corresponding test files
(`tests/postprocessing/test_postprocessor.py`,
`tests/evaluation/test_evaluator.py`, `tests/evaluation/test_evaluation_utils.py`,
and their helpers). Full test suite green. Run `mist_postprocess` and
`mist_evaluate` on Stage 0 outputs and diff results against the golden
reference.

## Stage 3 — Migrate `analyze_data`

**Scope:** `mist/analyze_data/analyzer.py`,
`mist/analyze_data/analyzer_utils.py`, `mist/analyze_data/data_dump_utils.py`.
Highest call count (~30 ANTs calls) but runs once per dataset and only
produces `config.json` / `fg_bboxes.csv` — easy to regression-test by exact
diff.

**Release gate:** Rewrite `tests/analyze_data/test_analyzer.py`,
`tests/analyze_data/test_data_dump_utils.py`, and `tests/analyze_data/helpers.py`.
Run `mist_analyze` on the Stage 0 fixtures and assert the generated config
and bbox files are identical to the golden reference.

## Stage 4 — Migrate preprocessing core (highest risk)

**Scope:** `mist/preprocessing/preprocessing_utils.py`,
`mist/preprocessing/preprocess.py`. This is where the `ants_to_sitk` /
`sitk_to_ants` glue currently lives and where the `(x,y,z)` axis convention
and `target_spacing`/`fg_bbox` semantics matter most. Only start this stage
once Stages 1–3 have shipped and had real use time with no issues.

**Release gate:** The strictest gate in the plan. Run `mist_preprocess` on
all Stage 0 fixtures — especially the oblique-direction and anisotropic
cases — and assert numerical equality (or tight floating-point tolerance) of
every output `.npy` file against the golden reference, not just "tests
pass." Consider holding this release for an extra bake cycle before starting
Stage 5.

## Stage 5 — Migrate inference

**Scope:** `mist/inference/inference_runners.py`,
`mist/inference/inference_utils.py`. Depends on Stage 4, since
`back_to_original_space` / `probabilities_back_to_original_space` call into
`preprocess.resample_image` / `preprocess.resample_mask`.

**Release gate:** Rewrite `tests/inference/test_inference_utils.py` and
`tests/cli/test_ensemble_entrypoint.py`. Run `mist_predict` on the Stage 0
fixtures (and, if available, a couple of already-trained models on held-out
cases) and diff predictions against the golden reference — exact match or
Dice ≈ 1.0 against the old pipeline's output on the same input.

## Stage 6 — Cleanup

**Scope:**
- Remove `antspyx` from the `dependencies` list in `pyproject.toml`.
- Delete the now-dead `ants_to_sitk` / `sitk_to_ants` glue in
  `preprocessing_utils.py`.
- Sweep for any remaining `import ants` across `mist/` and `tests/`.
- Update any docs/CLAUDE.md references to ANTs.

**Release gate:** Full test suite green.
`grep -r "import ants" mist/ tests/` returns nothing (outside `build/`).

## Cross-cutting rule for every stage

Ship it, let it sit in real use for a while, confirm no issues surface
before starting the next stage. Some edge cases (unusual direction cosines,
sparse labels) may only show up on real, diverse datasets that the Stage 0
fixtures didn't anticipate — not just on the test suite.
