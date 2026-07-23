# ANTs → SimpleITK regression harness (Stage 0)

This is **Stage 0** of `ants_to_simpleitk_migration.md`: a small, deterministic
harness that runs the *current* pipeline over fixed edge-case fixtures and
captures golden artifacts. Later stages re-run it and diff against the golden
set to catch the migration's signature failure mode — a missed `(x, y, z)` ↔
`(z, y, x)` transpose that produces spatially wrong (but non-crashing) output.

## What it captures

`mist_analyze → mist_preprocess` over the fixtures, saving every artifact a
later stage must reproduce:

| Artifact | Produced by | Primary diff target for |
|---|---|---|
| `config.json` | analyze | Stage 3 |
| `train_paths.csv` | analyze | Stage 3 |
| `fg_bboxes.csv` | analyze | Stage 3 / Stage 4 |
| `numpy/images/<id>.npy` | preprocess | Stage 4 |
| `numpy/labels/<id>.npy` | preprocess | Stage 4 |
| `numpy/dtms/<id>.npy` | preprocess | Stage 4 |

Prediction artifacts (Stage 5) need a trained model, which the plan defers to
that stage. `harness.run_prediction` is a stub hook, deliberately not part of
the Stage 0 gate.

## Fixtures (`fixtures.py`)

Four patients, each stressing one edge case from the plan. Geometry is defined
with **SimpleITK only**, so the fixtures don't favor either implementation:

| Patient | Stresses |
|---|---|
| `iso_small` | Tiny isotropic, identity direction — baseline. |
| `anisotropic` | Spacing `(1, 1, 3)` — resample / target-spacing path. |
| `oblique` | Non-identity/oblique direction cosines — most likely to expose a reorient convention mismatch. |
| `sparse_labels` | Label `2` present on only two slices — label-aware resampling with a label absent from most slices. |

Fixtures are seeded with a stable (cross-process) hash so `capture` and `diff`
run in separate processes and still compare byte-for-byte.

## Running it

**The gate (release criterion for Stage 0):**

```bash
mamba run -n mist pytest tests/regression/ants_sitk/
```

`test_selfdiff_is_exactly_zero` is the plan's "diff reports zero differences
when diffed against itself" check. The `test_differ_detects_*` tests prove the
differ isn't trivially passing — they corrupt an artifact (including an axis
transpose) and assert it's reported.

**Capture a golden set once, diff future runs against it:**

```bash
# After confirming the CURRENT (ANTs) pipeline is the reference:
mamba run -n mist python -m tests.regression.ants_sitk.harness capture \
    --golden-dir path/to/golden

# After a migration stage, regenerate and diff:
mamba run -n mist python -m tests.regression.ants_sitk.harness diff \
    --golden-dir path/to/golden          # exits non-zero if anything differs
```

`capture` writes a `manifest.json` recording the capture-time working root so
`diff` can normalize absolute paths inside `config.json` / the CSVs.

## Tolerances

`diff` defaults to **exact** equality (`--atol 0 --rtol 0`), which is correct
for Stage 0: two runs of the same code must be bit-identical. Integer/label
arrays are always compared exactly; a shape mismatch is reported first, since a
transposed axis is the bug this exists to catch.

For the cross-implementation stages (especially **Stage 4**), loosen the
floating-point tolerance for the image/dtm arrays, e.g. `--atol 1e-5 --rtol
1e-5`, while keeping label arrays exact and predictions exact (Dice ≈ 1.0).
Commit to a specific per-artifact tolerance in that stage rather than a blanket
value.
