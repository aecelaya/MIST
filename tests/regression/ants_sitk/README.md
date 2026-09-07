# Shared regression fixture/pipeline harness

Originally built as the Stage 0 harness for the ANTs → SimpleITK migration: a
small, deterministic harness that ran the pipeline over fixed edge-case
fixtures and diffed later stages against golden artifacts, to catch the
migration's signature failure mode — a missed `(x, y, z)` ↔ `(z, y, x)`
transpose producing spatially wrong (but non-crashing) output.

That migration is complete (MIST no longer depends on `ants` at all), and the
migration-specific diffing/golden-set workflow is gone along with it. What's
still here — `fixtures.py` (`generate_dataset`) and `harness.py`
(`run_pipeline`) — is genuinely reused, load-bearing test infrastructure for
unrelated tests that need a small, real, deterministic dataset run through the
actual pipeline:

- `tests/data_loading/test_generic_loader.py`
- `tests/regression/cpu_rocm/test_stage0_baseline.py`
- `tests/regression/cpu_rocm/test_stage4_cpu_end_to_end.py`

## Fixtures (`fixtures.py`)

Four patients, each stressing one edge case from the original migration plan.
Geometry is defined with **SimpleITK only**:

| Patient         | Stresses                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| `iso_small`     | Tiny isotropic, identity direction — baseline.                                                      |
| `anisotropic`   | Spacing `(1, 1, 3)` — resample / target-spacing path.                                               |
| `oblique`       | Non-identity/oblique direction cosines — most likely to expose a reorient convention mismatch.      |
| `sparse_labels` | Label `2` present on only two slices — label-aware resampling with a label absent from most slices. |

`generate_dataset` writes these out as a real MIST-format dataset directory —
any test needing realistic, deterministic 3D imaging data can call it directly
rather than hand-rolling fixtures.

## Pipeline runner (`harness.py`)

`run_pipeline` drives `mist_analyze -> mist_preprocess` (and, where a caller
needs it, further stages) over a generated dataset, so a test can exercise the
real CLI entrypoints end to end instead of mocking pipeline internals.
