# 🚀 MIST RC1 — Release Notes

We're excited to share the first release candidate of MIST! This release brings
a large number of new features, training improvements, and quality-of-life
fixes across the entire pipeline. Whether you're upgrading from an earlier build
or trying MIST for the first time, read on for everything you need to know.

---

## ⚠️ Breaking changes

These changes require action if you are upgrading from an earlier version.

### 🔀 Dataset conversion commands renamed

`mist_convert_dataset` has been replaced by two focused commands:

| Old | New |
|-----|-----|
| `mist_convert_dataset --format msd ...` | `mist_convert_msd --source <dir> --output <dir>` |
| `mist_convert_dataset --format csv ...` | `mist_convert_csv --train-csv <file> --output <dir>` |

The `--format` dispatch flag is gone. Each command now has its own parser and
`--num-workers` flag for parallel conversion.

### 🗑️ `--pocket` flag removed

The `--pocket` training flag has been removed. The pocket nnU-Net variant is
now a first-class registered architecture:

```bash
# Old
mist_train ... --pocket

# New
mist_train ... --model nnunet-pocket
```

### 🗑️ `--use-dtms` flag removed

Distance transform maps (DTMs) are now computed automatically whenever a
DTM-aware loss function is selected (`hdos`, `gsl`, `volumetric_sddl`,
`vessel_sddl`). The `--use-dtms` flag has been removed from the CLI and from
`config.json`. Existing configs that contain `use_dtms` will be silently ignored
on the first run, but **users should re-run `mist_analyze` to regenerate a clean
config**.

### 📋 `composite_loss_weighting` config structure changed

The training loss config block has been flattened. If you edited `config.json`
by hand, update it:

```json
// Old
"loss": {
  "name": "cldice",
  "params": { "composite_loss_weighting": { "name": "cosine", ... } }
}

// New
"loss": {
  "name": "cldice",
  "composite_loss_weighting": { "name": "cosine", "params": { ... } }
}
```

### 📋 `apply_sequentially` → `per_label` in postprocessing strategy JSON

The `apply_sequentially` key in postprocessing strategy files has been renamed
to `per_label`. Update any custom strategy JSON files:

```json
// Old
{ "transform": "remove_small_objects", "apply_sequentially": true, ... }

// New
{ "transform": "remove_small_objects", "per_label": true, ... }
```

### 📁 `mist_postprocess` output structure changed

Postprocessed predictions are no longer written directly into the output
directory. The new layout is:

```
<output>/
  predictions/              ← postprocessed NIfTI files (was the root)
  strategy.json             ← copy of the strategy used
  postprocess_results.csv   ← evaluation results (if --paths-csv / --eval-config provided)
```

The `--eval-output-csv` flag has been removed; the results CSV path is now fixed.

### 🗑️ Legacy config format support removed

The backwards-compatibility shims that converted old `model_config.json`
checkpoint layouts and old-style flat evaluation configs are gone. If you have
checkpoints from a pre-RC build, re-run the full pipeline from `mist_analyze`.

---

## ✨ New features

### ▶️ Resume training

Training can now be interrupted and resumed without losing progress:

```bash
mist_train ... --resume
```

A checkpoint is written atomically after every epoch to
`results/checkpoints/fold_N_checkpoint.pt`. On resume, training picks up
from the next unfinished epoch; folds that already reached `--epochs` are
skipped entirely. `--resume` and `--overwrite` are mutually exclusive.

### 📈 Learning rate warmup

A linear LR warmup phase can be prepended to any LR schedule:

```bash
mist_train ... --warmup-epochs 5
```

The main scheduler runs for `epochs - warmup_epochs` so the total decay budget
is preserved. A warning is emitted if `--warmup-epochs` is changed on resume.

### 🔁 Transfer learning *(experimental)*

Encoder weights from a previously trained MIST model can now be used to
initialise training on a new task:

```bash
mist_train \
  --pretrained-weights /path/to/fold_0.pt \
  --pretrained-config  /path/to/config.json \
  --input-channel-strategy average   # or 'first' / 'skip'
```

`--input-channel-strategy` controls what happens when the source and target
datasets have a different number of input channels. Before fine-tuning, use
`mist_average_weights` to produce a single initialisation checkpoint from all
fold weights:

```bash
mist_average_weights \
  --models-dir /path/to/results/models \
  --config     /path/to/results/config.json \
  --output     averaged_weights.pt
```

> ⚠️ **Experimental:** Transfer learning is under active development. If you
> encounter unexpected behaviour, please open an issue on GitHub.

### 🧠 New model architectures

Four new architectures are available out of the box:

| Architecture | Notes |
|---|---|
| `nnunet-pocket` | Constant 32-filter-width nnU-Net; replaces the old `--pocket` flag |
| `swinunetr-v2-small` | SwinUNETR-V2 small variant |
| `swinunetr-v2-base` | SwinUNETR-V2 base variant |
| `swinunetr-v2-large` | SwinUNETR-V2 large variant |

### 🖥️ GPU-aware patch size selection

The automatic patch size algorithm has been redesigned to be smarter about your
hardware:

- **Voxel budget** is derived from minimum GPU VRAM across all CUDA devices,
  scaled by batch size — no more one-size-fits-all defaults.
- **3D isotropic mode** (spacing ratio ≤ 3): budget is distributed
  proportionally in physical space; axes that exceed the median image size
  are clamped and remaining budget is redistributed.
- **Quasi-2D mode** (spacing ratio > 3): the low-resolution axis receives a
  minimal patch; both in-plane axes get an equal square patch.
- All axes are snapped to the nearest multiple of 32.

### 🔬 Data dump (`--data-dump`)

Running `mist_analyze` with `--data-dump` now produces a `data_dump.md` report
alongside the standard analysis outputs. The report includes rich per-label
statistics: voxel counts, surface area, Isoperimetric Quotient (IQ), skeleton
ratio, PCA-based shape descriptors (linearity / planarity / sphericity), and a
shape-class label (tubular / planar / blob). Label sparsity is reported in two
tiers: < 1% → "very sparse", 1–5% → "sparse".

### ✅ Dataset verification (`--verify`)

`mist_analyze` now accepts `--verify` to perform a thorough integrity check of
the raw dataset — file existence, dtype consistency, spacing and header
agreement — before the full analysis run. Catch data problems early, before
they surface as cryptic errors mid-training.

### 📏 Lesion-wise metrics overhaul

`lesion_wise_dice` and `lesion_wise_surface_dice` have been rewritten with
correct BraTS-style aggregation:

- Score = Σ(per-lesion scores) / (N_GT_above_threshold + N_FP_predictions)
- False-positive predictions are explicitly penalised.
- GT lesion consolidation (`gt_consolidation_iters`) merges nearby lesions so
  they are treated as a single entity, matching the BraTS reference implementation.
- `reduction="none"` mode returns raw per-lesion result dicts for custom
  downstream analysis.

### 📊 Composite loss scheduling improvements

The documentation now includes a full summary table showing which losses support
alpha scheduling, whether they require DTMs, and their default α value. A new
`init_pause` parameter (available for `linear` and `cosine` schedulers) holds
α at `start_val` for a configurable warm-up period before decay begins.

### 📉 TensorBoard: learning rate and alpha logging

The per-fold TensorBoard writer now records two additional scalars:

- `learning_rate` — current LR after each epoch (rank 0 only)
- `alpha` — current composite loss weighting (only for losses with an alpha
  scheduler)

### ⚡ Parallel dataset analysis

`mist_analyze` now accepts `--num-workers-analyze N` to parallelise per-patient
stat collection, making analysis of large datasets significantly faster.

### 🔧 Postprocessing: `describe_transforms()`

`Postprocessor.describe_transforms()` returns structured metadata for all
registered transforms, including parameter types, descriptions, defaults, and
valid ranges — useful for programmatic introspection and building tooling on top
of the postprocessing pipeline.

### 🩺 Evaluation improvements

- **`--validate` flag** performs a pre-evaluation integrity check on every mask
  pair (3D shape, dtype, label set) before the parallel pipeline runs. Masks
  saved as float32 with integer-valued labels (common with ANTs and BraTS ground
  truth data) are now correctly accepted.
- **`--num-workers-evaluate` for `mist_postprocess`**: a separate parallelism
  flag for the optional evaluation step that follows postprocessing, independent
  of `--num-workers-postprocess`.
- **Clearer failure reporting**: when all patients fail evaluation, the evaluator
  now prints an explicit error message instead of a misleading success message.

---

## 📝 Minor changes and things to be aware of

### ⚙️ Default optimiser changed to AdamW with lr = 1e-4

The default training optimiser is now **AdamW** (was Adam) with a default
learning rate of **1e-4** (was 3e-4). Existing configs generated by older
versions of `mist_analyze` will continue to use the old defaults until you
re-run `mist_analyze` or manually update `config.json`.

### ✂️ Gradient clipping exposed via `--grad-clip-norm`

Gradient norm clipping can now be configured directly from the CLI:

```bash
mist_train ... --grad-clip-norm 1.0
```

The default (`None`) disables clipping, preserving the previous behaviour.

### 🧱 2D training support removed

The DALI data loading pipeline is now 3D-only. The `dimension` parameter has
been removed from `TrainPipeline`. MIST has always been primarily a 3D
segmentation tool; this removal simplifies the codebase without affecting any
documented workflow.

### 🗂️ `spatial_config` is now the single source of truth

`patch_size` and `target_spacing` are stored under a dedicated `spatial_config`
key in `config.json` rather than being duplicated across multiple sections.
Old configs that store these values elsewhere are no longer supported — re-run
`mist_analyze` to generate a clean config.

### 👷 `--num-workers` flags all default to 1

All worker-count CLI flags (`--num-workers-preprocess`, `--num-workers-convert`,
`--num-workers-analyze`, `--num-workers-postprocess`, `--num-workers-evaluate`,
`--num-workers-train-eval`) now default to **1** for safety on memory-constrained
machines. Increase these flags explicitly for faster runs on capable hardware.

### ⏭️ `skip=True` is now a true preprocessing pass-through

When `preprocessing.skip = true`, images are now read exactly as they are on
disk with no spatial transforms, header copying, or resampling applied.
Previously, some transforms were still applied in this mode. Predictions written
in skip mode now correctly carry the original image header.

### 🧪 Surface Dice Dilation (`sddl`) losses are experimental

`volumetric_sddl` and `vessel_sddl` are documented as experimental. They may
produce `UserWarning` messages from the underlying dilation operations on
certain inputs. These warnings are cosmetic and do not affect training
correctness.

### 🔤 CLI flags standardised across all commands

Flag names are now consistent across all MIST commands — `--num-workers-*`
format everywhere, `--output` used instead of mixed `--output-dir` / `--out`.
Scripts that use the old flag spellings will need to be updated.

### 🔢 ANTs `image_read` always returns float32

ANTs reads NIfTI files as float32 regardless of the on-disk dtype. MIST's
`--validate` flag and evaluation pipeline now handle this correctly, accepting
float32 arrays whose values are all integer-valued. This is not a bug in MIST;
it is an ANTs behaviour that users building custom evaluation pipelines outside
MIST should be aware of.
