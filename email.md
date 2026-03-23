# 🚀 MIST RC1 is here!

Hi everyone,

We're thrilled to announce the first release candidate of MIST! This release
is the result of months of work and represents a major step forward in
usability, training flexibility, and segmentation accuracy. We'd love your
feedback — if you run into anything unexpected, please open an issue on GitHub.

Here's what's new.

---

## ✨ What's new

**▶️ Resume training** — Interrupted runs are no longer lost. Use `--resume`
to pick up exactly where you left off. Checkpoints are written atomically after
every epoch so a crash can never corrupt your training state.

**📈 Learning rate warmup** — Add a linear warmup phase to any LR schedule
with `--warmup-epochs`. Great for fine-tuning and transfer learning.

**🔁 Transfer learning** *(experimental)* — Initialise training from a
previously trained MIST encoder using `--pretrained-weights`. Use the new
`mist_average_weights` command to produce a single initialisation checkpoint
from all fold weights before fine-tuning.

**🧠 New architectures** — Four new models are registered and ready to use:
`nnunet-pocket`, `swinunetr-v2-small`, `swinunetr-v2-base`, and
`swinunetr-v2-large`.

**🖥️ Smarter patch size selection** — The automatic patch size algorithm now
accounts for your GPU's VRAM and batch size, handles anisotropic data in a
dedicated quasi-2D mode, and snaps to multiples of 32 for architecture
compatibility.

**🔬 Data dump report** — Run `mist_analyze --data-dump` to generate a rich
per-label report including shape descriptors, skeleton ratio, surface area,
and sparsity tiers. A great first step for understanding a new dataset.

**✅ Dataset verification** — Run `mist_analyze --verify` to catch data
problems (missing files, inconsistent headers, dtype mismatches) before they
surface mid-training.

**📏 Better lesion-wise metrics** — `lesion_wise_dice` and
`lesion_wise_surface_dice` now use correct BraTS-style aggregation that
penalises both false negatives and spurious false positive predictions.

**📉 TensorBoard LR and alpha logging** — Learning rate and composite loss
alpha are now tracked automatically in TensorBoard alongside training losses.

---

## ⚠️ Migration notes

A few things require updates if you're upgrading from an earlier build.

- **`mist_convert_dataset` is gone.** Use `mist_convert_msd` or
  `mist_convert_csv` instead.
- **`--pocket` is gone.** Use `--model nnunet-pocket` instead.
- **`--use-dtms` is gone.** DTMs are now computed automatically based on your
  choice of loss function.
- **Postprocessing strategy JSON:** rename `apply_sequentially` → `per_label`
  in any custom strategy files.
- **`mist_postprocess` output layout changed:** predictions now go to
  `output/predictions/` instead of the root output directory.
- **Default optimiser is now AdamW with lr = 1e-4** (was Adam at 3e-4).
  Re-run `mist_analyze` to pick up the new defaults.
- **Old configs are no longer supported.** If you have results from a pre-RC
  build, re-run the pipeline from `mist_analyze` to generate a clean config.

Full details on every change — including code examples and config snippets —
are in [`release_notes.md`](release_notes.md).

---

Thanks for using MIST, and as always, we appreciate your feedback!
