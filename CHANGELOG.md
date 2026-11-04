# Changelog

All notable changes to MIST are listed here, newest first. MIST follows
[semantic versioning](https://semver.org/) from 2.3.0 onward.

## 2.5.0 — Unreleased

### Added

- `training.validate_every` (and `--validate-every`) runs validation every N
  epochs instead of every epoch. Validation always runs on the last epoch.
  Defaults to `1`, the previous behavior.

### Changed

- **Training augmentation now follows nnU-Net's default transforms.** New
  transforms: rotation (±30° per axis, or ±180° in-plane for quasi-2D patches),
  scaling from 0.7 to 1.4 (replacing zoom, which only zoomed in), simulated low
  resolution, and gamma (including on the inverted image). Noise, blur,
  brightness, and flips use nnU-Net's probabilities and ranges, and contrast now
  scales about the mean instead of acting as a second brightness. Rotation and
  scaling resample a larger patch cropped around the same center (images and
  DTMs linearly, labels per class with an argmax, as in nnU-Net), and apply to
  DTMs too (previously zoom was skipped when DTMs were used). Scaling keeps DTM
  distances valid: unnormalized DTMs are divided by the scale factor, and
  quasi-2D patches, which scale only in-plane, skip scaling when DTMs are used.
- Augmentation runs in PyTorch on each batch on the training device, so the
  DALI and generic loaders now apply identical transforms. The loaders only
  extract patches.
- The generic loader's training workers now last the whole fold instead of
  restarting at every pass over the data and every epoch, and each worker
  remembers where each case's foreground is (up to 10,000 voxels per class)
  after scanning its label once. Patches that run past the image edge no longer
  read the whole case. DALI is unaffected.
- New config keys under `training.augmentation.transforms`: `rotation`,
  `scaling`, `low_resolution`, and `gamma`. Missing keys mean enabled, so older
  configs get the new transforms; their `zoom` key is read as `scaling`.
- **Predictions are restored to the original image space by resampling class
  probabilities, then taking the argmax**, as nnU-Net does. Previously MIST
  took the argmax at the training spacing and resampled the labels. This helps
  most for small or thin structures when the original images are finer than
  the training spacing; along the low-resolution axis of anisotropic images
  both use nearest-neighbor interpolation, so nothing changes there. Classes
  are resampled one at a time, so memory doesn't grow with the number of
  classes. This applies to cross-validation predictions and to `mist_infer`,
  including for models trained with earlier versions.
- **Which model is saved to `models/` now depends on `val_percent`.** With the
  default `val_percent` of `0.0`, the saved model is the final-epoch model.
  Previously it was the model with the lowest loss on the held-out fold, which
  let the fold used for cross-validation choose its own model and made
  cross-validation results optimistic. nnU-Net also uses the final model by
  default. With `val_percent` above `0.0`, the saved model is still the best one
  on the validation split carved from the training cases, now chosen by the
  validation loss smoothed with an exponential moving average (decay 0.9) so
  that one noisy epoch doesn't pick it. Cross-validation scores from a default
  run may come out slightly lower than in 2.4.0; that is the bias being removed.
- MIST warns when `val_percent` leaves fewer than 5 validation cases in a fold.
- `--resume` warns if `--val-percent` or `--validate-every` differ from the
  saved run.

### Fixed

- The transfer summary printed at the start of training no longer double-counts
  parameters registered under two names (e.g., `resenc`'s convolutions), which
  made nnFoundationCNN report about 180.6M loaded parameters instead of about
  90.3M. Only the printout was affected.

## 2.4.0 — 2026-11-02

### Added

- **ResEnc UNet** (`resenc`) — nnU-Net's Residual Encoder UNet (Isensee et al.,
  2024), with MIST's adaptive topology. Encoder weights are compatible with
  nnU-Net and nnssl checkpoints, including nnFoundationCNN.
- **Primus and PrimusV2** (`primus-medium`, `primus-large`, `primusv2-small`,
  `primusv2-base`, `primusv2-medium`, `primusv2-large`) — the
  transformer-centric architectures from Wald et al. (2025), plus `primus-x`,
  the 40-layer encoder of nnFoundationViT. Encoder weights are compatible with
  nnU-Net and nnssl checkpoints, and a learned position embedding pretrained at
  another patch size is resized automatically.
- **Checkpoints from other frameworks** — `--pretrained-weights` unwraps state
  dicts nested under `state_dict`, `model_state_dict`, `network_weights`, or
  `model`, so checkpoints from PyTorch Lightning, nnU-Net, and nnssl load
  directly.

- **Fine-tuning notebook** — the
  [nnFoundation fine-tuning notebook](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_nnfoundation_spleen_demo.ipynb)
  fine-tunes nnFoundationCNN on spleen CT in Colab.

### Changed

- Transfer learning is no longer marked experimental in the documentation.
- Transfer learning now raises an error if no encoder weights transfer, and
  warns if fewer than half do, instead of silently training from random
  initialization.
- Transfer learning also warns when the checkpoint's encoder is larger than the
  model's (e.g., nnFoundationViT loaded into `primus-large` instead of
  `primus-x`), since only part of it is used.
- For `resenc`, a patch size or spacing mismatch with `--pretrained-config` is a
  warning rather than an error, since shared stages still transfer exactly.
- `--pretrained-config` accepts MISFIT pretraining configs: a null
  `target_spacing` (pretrained without resampling) or a missing `spatial_config`
  warns that spacing can't be checked instead of failing.

### Removed

- The experimental `mist_analyze --data-dump` option (`data_dump.json` and
  `data_dump.md`). Dataset summaries for LLM-assisted configuration will live in
  a separate tool.

### Fixed

- Documentation: `--pretrained-config` is optional, not required.

## 2.3.0 — 2026-10-03

First stable release. No functional changes from 2.3.0-rc.

## 2.3.0-rc — 2026-09-07

### Added

- **CPU and AMD ROCm training** — the full pipeline, `mist_train` included, runs
  on CPU-only machines and AMD ROCm GPUs through a generic, pure-PyTorch data
  loader. The data loader and communication backend (`nccl`/RCCL or `gloo`) are
  detected automatically and saved to `config.json`.
- **`mist_finalize`** — supports one-fold-per-node training: after every
  per-fold job finishes, it aggregates out-of-fold predictions into
  `results.csv` and runs test-set inference. Safe to re-run.
- **Probability-level ensembling** — `mist_predict --output-probs` saves softmax
  probabilities, and `mist_ensemble --input-type probabilities` averages them
  before a single argmax.
- **Parallel ensembling** — `mist_ensemble --num-workers-ensemble`.

### Changed

- The `train` install extra is now `dali`; `[train]` remains as an alias.
- BF16 AMP is enabled only on hardware with BF16 matrix acceleration (NVIDIA
  Ampere and later, AMD CDNA and RDNA3 and later); other hardware falls back to
  float32 with a warning.

### Removed

- The ANTs (`antspyx`) dependency. Image I/O, preprocessing, postprocessing,
  evaluation, analysis, and inference run entirely on SimpleITK.

## 2.0.2-rc — 2026-07-09

### Added

- **Interactive Colab demo** — the
  [heart segmentation notebook](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb)
  runs the full pipeline on a free Colab GPU.
- AMP auto-detection and a training summary at the start of `mist_train`.

### Fixed

- `sw_batch_size` handling in sliding-window inference.

## 2.0.1-rc — 2026-06-12

### Added

- **`mist_ensemble`** — combines predictions from separately trained models by
  STAPLE (default) or majority vote.
- `mist_rank --significance-csv` for pairwise Wilcoxon significance testing.
- A tunable `sw_batch_size` for sliding-window inference.
- An official Docker image, `mistmedical/mist` (CUDA 12.8; NVIDIA driver ≥
  525.x).

### Changed

- BF16 automatic mixed precision replaces FP16 in training and inference.

### Fixed

- Lower peak memory use across the inference stack.

## 2.0.0-rc — 2026-05-07

### Added

- **`mist_rank`** — BraTS-style multi-strategy ranking of evaluation results.
- A structured postprocessing transform registry with `describe_transforms`.
- **Transfer learning** — `--pretrained-weights` and `mist_average_weights`.
- **Resume training** — `--resume`, with atomic checkpointing.
- **CPU inference** — `mist_predict` runs without an NVIDIA GPU.
- GPU-aware automatic patch size selection.

### Changed

- AdamW and gradient clipping are the defaults; the clipping threshold is set by
  `grad_clip_norm` in `config.json`.
- The codebase uses `pathlib` and modern type hints throughout.

## Earlier releases

Beta releases from 0.0.1-beta (October 2024) through 1.0.1-beta (February 2026)
are listed on the
[GitHub releases page](https://github.com/mist-medical/MIST/releases). They
include the MedNeXt models (`mednext-small`, `mednext-base`, `mednext-medium`,
`mednext-large`), added in 0.1.5-beta.
