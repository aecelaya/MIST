# Changelog

All notable changes to MIST are listed here, newest first. MIST follows
[semantic versioning](https://semver.org/) from 2.3.0 onward.

## 2.4.1 — 2026-11-09

Bug fixes only, backported from the 2.5.0 work.

### Fixed

- Multi-GPU training with `resenc` no longer crashes on the second step with
  "Expected to have finished reduction in the prior iteration" when the network
  has more decoder stages than it uses for deep supervision. Those segmentation
  layers are now frozen; they stay in the state dict, so checkpoints still load.
- Multi-GPU training now applies MIST's GPU math settings (cuDNN benchmark mode,
  fast matmul, and TF32 convolutions). They were only set in the parent process,
  so the spawned training processes ran with PyTorch's defaults. Multi-GPU speed
  and first-step memory may change.
- **Training on large images is much faster with DALI.** `mist_preprocess` saved
  labels in Fortran order, which made the DALI loader rearrange each whole label
  for every patch. On an A100 with LiTS-sized CT (about 512 × 512 × 480 voxels),
  this made DALI wait about 1.9 seconds per training step for data. Labels are
  now saved in C order. Data preprocessed the old way still trains, with a
  warning; run `mist_preprocess` again to speed it up.
- `mist_analyze` computes `median_resampled_image_size` in the target spacing's
  axis order. For images whose axes aren't stored in RAI order, the estimate,
  and with it the memory warning and the patch size, came out wrong.
- The generic loader's validation and test loaders no longer fill shared memory
  (`/dev/shm`). Their workers started when the fold began and each read whole
  volumes ahead, holding them until validation ran: about 10 GiB per GPU for
  LiTS, enough to crash a 2-GPU run in a container with a 10 GiB `/dev/shm`. They
  now start at the first batch, stop after each pass, and use at most 2 workers,
  each reading one volume ahead. This affects ROCm, CPU, and NVIDIA runs that
  use `"data_loader": "generic"`; DALI is the NVIDIA default.
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
  fine-tunes nnFoundationCNN on spleen CT on a free Colab GPU and compares it
  with training from scratch.

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
