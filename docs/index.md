Medical Imaging Segmentation Toolkit
===

MIST is a simple, scalable, end-to-end framework for 3D medical image
segmentation. It handles everything from raw NIfTI files to trained models and
evaluated predictions, with sensible defaults that work well out of the box and
a configuration file for when you need more control.

## Installation

**Training** (NVIDIA GPU required):
```console
pip install "mist-medical[train]"
```

**Inference only** (CPU-compatible, works on Mac):
```console
pip install mist-medical
```

## Key Features

- **Automatic configuration** — analysis step determines target spacing, patch
  size, normalization, and foreground cropping from your data and available GPU memory
- **Five-fold cross-validation** by default, with custom fold assignment support
- **Multi-GPU training** via PyTorch DDP; uses all visible GPUs automatically
- **GPU-accelerated data loading** via NVIDIA DALI during training
- **Sliding window inference** with configurable overlap and patch blending
- **Test-time augmentation** and **multi-model ensembling** at inference
- **Transfer learning** — initialize encoders from pretrained weights
- **Resume training** — continue interrupted runs from the last checkpoint
- **CPU inference** — `mist_predict` runs on any machine, including Macs

## Supported Architectures

| Model | Key |
|---|---|
| nnU-Net | `nnunet` |
| nnU-Net Pocket | `nnunet-pocket` |
| MedNeXt (small / base / medium / large) | `mednext-small`, `mednext-base`, `mednext-medium`, `mednext-large` |
| FMG-Net | `fmgnet` |
| W-Net | `wnet` |
| Swin UNETR (small / base / large) | `swinunetr-small`, `swinunetr-base`, `swinunetr-large` |

## What's New

* April 2026 — **CPU inference support** — `mist_predict` now runs on any
  machine, including Macs and laptops without an NVIDIA GPU. Install with
  `pip install mist-medical` (no GPU required).
* March 2026 — **Resume training** — interrupted runs can be continued from the
  last checkpoint with `--resume`, with atomic checkpointing to prevent corruption.
* March 2026 — **GPU-aware automatic patch size** — the analysis step now derives
  the patch size from available GPU memory, so the default configuration is
  hardware-appropriate without manual tuning.
* March 2026 — **Transfer learning** — initialize encoders from pretrained weights
  with `--pretrained-weights`, and average model weights across folds with
  `mist_average_weights`.
* March 2026 — **Better training defaults** — AdamW optimizer and gradient clipping
  are now the defaults, with the clipping threshold exposed via `grad_clip_norm`
  in `config.json`.
* September 2025 — **[BraTS 2025 adult glioma challenge @ MICCAI 2025](https://www.synapse.org/Synapse:syn64153130/wiki/633062)** — MIST takes 3rd place (repeat).
* November 2024 — **MedNeXt models** — small, base, medium, and large variants
  added (`mednext-small`, `mednext-base`, `mednext-medium`, `mednext-large`).
* October 2024 — **[BraTS 2024 adult glioma challenge @ MICCAI 2024](https://www.synapse.org/Synapse:syn53708249/wiki/630150)** — MIST takes 3rd place.

## Citation

If you use MIST in your work, please cite:

[A. Celaya et al. "MIST: A Simple and Scalable End-To-End 3D Medical Imaging Segmentation Framework," arXiv preprint arXiv:2407.21343](https://www.arxiv.org/abs/2407.21343)

[A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3224873.](https://ieeexplore.ieee.org/document/9964128)
