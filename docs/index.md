Medical Imaging Segmentation Toolkit
===

MIST is a simple, scalable, end-to-end framework for 3D medical image
segmentation. It handles everything from raw NIfTI files to trained models and
evaluated predictions, with sensible defaults that work well out of the box and
a configuration file for when you need more control.

!!! tip "Try MIST in your browser"

    The **[end-to-end Colab demo](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb)**
    runs the whole pipeline — analyze → preprocess → train → evaluate → predict →
    postprocess → rank → visualize — on a heart MRI dataset using a free Colab T4
    GPU, with no local setup required.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb)

## Installation

**CPU** (default — runs the full pipeline: train, predict, evaluate, everything.
Works on Mac):

```console
pip install mist-medical
```

**NVIDIA GPU** (recommended — adds DALI-accelerated data loading):

```console
pip install "mist-medical[dali]"
```

!!! note

    Upgrading from an older MIST release? `pip install "mist-medical[train]"`
    still works too — it's kept as an identical alias for `[dali]`, so
    nothing in an existing script or notebook breaks.

**AMD ROCm GPU** (install a ROCm-enabled PyTorch build first — PyPI's default
`torch` wheel has no ROCm support — matching your driver's ROCm version, then
MIST on top with no extra needed):

```console
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install mist-medical
```

See
[Accelerator support](advanced_topics.md#accelerator-support-nvidia-amd-rocm-cpu)
for details and how to pick the right `rocmX.Y` version for your machine.

## Key Features

- **Automatic configuration** — analysis step determines target spacing, patch
  size, normalization, and foreground cropping from your data and available GPU
  memory
- **Five-fold cross-validation** by default, with custom fold assignment support
- **Multi-GPU training** via PyTorch DDP on NVIDIA CUDA or AMD ROCm; uses all
  visible GPUs automatically
- **GPU-accelerated data loading** via NVIDIA DALI during training, with a
  generic CPU-based data loader used automatically on AMD ROCm or CPU-only
  hardware — no NVIDIA-specific dependencies required to train at all
- **Sliding window inference** with configurable overlap and patch blending
- **Test-time augmentation** and **multi-model ensembling** at inference
- **Transfer learning** — initialize encoders from pretrained weights
- **Resume training** — continue interrupted runs from the last checkpoint
- **CPU training and inference** — the full pipeline, `mist_train` included,
  runs on any machine, including Macs and laptops without an NVIDIA GPU

## Supported Architectures

| Model                                    | Key                                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| nnU-Net                                  | `nnunet`                                                               |
| nnU-Net Pocket                           | `nnunet-pocket`                                                        |
| ResEnc UNet                              | `resenc`                                                               |
| Primus (medium / large / X)              | `primus-medium`, `primus-large`, `primus-x`                            |
| PrimusV2 (small / base / medium / large) | `primusv2-small`, `primusv2-base`, `primusv2-medium`, `primusv2-large` |
| MedNeXt (small / base / medium / large)  | `mednext-small`, `mednext-base`, `mednext-medium`, `mednext-large`     |
| FMG-Net                                  | `fmgnet`                                                               |
| W-Net                                    | `wnet`                                                                 |
| Swin UNETR (small / base / large)        | `swinunetr-small`, `swinunetr-base`, `swinunetr-large`                 |

## What's New

- November 2026 — **MIST 2.4.0** — new ResEnc UNet (`resenc`), Primus, and
  PrimusV2 architectures, and fine-tuning from the openly available
  nnFoundationCNN and nnFoundationViT pretrained encoders with
  `--pretrained-weights`.
- October 2026 — **MIST 2.3.0, the first stable release** — CPU and AMD ROCm
  training, `mist_finalize` for per-fold distributed training, probability-level
  and parallel ensembling, and no more ANTs dependency.
  `pip install --upgrade mist-medical`, or
  `pip install --upgrade "mist-medical[dali]"` for DALI-accelerated data loading
  on NVIDIA GPUs (highly recommended).
- July 2026 — **Interactive Colab demo** — run the full pipeline on a free Colab
  GPU with the
  [heart segmentation notebook](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb).
- 2024 and 2025 — **3rd place in the BraTS adult glioma challenge** at
  [MICCAI 2024](https://www.synapse.org/Synapse:syn53708249/wiki/630150) and
  [MICCAI 2025](https://www.synapse.org/Synapse:syn64153130/wiki/633062).

See the [changelog](changelog.md) for the full release history.

## Citation

If you use MIST in your work, please cite:

[A. Celaya et al. "MIST: A Simple and Scalable End-To-End 3D Medical Imaging Segmentation Framework," arXiv preprint arXiv:2407.21343](https://www.arxiv.org/abs/2407.21343)

[A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3224873.](https://ieeexplore.ieee.org/document/9964128)
