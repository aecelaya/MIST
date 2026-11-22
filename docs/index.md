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

```console
pip install mist-medical
```

This runs the whole pipeline on CPUs and NVIDIA GPUs. AMD GPUs need a ROCm
build of PyTorch first, and NVIDIA GPUs with PyTorch 2.9 need a newer cuDNN for
fast mixed precision. [Getting started](getting_started.md) covers both.
Upgrading from MIST 2.x? See [Upgrading to 3.0](upgrading.md).

## Key Features

- **Automatic configuration** — analysis chooses target spacing,
  normalization, and foreground cropping from your data; the first training run
  chooses patch size, batch size, and precision for the GPU
- **Five-fold cross-validation** by default, with custom fold assignment support
- **Multi-GPU training** via PyTorch DDP on NVIDIA CUDA or AMD ROCm; uses all
  visible GPUs automatically
- **Fast data loading** that reads only each training patch, with augmentation
  on the GPU — on NVIDIA, AMD ROCm, or CPU-only hardware, with no
  NVIDIA-specific dependencies
- **Sliding window inference** with configurable overlap and patch blending
- **Test-time augmentation** and **multi-model ensembling** at inference
- **Transfer learning** — initialize encoders from pretrained weights
- **Resume training** — continue interrupted runs from the last checkpoint
- **CPU training and inference** — the full pipeline, `mist_train` included,
  runs on any machine, including Macs and laptops without an NVIDIA GPU

## Supported Architectures

nnU-Net (`nnunet-plain`, the default), residual and pocket nnU-Net, ResEnc,
Primus and PrimusV2, MedNeXt, FMG-Net, W-Net, and SwinUNETR. See
[Network architectures](advanced_topics.md#network-architectures) for the keys
to use with `--model`.

## What's New

- November 2026 — **MIST 3.0** — NVIDIA DALI is gone, and MIST's own data
  loader is the only one, on NVIDIA, AMD, and CPU machines. It trained full LiTS
  in 13 s per epoch on an H100, against 67 s with DALI. Patch size, batch size, and
  precision are planned for the GPU at training time, FP16 works on any GPU, and
  `nnunet-plain` is the new default network. In a matched five-fold comparison
  on LiTS, MIST ranked ahead of nnU-Net overall and matched it on tumors. Models
  trained with 2.x keep working; see [Upgrading to 3.0](upgrading.md).
- November 2026 — **MIST 2.4.0** — new ResEnc UNet (`resenc`), Primus, and
  PrimusV2 architectures, and fine-tuning from the openly available
  nnFoundationCNN and nnFoundationViT pretrained encoders with
  `--pretrained-weights`. Try it in the
  [fine-tuning notebook](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_nnfoundation_spleen_demo.ipynb).
- October 2026 — **MIST 2.3.0, the first stable release** — CPU and AMD ROCm
  training, `mist_finalize` for per-fold distributed training, probability-level
  and parallel ensembling, and no more ANTs dependency.
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
