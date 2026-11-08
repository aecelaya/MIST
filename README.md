Medical Imaging Segmentation Toolkit
===

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mist-medical/MIST/python-publish.yml?style=flat)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen?style=flat)](https://mist-medical.readthedocs.io/en/latest/)
![GitHub Repo stars](https://img.shields.io/github/stars/mist-medical/MIST?style=flat)
![Coverage](coverage.svg)

MIST is a simple, scalable, end-to-end framework for 3D medical image
segmentation. It handles everything from raw NIfTI files to trained models and
evaluated predictions, with sensible defaults that work well out of the box and
a configuration file for when you need more control.

<!-- prettier-ignore -->
> [!TIP]
> **Try MIST in your browser — no setup required.** The
> [**end-to-end Colab demo**](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb)
> runs the whole pipeline — analyze → preprocess → train → evaluate → predict →
> postprocess → rank → visualize — on a heart MRI dataset using a free Colab T4
> GPU.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb)

## Installation

**CPU** (default — runs the full pipeline: train, predict, evaluate,
everything):

```bash
pip install mist-medical
```

The same command works on an **NVIDIA GPU**. The `[dali]` extra
(`pip install "mist-medical[dali]"`, also available as `[train]`) installs NVIDIA
DALI, which MIST only uses when `training.hardware.data_loader` is set to
`"dali"`.

**AMD ROCm GPU** (install a ROCm-enabled PyTorch build first — PyPI's default
`torch` wheel has no ROCm support — matching your driver's ROCm version, then
MIST on top with no extra needed; see
[Accelerator support](https://mist-medical.readthedocs.io/en/latest/advanced_topics/#accelerator-support-nvidia-amd-rocm-cpu)
for details):

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install mist-medical
```

**Development**:

```bash
git clone https://github.com/mist-medical/MIST.git
cd MIST
pip install -e .            # CPU or NVIDIA GPU
pip install -e ".[dali]"    # Adds NVIDIA DALI, for "data_loader": "dali"
```

**Docker** (NVIDIA GPU required, driver ≥ 525.x):

```bash
docker pull mistmedical/mist:latest
```

## Quick Start

**1. Prepare a dataset JSON** describing your data:

```json
{
  "task": "brats2023",
  "modality": "mr",
  "train-data": "/full/path/to/raw/data/train",
  "test-data": "/full/path/to/raw/data/validation",
  "mask": ["seg.nii.gz"],
  "images": {
    "t1": ["t1n.nii.gz"],
    "t2": ["t2w.nii.gz"],
    "tc": ["t1c.nii.gz"],
    "fl": ["t2f.nii.gz"]
  },
  "labels": [0, 1, 2, 3],
  "final_classes": { "WT": [1, 2, 3], "TC": [1, 3], "ET": [3] }
}
```

**2. Run the full pipeline** (analyze → preprocess → train → evaluate):

```bash
mist_run_all --data dataset.json \
             --numpy /path/to/numpy \
             --results /path/to/results
```

**3. Run inference** on new data:

```bash
mist_predict --models-dir /path/to/results/models \
             --config /path/to/results/config.json \
             --paths-csv /path/to/test.csv \
             --output /path/to/predictions
```

## Key Features

- **Automatic configuration** — analysis step determines target spacing, patch
  size, normalization, and foreground cropping from your data
- **Five-fold cross-validation** by default, with custom fold assignment support
- **Multi-GPU training** via PyTorch DDP on NVIDIA CUDA or AMD ROCm; uses all
  visible GPUs automatically
- **Fast data loading** that reads only each training patch, with augmentation
  on the GPU — on NVIDIA, AMD ROCm, or CPU-only hardware, with no
  NVIDIA-specific dependencies
- **Sliding window inference** with configurable overlap and patch blending
- **Test-time augmentation** and **multi-model ensembling** at inference
- **Postprocessing** with learnable, per-class morphological strategies
- **CPU training and inference** — the full pipeline, `mist_train` included,
  runs on any machine, including Macs and laptops without an NVIDIA GPU

## Supported Architectures

| Model                                    | Key                                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| nnU-Net (default)                        | `nnunet-plain`                                                         |
| nnU-Net Residual                         | `nnunet-res`, `nnunet`                                                 |
| nnU-Net Pocket                           | `nnunet-pocket`                                                        |
| ResEnc UNet                              | `resenc`                                                               |
| Primus (medium / large / X)              | `primus-medium`, `primus-large`, `primus-x`                            |
| PrimusV2 (small / base / medium / large) | `primusv2-small`, `primusv2-base`, `primusv2-medium`, `primusv2-large` |
| MedNeXt (small / base / medium / large)  | `mednext-small`, `mednext-base`, `mednext-medium`, `mednext-large`     |
| FMG-Net                                  | `fmgnet`                                                               |
| W-Net                                    | `wnet`                                                                 |
| Swin UNETR (small / base / large)        | `swinunetr-small`, `swinunetr-base`, `swinunetr-large`                 |

Change the architecture in `config.json` or pass `--model <key>` at the command
line.

## Supported Loss Functions

| Loss                         | Key               | Notes     |
| ---------------------------- | ----------------- | --------- |
| Dice                         | `dice`            |           |
| Dice + Cross-Entropy         | `dice_ce`         |           |
| clDice                       | `cldice`          | Composite |
| Boundary Loss                | `bl`              | Composite |
| Generalized Surface Loss     | `gsl`             | Composite |
| Hausdorff Distance One-Sided | `hdos`            | Composite |
| Volumetric SDDL              | `volumetric_sddl` | Composite |
| Vessel SDDL                  | `vessel_sddl`     | Composite |

Composite losses blend a region-based term with a boundary/distance term,
weighted by a scheduled alpha. Schedules: `constant`, `linear`, `cosine`.

## Pipeline Commands

| Command                | Description                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `mist_run_all`         | Run the full pipeline end-to-end                                                             |
| `mist_analyze`         | Analyze dataset and generate `config.json`                                                   |
| `mist_preprocess`      | Preprocess images into NumPy arrays                                                          |
| `mist_train`           | Train models                                                                                 |
| `mist_finalize`        | Aggregate out-of-fold predictions into `results.csv` and run test-set inference              |
| `mist_predict`         | Run inference on new data                                                                    |
| `mist_ensemble`        | Combine predictions from multiple models via STAPLE, majority vote, or probability averaging |
| `mist_evaluate`        | Evaluate predictions against ground truth                                                    |
| `mist_postprocess`     | Apply postprocessing strategies                                                              |
| `mist_rank`            | Rank multiple evaluation result CSVs BraTS-style                                             |
| `mist_average_weights` | Average model weights across folds                                                           |
| `mist_convert_msd`     | Convert Medical Segmentation Decathlon datasets                                              |
| `mist_convert_csv`     | Convert CSV-formatted datasets                                                               |

## Documentation

Full documentation, including configuration reference and advanced topics, is at
[**mist-medical.readthedocs.io**](https://mist-medical.readthedocs.io/).

## What's New

- November 2026 — **MIST 2.4.0** — new ResEnc UNet (`resenc`), Primus, and
  PrimusV2 architectures, and fine-tuning from the openly available
  nnFoundationCNN and nnFoundationViT pretrained encoders with
  `--pretrained-weights`. Try it in the
  [fine-tuning notebook](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_nnfoundation_spleen_demo.ipynb).
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

See the [changelog](CHANGELOG.md) for the full release history.

## Citation

If you use MIST in your work, please cite:

```bibtex
@article{celaya2024mist,
  title   = {MIST: A Simple and Scalable End-To-End 3D Medical Imaging Segmentation Framework},
  author  = {Celaya, Adrian and others},
  journal = {arXiv preprint arXiv:2407.21343},
  year    = {2024}
}

@article{celaya2022pocketnet,
  title   = {PocketNet: A Smaller Neural Network For Medical Image Analysis},
  author  = {Celaya, Adrian and others},
  journal = {IEEE Transactions on Medical Imaging},
  doi     = {10.1109/TMI.2022.3224873},
  year    = {2022}
}
```
