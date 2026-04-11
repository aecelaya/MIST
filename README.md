Medical Imaging Segmentation Toolkit
===

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mist-medical/MIST/python-publish.yml)
![Read the Docs](https://img.shields.io/readthedocs/mist-medical?style=flat)
![GitHub Repo stars](https://img.shields.io/github/stars/mist-medical/MIST?style=flat)
![Coverage](coverage.svg)

MIST is a simple, scalable, end-to-end framework for 3D medical image
segmentation. It handles everything from raw NIfTI files to trained models and
evaluated predictions, with sensible defaults that work well out of the box and
a configuration file for when you need more control.

## Installation

**Training** (NVIDIA GPU required):
```bash
pip install "mist-medical[train]"
```

**Inference only** (CPU-compatible, works on Mac):
```bash
pip install mist-medical
```

**Development**:
```bash
git clone https://github.com/mist-medical/MIST.git
cd MIST
pip install -e ".[train]"   # full
pip install -e .             # inference only
```

## Quick Start

**1. Prepare a dataset JSON** describing your data:
```json
{
    "task":        "brats2023",
    "modality":    "mr",
    "train-data":  "/full/path/to/raw/data/train",
    "test-data":   "/full/path/to/raw/data/validation",
    "mask":        ["seg.nii.gz"],
    "images":      {"t1": ["t1n.nii.gz"],
                    "t2": ["t2w.nii.gz"],
                    "tc": ["t1c.nii.gz"],
                    "fl": ["t2f.nii.gz"]},
    "labels":      [0, 1, 2, 3],
    "final_classes": {"WT": [1, 2, 3],
                      "TC": [1, 3],
                      "ET": [3]}
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
- **Multi-GPU training** via PyTorch DDP; uses all visible GPUs automatically
- **GPU-accelerated data loading** via NVIDIA DALI during training
- **Sliding window inference** with configurable overlap and patch blending
- **Test-time augmentation** and **multi-model ensembling** at inference
- **Postprocessing** with learnable, per-class morphological strategies
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

Change the architecture in `config.json` or pass `--model <key>` at the command line.

## Supported Loss Functions

| Loss | Key | Notes |
|---|---|---|
| Dice | `dice` | |
| Dice + Cross-Entropy | `dice_ce` | |
| clDice | `cldice` | Composite |
| Boundary Loss | `bl` | Composite |
| Generalized Surface Loss | `gsl` | Composite |
| Hausdorff Distance One-Sided | `hdos` | Composite |
| Volumetric SDDL | `volumetric_sddl` | Composite |
| Vessel SDDL | `vessel_sddl` | Composite |

Composite losses blend a region-based term with a boundary/distance term,
weighted by a scheduled alpha. Schedules: `constant`, `linear`, `cosine`.

## Pipeline Commands

| Command | Description |
|---|---|
| `mist_run_all` | Run the full pipeline end-to-end |
| `mist_analyze` | Analyze dataset and generate `config.json` |
| `mist_preprocess` | Preprocess images into NumPy arrays |
| `mist_train` | Train models |
| `mist_predict` | Run inference on new data |
| `mist_evaluate` | Evaluate predictions against ground truth |
| `mist_postprocess` | Apply postprocessing strategies |
| `mist_average_weights` | Average model weights across folds |
| `mist_convert_msd` | Convert Medical Segmentation Decathlon datasets |
| `mist_convert_csv` | Convert CSV-formatted datasets |

## Documentation

Full documentation, including configuration reference and advanced topics, is at
[**mist-medical.readthedocs.io**](https://mist-medical.readthedocs.io/).

## What's New

* April 2026 — **CPU inference support** — `mist_predict` now runs on any
  machine, including Macs and laptops without an NVIDIA GPU. Install with
  `pip install mist-medical` (no GPU required).
* March 2026 — **Resume training** — interrupted runs can be continued from the last
  checkpoint with `--resume`, with atomic checkpointing to prevent corruption.
* March 2026 — **GPU-aware automatic patch size** — the analysis step now derives the
  patch size from available GPU memory, so the default configuration is
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
