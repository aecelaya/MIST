Usage
===

## Overview

MIST is a **command-line tool** for medical image segmentation. The pipeline
consists of three main stages:

1. **Analysis** – Gathers dataset parameters such as target spacing,
normalization settings, and patch size. Produces a `config.json` file, which is
required for the rest of the pipeline.

2. **Preprocessing** – Uses the parameters learned during analysis to preprocess
the data (reorient, resample, normalize, etc.) and convert it into NumPy arrays.

3. **Training** – Trains models on the preprocessed data using five-fold cross
validation, producing a set of models for inference.

MIST also provides auxiliary commands for **postprocessing**,
**test-time prediction**, **evaluation**, and **dataset conversion**.

## Running the full pipeline

To run the entire pipeline with default arguments, use the `mist_run_all`
command:

- `--data` (**required**): Path to your dataset JSON file.  
- `--numpy`: Path to save preprocessed NumPy files. *(default: `./numpy`)*
- `--results`: Path to save pipeline outputs. *(default: `./results`)*

!!! note
    The `numpy` and `results` directories will be created automatically if they
    do not already exist.

### Example

Run the entire MIST pipeline with default arguments.

```console
mist_run_all --data /path/to/dataset.json \
             --numpy /path/to/preprocessed/data \
             --results /path/to/results
```

See below for more details about each command and how to run them individually.

## Output

The output of the MIST pipeline is stored under the `./results` directory, with
the following structure:

```text
results/
    checkpoints/                (fold checkpoints for --resume)
    logs/                       (TensorBoard logs)
    models/                     (trained model weights)
    predictions/                (cross-validation and test predictions)
    config.json
    fg_bboxes.csv
    train_paths.csv
    test_paths.csv              (only if test-data is set in dataset JSON)
    evaluation_paths.csv
    results.csv
    data_dump.json              (only with --data-dump)
    data_dump.md                (only with --data-dump)
```

### Breakdown of outputs

| File/Directory         | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `checkpoints/`         | Per-fold training checkpoints used by `--resume` to continue interrupted runs. |
| `logs/`                | TensorBoard logs for each fold.                                             |
| `models/`              | Trained PyTorch model weights for each fold (`fold_0.pt`, `fold_1.pt`, …). |
| `predictions/`         | Cross-validation predictions and test-set predictions (if a test set was provided). |
| `config.json`          | Full pipeline configuration (target spacing, normalization, patch size, model, loss, etc.). Required by all downstream commands. |
| `fg_bboxes.csv`        | Foreground bounding box for each training image, used to restore predictions to original space. |
| `train_paths.csv`      | Paths to training images/masks with assigned fold numbers.                  |
| `test_paths.csv`       | Paths to test images/masks. Only produced when `test-data` is set in the dataset JSON. |
| `evaluation_paths.csv` | Ground truth and prediction paths used for cross-validation evaluation.     |
| `results.csv`          | Per-patient and aggregate evaluation metrics from cross-validation.         |
| `data_dump.json`       | Structured dataset statistics (machine-readable). Only produced with `--data-dump`. |
| `data_dump.md`         | Narrative dataset summary. Only produced with `--data-dump`.                |

## Analysis

The **analysis step** computes dataset parameters (target spacing, normalization
, patch size, etc.) and saves them to `config.json`.

!!! note
    The `config.json` file is **required** for all subsequent stages, including
    inference.

Run analysis alone with the `mist_analyze` command. This command has the
following arguments:

- `--data` (**required**): Path to your dataset JSON file.
- `--results`: Directory to save analysis outputs. *(default: `./results`)*
- `--nfolds`: How many folds to split the dataset into. *(default: 5)*
- `--num-workers-analyze`: Number of parallel workers for dataset analysis.
*(default: 1)*
- `--verify`: Verify dataset integrity before analysis (checks headers,
  dimensions, and that all declared files are present).
- `--data-dump`: Save extended dataset statistics alongside `config.json`
  (`data_dump.json` and `data_dump.md`). See [Data Dump](#data-dump) below.
- `--overwrite`: Overwrite previous results/configuration.

!!!note
    Paths in the dataset JSON file (i.e., `train-data` and `test-data`) can be
    absolute or relative. Relative paths are resolved relative to the **location
    of the JSON file itself**, so the JSON and its data can be moved to a
    different location or machine without editing the paths, as long as their
    relative structure is preserved.

### Example

Run the MIST analysis pipeline.

```console
mist_analyze --data /path/to/dataset.json \
             --results /path/to/analysis/results
```

### Data Dump

When the `--data-dump` flag is passed, the analysis step produces two additional
files alongside `config.json`: `data_dump.json` and `data_dump.md`.

`data_dump.json` contains a full structured summary of the dataset statistics,
including:

- **Spacing and anisotropy** – per-axis voxel spacing statistics and anisotropy
ratio.
- **Image dimensions** – original and estimated resampled dimensions.
- **Intensity distributions** – per-channel foreground intensity statistics
(mean, std, and key percentiles).
- **Label statistics** – per-label voxel counts, presence rates, volume
fractions (relative to both foreground and the effective image region), and
shape descriptors:
    - *PCA-based descriptors* — three shape metrics derived from the eigenvalues
    of the label's spatial covariance matrix:
        - *Linearity*: how much the shape extends along a single axis (high for
          elongated structures such as vessels).
        - *Planarity*: how much the shape lies in a plane (high for disc-like
          structures).
        - *Sphericity*: how uniformly the shape extends in all directions (high
          for compact, roughly spherical structures).
    - *Isoperimetric Quotient (IQ)* measuring compactness relative to a sphere.
    - *Skeleton ratio* — the fraction of label voxels on the morphological
    medial axis, which is the primary signal for thin, branching structures
    such as vessels or airways.
- **Observations** – auto-generated notes flagging anisotropy, sparse labels,
thin/branching structures, and other dataset characteristics that may influence
architecture or loss function choices.

`data_dump.md` is a human-readable Markdown version of the same statistics,
pre-filled with metric definitions and auto-generated observations. It is
intended to be reviewed and annotated by the user before being passed to an LLM
for architecture and training configuration advice.

## Preprocessing

The second step in the MIST pipeline is to take the parameters gathered from the
analysis step and use them to preprocess the dataset. This step converts raw
NIfTI files into NumPy arrays, which will be used for training.

The preprocessing stage requires the `config.json` file produced during the
analysis step.

To run the preprocessing portion of the MIST pipeline only, use the
`mist_preprocess` command. This command has the following arguments:

- `--results`: Path to the output of the analysis step. *(default: `./results`)*
- `--numpy`: Path to save the preprocessed NumPy files. *(default: `./numpy`)*
- `--num-workers-preprocess`: Number of parallel workers for preprocessing.
  *(default: 1)*
- `--compute-dtms`: Compute per-class Distance Transform Maps (DTMs) from ground
truth masks. DTMs encode each voxel's signed distance to the nearest label
boundary and are required by certain loss functions (`bl`, `hdos`, `gsl`).
- `--no-preprocess`: Skip preprocessing steps and only convert raw NIfTI files
into NumPy format.
- `--overwrite`: Overwrite previous preprocessing output.

!!!note
  Use `--no-preprocess` when your images are already fully preprocessed
  externally and stored as NIfTI files. MIST will read each image as-is and
  convert it directly to NumPy format — reorientation, cropping, resampling,
  and normalization are all skipped. DTMs are still computed if
  `--compute-dtms` is also passed.

### Example

Run the MIST preprocessing pipeline and compute DTMs.

```console
mist_preprocess --results /path/to/analysis/results \
                --numpy /path/to/preprocessed/data \
                --compute-dtms
```

## Training

The next step in the MIST pipeline is to take the preprocessed data and train
models using a cross validation scheme. Training produces a set of models that
can later be used for inference or ensemble prediction.

To run the training stage only, use the `mist_train` command. This command has
the following arguments:

- `--numpy`: Path to the preprocessed NumPy data. *(default: `./numpy`)*
- `--results`: Path to save training outputs (models, logs, predictions, etc.).
  *(default: `./results`)*. This should also contain the output of the analysis
  pipeline.
- `--overwrite`: Overwrite previous configuration/results.

**Hardware:**

- `--num-workers-evaluate`: Number of parallel workers for the post-training
  evaluation step. *(default: `1`)*

!!! note
    MIST uses all GPUs visible to the process. To restrict which GPUs are used,
    set `CUDA_VISIBLE_DEVICES` before running MIST (e.g.,
    `CUDA_VISIBLE_DEVICES=0,1 mist_train ...`). On HPC clusters, the job
    scheduler sets this automatically. See
    [Multi-GPU training](advanced_topics.md#multi-gpu-training) for details.

**Model:**

- `--model`: Network architecture. *(default: `nnunet`)*
- `--patch-size`: Patch size as three integers: `X Y Z`. This will overwrite the
  choice of patch size determined by the analysis pipeline.

**Loss function:**

- `--loss`: Loss function for training. *(default: `dice_ce`)*
- `--composite-loss-weighting`: Weighting schedule for composite losses.
*(default: `None`)*

**Training loop:**

- `--epochs`: Number of epochs per fold. *(default: `1000`)*
- `--batch-size-per-gpu`: Batch size per GPU worker. *(default: `2`)*
- `--learning-rate`: Initial learning rate. *(default: `0.001`)*
- `--lr-scheduler`: Learning rate scheduler *(default: `cosine`)*.
- `--warmup-epochs`: Number of linear warmup epochs before the main LR
  schedule begins. *(default: `0`)*
- `--optimizer`: Optimizer *(default: `adamw`)*.
- `--l2-penalty`: L2 penalty (weight decay). *(default: `0.0001`)*
- `--folds`: Specify which folds to run. If not provided, all folds are trained.
- `--val-percent`: Specify a percentage of the training data to set aside as a
  validation set. If not specified, we use the entire held out fold as a
  validation set during training.
- `--resume`: Resume training from the latest checkpoint. See
  [Resuming training](#resuming-training) for details.

**Transfer learning** *(experimental)*:

!!! warning "Experimental feature"
    Transfer learning is still under active development. You are welcome to
    try it, but you may encounter rough edges. If you run into problems,
    please [open an issue on GitHub](https://github.com/mist-medical/MIST/issues).

- `--pretrained-weights`: Path to a pretrained checkpoint to initialize the
  encoder from. Accepts a single fold checkpoint or the output of
  `mist_average_weights`. See
  [Transfer learning](advanced_topics.md#transfer-learning) for details.
- `--pretrained-config`: Path to the source model's `config.json`. Required
  when `--pretrained-weights` is set — used to validate encoder compatibility
  between source and target models.
- `--input-channel-strategy`: How to handle an input-channel mismatch between
  the source and target encoder. Choices: `average` (mean over source channels),
  `first` (use first source channel only), `skip` (keep random initialization).
  *(default: `average`)*

!!! note
    Gradient clipping norm (`training.grad_clip_norm`, default `1.0`) is
    configurable directly in `config.json` but is not exposed as a CLI flag.
    See [Optimizers](advanced_topics.md#optimizers) in the advanced topics
    for details.

### Example

Run the MIST training pipeline with custom training hyperparameters.

```console
mist_train --numpy /path/to/preprocessed/data \
           --results /path/to/results \
           --model mednext-base \
           --epochs 200 \
           --batch-size-per-gpu 4 \
           --learning-rate 1e-4 \
           --optimizer adamw
```

At the end of the training loop, MIST will run inference on the held out fold,
write the predictions to `./results/predictions/train/raw`, and then evaluate
the results with the metrics specified in the `evaluation` entry of the
configuration file. The computed metrics will be saved in
`./results/results.csv`.

### Resuming training

If a training run is interrupted (e.g., due to a preempted job, out-of-memory
error, or system crash), it can be resumed from the last completed epoch using
the `--resume` flag.

```console
mist_train --numpy /path/to/preprocessed/data \
           --results /path/to/results \
           --resume
```

MIST saves a checkpoint at the end of every completed epoch to
`results/checkpoints/fold_{fold}_checkpoint.pt`. The checkpoint stores the
full training state: model weights, optimizer state, learning rate scheduler
state, AMP scaler state, epoch index, global step, and best validation loss.

On resume:

- **Interrupted folds** are continued from the epoch after the last completed
  one. All training state is restored exactly, including the learning rate
  schedule.
- **Completed folds** (where the saved epoch equals the final epoch) are
  skipped automatically.
- **Missing checkpoints** (e.g., a fold that never started) fall back to
  training from scratch with a warning.

!!! warning
    `--resume` and `--overwrite` are mutually exclusive. Passing both will
    raise an error.

!!! note
    Checkpoints are written atomically — a temporary file is written first
    and then renamed into place, so a crash during the save itself will never
    leave a corrupted checkpoint on disk.

## Inference

The main MIST pipeline is responsible for training and evaluating models. The
`mist_predict` command performs inference using trained MIST models on new data.

!!! note
	To use `mist_predict`, you need the models directory and config.json file from
  the output of the main MIST pipeline.

The `mist_predict` command uses the following arguments:

- `--models-dir`: (**required**) Path to the `./results/models` directory.
- `--config`: (**required**) Path to the `./results/config.json` file.
- `--paths-csv`: (**required**) Path to CSV file containing patient IDs and
  paths to imaging data. Must have an `id` column and one column per image
  type matching the dataset's image keys (e.g., `t1`, `t2`). See the table
  below for the required format.
- `--output`: (**required**) Path to directory containing predictions.
- `--device`: Device to run inference with. This can be `cpu`, `cuda`, or the
integer ID of a specific GPU (i.e., `1`). *(default: `cuda`)*.
- `--postprocess-strategy`: Path to postprocessing strategy JSON file. See below
for more details on defining postprocessing strategies in MIST.

For CSV formatted data, the CSV file must, at a minimum, have an `id` column
with the new patient IDs and one column for each image type. For example, for
the BraTS dataset, our CSV header would look like the following.

| id         | t1               | t2               | tc               | fl               |
|------------|------------------|------------------|------------------|------------------|
| Patient ID | Path to t1 image | Path to t2 image | Path to tc image | Path to fl image |

### Example

Run inference with a postprocessing strategy file on GPU `2`.

```console
mist_predict --models-dir /path/to/models \
             --config /path/to/config.json \
             --paths-csv /path/to/data/paths.csv \
             --output /path/to/output/folder \
             --device 2 \
             --postprocess-strategy /path/to/postprocess.json
```

## Postprocessing

MIST includes a flexible postprocessing utility that allows users to apply
custom postprocessing strategies to prediction masks. These strategies are
defined via a JSON file and support operations like removing small objects,
extracting connected components, and filling holes. This enables experimentation
with a range of postprocessing techniques to improve segmentation accuracy.

Postprocessing is run using the `mist_postprocess` command and uses the following
arguments:

- `--base-predictions` (**required**): Path to directory containing the base
predictions to postprocess.
- `--output` (**required**): Root output directory. See
[Output structure](#output-structure) below for details.
- `--postprocess-strategy` (**required**): Path to JSON file defining the
sequence of postprocessing steps to apply.
- `--num-workers-postprocess` *(optional)*: Number of parallel workers for
postprocessing. Defaults to `1`.
- `--num-workers-evaluate` *(optional)*: Number of parallel workers for
evaluating postprocessed predictions. Only used when `--paths-csv` and
`--eval-config` are provided. Defaults to `1`.
- `--paths-csv` *(optional)*: CSV with `id` and `mask` columns containing
patient IDs and paths to ground truth masks. When provided alongside
`--eval-config`, evaluation is automatically run on the postprocessed
predictions. The `train_paths.csv` generated by `mist_analyze` can be passed
here directly — any extra columns (e.g. image channel paths) are ignored.
- `--eval-config` *(optional)*: Path to an evaluation config JSON. Required
when `--paths-csv` is provided. Accepts a full MIST `config.json` (the
`evaluation` key is extracted automatically) or a standalone evaluation config.

### Output structure

Every `mist_postprocess` run produces the following layout under `--output`:

```
output/
├── predictions/        # postprocessed NIfTI masks (one per patient)
├── strategy.json       # copy of the strategy file used (for reproducibility)
└── postprocess_results.csv   # evaluation results (only when --paths-csv and
                              # --eval-config are provided)
```

### Strategy-based postprocessing

Postprocessing is configured using a JSON strategy file. Each strategy is a list
of steps, where each step includes the transformation name, the target labels, a
flag for whether the operation should be applied per label or across grouped
labels, and any additional parameters.

### Strategy file format

The strategy file is a JSON file containing a list of postprocessing steps. Each
step is a dictionary with the following required fields:

- **`transform`** (`str`):
  Name of the postprocessing transformation. Currently supported transformations
  are:
  - `remove_small_objects`: Remove connected components below a size threshold.
  - `fill_holes_with_label`: Fill interior holes in a mask with a specified
    label.
  - `get_top_k_connected_components`: Keep only the `k` largest connected
    components.
  - `replace_small_objects_with_label`: Replace small components with a
    different label instead of zeroing them out.

  Each transformation can be applied either **per label** (independently to each
  label) or **grouped** (treating all specified labels as one binary mask),
  controlled via the `per_label` flag.

  Each transform is registered in `transform_registry.py`. Custom transforms can
  be added by implementing a function there and decorating it with
  `@register_transform('name', metadata={...})`.

- **`apply_to_labels`** (`List[int]`):
  A list of label integers to which the transform should be applied.
  For example, `[1, 2]` applies the transform to labels 1 and 2.
  Use `[-1]` to apply to all non-zero labels.

- **`per_label`** (`bool`):
  Controls how the transform is applied to `apply_to_labels`:
  - `true` — apply the transform independently to each label.
  - `false` — group all specified labels into a single binary mask and apply
    the transform once.

  > **Note:** `replace_small_objects_with_label` always requires `per_label:
  > true` because each component must retain its original label value.

- **`kwargs`** *(optional, `Dict[str, Any]`)*:
  Transform-specific keyword arguments. Valid kwargs for each transform are:

  | Transform | kwarg | Description | Default |
  |---|---|---|---|
  | `remove_small_objects` | `small_object_threshold` | Minimum component size (voxels) to retain | `64` |
  | `get_top_k_connected_components` | `top_k_connected_components` | Number of largest components to keep | `1` |
  | `get_top_k_connected_components` | `apply_morphological_cleaning` | Apply erosion before and dilation after component selection | `false` |
  | `get_top_k_connected_components` | `morphological_cleaning_iterations` | Number of erosion/dilation iterations | `2` |
  | `fill_holes_with_label` | `fill_holes_label` | Label value to assign to filled holes | `0` |
  | `replace_small_objects_with_label` | `small_object_threshold` | Maximum component size (voxels) to replace | `64` |
  | `replace_small_objects_with_label` | `replacement_label` | Label to assign to small components | `0` |

Below is an example strategy file that demonstrates several transformations:

```json
[
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [1],
    "per_label": true,
    "kwargs": {
      "small_object_threshold": 64
    }
  },
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [2, 4],
    "per_label": false,
    "kwargs": {
      "small_object_threshold": 100
    }
  },
  {
    "transform": "fill_holes_with_label",
    "apply_to_labels": [1, 2],
    "per_label": false,
    "kwargs": {
      "fill_holes_label": 1
    }
  },
  {
    "transform": "get_top_k_connected_components",
    "apply_to_labels": [4],
    "per_label": true,
    "kwargs": {
      "top_k_connected_components": 1,
      "apply_morphological_cleaning": true,
      "morphological_cleaning_iterations": 1
    }
  },
  {
    "transform": "replace_small_objects_with_label",
    "apply_to_labels": [1, 2, 4],
    "per_label": true,
    "kwargs": {
      "small_object_threshold": 50,
      "replacement_label": 0
    }
  }
]
```

### Examples

Run the postprocessing pipeline without evaluation:

```console
mist_postprocess --base-predictions /path/to/predictions \
                 --output /path/to/output \
                 --postprocess-strategy /path/to/strategy.json
```

Run the postprocessing pipeline and evaluate the results:

```console
mist_postprocess --base-predictions /path/to/predictions \
                 --output /path/to/output \
                 --postprocess-strategy /path/to/strategy.json \
                 --paths-csv /path/to/paths.csv \
                 --eval-config /path/to/config.json
```

## Evaluation

MIST provides a flexible command-line tool to evaluate prediction masks against
ground truth using various metrics. Metrics and their parameters are defined
entirely in a config JSON, giving you full per-class control without any
additional CLI flags.

To run the stand-alone evaluation pipeline, use `mist_evaluate` with the
following arguments:

- `--config` (**required**): Path to an evaluation config JSON. Accepts either a
full MIST `config.json` (the `evaluation` key is extracted automatically) or a
standalone evaluation config with the nested per-class structure shown below.
- `--paths-csv` (**required**): Path to CSV file containing patient IDs and
paths to ground truth and predicted masks.
- `--output-csv` (**required**): Path to output CSV containing the computed
metrics for each patient.
- `--num-workers-evaluate` *(optional)*: Number of parallel workers. *(default: 1)*
- `--validate` *(optional)*: Validate each mask pair before evaluation. Checks
that images are 3D, have an integer or boolean dtype, and contain only labels
defined in the config. Recommended for external data.

The paths CSV for the evaluation tool should have the following format:

| id         | mask                       | prediction         |
|------------|----------------------------|--------------------|
| Patient ID | Path to ground truth mask  | Path to prediction |

### Evaluation config format

The `evaluation` entry in `config.json` (or a standalone config file) defines
one or more classes to evaluate. Each class specifies which label values to
include and which metrics to compute, along with any metric-specific parameters:

```json
{
  "class_name": {
    "labels": [1, 2, 3],
    "metrics": {
      "metric_name": {"param": value}
    }
  }
}
```

### Available metrics

| Metric key             | Description                                   | Parameters |
|------------------------|-----------------------------------------------|------------|
| `dice`                 | Volumetric Sørensen–Dice coefficient          | — |
| `haus95`               | 95th-percentile Hausdorff distance (mm)       | — |
| `avg_surf`             | Average symmetric surface distance (mm)       | — |
| `surf_dice`            | Surface Dice at a configurable tolerance      | `tolerance` (mm, default `1.0`) |
| `lesion_wise_dice`     | BraTS-style lesion-wise Dice                  | see below |
| `lesion_wise_haus95`   | BraTS-style lesion-wise HD95 (mm)             | see below |
| `lesion_wise_surf_dice`| BraTS-style lesion-wise surface Dice          | see below |

#### Lesion-wise metric parameters

Lesion-wise metrics evaluate each GT lesion individually, track false positives,
and aggregate using `sum(scores) / (num_gt_above_thresh + num_fp)` — the same
formula used by the BraTS (Brain Tumor Segmentation) challenge. This scoring
penalizes both missed lesions and spurious predictions equally, regardless of
lesion size.

| Parameter                 | Default | Description |
|---------------------------|---------|-------------|
| `min_lesion_volume`       | `10.0`  | Minimum GT lesion volume in mm³. Lesions smaller than this are excluded. |
| `dilation_iters`          | `3`     | Dilation iterations used to match predicted components to a GT lesion. |
| `gt_consolidation_iters`  | `0`     | Dilation iterations for merging nearby GT lesions before analysis. Set equal to `dilation_iters` to replicate BraTS-style consolidation. `0` disables consolidation. |
| `tolerance`               | `1.0`   | Surface Dice tolerance in mm (`lesion_wise_surf_dice` only). |

> **Penalization rules:** An undetected GT lesion (false negative) contributes
> `0` to the Dice / surface Dice numerator, or the image diagonal to the HD95
> numerator, and `1` to the denominator. Each spurious predicted lesion (false
> positive) is penalized identically.

### Example

Run the evaluation pipeline with Dice and HD95:

```console
mist_evaluate --config /path/to/config.json \
              --paths-csv /path/to/evaluation/paths.csv \
              --output-csv /path/to/output.csv
```

### BraTS-style lesion-wise evaluation example

The following standalone evaluation config replicates the BraTS glioma (GLI)
lesion-wise evaluation protocol for Whole Tumor (WT), Tumor Core (TC), and
Enhancing Tumor (ET). BraTS glioma label conventions: `1` = necrotic core,
`2` = peritumoral edema, `3` = enhancing tumor.

```json
{
  "whole_tumor": {
    "labels": [1, 2, 3],
    "metrics": {
      "lesion_wise_dice": {
        "min_lesion_volume": 50.0,
        "dilation_iters": 3,
        "gt_consolidation_iters": 3
      },
      "lesion_wise_haus95": {
        "min_lesion_volume": 50.0,
        "dilation_iters": 3,
        "gt_consolidation_iters": 3
      }
    }
  },
  "tumor_core": {
    "labels": [1, 3],
    "metrics": {
      "lesion_wise_dice": {
        "min_lesion_volume": 50.0,
        "dilation_iters": 3,
        "gt_consolidation_iters": 3
      },
      "lesion_wise_haus95": {
        "min_lesion_volume": 50.0,
        "dilation_iters": 3,
        "gt_consolidation_iters": 3
      }
    }
  },
  "enhancing_tumor": {
    "labels": [3],
    "metrics": {
      "lesion_wise_dice": {
        "min_lesion_volume": 50.0,
        "dilation_iters": 3,
        "gt_consolidation_iters": 3
      },
      "lesion_wise_haus95": {
        "min_lesion_volume": 50.0,
        "dilation_iters": 3,
        "gt_consolidation_iters": 3
      }
    }
  }
}
```

Save this as `brats_eval_config.json` and run:

```console
mist_evaluate --config brats_eval_config.json \
              --paths-csv /path/to/evaluation/paths.csv \
              --output-csv /path/to/brats_results.csv
```

## Converting CSV and MSD Data

Several popular formats exist for different datasets, like the Medical
Segmentation Decathlon (MSD) or simple CSV files with file paths to images and
masks. To bridge the usability gap between these kinds of datasets and MIST, we
provide two dedicated conversion commands.

Both commands copy data into a MIST-compatible directory structure and generate
a `dataset.json` file. Paths inside the generated `dataset.json` are written
as relative paths, making the converted dataset portable across machines and
cloud environments.

### `mist_convert_msd`

Converts a Medical Segmentation Decathlon dataset.

| Argument | Required | Description |
|---|---|---|
| `--source` | Yes | Path to the MSD dataset directory (must contain `dataset.json`). |
| `--output` | Yes | Directory to save the converted MIST-format dataset. |
| `--num-workers-conversion` | No | Number of parallel threads for file copying. *(default: 1)* |

```console
mist_convert_msd --source /path/to/msd/dataset \
                 --output /path/to/mist/dataset
```

The MSD `dataset.json` is used to automatically populate the task name,
modality, labels, and class definitions in the generated MIST `dataset.json`.

### `mist_convert_csv`

Converts a CSV-format dataset.

| Argument | Required | Description |
|---|---|---|
| `--train-csv` | Yes | Path to training CSV with columns: `id`, `mask`, `image1` [, `image2`, ...]. |
| `--output` | Yes | Directory to save the converted MIST-format dataset. |
| `--test-csv` | No | Path to optional test CSV with columns: `id`, `image1` [, `image2`, ...]. |
| `--num-workers-conversion` | No | Number of parallel threads for file copying. *(default: 1)* |

```console
mist_convert_csv --train-csv /path/to/train.csv \
                 --output /path/to/mist/dataset \
                 --test-csv /path/to/test.csv
```

!!! note
    CSV conversion copies the data into MIST format but cannot infer task
    name, modality, labels, or class definitions automatically. After
    conversion, open the generated `dataset.json` and fill in the `task`,
    `modality`, `labels`, and `final_classes` fields before running
    `mist_analyze`.
