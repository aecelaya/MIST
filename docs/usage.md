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
    logs/
    models/
    predictions/
    config.json
    data_dump.json
    data_dump.md
    results.csv
    train_paths.csv
    evaluation_paths.csv
    test_paths.csv (if a test set is specified in dataset.json)
    fg_bboxes.csv
```

### Breakdown of outputs

| File/Directory         | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `logs/`                | TensorBoard logs for each fold.                                             |
| `models/`              | Trained PyTorch models for each fold.                                       |
| `predictions/`         | Predictions from cross validation and test set (if specified).              |
| `config.json`          | Dataset configuration (target spacing, normalization, patch size, etc.).    |
| `data_dump.json`       | Full structured dataset statistics produced by the analysis step (machine-readable). |
| `data_dump.md`         | Narrativized dataset summary optimized for review and LLM consumption.      |
| `results.csv`          | Evaluation results from five-fold cross validation.                         |
| `train_paths.csv`      | CSV with `id`, `fold`, and paths to images/masks for training.              |
| `evaluation_paths.csv` | CSV with `id`, `mask`, and `prediction` paths for evaluation.               |
| `test_paths.csv`       | Same as `train_paths.csv`, but for test set (no `fold` column).             |
| `fg_bboxes.csv`        | Bounding box information for the foreground region of each image.           |

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

After computing dataset parameters, the analysis step produces two additional
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
    - *PCA-based descriptors* (linearity, planarity, sphericity) characterising
    the global geometry of each label.
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
for architecture and training configuration advice (this feature is currently in progress, stay tuned!).

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
- `--compute-dtms`: Compute per-class distance transform maps (DTMs) from ground
truth masks.
- `--no-preprocess`: Skip preprocessing steps and only convert raw NIfTI files
into NumPy format.
- `--overwrite`: Overwrite previous preprocessing output.

!!!note
  The `--no-preprocess` flag does not completely turn off all of the
  preprocessing steps. With this flag, the preprocessing pipeline will still
  reorient the images to RAI, crop to the foreground (if called for by the
  analysis pipeline), and compute DTMs (if called for by the user).

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

- `--gpus`: IDs of GPUs to use; use `-1` for all GPUs. *(default: `-1`)*

**Model:**

- `--model`: Network architecture. *(default: `nnunet`)*  
- `--pocket`: Flag to enable the pocket version of the model (if available).
- `--patch-size`: Patch size as three integers: `X Y Z`. This will overwrite the
the choice of patch size determined by the analysis pipeline.

**Loss function:**

- `--loss`: Loss function for training. *(default: `dice_ce`)*  
- `--use-dtms`: Flag to use distance transform maps during training.  
- `--composite-loss-weighting`: Weighting schedule for composite losses.
*(default: `None`)*

**Training loop:**

- `--epochs`: Number of epochs per fold. *(default: `1000`)*
- `--batch-size-per-gpu`: Batch size per GPU worker. *(default: `2`)*
- `--learning-rate`: Initial learning rate. *(default: `0.001`)*
- `--lr-scheduler`: Learning rate scheduler *(default: `cosine`)*.
- `--optimizer`: Optimizer *(default: `adam`)*.
- `--l2-penalty`: L2 penalty (weight decay). *(default: `0.00001`)*
- `--folds`: Specify which folds to run. If not provided, all folds are trained.
- `--val-percent`: Specify a percentage of the training data to set aside as a
validation set. If not specified, the we use the entire held out fold as a
a validation set during training.

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

## Inference

The main MIST pipeline is responsible for training and evaluating models. The
`mist_predict` command performs inference using trained MIST models on new data.

!!! note
	To use `mist_predict`, you need the models directory and config.json file from
  the output of the main MIST pipeline.

The `mist_predict` command uses the following arguments:

- `--models-dir`: (**required**) Path to the `./results/models` directory.
- `--config`: (**required**) Path to the `./results/config.json` file.
- `--paths-csv`: (**required**) Path to CVS containing patient IDs and paths to
imaging data (see below for more details).
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

- `--base-predictions` (**required**): Path to directory containing the
predictions which we will apply postprocessing.
- `--output` (**required**): Path to directory where we will write the
postprocessed predictions. This directory will be created if it does not exist.
- `--postprocess-strategy` (**required**): Path to JSON file defining the
sequence of postprocessing steps that we will apply.

### Strategy-based postprocessing

Postprocessing is configured using a JSON strategy file. Each strategy is a list
of steps, where each step includes the transformation name, the target labels, a
flag for whether the operation should be applied sequentially per label, and any
additional parameters.

### Strategy file format
The strategy file is a JSON file containing a list of postprocessing steps. Each
step is a dictionary with the following required fields:

- **`transform`** (`str`):  
  Name of the postprocessing transformation. Currently supported transformations
  are
  	- `remove_small_objects`: Remove connected components below a given size
    threshold.
	- `fill_holes_with_label`: Fill holes in a mask with a specified label.
	- `get_top_k_connected_components`: Keep the largest `k` connected components.
	- `replace_small_objects_with_label`: Replace the label of small objects with
  a different label.

  Each transformation can be applied either **sequentially per label** or
  **across a grouped set of labels**, controlled via the `apply_sequentially`
  flag.

  Each transform is registered in the `transform_registry.py` file. You can add
  custom postprocessing transforms by implementing them in this file and
  registering them with the `@register_transform('name')` decorator.

- **`apply_to_labels`** (`List[int]`):  
  A list of label integers to which the transform should be applied.  
  For example, `[1, 2]` will apply the transform to labels 1 and 2.

- **`apply_sequentially`** (`bool`):  
  Indicates whether to apply the transform to each label individually (`true`),  
  or to all the labels in the group at once (`false`).

- **`kwargs`** *(optional, `Dict[str, Any]`)*:  
  Additional keyword arguments passed directly to the transform function. These
  are transform-specific. For the currently available transforms, the following
  keyword arguments are valid:

  - `small_object_threshold` – A threshold for removing or replacing small
  objects.
    Used in `remove_small_objects` and `replace_small_objects_with_label`. A
    common default value is `64`.

  - `top_k` – The number of largest connected components to retain.
    Used in `get_top_k_connected_components`. For example, `top_k: 1` retains
    only the largest component.

  - `fill_label` – The label value to use when filling holes.  
    Used in `fill_holes_with_label`. For example, `fill_label: 1` will fill
    holes using label `1`.

  - `replacement_label` – The label used to replace small components. Used in
  `replace_small_objects_with_label`. For example, `replacement_label: 1` will
  update the label of objects smaller than `small_object_threshold` with `1`.

  - `morph_cleanup` – Whether to apply morphological erosion/dilation during
  `get_top_k_connected_components`. Set to `true` to activate cleanup.

  - `morph_cleanup_iterations` – The number of iterations to use for
  morphological erosion/dilation if `morph_cleanup` is enabled. Default is
  usually `2`.

Below is an example strategy file that demonstrates several transformations:

```json
[
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [1],
    "apply_sequentially": true,
    "kwargs": {
      "small_object_threshold": 64
    }
  },
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [2, 4],
    "apply_sequentially": false,
    "kwargs": {
      "small_object_threshold": 100
    }
  },
  {
    "transform": "fill_holes_with_label",
    "apply_to_labels": [1, 2],
    "apply_sequentially": false,
    "kwargs": {
      "fill_label": 1
    }
  },
  {
    "transform": "get_top_k_connected_components",
    "apply_to_labels": [4],
    "apply_sequentially": true,
    "kwargs": {
      "top_k": 1,
      "morph_cleanup": true,
      "morph_cleanup_iterations": 1
    }
  },
  {
    "transform": "replace_small_objects_with_label",
    "apply_to_labels": [1, 2, 4],
    "apply_sequentially": true,
    "kwargs": {
      "small_object_threshold": 50,
      "replacement_label": 0
    }
  }
]
```

### Example

Run the postprocessing pipeline.

```console
mist_postprocess --base-predictions /path/to/original/predictions \
                 --output /folder/to/save/postprocessed/predictions \
                 --postprocess-strategy /path/to/strategy.json
```

### Notes

- This version of `mist_postprocess` does **not** perform evaluation or update
the `config.json` file. It is designed to decouple strategy testing from
evaluation logic.
- Evaluation of postprocessing performance can be done separately using
`mist_evaluate`.

## Evaluation

MIST provides a flexible command-line tool to evaluate prediction masks against
ground truth using various metrics. The evaluation script supports several
metrics and outputs a detailed summary of the evaluation in CSV format.

To run the stand-alone evaluation pipeline, use the `mist_evaluate` with the
following arguments:

- `--config` (**required**): Path to the `./results/config.json` file. The
configuration file defines the evaluation classes in the `evaluation` entry.
- `--paths-csv` (**required**): Path to CSV file containing patient IDs and
paths to ground truth and predicted masks.
- `--output-csv` (**required**): Path to output CSV containing the computed
metrics for each patient.
- `--metrics`: Metrics to compute for each ground truth/prediction pair. Choices
are Dice (`dice`), 95th percentile Hausdorff distance (`haus95`),
average surface distance (`avg_surf`), and surface Dice (`surf_dice`).
*(default: `dice haus95`)*
- `--surf-dice-tol`: Tolerance (mm) for the surface Dice metric.
*(default: `1.0`)*

The paths CSV for the evaluation tool should have the following format:

| id         | mask                       | prediction         |
|------------|----------------------------|--------------------|
| Patient ID | Path to ground truth mask  | Path to prediction |

### Example

Run the evaluation pipeline with the surface Dice using a 1.5 mm tolerance.

```console
mist_evaluate --config /path/to/config.json \
              --paths-csv /path/to/evaluation/paths.csv \
              --output-csv /path/to/output.csv \
              --metrics dice haus95 surf_dice \
              --surf-dice-tol 1.5
```

## Converting CSV and MSD Data

Several popular formats exist for different datasets, like the Medical
Segmentation Decathlon (MSD) or simple CSV files with file paths to images and
masks. To bridge the usability gap between these kinds of datasets and MIST, we
provide a conversion tool called `mist_convert_dataset` to take MSD or CSV
formatted data and convert it to MIST-compatible data.

The `mist_convert_dataset` uses the following arguments:

- `--format`: (**required**) The format of the given dataset, which can be
either `msd` or `csv`.
- ```--output```: (**required**) Path to the new MIST formatted dataset,
- At least one of the following are required depending on the chosen format:
  - `--msd-source`: Path to the MSD dataset, if that is what you are converting.
  - `--train-csv`: Path to the CSV file containing training data, if that is
  what you are using
  - `--test-csv`: Path to CSV file containing test data.

The format for CSV data should be as follows:

| id         | mask                       | images             |
|------------|----------------------------|--------------------|
| Patient ID | Path to ground truth mask  | Path to images     |

!!! note
	If converting a CSV file, this command will reformat the CSV dataset to a
  MIST-compatible one, but will require the user to fill in details in its
  corresponding dataset JSON file.

### Example

Convert a MSD dataset into a MIST-compatible dataset.

```console
mist_convert_dataset --format msd \
                     --output /path/to/mist/dataset \
                     --msd-source /path/to/msd/top_level/directory \
```

Convert a CSV dataset into a MIST-compatible dataset.

```console
mist_convert_dataset --format csv \
                     --output /path/to/mist/dataset \
                     --train-csv /path/to/training/data.csv \
                     --test-csv /path/to/test/data.csv
```
