Getting Started
=====

!!! tip "Prefer to try before installing?"

    The **[end-to-end Colab demo](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_heart_demo.ipynb)**
    runs the entire pipeline on a free GPU in your browser — no local setup
    required. It's the fastest way to see how MIST goes from raw NIfTI files to
    trained models, predictions, and evaluation.

## System Requirements

MIST runs every step, training included, on a CPU, an NVIDIA GPU, or an AMD
ROCm GPU. A GPU is optional but makes training far faster. On a CPU, use a
smaller patch, batch size, and number of epochs. You also need enough RAM to
hold your 3D images.

## Install

### Default

Runs the full pipeline, including training, on any machine. This includes
laptops, Linux workstations, HPC nodes, etc.

```console
pip install mist-medical
```

### NVIDIA GPU

Use the same command as for CPU. With PyTorch 2.9, also upgrade cuDNN:

!!! warning "PyTorch 2.9 and cuDNN"

    PyTorch 2.9's CUDA wheels come with cuDNN 9.10, which runs 3D convolutions
    in BF16 and FP16 about 3x slower than FP32, so training gets nothing from
    mixed precision. `mist_train` warns when it finds this. Install cuDNN 9.15
    or later alongside PyTorch (use `cu13` instead of `cu12` with CUDA 13
    builds):

    ```console
    pip install -U "nvidia-cudnn-cu12>=9.15"
    ```

    pip will report that torch requires an older cuDNN. That's expected, and
    training works. To check which cuDNN PyTorch loads (91500 or higher is
    9.15 or later):

    ```console
    python -c "import torch; print(torch.backends.cudnn.version())"
    ```

    The [Docker image](advanced_topics.md#docker) already has the newer cuDNN.

### AMD ROCm GPU

PyPI's default `torch` wheel has no ROCm support, so install a ROCm-enabled
PyTorch build _first_ — matching your machine's installed ROCm version, which
you can check with `cat /opt/rocm/.info/version` — then install MIST on top of
it. No install extra is needed:

```console
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install mist-medical
```

Swap `rocm6.4` for whichever ROCm release matches your driver (see the
[ROCm PyTorch install docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
for the full list of supported versions). To check that PyTorch sees the GPU
through ROCm:

```console
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

You want a version string ending in `+rocmX.Y`, a `torch.version.hip` that
isn't `None`, and `True`. MIST treats the GPU as AMD ROCm when
`torch.cuda.is_available()` is `True` and `torch.version.hip` is set.

### Development install

To install MIST and customize the underlying code (e.g., add a loss function or
new architecture), clone the repo and install in editable mode. On AMD ROCm,
install the ROCm PyTorch wheel first, as above:

```console
git clone https://github.com/mist-medical/MIST.git
cd MIST
pip install -e .          # CPU, NVIDIA, or AMD ROCm
```

## Data Format

The MIST pipeline assumes that your train and test data directories are set up
in the following structure.

```console
data/
    patient_1/
        image_1.nii.gz
        image_2.nii.gz
        ...
        image_n.nii.gz
        mask.nii.gz
    patient_2/
        image_1.nii.gz
        image_2.nii.gz
        ...
        image_n.nii.gz
        mask.nii.gz
```

!!! note

    The naming convention is for this example only. MIST does not enforce any
    specific naming conventions for the files inside of your dataset — only that
    filenames are consistent across patient directories and that each file can be
    identified by at least one unique substring (e.g., `"t1n.nii.gz"` to match
    all T1 images, or `"seg.nii.gz"` to match all mask files).

MIST offers support for MSD and CSV formatted datasets via `mist_convert_msd`
and `mist_convert_csv`. For more details, please see
[Converting CSV and MSD Data](usage.md#converting-csv-and-msd-data).

Once your dataset is in the correct format, the final step is to prepare a small
JSON file containing the details of the dataset. We specifically ask for the
following key-value pairs.

| Key             | Value                                                                                                                                                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`          | Name of task (i.e., brats, lits, etc.).                                                                                                                                                                                     |
| `modality`      | Options are `ct`, `mr`, or `other`.                                                                                                                                                                                         |
| `train-data`    | Path to training data directory. Can be absolute or relative to the dataset JSON file.                                                                                                                                      |
| `test-data`     | Path to test data directory (optional). Can be absolute or relative to the dataset JSON file.                                                                                                                               |
| `mask`          | List containing identifying strings for the segmentation mask (ground truth) files.                                                                                                                                         |
| `images`        | Dictionary where each key is an image type (i.e., T1, T2, CT, etc.) and each value is a list containing identifying strings for that image type.                                                                            |
| `labels`        | List of labels in dataset (starting with 0).                                                                                                                                                                                |
| `final_classes` | _(optional)_ Dictionary where each key is the name of the final segmentation class (i.e., WT, ET, TC for BraTS) and each value is a list of the labels in that class. If omitted, each label is evaluated as its own class. |

Here is an example for the BraTS 2023 dataset using absolute paths.

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

The same dataset JSON using relative paths:

```json
{
  "task": "brats2023",
  "modality": "mr",
  "train-data": "relative/to/dataset/json/train",
  "test-data": "relative/to/dataset/json/validation",
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

!!! note

    Relative paths in the dataset JSON are resolved relative to the **location
    of the JSON file itself**, not the working directory from which you run MIST.
    This means the JSON and its data directories can be moved together to a new
    location (or a different machine) without needing to edit the paths, as long
    as the relative structure between the JSON file and the data directories is
    preserved.

## Next steps

Run the whole pipeline with `mist_run_all`, or each step on its own. See
[Usage](usage.md).
