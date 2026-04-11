Getting Started
=====

### System Requirements
**Training** requires at least one NVIDIA GPU and sufficient memory to handle
3D medical images.

**Inference** (`mist_predict`) runs on any machine, including CPU-only systems
and Macs, and does not require an NVIDIA GPU.

### Install

#### Inference only (CPU-compatible)
To run `mist_predict` on any machine — including laptops and Macs without an
NVIDIA GPU — install the base package:

```console
pip install mist-medical
```

#### Training (NVIDIA GPU required)
To train models, install the `train` extra, which includes NVIDIA DALI for
GPU-accelerated data loading:

```console
pip install "mist-medical[train]"
```

#### Development install
To install MIST and customize the underlying code (e.g., add a loss function
or new architecture), clone the repo and install in editable mode. Add
`[train]` if you need to run training:

```console
git clone https://github.com/mist-medical/MIST.git
cd MIST
pip install -e .          # inference only
pip install -e ".[train]" # training
```

### Data Format
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

!!!note
    The naming convention is for this example only. MIST does not enforce any
    specific naming conventions for the files inside of your dataset — only that
    filenames are consistent across patient directories and that each file can be
    identified by at least one unique substring (e.g., `"t1n.nii.gz"` to match
    all T1 images, or `"seg.nii.gz"` to match all mask files).

MIST offers support for MSD and CSV formatted datasets via `mist_convert_msd`
and `mist_convert_csv`. For more details, please see
[Converting CSV and MSD Data](usage.md#converting-csv-and-msd-data).

Once your dataset is in the correct format, the final step is to prepare a small
JSON  file containing the details of the dataset. We specifically ask for the
following key-value pairs.

| Key | Value |
|---|---|
| ```task``` | Name of task (i.e., brats, lits, etc.). |
| ```modality``` | Options are ``ct``, ``mr``, or ``other``. |
| ```train-data``` | Path to training data directory. Can be absolute or relative to the dataset JSON file. |
| ```test-data``` | Path to test data directory (optional). Can be absolute or relative to the dataset JSON file. |
| ```mask``` | List containing identifying strings for the segmentation mask (ground truth) files. |
| ```images``` | Dictionary where each key is an image type (i.e., T1, T2, CT, etc.) and each value  is a list containing identifying strings for that image type. |
| ```labels``` | List of labels in dataset (starting with 0). |
| ```final_classes``` | *(optional)* Dictionary where each key is the name of the final segmentation class (i.e., WT, ET, TC for BraTS) and each value is a list of the labels in that class. If omitted, each label is evaluated as its own class. |

Here is an example for the BraTS 2023 dataset using absolute paths.

```json
{
    "task": "brats2023",
    "modality": "mr",
    "train-data": "/full/path/to/raw/data/train",
    "test-data": "/full/path/to/raw/data/validation",
    "mask": ["seg.nii.gz"],
    "images": {"t1": ["t1n.nii.gz"],
               "t2": ["t2w.nii.gz"],
               "tc": ["t1c.nii.gz"],
               "fl": ["t2f.nii.gz"]},
    "labels": [0, 1, 2, 3],
    "final_classes": {"WT": [1, 2, 3],
                      "TC": [1, 3],
                      "ET": [3]}
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
    "images": {"t1": ["t1n.nii.gz"],
               "t2": ["t2w.nii.gz"],
               "tc": ["t1c.nii.gz"],
               "fl": ["t2f.nii.gz"]},
    "labels": [0, 1, 2, 3],
    "final_classes": {"WT": [1, 2, 3],
                      "TC": [1, 3],
                      "ET": [3]}
}
```

!!!note
    Relative paths in the dataset JSON are resolved relative to the **location
    of the JSON file itself**, not the working directory from which you run MIST.
    This means the JSON and its data directories can be moved together to a new
    location (or a different machine) without needing to edit the paths, as long
    as the relative structure between the JSON file and the data directories is
    preserved.
