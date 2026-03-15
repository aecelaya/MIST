Getting Started
=====

### System Requirements
MIST assumes that your system as at least one NVIDIA GPU and sufficient memory
to handle 3D medical images.

### Install
To install the latest version of MIST as an out-of-the-box segmentation
pipeline, use

```console
pip install mist-medical
```

If you want to install MIST and customize the underlying code (i.e., add a loss
function or new architecture), then clone the MIST repo and install as follows:

```console
git clone https://github.com/mist-medical/MIST.git
cd MIST
pip install -e .
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
    specific naming conventions for the files inside of you dataset - only that
    the names of the images/masks be consistent within each patient directory
    and that they are identifiable by a list of identifier strings.

MIST offers support for MSD and CSV formatted datasets. For more details, please
see [MSD and CSV Formatted Data](usage.md).

Once your dataset is in the correct format, the final step is to prepare a small
JSON  file containing the details of the dataset. We specifically ask for the
following key-value pairs.

| Key | Value |
|---|---|
| ```task``` | Name of task (i.e., brats, lits, etc.). |
| ```modality``` | Options are ``ct``, ``mr``, or ``other``. |
| ```train-data``` | Path to training data directory. Can be absolute or relative to the dataset JSON file. |
| ```test-data``` | Path to test data directory (optional). Can be absolute or relative to the dataset JSON file. |
| ```mask``` | List containing identifying strings for mask or ground truth images in dataset. |
| ```images``` | Dictionary where each key is an image type (i.e., T1, T2, CT, etc.) and each value  is a list containing identifying strings for that image type. |
| ```labels``` | List of labels in dataset (starting with 0). |
| ```final_classes``` | Dictionary where each key is the name of the final segmentation class  (i.e., WT, ET, TC for BraTS) and each value is a list of the labels in that class. |

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
    "train-data": "relative/train",
    "test-data": "relative/validation",
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
