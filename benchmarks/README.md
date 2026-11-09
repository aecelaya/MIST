# Benchmarks

Measurements of MIST on real GPUs, run by hand. These aren't tests: nothing here
runs in CI or passes or fails on its own. Each one produces numbers to read, and
some of MIST's constants (the planner's memory estimate, for example) come from
them. For MIST demos, see [`examples/`](../examples/).

## Colab notebooks

Open a notebook on Colab with its badge, choose a GPU under *Runtime > Change
runtime type*, then *Runtime > Run all*. Each saves a CSV named after the GPU and
downloads it.

| Notebook | What it measures | When to run it |
| --- | --- | --- |
| [`mist_compile_benchmark.ipynb`](mist_compile_benchmark.ipynb) | Speed, memory, and compile time of `torch.compile` for each network and precision, and the effect of cuDNN's algorithm search | After changing networks or upgrading PyTorch |
| [`mist_planning_calibration.ipynb`](mist_planning_calibration.ipynb) | Real GPU memory against the patch planner's estimate, for each network and precision | After changing a network or the planner's constants, once per GPU |
| [`mist_data_loading_benchmark.ipynb`](mist_data_loading_benchmark.ipynb) | Data loader speed (DALI, generic, and nnU-Net's) and training steps on synthetic LiTS- and BraTS-sized data | After changing data loading, preprocessing, or augmentation |
| [`mist_precision_memory.ipynb`](mist_precision_memory.ipynb) | GPU memory and speed of FP32, BF16, and FP16 training, one setting changed at a time | After changing mixed precision or the loss path |

## Scripts

Run these on a machine with a GPU and MIST installed.

| Script | What it checks |
| --- | --- |
| [`calibrate_planning.py`](calibrate_planning.py) | The planner's memory estimate against peak memory, for every network |
| [`check_patch_reads.py`](check_patch_reads.py) | How fast training patches read from preprocessed data on this machine's filesystem, through a memory map and with positioned reads |
| [`check_blosc2_reads.py`](check_blosc2_reads.py) | Whether Blosc2 (nnU-Net's format) makes training patches fast to read from this filesystem: compression ratio and patch read times for several chunk/block layouts and codecs, against the memory map and positioned reads. Needs `pip install blosc2` |
| [`check_augmentation.py`](check_augmentation.py) | Training augmentation with each data loader on real preprocessed data, with patches saved to inspect |
| [`check_training.py`](check_training.py) | That the default pipeline analyzes, preprocesses, trains, and predicts on a real dataset |

Each script's docstring gives its command line.
