"""Manual check of how fast training patches read from preprocessed data.

The generic loader used to copy each patch through a memory map of its .npy file.
On network filesystems such as NFS, that's tens of thousands of page faults per
LiTS-sized patch. It now reads each patch with one positioned read per position
along the first axis. This times both on real preprocessed data, on the machine
and filesystem that will train:

    python benchmarks/check_patch_reads.py --numpy /path/to/numpy

The two methods alternate on random cases and windows, so both see the same file
cache. It only needs NumPy, so it runs with any MIST version installed.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def inside(array, starts, shape):
    """Slices of a (D, H, W, C) array inside a patch window."""
    return tuple(
        slice(min(max(0, s), d), max(min(d, s + n), min(max(0, s), d)))
        for s, n, d in zip(starts, shape, array.shape[:3], strict=True)
    )


def memory_map(array, starts, shape):
    """Copy the patch through the memory map, as the loader did."""
    return np.array(array[inside(array, starts, shape)])


def positioned_reads(array, starts, shape):
    """Read the patch with one read per first-axis position, as the loader now does."""
    window = inside(array, starts, shape)
    depth = window[0].stop - window[0].start
    rows = window[1].stop - window[1].start
    _, height, width, channels = array.shape
    row_bytes = width * channels * array.itemsize
    block = np.empty((depth, rows, width, channels), dtype=array.dtype)
    with open(array.filename, "rb", buffering=0) as file:
        for i in range(depth):
            file.seek(array.offset + ((window[0].start + i) * height + window[1].start) * row_bytes)
            target = memoryview(block[i]).cast("B")
            done = 0
            while done < len(target):
                done += file.readinto(target[done:])
    return np.array(block[:, :, window[2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--numpy", required=True, help="mist_preprocess output folder.")
    parser.add_argument("--patches", type=int, default=40, help="Patches per method.")
    parser.add_argument("--size", type=int, default=206, help="Patch side, with margin.")
    args = parser.parse_args()

    paths = sorted((Path(args.numpy) / "images").glob("*.npy"))
    if not paths:
        raise SystemExit(f"No images found in {Path(args.numpy) / 'images'}.")
    rng = np.random.default_rng()
    shape = [args.size] * 3
    times = {"memory map": [], "positioned reads": []}
    for _ in range(args.patches):
        for name, read in (("memory map", memory_map), ("positioned reads", positioned_reads)):
            array = np.load(rng.choice(paths), mmap_mode="r")
            if not array.flags["C_CONTIGUOUS"]:
                raise SystemExit("These images are Fortran-ordered; rerun mist_preprocess.")
            starts = [
                int(rng.integers(-shape[0] // 4, max(1, d - shape[0] // 2)))
                for d in array.shape[:3]
            ]
            started = time.perf_counter()
            read(array, starts, shape)
            times[name].append(time.perf_counter() - started)

    for name, values in times.items():
        print(
            f"{name:17}: median {statistics.median(values) * 1000:7.1f} ms, "
            f"mean {statistics.fmean(values) * 1000:7.1f} ms per {args.size}^3 image patch"
        )


if __name__ == "__main__":
    main()
