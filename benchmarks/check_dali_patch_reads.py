"""Manual check of whether DALI can train from large .npy files by reading only patches.

MIST's DALI pipeline reads each whole image and label for every patch: its
reader sets dont_use_mmap, pads the whole volume, finds foreground on the whole
label, then crops. On LiTS-sized CT that is about 0.5 GB per sample. This times
that pipeline against two DALI pipelines that read only the patch, on real
preprocessed data:

- "region reads": DALI's numpy reader with roi_start and roi_shape on memory-
  mapped files. Patch windows are chosen in Python, as the generic loader
  chooses them, and fed in with external_source. DALI still asks the OS to
  read ahead each whole array (MADV_WILLNEED), so this may save copying but
  not reading; the timing shows which.
- "python reads": DALI's parallel external_source, where Python workers read
  each patch with positioned reads, as the generic loader does, and DALI
  batches the patches and moves them to the GPU.

Both read exactly the same patches, and the check compares them voxel for
voxel. Run it on a Linux machine with an NVIDIA GPU, DALI, and this branch of
MIST installed:

    python benchmarks/check_dali_patch_reads.py --numpy /path/to/numpy \\
        --patch 128 128 128 --margin 39 39 39 --labels 1 2

Steps include only data loading and the copy to the GPU, not training. Labels
default to those saved with the foreground locations, if any.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from mist.data_loading import generic_loader
from mist.preprocessing import foreground_locations

WARMUP_STEPS = 5


def choose_windows(images, labels_paths, classes, args, count, seed):
    """Choose cases and patch windows as the generic loader would."""
    rng = np.random.default_rng(seed)
    cases = rng.integers(0, len(images), size=count)
    windows = []
    cached = {}
    for case in cases:
        label = np.load(labels_paths[case], mmap_mode="r")
        if case not in cached:
            saved = foreground_locations.load(
                foreground_locations.path_for(labels_paths[case]), classes
            )
            cached[case] = (
                saved if saved is not None else foreground_locations.find(label, classes, rng)
            )
        starts, shape = generic_loader._patch_window(
            label,
            tuple(args.patch),
            classes,
            args.oversampling,
            rng,
            tuple(args.margin),
            lambda case=case: cached[case],
        )
        windows.append((int(case), np.asarray(starts, dtype=np.int32), shape))
    return windows


def read_patch(path, starts, shape):
    """Read a zero-padded (D, H, W, C) patch with positioned reads."""
    array = np.load(path, mmap_mode="r")
    inside, pad = generic_loader._read_inside(array, starts, shape)
    return np.pad(inside, [*pad, (0, 0)])


class PatchSource:
    """Per-sample source for DALI's parallel external_source."""

    def __init__(self, images, labels_paths, windows):
        self.images = images
        self.labels = labels_paths
        self.windows = windows

    def __call__(self, sample_info):
        case, starts, shape = self.windows[sample_info.idx_in_epoch % len(self.windows)]
        return read_patch(self.images[case], starts, shape), read_patch(
            self.labels[case], starts, shape
        )


def region_pipeline(images, labels_paths, windows, args):
    """DALI numpy readers that read only each patch's region."""
    from nvidia.dali import fn, pipeline_def

    order_images = [images[case] for case, _, _ in windows]
    order_labels = [labels_paths[case] for case, _, _ in windows]
    starts = [window[1] for window in windows]
    shape = list(windows[0][2])

    @pipeline_def(batch_size=args.batch_size, num_threads=args.threads, device_id=0)
    def pipe():
        start = fn.external_source(
            source=lambda info: starts[info.idx_in_epoch % len(starts)], batch=False
        )
        reader = {
            "dont_use_mmap": False,
            "read_ahead": False,
            "random_shuffle": False,
            "shuffle_after_epoch": False,
            "roi_start": start,
            "roi_shape": shape,
            "roi_axes": (0, 1, 2),
            "out_of_bounds_policy": "pad",
        }
        image = fn.readers.numpy(files=order_images, name="image_reader", **reader)
        label = fn.readers.numpy(files=order_labels, name="label_reader", **reader)
        return image.gpu(), label.gpu()

    return pipe()


def python_pipeline(images, labels_paths, windows, args):
    """DALI's parallel external_source over Python workers doing positioned reads."""
    from nvidia.dali import fn, pipeline_def

    @pipeline_def(
        batch_size=args.batch_size,
        num_threads=args.threads,
        device_id=0,
        py_num_workers=args.workers,
        py_start_method="spawn",
    )
    def pipe():
        image, label = fn.external_source(
            source=PatchSource(images, labels_paths, windows),
            num_outputs=2,
            batch=False,
            parallel=True,
        )
        return image.gpu(), label.gpu()

    return pipe()


def mist_iterator(images, labels_paths, classes, args):
    """MIST's current DALI training loader."""
    from mist.data_loading import dali_loader

    return dali_loader.get_training_dataset(
        image_paths=images,
        label_paths=labels_paths,
        dtm_paths=None,
        batch_size=args.batch_size,
        roi_size=tuple(args.patch),
        labels=classes,
        oversampling=args.oversampling,
        seed=args.seed,
        num_workers=args.threads,
        rank=0,
        world_size=1,
        patch_margin=tuple(args.margin),
    )


def time_steps(iterator, steps):
    """Median and mean seconds per batch, after warmup."""
    import torch

    def step():
        try:
            batch = next(iterator)
        except StopIteration:
            iterator.reset()
            batch = next(iterator)
        torch.cuda.synchronize()
        return batch

    for _ in range(WARMUP_STEPS):
        step()
    times = []
    for _ in range(steps):
        started = time.perf_counter()
        step()
        times.append(time.perf_counter() - started)
    return statistics.median(times), statistics.fmean(times)


def check_same(region, python, count):
    """Compare the first batches of the two patch-only pipelines voxel for voxel."""
    for _ in range(count):
        a, b = next(region)[0], next(python)[0]
        for key in ("image", "label"):
            if not np.array_equal(a[key].cpu().numpy(), b[key].cpu().numpy()):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--numpy", required=True, help="mist_preprocess output folder.")
    parser.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--margin", type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument("--labels", type=int, nargs="+", help="Foreground classes.")
    parser.add_argument("--oversampling", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8, help="DALI CPU threads.")
    parser.add_argument("--workers", type=int, default=8, help="Python reader workers.")
    parser.add_argument("--steps", type=int, default=40, help="Timed batches per loader.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-mist", action="store_true", help="Skip MIST's whole-volume pipeline."
    )
    args = parser.parse_args()

    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    root = Path(args.numpy)
    images = sorted(str(path) for path in (root / "images").glob("*.npy"))
    labels_paths = [str(root / "labels" / Path(path).name) for path in images]
    if not images:
        raise SystemExit(f"No images found in {root / 'images'}.")
    for path in (images[0], labels_paths[0]):
        if not np.load(path, mmap_mode="r").flags["C_CONTIGUOUS"]:
            raise SystemExit(f"{path} is Fortran-ordered; rerun mist_preprocess.")
    classes = args.labels
    if classes is None:
        saved = foreground_locations.path_for(labels_paths[0])
        if not saved.is_file():
            raise SystemExit("No saved foreground locations; pass --labels.")
        with np.load(saved) as data:
            classes = data["labels"].tolist()

    count = (WARMUP_STEPS + args.steps + 2) * args.batch_size
    windows = choose_windows(images, labels_paths, classes, args, count, args.seed)
    keys = ["image", "label"]
    region = DALIGenericIterator(region_pipeline(images, labels_paths, windows, args), keys)
    python = DALIGenericIterator(python_pipeline(images, labels_paths, windows, args), keys)
    same = check_same(region, python, 2)
    print(f"Region reads and python reads give the same patches: {same}")

    # Fresh loaders, so each timing starts from the same samples.
    results = {
        "region reads": DALIGenericIterator(
            region_pipeline(images, labels_paths, windows, args), keys
        ),
        "python reads": DALIGenericIterator(
            python_pipeline(images, labels_paths, windows, args), keys
        ),
    }
    if not args.skip_mist:
        results = {"MIST now": mist_iterator(images, labels_paths, classes, args), **results}
    size = [p + 2 * m for p, m in zip(args.patch, args.margin, strict=True)]
    for name, iterator in results.items():
        median, mean = time_steps(iterator, args.steps)
        print(
            f"{name:13}: median {median:6.3f} s, mean {mean:6.3f} s per batch of "
            f"{args.batch_size} {'x'.join(map(str, size))} patches"
        )


if __name__ == "__main__":
    main()
