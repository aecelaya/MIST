"""Manual check of whether Blosc2 makes training patches fast to read from network storage.

nnU-Net 2.8 stores preprocessed data as Blosc2 arrays (blosc2.NDArray), compressed
in chunks that are split into blocks, so a patch read only touches the blocks it
overlaps. This converts a few preprocessed cases to Blosc2 with several chunk and
block layouts and codecs, next to the original .npy files, then times random
training patches read from each, against the memory map and positioned reads the
generic loader uses:

    pip install blosc2
    python benchmarks/check_blosc2_reads.py --numpy /path/to/numpy

It reports the compression ratio and write time of each variant, then the median
and mean read time per image and label patch. Every Blosc2 patch is checked against
the .npy patch. The converted files go in a temporary folder beside --numpy (or in
--out), on the same filesystem, and are deleted at the end unless --keep is given.

Reads use one decompression thread, as one of a GPU's loader workers would. On
Linux, the script asks the kernel to drop the files it just wrote and read from the
page cache, so reads come from storage. Elsewhere, reads may hit the cache.

It only needs NumPy and Blosc2, so it runs with any MIST version installed.
"""

import argparse
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import blosc2
except ImportError:
    raise SystemExit("This script needs Blosc2: pip install blosc2") from None

CODECS = {"zstd": blosc2.Codec.ZSTD, "lz4": blosc2.Codec.LZ4}


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


def blosc2_read(array, starts, shape):
    """Decompress only the blocks the patch overlaps."""
    return array[inside(array, starts, shape)]


def as_4d(array):
    """View a (D, H, W) array as (D, H, W, 1); (D, H, W, C) arrays pass through."""
    return array if array.ndim == 4 else array[..., None]


def drop_cache(path):
    """Ask the kernel to drop a file from the page cache (Linux only)."""
    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def parse_layout(text):
    """'128/64' -> chunk side 128, block side 64."""
    chunk, block = (int(n) for n in text.split("/"))
    if block > chunk:
        raise argparse.ArgumentTypeError(f"Block {block} is larger than chunk {chunk}.")
    return chunk, block


def write(array, path, chunk, block, codec, threads):
    """Save a (D, H, W, C) array as Blosc2, chunked along the spatial axes only."""
    channels = array.shape[3]
    chunks = tuple(min(chunk, d) for d in array.shape[:3]) + (channels,)
    blocks = tuple(min(block, c) for c in chunks[:3]) + (channels,)
    cparams = blosc2.CParams(
        codec=CODECS[codec],
        clevel=5,
        typesize=array.itemsize,
        nthreads=threads,
        filters=[blosc2.Filter.SHUFFLE],
    )
    result = blosc2.asarray(
        np.ascontiguousarray(array),
        chunks=chunks,
        blocks=blocks,
        cparams=cparams,
        urlpath=str(path),
        mode="w",
    )
    return array.nbytes, result.schunk.cbytes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--numpy", required=True, help="mist_preprocess output folder.")
    parser.add_argument("--out", help="Folder for the converted files. Default: beside --numpy.")
    parser.add_argument("--cases", type=int, default=10, help="Cases to convert.")
    parser.add_argument("--patches", type=int, default=40, help="Patches per method.")
    parser.add_argument("--size", type=int, default=206, help="Patch side, with margin.")
    parser.add_argument(
        "--layouts",
        nargs="+",
        type=parse_layout,
        default=[(64, 32), (128, 64)],
        help="Chunk/block sides to try, as CHUNK/BLOCK. Default: 64/32 128/64.",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=sorted(CODECS),
        default=["zstd", "lz4"],
        help="Codecs to try, each with byte shuffle at level 5.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the converted files.")
    args = parser.parse_args()

    numpy_dir = Path(args.numpy)
    images = sorted((numpy_dir / "images").glob("*.npy"))
    if not images:
        raise SystemExit(f"No images found in {numpy_dir / 'images'}.")
    rng = np.random.default_rng()
    cases = [
        images[i]
        for i in sorted(rng.choice(len(images), min(args.cases, len(images)), replace=False))
    ]
    pairs = []
    for image in cases:
        label = numpy_dir / "labels" / image.name
        if not label.exists():
            raise SystemExit(f"No label for {image.name} in {numpy_dir / 'labels'}.")
        pairs.append((image, label))

    variants = [
        f"{codec} {chunk}/{block}" for codec in args.codecs for chunk, block in args.layouts
    ]
    out_parent = Path(args.out) if args.out else numpy_dir.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="blosc2_check_", dir=out_parent))
    threads = min(8, os.cpu_count() or 1)
    try:
        # Convert: each case is loaded once, then written in every variant.
        sizes = {v: {"images": [0, 0], "labels": [0, 0]} for v in variants}
        write_times = {v: [] for v in variants}
        files = {v: [] for v in variants}
        print(f"Converting {len(pairs)} cases to {len(variants)} variants in {out}")
        for image, label in pairs:
            data = {"images": as_4d(np.load(image)), "labels": as_4d(np.load(label))}
            for kind, path in (("images", image), ("labels", label)):
                if not data[kind].flags["C_CONTIGUOUS"]:
                    raise SystemExit(f"{path} is Fortran-ordered; rerun mist_preprocess.")
            for codec in args.codecs:
                for chunk, block in args.layouts:
                    variant = f"{codec} {chunk}/{block}"
                    folder = out / f"{codec}_{chunk}_{block}"
                    paths = {}
                    started = time.perf_counter()
                    for kind in ("images", "labels"):
                        (folder / kind).mkdir(parents=True, exist_ok=True)
                        paths[kind] = folder / kind / f"{image.stem}.b2nd"
                        raw, compressed = write(
                            data[kind], paths[kind], chunk, block, codec, threads
                        )
                        sizes[variant][kind][0] += raw
                        sizes[variant][kind][1] += compressed
                    write_times[variant].append(time.perf_counter() - started)
                    files[variant].append((paths["images"], paths["labels"]))
                    for path in paths.values():
                        drop_cache(path)
            del data
            drop_cache(image)
            drop_cache(label)
            print(f"  {image.stem}")

        print(f"\n{'variant':16} {'image ratio':>12} {'label ratio':>12} {'write s/case':>13}")
        for v in variants:
            ratios = [sizes[v][k][0] / max(1, sizes[v][k][1]) for k in ("images", "labels")]
            print(
                f"{v:16} {ratios[0]:11.2f}x {ratios[1]:11.2f}x {statistics.fmean(write_times[v]):13.2f}"
            )

        # Time reads: methods alternate, each on its own random case and window.
        blosc2.set_nthreads(1)
        dparams = blosc2.DParams(nthreads=1)
        shape = [args.size] * 3
        methods = ["memory map", "positioned reads"] + variants
        times = {m: {"images": [], "labels": []} for m in methods}
        checks = []
        for _ in range(args.patches):
            for method in methods:
                i = int(rng.integers(len(pairs)))
                for k, kind in enumerate(("images", "labels")):
                    if method in variants:
                        array = blosc2.open(str(files[method][i][k]), mode="r", dparams=dparams)
                    else:
                        # A memmap view keeps the file name and offset positioned reads need.
                        array = as_4d(np.load(pairs[i][k], mmap_mode="r"))
                    if kind == "images":
                        starts = [
                            int(rng.integers(-shape[0] // 4, max(1, d - shape[0] // 2)))
                            for d in array.shape[:3]
                        ]
                    read = {"memory map": memory_map, "positioned reads": positioned_reads}.get(
                        method, blosc2_read
                    )
                    started = time.perf_counter()
                    read(array, starts, shape)
                    times[method][kind].append(time.perf_counter() - started)
                    if method in variants:
                        checks.append((method, i, k, starts))

        print(
            f"\n{'method':16} {'image median':>13} {'mean':>8} {'label median':>13} {'mean':>8}  (ms per {args.size}^3 patch)"
        )
        for m in methods:
            image_t, label_t = (np.array(times[m][k]) * 1000 for k in ("images", "labels"))
            print(
                f"{m:16} {np.median(image_t):13.1f} {image_t.mean():8.1f}"
                f" {np.median(label_t):13.1f} {label_t.mean():8.1f}"
            )

        # Check, after timing so the timed reads saw no extra cache.
        for method, i, k, starts in checks:
            expected = memory_map(as_4d(np.load(pairs[i][k], mmap_mode="r")), starts, shape)
            array = blosc2.open(str(files[method][i][k]), mode="r", dparams=dparams)
            if not np.array_equal(blosc2_read(array, starts, shape), expected):
                raise SystemExit(
                    f"{method} patch differs from .npy for {pairs[i][k].name} at {starts}."
                )
        print(f"\nAll {len(checks)} Blosc2 patches match the .npy patches exactly.")
    finally:
        if args.keep:
            print(f"Kept the converted files in {out}")
        else:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
