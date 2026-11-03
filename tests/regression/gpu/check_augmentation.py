"""Manual GPU check for MIST's training augmentation.

The test suite fakes DALI, so the DALI pipeline only really runs on a CUDA
machine. Run this there after mist_analyze and mist_preprocess:

    python tests/regression/gpu/check_augmentation.py \\
        --results /path/to/results --numpy /path/to/numpy --out /tmp/aug_check

For each data loader backend available ("dali" and "generic") it builds the
training loader the way training does, pulls batches through the augmenter,
checks their shapes and label values, times batches with and without
augmentation, and saves a few augmented patches as NIfTI files to eyeball
(image and label should still line up after rotation and scaling).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from mist.data_loading import augmentation, data_loader_registry


def _paths(numpy_dir: Path, subdir: str, ids: list[str]) -> list[str]:
    return [str(numpy_dir / subdir / f"{pid}.npy") for pid in ids]


def _time_batches(iterator, count: int) -> float:
    """Return the mean seconds per batch over `count` batches."""
    iterator.next()  # Warm up.
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(count):
        iterator.next()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--numpy", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("aug_check"))
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--dtms", action="store_true", help="Also load DTMs.")
    args = parser.parse_args()

    config = json.loads((args.results / "config.json").read_text())
    training = config["training"]
    patch_size = tuple(config["spatial_config"]["patch_size"])
    labels = config["dataset_info"]["labels"]
    ids = list(pd.read_csv(args.results / "train_paths.csv")["id"])
    settings = augmentation.settings_from_config(training["augmentation"])
    margin = augmentation.spatial_margin(patch_size, settings)
    print(f"Patch size {patch_size}, margin {margin}, settings {settings}")

    common = {
        "image_paths": _paths(args.numpy, "images", ids),
        "label_paths": _paths(args.numpy, "labels", ids),
        "dtm_paths": _paths(args.numpy, "dtms", ids) if args.dtms else None,
        "batch_size": training["batch_size_per_gpu"],
        "roi_size": patch_size,
        "labels": labels[1:],
        "oversampling": training["dali_foreground_prob"],
        "seed": training["seed"],
        "num_workers": training["hardware"]["num_cpu_workers"],
        "rank": 0,
        "world_size": 1,
    }
    args.out.mkdir(parents=True, exist_ok=True)

    for name in data_loader_registry.list_registered_data_loaders():
        loader = data_loader_registry.get_data_loader_from_registry(name)
        print(f"\n== {name} ==")

        plain = loader.get_training_dataset(**common, patch_margin=(0, 0, 0))
        enlarged = loader.get_training_dataset(**common, patch_margin=margin)
        raw = enlarged.next()[0]
        expected_raw = tuple(p + 2 * m for p, m in zip(patch_size, margin, strict=True))
        assert tuple(raw["image"].shape[2:]) == expected_raw, raw["image"].shape
        print(f"Enlarged patch {tuple(raw['image'].shape)} on {raw['image'].device}")

        augmented = augmentation.AugmentedIterator(
            enlarged, augmentation.BatchAugmenter(patch_size, settings, seed=0)
        )
        foreground = []
        for i in range(args.batches):
            batch = augmented.next()[0]
            assert tuple(batch["image"].shape[2:]) == patch_size, batch["image"].shape
            assert tuple(batch["label"].shape[2:]) == patch_size, batch["label"].shape
            values = set(batch["label"].unique().int().tolist())
            assert values <= set(labels), f"unexpected label values {values - set(labels)}"
            assert torch.isfinite(batch["image"]).all(), "non-finite image values"
            foreground.append((batch["label"] > 0).float().mean().item())
            if i == 0:
                for n in range(min(2, batch["image"].shape[0])):
                    for key in ("image", "label"):
                        array = batch[key][n, 0].float().cpu().numpy()
                        sitk.WriteImage(
                            sitk.GetImageFromArray(array.T),
                            str(args.out / f"{name}_sample{n}_{key}.nii.gz"),
                        )
        print(f"{args.batches} augmented batches OK; mean foreground {np.mean(foreground):.3f}")

        seconds_plain = _time_batches(plain, args.batches)
        seconds_augmented = _time_batches(augmented, args.batches)
        print(
            f"Seconds per batch: no augmentation {seconds_plain:.4f}, "
            f"with augmentation {seconds_augmented:.4f}"
        )

    print(f"\nSaved sample patches to {args.out}")


if __name__ == "__main__":
    main()
