# Upgrading to 3.0

MIST 3.0 removes NVIDIA DALI and makes MIST's own data loader the only one. It
also includes everything from 2.5.0rc0: planning at training time, FP16,
nnU-Net-style augmentation, `nnunet-plain`, and probability resampling. 2.5.0
was never released as final; 2.5.0rc0 became 3.0.

In short:

- **Your trained models keep working.** Inference on results folders from 2.x
  and 2.5.0rc0 runs as before.
- **Re-run `mist_preprocess`** before training on data preprocessed by 2.x.
  Old data still trains, just more slowly.
- **Expect different training results** from 2.x with the same config.
- With PyTorch 2.9 on NVIDIA GPUs, **upgrade cuDNN** to 9.15 or later.

## Installing

```console
pip install -U mist-medical
```

`pip install "mist-medical[dali]"` and `pip install "mist-medical[train]"`
still work, but install nothing extra. Both extras will be removed in 4.0, so
drop them from your scripts.

### PyTorch 2.9 and cuDNN

PyTorch 2.9's CUDA wheels come with cuDNN 9.10, which runs most 3D convolutions
in BF16 and FP16 without cuDNN. On an H100, `nnunet-plain` trained about 3x
slower in BF16 than in FP32, so MIST's precision test chose FP32. With cuDNN
9.15 or later, BF16 trained twice as fast as FP32 with half the memory:

```console
pip install -U "nvidia-cudnn-cu12>=9.15"
```

pip reports that torch requires cuDNN 9.10. That's expected, and training works.
Use `nvidia-cudnn-cu13` with CUDA 13 builds of PyTorch. `mist_train` warns when
it finds PyTorch 2.9 with an older cuDNN. The MIST Docker image already has the
newer cuDNN.

If a config's `training.amp` was set by the precision test under the old cuDNN,
it may say `false`. Set it back to `"auto"` to run the test again.

## What keeps working

**Inference on results folders from 2.x and 2.5.0rc0.** `mist_predict`,
`test_on_fold`, and `mist_evaluate` read old `config.json` files and
checkpoints. The network 2.x called `nnunet` is still found under that name
(3.0 calls it `nnunet-res`).

Predicted labels may differ slightly at boundaries for results folders from
before 2.5.0. Those versions took the argmax at the training spacing and
resampled the labels back to the original image; MIST now resamples the class
probabilities, then takes the argmax. This helps small and thin structures when
the original images are finer than the training spacing. Along the
low-resolution axis of anisotropic images, both use nearest-neighbor
interpolation, so nothing changes there.

MIST's test suite checks this against results folders trained by 2.4.1 and
2.5.0rc0 (`tests/compat`).

**Old config settings.** These still work:

| Setting in a 2.x config | In 3.0 |
| --- | --- |
| `training.hardware.data_loader: "dali"` | Uses the generic loader, with a warning. Set it to `"generic"` to remove the warning. |
| `training.dali_foreground_prob` | Read as `training.foreground_prob`. |
| `training.augmentation.transforms.zoom` | Read as `scaling`. |
| `training.hardware.num_cpu_workers: 8` | Kept. New configs use 4 (see below). |

## Re-run `mist_preprocess`

We strongly recommend running `mist_preprocess` again on data preprocessed by
MIST 2.x, including 2.5.0rc0:

- 3.0 saves the locations of each foreground class in a `foreground` folder next
  to the labels. Without it, each data loader worker scans every label it reads
  to find foreground, which is slow for large images such as abdominal CT.
- MIST 2.4.0 and earlier saved labels in Fortran order. The data loader can't
  read patches from those with positioned reads, so it copies them through a
  memory map instead, which is much slower on network filesystems such as NFS.

Old data still trains. `mist_train` warns about each of these and goes on.

## Why training results differ from 2.x

Training the same config with 3.0 gives different models than 2.x, beyond the
usual run-to-run variation:

- **Sampling.** The generic loader centers a patch on foreground with
  probability `foreground_prob`, choosing among the classes present with equal
  weight. DALI placed patches around connected foreground objects instead.
- **Augmentation.** The transforms, probabilities, and ranges follow nnU-Net's
  defaults: rotation, scaling from 0.7 to 1.4, simulated low resolution, and
  gamma are new, and noise, blur, brightness, contrast, and flips changed.
- **Planning.** The patch and batch size are chosen at the first `mist_train`
  for the GPU, network, and precision, within an 8 GiB memory budget. Some
  anisotropic datasets get a coarser target spacing. Configs that already have
  a patch size and target spacing keep them.
- **The saved model.** With the default `val_percent` of 0, `models/` gets the
  final-epoch model instead of the one with the lowest loss on the held-out
  fold. Cross-validation scores may come out slightly lower; that is the bias
  being removed.
- **Precision and speed.** `training.amp` defaults to `"auto"`, which times
  BF16, FP32, and FP16 and picks one. New configs compile the model and validate
  every 25 epochs.

Finish runs started with 2.x using the version that started them rather than
resuming them with 3.0.

## Data loader workers

New configs use 4 workers per GPU (`training.hardware.num_cpu_workers`) instead
of 8. On two A100s with LiTS, 4 workers trained as fast as all cores (25 against
24 s per epoch), the first epoch dropped from 87 to 40 s, and validation from
1:48 to 1:31. Fewer workers also need less shared memory (`/dev/shm`). Configs
from 2.x keep their 8.

## Staying on DALI

MIST 2.4.1 is the last release with DALI:

```console
pip install "mist-medical[dali]==2.4.1"
```

In MIST's benchmarks the generic loader was faster: 13 against 67 s per epoch
on full LiTS on an H100, 24 against 88 s on two A100s, and a tie on BraTS.
