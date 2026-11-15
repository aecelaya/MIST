"""Run MIST's real training on two spawned processes, without a GPU.

mist_train only spawns on GPUs, so this calls what its fit() would call. It's a
script because spawned processes re-import the main module, which shouldn't be
pytest.
"""

import os
import sys
from pathlib import Path

import torch.multiprocessing as mp

from mist.cli.train_entrypoint import _create_train_dirs, _parse_train_args
from mist.training.trainers.patch_3d_trainer import Patch3DTrainer

NUM_PROCESSES = 2


def main() -> None:
    """Train with the mist_train arguments given on the command line."""
    # Keep the processes from competing for the same CPU cores.
    os.environ["OMP_NUM_THREADS"] = "1"
    args = _parse_train_args(sys.argv[1:])
    _create_train_dirs(Path(args.results), has_test_paths=False)
    trainer = Patch3DTrainer(args)
    mp.spawn(
        trainer.run_cross_validation,
        args=(NUM_PROCESSES,),
        nprocs=NUM_PROCESSES,
        join=True,
    )


if __name__ == "__main__":
    main()
