# coding=utf-8
"""Data utilities for the GeoSim FNO baseline.

This is a clean-room implementation for the single-file 2D mantle-convection
reader used by the released FNO configuration.  It intentionally contains no
code from the NEC/PDEBench ``utils.py``; the release licence is MIT.

The reader expects HDF5 files with ``ro``, ``pr``, ``tk``, ``nu``, ``vx``,
``vy``, ``x-coordinate``, and ``y-coordinate`` datasets.  The on-disk layout
is ``(sample, time, x, y)``; batches are returned as
``(sample, x, y, time, channel)``.
"""
from __future__ import annotations

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = ("ro", "pr", "tk", "nu", "vx", "vy")


class FNODatasetSingle(Dataset):
    """Single-file reader for the 2D mantle-convection fields.

    Parameters mirror the released FNO configuration.  The train/validation
    split follows the original benchmark convention: the first
    ``int(n_samples * test_ratio)`` trajectories are validation and the rest
    are training; ``if_test=True`` returns the validation part.
    """

    def __init__(self, filename, initial_step=10, t_train=21,
                 saved_folder="./data/", reduced_resolution=1,
                 reduced_resolution_t=1, reduced_batch=1, if_test=False,
                 test_ratio=0.1, num_samples_max=-1,
                 pde_name="2D_Mantle_Convection"):
        if pde_name not in ("2D_Mantle_convection", "2D_Mantle_Convection"):
            raise ValueError(
                "the release FNO reader supports only 2D mantle convection; "
                "got {!r}".format(pde_name)
            )
        self.initial_step = int(initial_step)
        self.t_train = int(t_train)
        path = os.path.join(saved_folder, filename)
        with h5py.File(path, "r") as f:
            missing = [k for k in FIELDS + ("x-coordinate", "y-coordinate")
                       if k not in f]
            if missing:
                raise KeyError("missing HDF5 datasets: {}".format(missing))
            arrays = []
            for key in FIELDS:
                arr = np.asarray(f[key], dtype=np.float32)
                arr = arr[::reduced_batch, ::reduced_resolution_t,
                          ::reduced_resolution, ::reduced_resolution]
                # (sample, time, x, y) -> (sample, x, y, time)
                arrays.append(np.transpose(arr, (0, 2, 3, 1)))
            data = np.stack(arrays, axis=-1)  # (sample, x, y, time, 6)
            x = torch.as_tensor(np.asarray(f["x-coordinate"], dtype=np.float32))
            y = torch.as_tensor(np.asarray(f["y-coordinate"], dtype=np.float32))

        if num_samples_max is not None and num_samples_max > 0:
            data = data[:min(num_samples_max, data.shape[0])]
        n = data.shape[0]
        val_n = int(n * test_ratio)
        data = data[:val_n] if if_test else data[val_n:]
        self.data = torch.from_numpy(np.ascontiguousarray(data))
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.grid = torch.stack((X, Y), dim=-1)[::reduced_resolution,
                                                ::reduced_resolution]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (sample[..., :self.initial_step, :],
                sample[..., :self.t_train, :],
                self.grid)


class FNODatasetMult(Dataset):
    """Placeholder for the multi-file reader.

    The released GeoSim configurations use ``single_file: True``; other
    layouts are intentionally not part of this release.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GeoSim FNO release supports single_file: True only"
        )

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(idx)


def setup_seed(seed):
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_physical_loss(model_output, x, grid, args):
    """Weak mass-conservation term used by the original FNO training loop.

    It is retained for API compatibility; the released GeoSim FNO
    configurations set ``if_physical_loss: False``.
    """
    if args.get("pde_name") in ("2D_Mantle_convection", "2D_Mantle_Convection"):
        pred_mass = model_output.squeeze(-1)[..., 0].sum(dim=[1, 2, 3])
        input_mass = x.squeeze(-1)[..., 0].sum(dim=[1, 2, 3])
        return (pred_mass - input_mass).pow(2).mean()
    return torch.zeros((), device=model_output.device)
