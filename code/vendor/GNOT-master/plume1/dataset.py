"""
Plume1 dataset for GNOT: creates DGL graphs from plume1 HDF5 data.
Matches original GNOT MIODataset format: (g, u_p, inputs) where:
  - g: DGL graph with ndata['x'] (trunk), ndata['y'] (target), no edges
  - u_p: global parameter tensor (step index)
  - inputs: MultipleTensors of branch input functions

Graphs are created on-the-fly in __getitem__ to avoid memory duplication.
"""
import os
import sys
import numpy as np
import torch
import h5py
import dgl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MultipleTensors


class Plume1Dataset:
    """Dataset that creates DGL graphs from plume1 HDF5 for GNOT training.

    Each item is a (g, u_p, inputs) tuple matching the original GNOT format.
    Graph has N grid nodes with no edges (GNOT uses linear attention, not GNN).

    Trunk (g.ndata['x']): spatial coordinates [x, y] per node
    Target (g.ndata['y']): next tw timesteps of all 6 fields, flattened per node
    u_p: time step index as [1] tensor
    inputs (branch): MultipleTensors of [input tw timesteps * 6 fields]
    """

    def __init__(self,
                 file_name: str,
                 saved_folder: str,
                 time_window: int = 10,
                 total_time_steps: int = 21,
                 if_test: bool = False,
                 test_ratio: float = 0.1):
        super().__init__()
        self.tw = time_window
        self.nt = total_time_steps

        file_path = os.path.join(saved_folder, file_name)
        with h5py.File(file_path, 'r') as f:
            x = torch.from_numpy(f["x-coordinate"][:])
            y = torch.from_numpy(f["y-coordinate"][:])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            self.coordinates = torch.stack([X.ravel(), Y.ravel()], dim=-1)  # [nx*ny, 2]

            field_keys = ["ro", "pr", "tk", "nu", "vx", "vy"]
            for i, key in enumerate(field_keys):
                _data = np.array(f[key], dtype=np.float32)
                if i == 0:
                    data_shape = list(_data.shape)
                    data_shape.append(6)
                    self.data = np.empty(data_shape, dtype=np.float32)
                self.data[..., i] = _data

        self.data = torch.tensor(self.data)  # [num_samples, nt_all, nx, ny, 6]
        self.nx = self.data.shape[2]
        self.ny = self.data.shape[3]
        self.n_nodes = self.nx * self.ny
        self.n_vars = 6

        # Train/val split
        num_samples = self.data.shape[0]
        test_idx = int(num_samples * (1 - test_ratio))
        if if_test:
            self.data = self.data[test_idx:num_samples]
        else:
            self.data = self.data[:test_idx]

        # Build flat index over (sample, step) pairs
        # Valid start steps: tw to nt-tw (only step=10 for tw=10, nt=21)
        self.valid_steps = list(range(self.tw, self.nt - self.tw + 1))
        self._indices = [(s, step) for s in range(self.data.shape[0])
                         for step in self.valid_steps]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        s, step = self._indices[idx]

        g = dgl.DGLGraph()
        g.add_nodes(self.n_nodes)

        # Trunk: spatial coordinates [n_nodes, 2]
        g.ndata['x'] = self.coordinates.float()

        # Target: next tw timesteps → [n_nodes, tw * 6]
        target = self.data[s, step:step + self.tw]  # [tw, nx, ny, 6]
        target = target.permute(1, 2, 0, 3).reshape(self.n_nodes, -1)  # [n_nodes, tw*6]
        g.ndata['y'] = target.float()

        # u_p: step index
        u_p = torch.tensor([step], dtype=torch.float32)  # [1]

        # Branch inputs: input tw timesteps → MultipleTensors([n_nodes, tw*6])
        input_data = self.data[s, step - self.tw:step]  # [tw, nx, ny, 6]
        input_data = input_data.permute(1, 2, 0, 3).reshape(self.n_nodes, -1)  # [n_nodes, tw*6]
        inputs = MultipleTensors([input_data.float()])

        return g, u_p, inputs
