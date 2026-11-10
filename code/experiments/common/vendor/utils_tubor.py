import h5py
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_cluster import radius_graph
from typing import Tuple
import random

class PDE(object):
    def __init__(self, 
                 name: str,
                 variables: dict,
                 temporal_domain: Tuple,
                 resolution_t: int,
                 spatial_domain: list,
                 resolution: list,
                 reduced_resolution_t: int=1,
                 reduced_resolution: int=1):
        super().__init__()
        self.name = name
        self.tmin = temporal_domain[0]
        self.tmax = temporal_domain[1]
        self.resolution_t = resolution_t // reduced_resolution_t
        self.spatial_domain = spatial_domain
        # self.resolution = [res // reduced_resolution for res in resolution] # 在整除不了时，和resolution数据降采样结果对不齐
        self.resolution = [(res-1) // reduced_resolution +1 for res in resolution]
        self.spatial_dim = len(spatial_domain)
        self.variables = variables

    def __repr__(self):
        return self.name

class MPNNDatasetSingle(Dataset):
    """2D 地幔对流单文件数据集（发布包唯一使用的数据口径）。

    文件键：ro / pr / tk / nu / vx / vy，形状 (n_traj, T, H, W)；
    坐标 x/y 为一维网格，``t-coordinate`` 是逐轨迹的真实时间轴 [N, T]。
    """
    def __init__(self,
                 file_name: str,
                 saved_folder: str,
                 reduced_resolution: int = 1,
                 reduced_resolution_t: int = 1,
                 reduced_batch: int = 1,
                 if_test: bool = False,
                 test_ratio: float = 0.1,
                 variables: dict = {},
                 pde_name: str = '2D_Mantle_Convection',
                 total_time_steps: int = -1) -> None:

        super().__init__()
        self.pde_name = pde_name
        self.total_time_steps = total_time_steps
        file_path = os.path.abspath(saved_folder + file_name)

        with h5py.File(file_path, 'r') as f:
            assert pde_name == '2D_Mantle_Convection', pde_name
            assert len(f["ro"].shape) - 2 == 2, f["ro"].shape
            x = torch.from_numpy(f["x-coordinate"][::reduced_resolution])
            y = torch.from_numpy(f["y-coordinate"][::reduced_resolution])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            self.coordinates = torch.stack([X.ravel(), Y.ravel()], dim=-1)
            self.t = torch.from_numpy(f["t-coordinate"][:])
            self.data = None
            for i, key in enumerate(["ro", "pr", "tk", "nu", "vx", "vy"]):
                _data = np.array(f[key], dtype=np.float32)
                _data = _data[::reduced_batch, ::reduced_resolution_t,
                              ::reduced_resolution, ::reduced_resolution]
                if self.data is None:
                    self.data = np.empty(list(_data.shape) + [6], dtype=np.float32)
                self.data[..., i] = _data

        self.variables = variables

        # 训练/验证按轨迹切分。NOTE: self.t 必须与 self.data 一起切，
        # 否则 val_data[i] 会用样本 i 的时间轴去配样本 (87+i) 的场。
        test_idx = int(self.data.shape[0] * (1 - test_ratio))
        if if_test:
            self.data = self.data[test_idx:]
            self.t = self.t[test_idx:]
        else:
            self.data = self.data[:test_idx]
            self.t = self.t[:test_idx]

        self.data = torch.tensor(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return (
            torch.flatten(self.data[idx, :self.total_time_steps, ...], start_dim=1, end_dim=-2),
            self.coordinates,
            self.t[idx, :self.total_time_steps],
            self.variables,
        )


class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 25) -> None:
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.nt = pde.resolution_t
        # 静态图缓存（第一次创建后复用）
        self.static_graph = None
        self.radius = None

    def _compute_radius(self):
        """计算半径（基于 2D 网格分辨率）"""
        x_min, x_max = self.pde.spatial_domain[0]
        res = self.pde.resolution[0]
        dx = (x_max - x_min) / res
        return self.n * dx * np.sqrt(2) + dx / 10

    # 原有的 create_data 方法保持不变（仅用于切片数据）
    def create_data(self, datapoints: torch.Tensor, steps: list, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 使用预分配 + 列表收集（避免cat）
        data_list = []
        labels_list = []
        
        for dp, step in zip(datapoints, steps):

            data_list.append(dp[step - self.tw:step])
            labels_list.append(dp[step:step + self.tw])

        data = torch.stack(data_list)    # [bs, tw, nx, v]
        labels = torch.stack(labels_list) # [bs, tw, nx, v]
        
        if device is not None:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
        return data, labels
    
    def build_static_graph(self, x: torch.Tensor,  data:  torch.Tensor, labels:  torch.Tensor, t: torch.Tensor, steps: list, batch_size: int, variables: dict, device: torch.device):
        """
        构建静态图部分（边索引、节点坐标、批次索引），并移至 device。
        只需调用一次，后续复用（动态字段用 update_graph_input 每 batch 更新）。
        x: 空间坐标，形状 [nx, d]（所有样本相同）
        batch_size: 批次大小（通常为1）
        device: 计算设备
        """
        nx = x.shape[0]
        # 节点坐标：复制 batch_size 份 -> [bs*nx, d]
        x_pos = x.repeat(batch_size, 1).to(device)
        # 批次索引：每个节点所属的样本编号
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(nx)
        # 根据实际坐标间距计算半径（支持不同分辨率评估）
        edge_index = None
        if self.radius is None:
            x_sorted = torch.sort(torch.unique(x[:, 0]))[0]
            if len(x_sorted) > 1:
                actual_dx = (x_sorted[1] - x_sorted[0]).item()
                self.radius = self.n * actual_dx * np.sqrt(2) + actual_dx / 10
            else:
                self.radius = self._compute_radius()
        edge_index = radius_graph(x_pos, r=self.radius, batch=batch_idx, loop=False)
        # 缓存静态部分
        v = data.shape[-1]
        x_data = data.permute(0, 2, 1, 3).reshape(-1, self.tw, v).to(device, non_blocking=True)
        y_data = labels.permute(0, 2, 1, 3).reshape(-1, self.tw, v).to(device, non_blocking=True)
        steps_tensor = torch.tensor(steps, device=device, dtype=torch.long)
        t_vals = t.to(device, dtype=torch.float32).gather(1, steps_tensor.unsqueeze(1)).squeeze(1)

        self.static_graph = Data(x=x_data, edge_index=edge_index)  # x 后续动态更新
        self.static_graph.y = y_data
        self.static_graph.x_pos = x_pos
        self.static_graph.t_pos = t_vals.repeat_interleave(nx)
        self.static_graph.batch = batch_idx
        # 预分配节点特征和目标值的存储（形状固定：[bs*nx, tw, v]）
        # 具体形状在第一次更新数据时确定，因为 v 未知

        var_list = []
        for k in variables:
            val = variables[k]
            if isinstance(val, torch.Tensor):
                var_list.append(val.float().to(device))
            else:
                var_list.append(torch.tensor(val, dtype=torch.float32, device=device))
        var_tensor = torch.stack(var_list, dim=1)  # [bs, num_vars]
        self.static_graph.variables = var_tensor.repeat_interleave(nx, dim=0)
        # 关键：模型 forward 用 data.steps 算真实 per-sample dt（非均匀时间轴）。
        # update_graph_input/update_graph_data 都设置了 graph.steps，唯独 build_static_graph 漏了 →
        # 任何直接 build 的 eval 路径（evaluate.py）会 missing-steps（旧模型）或 uniform-dt 回退（phase01 模型），
        # 导致 vx/vy 误差放大 ~5-6 倍。与 update_* 保持一致。
        self.static_graph.steps = steps

        return self.static_graph, batch_size

    def update_graph_input(self, graph: Data,
                           data: torch.Tensor,     # [bs, tw, nx, v] (任意设备)
                           labels: torch.Tensor,   # [bs, tw, nx, v] (任意设备)
                           steps: list,
                           t: torch.Tensor,        # [bs, nt] (任意设备)
                           device: torch.device):
        """每个 batch 更新动态字段（输入 x、标签 y、时间 t_pos），复用静态图（x_pos/edge_index/batch/variables）。

        与 build_static_graph 配对：静态图只建一次，本方法每 batch 轻量更新动态数据。
        （variables 为数据集固定值，所有样本共享，无需逐 batch 更新。）
        """
        tw = self.tw
        v = data.shape[-1]
        nx = data.shape[2]  # 每样本节点数

        # 输入/标签 reshape 成 [bs*nx, tw, v] 并移至 device
        x_data = data.permute(0, 2, 1, 3).reshape(-1, tw, v).to(device, non_blocking=True)
        y_data = labels.permute(0, 2, 1, 3).reshape(-1, tw, v).to(device, non_blocking=True)
        # 时间位置（与 build_static_graph 一致）
        steps_tensor = torch.tensor(steps, device=device, dtype=torch.long)
        t_vals = t.to(device, dtype=torch.float32).gather(1, steps_tensor.unsqueeze(1)).squeeze(1)

        graph.x = x_data
        graph.y = y_data
        graph.t_pos = t_vals.repeat_interleave(nx)
        graph.steps = steps
        return graph

    def update_graph_data(self, graph: Data,
                        pred: torch.Tensor,      # [bs*nx, tw, v] (任意设备)
                        labels: torch.Tensor,    # [bs, tw, nx, v] (任意设备)
                        steps: list,
                        t: torch.Tensor,         # [bs, nt] (任意设备)
                        device: torch.device):
        batch_size = len(steps)
        nx = pred.shape[0] // batch_size
        tw = self.tw
        v = pred.shape[-1]

        # 关键修复：确保所有张量在同一设备
        pred = pred.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        # 预测值直接赋值
        graph.x = pred

        # 标签处理（已移到GPU）
        y_data = labels.permute(0, 2, 1, 3).reshape(-1, tw, v).to(device, non_blocking=True)
        graph.y = y_data
        t = t.to(device, non_blocking=True)  # ← 加上这一行，修复 gather 设备错误
        # 时间位置（优化：避免Python循环和.item()）
        steps_tensor = torch.tensor(steps, device=device, dtype=torch.long)
        # 如果 t 在CPU，先收集再移；如果在GPU，直接操作
        t_vals = t.gather(1, steps_tensor.unsqueeze(1)).squeeze(1).to(device, dtype=torch.float32)
        graph.t_pos = t_vals.repeat_interleave(nx)

        graph.steps = steps
        return graph

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def to_PDEBench_format(graph_data: torch.Tensor, batch_size: int, pde: PDE, spatial_resolution=None):
    """Convert graph data to PDEBench formart [bs, x1, ..., xd, t, v]
    Args:
        graph_data (torch.Tensor): input/output data of model with shape [bs*nx, t, v]
        batch_size (int): batch size
        pde (PDE): PDE to solve
        spatial_resolution (list, optional): override spatial dims, e.g. [H, W]
    """
    assert (len(graph_data.shape) == 3)
    output_shape = [batch_size]
    res = spatial_resolution if spatial_resolution is not None else pde.resolution
    for v in res:
        output_shape.append(v)
    output_shape.append(graph_data.shape[-2])
    output_shape.append(graph_data.shape[-1])

    return graph_data.reshape(output_shape)
