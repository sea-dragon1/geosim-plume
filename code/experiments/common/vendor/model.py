# coding=utf-8
import torch
from torch import nn
from torch_geometric.data import Data
from utils import PDE
from model_utils import *


def compute_cumulative_dt(data, t, time_window, pde, device):
    """[本项目] 真实时间轴下的累积时间增量 dt，形状 [bs, tw]。

    物理含义：解码是 ``out = u_last + dt * diff``，其中 ``u_last`` 是**输入窗口
    最后一帧**在时刻 ``t_last`` 的状态。因此 dt 应取相对 t_last 的时间差：

        dt[b, q] = t[b, base+q] - t[b, base]
        base     = steps[b] + time_window - 1          # 输入窗口最后一帧的索引
        q        = 1 .. time_window

    各样本用各自的真实时间轴，所以 dt 是 **per-sample** 的 ``[bs, tw]``，
    而不是上游那种 ``[1, tw]`` 的均匀因子。

    为什么必须这么做：原始模拟的时间间隔**非均匀**（实测起动段 0.27~0.75、
    主体 1.0、尾段 0.5 Myr）。用 ``(tmax-tmin)/resolution_t`` 的 cumsum 会
    系统性偏离真实积分步长 —— 对 20 帧数据约偏 60%。

    回退：``t`` 或 ``data.steps`` 缺失时退回均匀因子，并返回
    ``uniform_fallback=True`` 让调用方记录，**不静默**。
    """
    steps = getattr(data, "steps", None)
    if t is None or steps is None:
        dt0 = (pde.tmax - pde.tmin) / pde.resolution_t
        dt = torch.ones(1, time_window, device=device) * dt0
        return torch.cumsum(dt, dim=1), True

    steps_t = torch.as_tensor(steps, device=t.device, dtype=torch.long).reshape(-1)
    base = steps_t + (time_window - 1)                          # [bs]
    q = torch.arange(1, time_window + 1, device=t.device)       # [tw]
    idx = (base.unsqueeze(1) + q.unsqueeze(0)).clamp(max=t.shape[1] - 1)
    dt_abs = torch.gather(t, 1, idx).to(device=device, dtype=torch.float32)
    t_last = torch.gather(t, 1, base.unsqueeze(1)).to(device=device, dtype=torch.float32)
    return dt_abs - t_last, False


def expand_dt_to_nodes(dt, total_nodes, batch_index=None):
    """把 per-sample 的 dt 扩展到 per-node，与 ``[bs*nx, tw]`` 的解码输出对齐。

    均匀回退时 dt 形状为 ``[1, tw]``，直接广播；真实 dt 为 ``[bs, tw]``，
    按 ``data.batch``（每个节点所属的样本索引）取值。
    """
    if dt.shape[0] == 1:
        return dt.expand(total_nodes, -1)
    if batch_index is None:
        raise ValueError("真实 dt 需要 data.batch 才能扩展到节点维")
    return dt[batch_index]


class MPNN(nn.Module):
    def __init__(self,
                 pde: PDE,
                 time_window: int = 10,
                 hidden_features: int = 128,
                 hidden_layers: int = 6,
                 eq_variables: dict = {}):
        """Initialize MPNN
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE to solve
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MPNN, self).__init__()
        # arguments
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.time_window = time_window
        self.eq_variables = eq_variables

        # encoder
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window+self.pde.spatial_dim+1+len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )

        # processor
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            spatial_dim=self.pde.spatial_dim,
            n_variables=len(self.eq_variables)+1 # (time is treated as equation variable)
            ) for _ in range(self.hidden_layers)))
        
        # decoder —— 发布包只保留论文口径 time_window=10。
        # Conv1d 的 kernel/stride 必须让输出长度恰好等于 tw：
        #   hidden 200（A 组配置）: (200-18)//5+1=37 → 37-28+1=10
        #   hidden 128（tests/test_dt.py）: (128-18)//5+1=23 → 23-14+1=10
        if self.time_window != 10:
            raise ValueError(
                "MPNN 只支持 time_window=10（发布包按论文实验裁剪），收到 {}".format(
                    self.time_window))
        kernel2 = 28 if self.hidden_features == 200 else 14
        self.output_mlp = nn.Sequential(
            nn.Conv1d(1, 8, 18, stride=5),
            Swish(),
            nn.Conv1d(8, 1, kernel2, stride=1)
            )

    def forward(self, data: Data, t: torch.Tensor=None, v: int=0, bs=1) -> torch.Tensor:
        """Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window, v].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
            t (Tensor): [bs, nt] 真实时间轴（phys1e6，单位 Myr）。[本项目] 新增，
                用于按真实时间间隔做残差积分；缺省 None 时退回均匀因子。
            v (int): physics variable to solve
        Returns:
            torch.Tensor: predictive solutions with shape [batch*n_nodes, time_window]
        """
        u = data.x[..., v] # [bs*nx, tw]
        x_pos = data.x_pos # [bs*nx, spatial_dim]
        t_pos = data.t_pos # [bs*nx]
        edge_index = data.edge_index # [2, num_edges]
        batch = data.batch # [bs*nx]
        variables = data.variables # [bs*nx, num_variables]

        # normalize temporal and spatial coordinate
        # [本项目] 不做时间归一化：t 以数据自带标尺（phys1e6，单位 Myr）原值
        # 进入编码器与消息条件。原实现除以 pde.tmax 会把它压到 [0,1]，
        # 与"时间以真实物理尺度参与"的口径不符。
        # t_pos = t_pos / self.pde.tmax
        for d in range(self.pde.spatial_dim):
            x_pos[:, d] = x_pos[:, d] / (self.pde.spatial_domain[d][1] - self.pde.spatial_domain[d][0])

        # encode
        node_input = torch.cat((u, x_pos, t_pos.unsqueeze(1), variables), dim=-1)
        f = self.embedding_mlp(node_input) # [bs*nx, hidden_dim]

        # process
        for i in range(self.hidden_layers):
            # f = self.gnn_layers[i](f, u, x_pos, variables, edge_index, batch)
            f = self.gnn_layers[i](f, u, x_pos, torch.cat((t_pos.unsqueeze(1), variables), dim=-1), edge_index, batch)

        # decode
        # [本项目] 真实时间轴 dt（per-sample [bs, tw] -> 节点维），见 compute_cumulative_dt
        dt, _uniform = compute_cumulative_dt(data, t, self.time_window, self.pde, f.device)
        dt = expand_dt_to_nodes(dt, u.shape[0], getattr(data, "batch", None))
        # [bs*nx, hidden_dim] -> [bs*nx, 1, hidden_dim]) -> conv1d -> [bs*nx, 1, tw] -> [bs*nx, tw] (diff)
        diff = self.output_mlp(f[:, None]).squeeze(1)
        # [bs*nx] -> [tw, bs*nx] -> [bs*nx, tw]
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff
        # print(out.shape)
        return out


class MFGNO(nn.Module):
    """MFGNO：显式建模多物理场交互"""
    def __init__(self,
                 pde: PDE,
                 time_window: int = 10,
                 hidden_features: int = 128,
                 hidden_layers: int = 6,
                 num_physics_vars: int = 6,  # ro, pr, tk, nu, vx, vy
                 eq_variables: dict = {},
                 zero_var_diff: bool = False):
        super().__init__()
        self.pde = pde
        self.time_window = time_window
        self.num_physics_vars = num_physics_vars
        self.hidden_features = hidden_features
        # [E02 消融] False = C（完整 MFGNO），True = B（显式差分槽置零）。
        # 两者结构、宽度、层数与参数量完全一致，仅差分槽取值不同。
        self.zero_var_diff = zero_var_diff
        
        # ===== 1. 多变量联合编码器（关键改进）=====
        # 输入: [u_all (tw*num_physics_vars), x_pos, t_pos, eq_vars] 
        encoder_input_dim = time_window * num_physics_vars + \
                           pde.spatial_dim + 1 + len(eq_variables) # 10*6+1+1+3 = 66

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_features * 2),
            Swish(),
            nn.Linear(hidden_features * 2, hidden_features),
            Swish()
        )
        
        # ===== 2. 物理感知GNN层（核心改进）=====
        self.gnn_layers = nn.ModuleList([
            PhysicsAwareGNNLayer(
                in_features=hidden_features,
                hidden_features=hidden_features,
                out_features=hidden_features,
                spatial_dim=pde.spatial_dim,
                time_window=time_window,
                num_physics_vars=num_physics_vars,  # 传递变量数
                n_eq_vars=len(eq_variables) + 1
            ) for _ in range(hidden_layers)
        ])
        
        # ===== 3. 多变量联合解码器 =====
        # 输出: [bs*nx, time_window, num_physics_vars] 的导数预测
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            Swish(),
            nn.Linear(hidden_features, time_window * num_physics_vars)
        )
        
        # [E02 消融] B 变体：把每层的显式多场差分槽置零。
        # 只挡取值、不删权重列，故参数量与 C 严格相同。
        if zero_var_diff:
            for _layer in self.gnn_layers:
                _layer.zero_var_diff = True
        
        # 物理约束权重（可学习）
        self.physics_weight = nn.Parameter(torch.ones(num_physics_vars) * 0.1)

    def forward(self, data: Data, t: torch.Tensor, bs: int = 1) -> torch.Tensor:
        # === 输入准备 ===

        u_all = data.x    # data.x: [bs*nx, time_window, num_physics_vars] (所有物理场历史) [31155, 10, 6]
        # print("[DEBUG] u_all shape:", u_all.shape)
        x_pos = data.x_pos # x_pos shape: [bs*nx, spatial_num] [31155, 2]
        t_pos = data.t_pos
        variables = data.variables
        
        # 归一化（同原代码）
        # [本项目] 不做时间归一化：t 以数据自带标尺（phys1e6，单位 Myr）原值
        # 进入编码器与消息条件。原实现除以 pde.tmax 会把它压到 [0,1]，
        # 与"时间以真实物理尺度参与"的口径不符。
        # t_pos = t_pos / self.pde.tmax
        for d in range(self.pde.spatial_dim):
            x_pos[:, d] = x_pos[:, d] / (self.pde.spatial_domain[d][1] - self.pde.spatial_domain[d][0])
        
        # === 1. 多变量联合编码 ===
        u_flat = u_all.reshape(u_all.shape[0], -1)  # [bs*nx, tw*num_physics_vars]
        # print(f"u_flat shape: {u_flat.shape}, x_pos shape:{x_pos.shape}, t_pos shape: {t_pos.shape}, variable shape: {variables.shape}")
        node_input = torch.cat([u_flat, x_pos, t_pos.unsqueeze(1), variables], dim=-1) # [bs*nx, tw*num_physics_vars + 1 + spatial_num + virables_num] [31155,66]
        h = self.encoder(node_input)  # h [bs*nx, hidden_features] [31155, 128]
        # === 2. 物理感知GNN处理（含变量交互）===
        edge_index = data.edge_index # [2,E] [2, 123908]
        batch = data.batch # [bs*nx] [31155]
        time_vars = torch.cat([t_pos.unsqueeze(1), variables], dim=-1) # time_vars shape: [bs*nx, 1+virables_num] [31155, 4]
        for gnn in self.gnn_layers:
            h = gnn(h, u_flat, x_pos, time_vars, edge_index, batch) 
            # h.shape  [bs*nx, hidden_features] [31155, 128]

        # === 3. 多变量联合解码 ===
        diff_flat = self.decoder(h)  # [bs*nx, tw*num_physics_vars]
        diff = diff_flat.reshape(u_all.shape[0], self.time_window, self.num_physics_vars)

        # === 4. 物理约束积分（保留Euler积分思想）===
        # [本项目] 真实时间轴 dt（per-sample -> 节点维），见 compute_cumulative_dt。
        # 原实现在 t 非空时用均匀因子，else 分支引用了未定义的 u，从未生效。
        dt, _uniform = compute_cumulative_dt(data, t, self.time_window, self.pde, u_all.device)
        dt = expand_dt_to_nodes(dt, u_all.shape[0], getattr(data, "batch", None))

        
        # 对每个变量应用物理约束权重
        u_last = u_all[:, -1, :]  # [bs*nx, num_physics_vars]
        u_last_exp = u_last.unsqueeze(1).expand(-1, self.time_window, -1)
        out = u_last_exp + (dt.unsqueeze(-1) * diff * self.physics_weight)  # [bs*nx, tw, num_physics_vars]
        
        return out
