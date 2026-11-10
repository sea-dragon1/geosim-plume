#!/usr/bin/env python3
# coding=utf-8
"""长程滚动引擎（E03 核心）。

**为什么必须自写**：上游训练/评估里那个自回归循环

    for step in range(2*tw, nt - tw + 1, tw)          # tw=10, nt=21 → range(20,12,10)

在 first21 数据下是**空迭代**（``KNOWN_ISSUES.md`` K6）—— 一次都不会滚动。
本项目保留原处不动（改了会变更 E01/E02 已完成的评估口径），长程在这里另写。

两种模式（``MATH.md`` 的递推）::

    自治 auto     :  Ŵ_k = F_θ(W_{k-1}, t_k)    W_k = Ŵ_k          只用预测回填
    teacher-forced:  Ŵ_k = F_θ(W_k^true, t_k)  每次输入都取自真值

⚠ **禁止**在自治模式下把任何真实未来状态注入 ``W_k``（k ≥ 1）。本模块里
「真值」只出现在两个地方：① ``k=0`` 的初始窗口；② 评分用的 labels。
``labels`` 会传给 ``update_graph_data``，但那只是**写进 graph.y 供取用** ——
``MFGNO.forward`` 只读 ``data.x / x_pos / t_pos / variables / edge_index /
batch / steps``，**不读 graph.y**，所以自治性不会被污染（已核对模型 forward）。

模型家族的差异
--------------
GNN 系：一次前向直接出 ``[N, tw, C]``（10 帧）；
外部模型（FNO / Transolver）：每次前向只出 **1 帧**，出 10 帧要 10 次前向。
所以 ``RolloutResult.n_forward`` 与 ``n_window`` 在两类模型上差 10 倍 ——
两者都要报（``protocol/metrics.md`` 要求）。

⚠ GNOT 不在 E03 的评估范围内（dgl CPU-only，长程太慢），但引擎留了分支。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E03_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E03_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
for _p in (COMMON, HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from dataclasses import dataclass, field     # noqa: E402

import numpy as np                           # noqa: E402
import torch                                 # noqa: E402

import eval_metrics as em                    # noqa: E402
import model_zoo as mz                       # noqa: E402
from data_full import TW, FullData, n_windows, window_slice   # noqa: E402


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """一次完整 rollout 的产物。

    张量布局统一为 ``[n_traj, K*tw, H, W, C]``，与 ``eval_metrics`` 的
    ``[n_traj, n_frame, X, Y, C]`` 约定一致，可以直接喂给
    ``relative_l2_per_frame`` / ``rmse_per_frame``。
    """

    pred: np.ndarray            # (n, K*tw, H, W, C) float32
    truth: np.ndarray           # 同上
    t_out: np.ndarray           # (n, K*tw) 每个**预测帧**的物理时间（Myr）
    traj_ids: list              # 每条轨迹在源文件里的索引
    n_window: int               # 窗口推进次数 K
    n_forward: int              # 真实网络前向次数
    family: str
    mode: str
    nonfinite: np.ndarray       # (n, K*tw) 每帧的非有限值个数
    pred_has_nonfinite: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_frame(self) -> int:
        return int(self.pred.shape[1])

    @property
    def n_traj(self) -> int:
        return int(self.pred.shape[0])

    def summary(self) -> str:
        return ("{} K={} 帧={} 前向={} 轨迹={} 非有限帧={}/{}").format(
            self.family, self.n_window, self.n_frame, self.n_forward,
            self.n_traj, int((self.nonfinite > 0).sum()), self.n_frame * self.n_traj)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _to_nchw(out: torch.Tensor, H: int, W: int, n_var: int) -> np.ndarray:
    """``(H*W, tw, C)`` 或 ``(H*W, C)`` → ``(tw, H, W, C)`` numpy。

    节点展平序是 ``i*W + j``（``meshgrid 'ij'`` 后 ravel），所以直接
    ``reshape(H, W, ...)`` 就对了 —— 不要用 ``.T`` 或转置，那会引入
    训练管线里没有的坐标约定。
    """
    arr = out.detach().float().cpu()
    if arr.dim() == 2:                      # (HW, C) → 单帧
        arr = arr.unsqueeze(1)
    tw = arr.shape[1]
    return arr.reshape(H, W, tw, n_var).permute(2, 0, 1, 3).numpy()


def _assert_xpos_intact(graph, coords: torch.Tensor, where: str) -> None:
    """守护 ``x_pos`` 不被就地修改。

    ``vendor/model.py:182`` 有 ``x_pos[:, d] = x_pos[:, d] / span`` —— **就地**
    改 ``data.x_pos``，而 ``update_graph_data`` 复用静态图的 ``x_pos``。
    本项目 ``spatial_domain=[(0,1),(0,1)]`` ⇒ span=1.0 ⇒ 除法不改变值，
    所以当前无害；但一旦有人把 spatial_domain 改成别的区间，长程第 2 次前向
    起坐标就会被反复相除、**静默**给出错误结果。

    这里不做修复（不改上游），只做**检测**：不等就抛，把静默错误变成显式错误。
    """
    if graph.x_pos is None:
        return
    if graph.x_pos.shape[0] != coords.shape[0]:
        return                              # batch>1 时是 repeat 过的，跳过
    # ⚠ 必须**同时**对齐 device 与 dtype：``fd.coords`` 是 CPU 张量，而图建在
    #   ``device`` 上。只写 ``.to(dtype)`` 会在 GPU 上抛
    #   "Expected all tensors to be on the same device"（本项目已实测踩过）。
    ref = coords.to(device=graph.x_pos.device, dtype=graph.x_pos.dtype)
    if not torch.allclose(graph.x_pos, ref, rtol=0, atol=1e-6):
        raise RuntimeError(
            "graph.x_pos 已被就地修改（{}）—— spatial_domain 的 span 不是 1.0，"
            "长程滚动会静默出错。需要在本模块里每次前向前重置 x_pos。".format(where))


# --------------------------------------------------------------------------
# GNN 家族
# --------------------------------------------------------------------------

@torch.no_grad()
def rollout_gnn(loaded, fd: FullData, mode: str = "auto", k_max: int = None,
                verbose: bool = True, traj_max: int = None) -> RolloutResult:
    """GNN 系（MFGNO / MPNN）的长程滚动。

    图**只建一次**（``build_static_graph`` 含 ``radius_graph``，是主要开销），
    后续每步走 ``update_graph_data`` 增量更新 —— 与 vendor 训练时的
    unrolling 路径逐行同构。

    逐轨迹处理（batch=1）：``create_data`` / ``update_graph_data`` 都是按
    batch 整体操作的，逐轨迹走能保证每条轨迹的窗口切片独立、时间轴不串味。
    """
    if loaded.family not in mz.GNN_FAMILIES:
        raise ValueError("rollout_gnn 只处理 GNN 家族，收到 {}".format(loaded.family))
    if mode not in ("auto", "teacher"):
        raise ValueError("mode 必须是 auto 或 teacher，收到 {!r}".format(mode))

    gc = loaded.ctx["graph_creator"]
    model = loaded.model
    device = loaded.device
    # ⚠ 必须过 mz.variables_as_batch —— 配置里 variables 是 0 维标量，直接传会让
    #   build_static_graph 的 torch.stack(var_list, dim=1) 抛 IndexError。
    variables = mz.variables_as_batch(loaded.args)

    tw = int(gc.tw)
    n_var = int(loaded.args["num_outputs"])
    H, W = fd.shape_hw
    u_flat = fd.flat()                      # (n, T, HW, C)
    k_max = int(k_max if k_max is not None else n_windows(fd.n_frame))
    if k_max > n_windows(fd.n_frame):
        raise ValueError("k_max={} 超出该数据能支持的 {} 个窗口".format(
            k_max, n_windows(fd.n_frame)))

    preds, truths, times, nonfinite, traj_ids = [], [], [], [], []

    n_use = fd.n_traj if traj_max is None else min(int(traj_max), fd.n_traj)
    for i in range(n_use):
        u_i = u_flat[i:i + 1]               # (1, T, HW, C)
        t_i = fd.t[i:i + 1]                 # (1, T)
        coords = fd.coords                  # (HW, 2)

        # ── k=0：用真值初始窗口建图 ──────────────────────────────────
        steps = [tw]
        data, labels = gc.create_data(u_i, steps)
        graph, _ = gc.build_static_graph(
            coords, data, labels, t_i, steps, 1, variables, device)

        p_list, y_list, t_list, nf_list = [], [], [], []

        for k in range(k_max):
            _assert_xpos_intact(graph, coords, "k={} traj={}".format(k, fd.rows[i]))

            out = gnn_window_forward(loaded, graph, t_i, n_var)

            y_true = graph.y                                          # (HW, tw, C)
            p_list.append(_to_nchw(out, H, W, n_var))
            y_list.append(_to_nchw(y_true, H, W, n_var))

            # 该窗口每个预测帧的物理时间：真值时间轴是外生给定量，不是"状态"
            _, out_sl = window_slice(k, tw)
            t_list.append(t_i[0, out_sl].numpy().copy())

            o = out.detach().float()
            nf_list.append((~torch.isfinite(o)).sum(dim=(0, 2)).cpu().numpy())  # (tw,)

            if k + 1 >= k_max:
                break
            # ── 推进到下一步 ────────────────────────────────────────
            steps_next = [tw * (k + 2)]
            data_next, labels_next = gc.create_data(u_i, steps_next)
            if mode == "auto":
                # Predictions already have node-major layout [HW, tw, C].
                graph = gc.update_graph_data(
                    graph, out, labels_next, steps_next, t_i, device)
            else:
                # Ground truth has batch-major layout [1, tw, HW, C].
                # This API converts both inputs and labels to node-major layout
                # and updates timestamps/steps while retaining the static edges.
                graph = gc.update_graph_input(
                    graph, data_next, labels_next, steps_next, t_i, device)

        preds.append(np.concatenate(p_list, axis=0))
        truths.append(np.concatenate(y_list, axis=0))
        times.append(np.concatenate(t_list, axis=0))
        nonfinite.append(np.concatenate(nf_list, axis=0))
        traj_ids.append(fd.rows[i])

        if verbose:
            # ⚠ 这里**不**算 L2RE —— ε_c 还没拟合，拿占位 ε 打出来的数字会误导。
            # 只报一个不含 ε 的粗相对误差，纯作进度指示；真正的指标在评分阶段算。
            d = preds[-1][-1] - truths[-1][-1]
            r = np.linalg.norm(d) / max(np.linalg.norm(truths[-1][-1]), 1e-12)
            print("      traj {:>3}: 末窗口粗相对误差 {:.3e}".format(fd.rows[i], r))

    pred = np.stack(preds)                   # (n, K*tw, H, W, C)
    truth = np.stack(truths)
    res = RolloutResult(
        pred=pred, truth=truth, t_out=np.stack(times), traj_ids=traj_ids,
        n_window=k_max, n_forward=mz.n_forward_of(loaded.family, k_max),
        family=loaded.family, mode=mode, nonfinite=np.stack(nonfinite),
        pred_has_nonfinite=bool(np.any(np.stack(nonfinite) > 0)) if nonfinite else False,
        meta={"exp_id": loaded.exp_id, "ckpt_epoch": loaded.ckpt_epoch,
              "tw": tw, "n_var": n_var, "subset": fd.subset, "split": fd.split},
    )
    return res



def gnn_window_forward(loaded, graph, t, n_var):
    """Repeat the trained 21-frame window operator without changing weights.

    Legacy compute_cumulative_dt uses base=step+tw-1 and clamps its future
    indices. Limit timestamps to the current 2*tw+1 window's right boundary
    to retain that trained clamp. Absolute t_pos and time origin are unchanged.
    This is checkpoint compatibility, NOT corrected physical Euler integration.
    """
    tw = int(loaded.ctx["graph_creator"].tw)
    right = max(int(step) for step in graph.steps) + tw + 1
    if right > t.shape[1]:
        raise ValueError("Legacy checkpoint requires one timestamp beyond the output window")
    t_window = t[:, :right]
    if loaded.family == "gnn_joint":
        return loaded.model(graph, t_window, 1)
    return _forward_perfield(loaded.model, graph, t_window, n_var)


def _forward_perfield(model, graph, t, n_var):
    """A 组：6 个独立 MPNN，每场一次前向（故 n_forward = 6 × n_window）。"""
    out = torch.empty_like(graph.y)
    for i in range(n_var):
        out[..., i] = model[i](graph, t, i, 1)
    return out


# --------------------------------------------------------------------------
# 外部家族（FNO / Transolver）
# --------------------------------------------------------------------------

@torch.no_grad()
def rollout_external(loaded, k_max: int = None, mode: str = "auto",
                     verbose: bool = True, traj_max: int = None) -> RolloutResult:
    """FNO / Transolver 的长程滚动。

    这两个模型**每次前向只出 1 帧**，所以「一个 10 帧窗口」= 10 次前向、
    窗口边界是滑动缓冲的自然产物。这与 ``PLAN.md`` 说的"每次模型调用输出
    10 帧"在**帧数**上等价、在**调用次数**上差 10 倍 —— 报告时必须两个数都给。

    ⚠ 数据来自 ``loaded.ctx["data"]``（各自 dataset 已按全时程读入），
    **不是** ``FullData`` —— 因为两个 dataset 的布局不同
    （FNO 是 ``(N,H,W,T,C)``，Transolver 是 ``(n,T,H,W,C)``）。
    """
    if loaded.family not in ("fno", "transolver"):
        raise ValueError("rollout_external 只处理 fno/transolver，收到 {}".format(loaded.family))
    if mode not in ("auto", "teacher"):
        raise ValueError("mode 必须是 auto 或 teacher，收到 {!r}".format(mode))

    tw = TW
    model, device = loaded.model, loaded.device
    data = loaded.ctx["data"]

    # ⚠ ``grid`` 两个家族都要，且都在 ``ctx`` 里（model_zoo: FNO ctx 与
    #   Transolver ctx 各自带 "grid"）。**取在分支外**：原先写在 fno 分支里、
    #   给 transolver 置 None，导致下方 ``grid.unsqueeze(0)`` 抛 AttributeError。
    #   ``_external_forward_time``（能跑通的那条路）也是两个家族都取 ctx["grid"]。
    grid = loaded.ctx["grid"]
    if loaded.family == "fno":
        n, H, W, T, C = data.shape
    else:
        n, T, H, W, C = data.shape

    k_max = int(k_max if k_max is not None else n_windows(T))
    n_out = k_max * tw                        # 要预测的总帧数

    preds, truths, times, nonfinite, traj_ids = [], [], [], [], []

    n_use = n if traj_max is None else min(int(traj_max), n)
    for i in range(n_use):
        if loaded.family == "fno":
            y = data[i].unsqueeze(0).to(device)          # (1,H,W,T,C)
            x = y[..., :tw, :].clone()
            outs = []
            for q in range(tw, tw + n_out):
                out = model(x.reshape(1, H, W, -1), grid)      # (1,H,W,1,C)
                outs.append(out)
                if mode == "auto" or (q + 1) % tw != 0:
                    x = torch.cat((x[..., 1:, :], out), dim=-2)
                else:
                    x = y[..., q - tw + 1:q + 1, :]
            p = torch.cat(outs, dim=-2)[0]                    # (H,W,n_out,C)
            p = p.permute(2, 0, 1, 3)
            true = y[0, ..., tw:tw + n_out, :].permute(2, 0, 1, 3)
        else:
            y = data[i].to(device)                            # (T,H,W,C)
            N = H * W
            cur = y[:tw].permute(1, 2, 0, 3).reshape(N, -1)   # (N, tw*C)
            g = grid.unsqueeze(0)
            cur = cur.unsqueeze(0)                            # (1,N,tw*C)
            outs = []
            for q in range(tw, tw + n_out):
                out = model(g, cur)                           # (1,N,C)
                outs.append(out)
                if mode == "auto" or (q + 1) % tw != 0:
                    cur = torch.cat([cur[:, :, C:], out], dim=-1)
                else:
                    cur = y[q - tw + 1:q + 1].permute(1, 2, 0, 3).reshape(
                        1, N, -1)
            p = torch.stack(outs, dim=2)[0]                   # (N,n_out,C)
            p = p.reshape(H, W, n_out, C).permute(2, 0, 1, 3)
            true = y[tw:tw + n_out]                           # (n_out,H,W,C)

        preds.append(p.detach().float().cpu().numpy())
        truths.append(true.detach().float().cpu().numpy())
        times.append(fd_time_of(loaded, i)[tw:tw + n_out])
        nonfinite.append((~torch.isfinite(p)).sum(dim=(1, 2, 3)).cpu().numpy())
        traj_ids.append(i)

        if verbose:
            print("      traj {:>3}: 完成 {} 帧".format(i, n_out))

    return RolloutResult(
        pred=np.stack(preds), truth=np.stack(truths), t_out=np.stack(times),
        traj_ids=traj_ids, n_window=k_max,
        n_forward=mz.n_forward_of(loaded.family, k_max),
        family=loaded.family, mode=mode, nonfinite=np.stack(nonfinite),
        pred_has_nonfinite=bool(np.any(np.stack(nonfinite) > 0)),
        meta={"exp_id": loaded.exp_id, "ckpt_epoch": loaded.ckpt_epoch,
              "tw": tw, "n_var": C},
    )


_TIME_CACHE: dict = {}


def _t_axis_of(loaded) -> np.ndarray:
    """该 run 所属子集的**测试集**时间轴 ``(n_traj, T)``，单位 Myr。

    按子集缓存 —— 否则 ``rollout_external`` 的每条轨迹都会重读一次
    ``t-coordinate``（plume2 的 25 条就是 25 次 hdf5 打开）。

    ⚠ FNO / Transolver 的 dataset 里没有统一的 ``t`` 访问口，所以这里按子集
    重推一遍 —— 必须与 ``data_full.read_split`` 用**同一个** ``t_divisor``
    （plume2 是 1e6），不要另写一份换算。
    """
    import h5py
    from data_full import spec_of
    if "time_axis" in loaded.ctx:
        return loaded.ctx["time_axis"]
    subset = mz.subset_of_exp_id(loaded.exp_id)
    if subset not in _TIME_CACHE:
        s = spec_of(subset)
        path = os.path.join(s["data_dir"], s["test_file"])
        with h5py.File(path, "r") as f:
            _TIME_CACHE[subset] = (
                np.asarray(f["t-coordinate"][:], dtype=np.float64) / float(s["t_divisor"]))
    return _TIME_CACHE[subset]


def fd_time_of(loaded, traj: int) -> np.ndarray:
    """单条轨迹的时间轴（Myr）。"""
    return _t_axis_of(loaded)[traj]


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------

def rollout_one(loaded, fd: FullData = None, mode: str = "auto",
                k_max: int = None, verbose: bool = True) -> RolloutResult:
    """按家族分派。

    GNN 家族需要 ``fd``（``FullData``）；外部家族不需要（数据在 ``loaded.ctx``）。
    """
    if loaded.family in mz.GNN_FAMILIES:
        if fd is None:
            raise ValueError("GNN 家族需要传 fd（FullData）")
        return rollout_gnn(loaded, fd, mode=mode, k_max=k_max, verbose=verbose)
    if loaded.family == "gnot":
        if fd is None:
            raise ValueError("GNOT requires FullData")
        return rollout_gnot(loaded, fd, mode=mode, k_max=k_max, verbose=verbose)
    return rollout_external(loaded, k_max=k_max, mode=mode, verbose=verbose)

@torch.no_grad()
def rollout_gnot(loaded, fd: FullData, mode="auto", k_max=None,
                 verbose=True, traj_max=None):
    """GNOT: one forward per ten-frame window; raw frame index conditioning."""
    import dgl
    if loaded.family != "gnot" or mode not in ("auto", "teacher"):
        raise ValueError("Expected GNOT and auto/teacher mode")
    tw = TW
    k_max = n_windows(fd.n_frame) if k_max is None else int(k_max)
    if not 1 <= k_max <= n_windows(fd.n_frame):
        raise ValueError("Invalid rollout horizon")
    H, W = fd.shape_hw
    C = fd.u.shape[-1]
    n_use = fd.n_traj if traj_max is None else min(int(traj_max), fd.n_traj)
    wrap = loaded.ctx["MultipleTensors"]
    preds, truths, times, nonfinite = [], [], [], []
    for i in range(n_use):
        # DGL is CPU-only; the model transfers coordinates internally to GPU.
        # No target field is present on the inference graph.
        graph = dgl.graph(([], []), num_nodes=H * W)
        graph.ndata["x"] = fd.coords.float().cpu()
        cur = fd.u[i, :tw].to(loaded.device)
        parts = []
        for k in range(k_max):
            step = tw * (k + 1)
            if mode == "teacher":
                cur = fd.u[i, step-tw:step].to(loaded.device)
            branch = cur.permute(1, 2, 0, 3).reshape(1, H * W, tw * C)
            u_p = torch.tensor([step], dtype=torch.float32, device=loaded.device)
            output = loaded.model(graph, u_p, wrap([branch]))
            if tuple(output.shape) != (H * W, tw * C):
                raise ValueError("Unexpected GNOT output shape: "+str(output.shape))
            cur = output.reshape(H, W, tw, C).permute(2, 0, 1, 3)
            parts.append(cur.detach().float().cpu().numpy())
        prediction = np.concatenate(parts)
        preds.append(prediction)
        truths.append(fd.u[i, tw:tw*(k_max+1)].numpy())
        times.append(fd.t[i, tw:tw*(k_max+1)].numpy())
        nonfinite.append((~np.isfinite(prediction)).sum(axis=(1,2,3)))
        if verbose:
            print("GNOT trajectory {}: {} windows / {} frames".format(fd.rows[i],k_max,k_max*tw),flush=True)
    nf = np.stack(nonfinite)
    return RolloutResult(
        pred=np.stack(preds), truth=np.stack(truths), t_out=np.stack(times),
        traj_ids=fd.rows[:n_use], n_window=k_max, n_forward=k_max,
        family="gnot", mode=mode, nonfinite=nf,
        pred_has_nonfinite=bool(np.any(nf > 0)),
        meta={"exp_id":loaded.exp_id,"ckpt_epoch":loaded.ckpt_epoch,
              "tw":tw,"n_var":C,"subset":fd.subset,"split":fd.split,
              "time_input":"raw frame index; no new normalization"})
