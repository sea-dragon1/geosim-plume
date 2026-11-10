#!/usr/bin/env python3
# coding=utf-8
"""E03 / E04 共用的模型装载层。

**为什么单独抽一层**：E03（长程滚动）与 E04（诊断/成本）都要在「不训练」的前提下
把 E01/E02 训好的 checkpoint 装起来做推理，而五个模型的装载方式并不统一：

- GNN 系（``MFGNO`` / ``MPNN``）走 ``common/vendor`` 的训练栈，
  需要 ``PDE`` + ``GraphCreator`` 两个上下文对象；
- FNO / Transolver / GNOT 各在自己的目录里，且**三者不能共存于同一进程**
  （各有 ``utils.py`` / ``train.py``，见 ``predict_external.isolate_paths`` 的说明）。

本模块把差异收敛到一个 ``load_run()`` 后面，调用方只面对 ``Loaded`` 这一个结构。

⚠ **一个进程只能装载一个家族**。原因不只是模块名遮蔽：
``isolate_paths`` 会把 ``REPO`` 从 ``sys.path`` 里摘掉，而 vendor 的 ``train_tubor``
依赖 ``REPO``（要 ``import metrics``，仓库根那个）。所以先装 GNN 再装外部模型会坏。
本模块用 ``_LOADED_FAMILY`` 记录已装载的家族并在跨界时**直接报错**，
而不是让调用方拿到一个诡异的 ImportError。

用法::

    sys.path.insert(0, os.path.join(EXPERIMENTS, "common"))
    import model_zoo as mz

    loaded = mz.load_run(run_dir, device, which="best")
    print(loaded.family, mz.count_params(loaded.model))
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
REPO = paths.VENDOR

# --------------------------------------------------------------------------
# 家族元数据
# --------------------------------------------------------------------------

# model_id（配置里的值） → 家族名
FAMILY_OF = {
    "MFGNO": "gnn_joint",
    "MPNN": "gnn_perfield",
    "fno": "fno",
    "Transolver": "transolver",
    "GNOT": "gnot",
}

# GNN 家族（走 vendor 训练栈，需要 GraphCreator）
GNN_FAMILIES = {"gnn_joint", "gnn_perfield"}
# 外部家族（走各自脚本，需要 isolate_paths）
EXTERNAL_FAMILIES = {"fno", "transolver", "gnot"}

FRAMES_PER_CALL = 10
"""一次「窗口调用」产出的帧数。五个模型都是 10，但**语义不同**（见下）。"""

FORWARDS_PER_CALL = {
    "gnn_joint": 1,      # MFGNO 一次前向直接出 [N, tw, 6]
    "gnn_perfield": 6,   # 6 个独立 MPNN，每场一次前向
    "fno": 10,           # 逐帧自回归，出 10 帧要 10 次前向
    "transolver": 10,    # 同上（out_dim 是单帧通道数）
    "gnot": 1,           # 一次输出 TW*C 后 reshape
}
"""产出 10 帧所需的**真实网络前向次数**。

⚠ 报告「模型调用次数」时必须说清用的是哪个口径：
``n_window``（窗口推进次数，跨模型同义）与 ``n_forward``（真实前向次数）在
FNO / Transolver 上差 10 倍。``protocol/metrics.md`` 明确要求二者都给。
"""

_LOADED_FAMILY: Optional[str] = None


def family_of(model_id: str) -> str:
    if model_id not in FAMILY_OF:
        raise KeyError("未知 model_id {!r}，可选: {}".format(
            model_id, sorted(FAMILY_OF)))
    return FAMILY_OF[model_id]


def n_forward_of(family: str, n_window: int) -> int:
    """``n_window`` 次窗口推进对应多少次真实前向。"""
    return int(n_window) * FORWARDS_PER_CALL[family]


# --------------------------------------------------------------------------
# 装载结果
# --------------------------------------------------------------------------

@dataclass
class Loaded:
    """一个装载好的模型及其全部上下文。

    ``ctx`` 是家族特定的上下文袋：GNN 家族放 ``graph_creator`` / ``pde``，
    外部家族放 ``dataset`` / ``grid`` 等。调用方按 ``family`` 分支取用 ——
    这是刻意的：不同家族的推理接口本就不同，硬套一个统一接口反而会藏坑。
    """

    run_dir: str
    exp_id: str
    model_id: str
    family: str
    subset: Optional[str]
    seed: Optional[int]
    ckpt_epoch: Optional[int]
    model: nn.Module
    device: torch.device
    args: dict = field(default_factory=dict)
    ctx: dict = field(default_factory=dict)

    @property
    def n_forward_per_call(self) -> int:
        return FORWARDS_PER_CALL[self.family]

    def describe(self) -> str:
        return "{:<46} {:<12} {} epoch={}".format(
            self.exp_id, self.family, self.subset, self.ckpt_epoch)


# --------------------------------------------------------------------------
# 公共工具
# --------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """参数总数 ``Σ_p numel(p)``。

    ⚠ **不要**改用 ``vendor/utils_visual.print_model_summary`` —— 它只统计
    ``requires_grad`` 的参数，而 ``E04/MATH.md`` 的定义是不筛选。
    MPNN（A 组）是 ``nn.ModuleList``，6 个子网要全部计入。
    """
    if isinstance(model, nn.ModuleList):
        return int(sum(p.numel() for sub in model for p in sub.parameters()))
    return int(sum(p.numel() for p in model.parameters()))


def subset_of_exp_id(exp_id: str) -> Optional[str]:
    """从 exp_id 里判子集。判不出返回 None（由调用方决定报错还是回退）。"""
    for s in ("plume1d", "plume2"):
        if "-{}-".format(s) in exp_id or exp_id.endswith("-" + s):
            return s
    return None


def seed_of_exp_id(exp_id: str) -> Optional[int]:
    """从 exp_id 末尾的 ``-seed{N}`` 取种子号；没有则返回 None。"""
    tail = str(exp_id).rsplit("-seed", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return int(tail[1])
    return None


def variables_as_batch(args):
    """把配置里的标量 ``variables`` 包成 ``{name: Tensor(1,)}``。

    ⚠ **任何直接调 ``GraphCreator.build_static_graph`` 的地方都必须先用它。**
    原因：``build_static_graph`` 内部做 ``torch.stack(var_list, dim=1)``，而配置里
    ``variables`` 存的是**0 维标量**（如 ``{"m": 0.1, "eta": 0.1, "zeta": 0.1}``），
    0 维张量在 ``dim=1`` 上会抛
    ``IndexError: Dimension out of range (expected to be in range of [-1, 0],
    but got 1)``。训练侧不会遇到，因为 dataset 提供的本来就是 ``(bs,)`` 形状
    （见 ``utils_tubor`` 里 ``variables: {parm1: (bs), parm2: (bs), ...}`` 的约定）。

    ``bs=1`` 时包成 ``(1,)`` 即与训练口径一致（图构建只用到每样本一个值）。

    踩过两次：``E04/cost.py`` 的 ``measure_one`` 与 ``_train_step_probe``、
    ``E03/rollout.py``。**新增调用点请从这里取，不要再抄一份。**
    """
    import torch
    return {k: torch.tensor([float(v)], dtype=torch.float32)
            for k, v in (args.get("variables") or {}).items()}


def find_config(run_dir: str) -> str:
    """run 目录里的配置副本（生成时复制进来的那份）。"""
    import glob
    cands = sorted(glob.glob(os.path.join(run_dir, "*.yaml")))
    if not cands:
        raise FileNotFoundError("run 目录里没有配置副本: {}".format(run_dir))
    return cands[0]


def checkpoint_path(run_dir: str, which: str = "best") -> str:
    """checkpoint 的完整路径。

    ⚠ 命名**不统一**：GNN 系与 GNOT 是 ``<run目录名>-best.pt``，
    但 FNO 是 ``..._FNO-best.pt``、Transolver 是 ``Transolver_lr0.0001_bs4-best.pt``。
    所以这里用 glob 而不是拼名字 —— ``eval_checkpoint.load_checkpoint`` 是拼名字的，
    只对 GNN 系安全。
    """
    import glob
    hits = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*-{}.pt".format(which))))
    if not hits:
        raise FileNotFoundError(
            "找不到 *-{}.pt（{}）".format(which, os.path.join(run_dir, "checkpoints")))
    return hits[0]


def checkpoint_is_live(run_dir: str, which: str = "best", quiet_seconds: int = 900) -> bool:
    """checkpoint 是否**可能仍在被训练写入**。

    判据是文件 mtime 距今小于 ``quiet_seconds``（默认 15 分钟）—— 训练脚本每
    ``save_period`` 个 epoch 存一次（配置里是 20），20 × 93.8 s ≈ 31 分钟一轮。
    取 15 分钟是保守的下界：只要比一个保存周期短，就不会把"正在训练"误判成"已停"。

    ⚠ 这是**启发式**，不是权威判据。真正的门禁是 ``squeue``；本函数只是让脚本
    在忘记检查时不会静默用到半成品 checkpoint。
    """
    p = checkpoint_path(run_dir, which)
    import time
    return (time.time() - os.path.getmtime(p)) < quiet_seconds


# --------------------------------------------------------------------------
# 装载：GNN 家族
# --------------------------------------------------------------------------

def _load_gnn(run_dir, device, which, frames, t_span):
    """GNN 系（MFGNO / MPNN）—— 走 vendor 训练栈。

    ⚠ **覆盖 ``resolution_t`` 必须发生在 ``build_env`` 之前**：
    ``GraphCreator.nt = pde.resolution_t``（``vendor/utils_tubor.py``），
    而它**不是**数据帧数。E01/E02 用 first21（21 帧）所以两者恰好相等，
    长程改用全时程文件后如果不覆盖，``create_data`` 的窗口切片会按 21 帧算，
    直接越界或静默取错帧。同理 ``temporal_domain`` 影响 ``compute_cumulative_dt``
    的均匀回退分支。
    """
    import eval_checkpoint as ec

    args = ec.load_args(run_dir)
    if frames is not None:
        args["resolution_t"] = int(frames)
        args["_resolution_t_overridden"] = True
    if t_span is not None:
        args["temporal_domain"] = "({}, {})".format(t_span[0], t_span[1])
        args["_temporal_domain_overridden"] = True

    model, graph_creator = ec.build_env(args, device)
    epoch, _ = ec.load_checkpoint(model, run_dir, which=which, device=device)
    model = model.to(device).eval()

    return Loaded(
        run_dir=run_dir, exp_id=args["exp_id"], model_id=args["model_id"],
        family=family_of(args["model_id"]), subset=subset_of_exp_id(args["exp_id"]),
        seed=seed_of_exp_id(args["exp_id"]), ckpt_epoch=epoch,
        model=model, device=device, args=args,
        ctx={"graph_creator": graph_creator, "pde": args["pde"]},
    )


# --------------------------------------------------------------------------
# 装载：外部家族（FNO / Transolver / GNOT）
#
# ⚠ 下面三个函数是 ``predict_external.predict_{fno,transolver,gnot}`` 的**装载部分**
#   的副本 —— 那边是"装载 + 固定 10 帧推理"揉在一起的，E03 需要单独拿模型做长程
#   滚动，所以不得不复制。**改动其中一处时必须同步另一处**（已在此登记）。
# --------------------------------------------------------------------------

def _override_dataset(args: dict, ds_file: str = None, ds_dir: str = None) -> dict:
    """把 ``dataset.file_name`` / ``saved_folder`` 换成调用方指定的值。

    为什么需要它：外部基线的配置指向 ``*_first21*.hdf5`` —— 那是**训练口径**
    （21 帧），长程滚动在 21 帧轴上切 ``[10:130]`` 只会得到 11 帧，评分时直接
    形状不匹配。E03 长程必须读**全时程**文件。

    ⚠ 文件名**不能靠字符串裁剪猜**：plume2 的全时程文件叫 ``_norm_corrected``，
    而 first21 叫 ``_phys1e6_first21_corrected``，去掉 ``_first21`` 得到的名字
    **不存在**。所以由调用方从 ``E03/data_full.SUBSET_SPEC`` 传入权威文件名，
    与 GNN 分支读的是同一份。
    """
    if not ds_file and not ds_dir:
        return args
    ds = dict(args.get("dataset") or {})
    if ds_file:
        ds["file_name"] = ds_file
    if ds_dir:
        ds["saved_folder"] = ds_dir
    out = dict(args)
    out["dataset"] = ds
    return out


def _load_fno(run_dir, device, which, frames, ds_file=None, ds_dir=None):
    import predict_external as pe

    fno_dir = os.path.join(REPO, "fno")  # REPO = paths.VENDOR
    pe.isolate_paths(fno_dir)
    sys.path.insert(0, fno_dir)
    from utils import FNODatasetSingle            # noqa: E402
    from model.fno import FNO2d                   # noqa: E402

    args = _override_dataset(pe.load_config(run_dir), ds_file, ds_dir)
    m = args["model"]
    model = FNO2d(num_channels=m["num_channels"], width=m["width"],
                  modes1=m["modes"], modes2=m["modes"],
                  initial_step=args["initial_step"]).to(device)

    # ⚠ ``t_train`` 只影响 ``__getitem__`` 的目标切片；``ds.data`` 是**整份**读入
    #   的（FNODatasetSingle 不按 t_train 截断 data）。长程滚动用的是 ``ds.data``，
    #   所以这里传全时程帧数即可。
    test_file, ds_dir_ = pe._resolve_data(args)
    ds = FNODatasetSingle(filename=test_file, saved_folder=ds_dir_,
                          initial_step=pe.TW,
                          t_train=int(frames) if frames else (pe.TW + pe.T_OUT),
                          reduced_resolution=1, reduced_resolution_t=1, reduced_batch=1,
                          if_test=False, test_ratio=0.0, pde_name=args["pde_name"])
    grid = ds.grid.to(device)
    if grid.dim() == 3:                     # (H, W, 2) → (1, H, W, 2)
        grid = grid.unsqueeze(0)

    epoch = pe.load_checkpoint(model, run_dir, which=which, device=device)
    model = model.to(device).eval()
    # FNO 的 ds.data 是 (N, H, W, T, C)，T 是文件里的全帧数，不受 t_train 截断
    return Loaded(
        run_dir=run_dir, exp_id=os.path.basename(os.path.abspath(run_dir)),
        model_id="fno", family="fno", subset=subset_of_exp_id(os.path.basename(run_dir)),
        seed=seed_of_exp_id(os.path.basename(run_dir)), ckpt_epoch=epoch,
        model=model, device=device, args=args,
        ctx={"dataset": ds, "grid": grid, "data": ds.data},
    )


def _load_transolver(run_dir, device, which, frames, ds_file=None, ds_dir=None):
    import predict_external as pe

    tslv_dir = os.path.join(REPO, "Transolver")
    pe.isolate_paths(tslv_dir)
    sys.path.insert(0, tslv_dir)
    import train as tslv                             # noqa: E402

    args = _override_dataset(pe.load_config(run_dir), ds_file, ds_dir)
    m = args["model"]
    test_file, data_dir = pe._resolve_data(args)

    # ⚠ TransolverDataset.__init__ 会 `self.t_train = min(t_train, n_t_total)` 并据此
    #   分配 self.data（np.zeros([n, t_train, ...])）—— 传小了**会把数据截断**。
    #   长程必须传文件真实帧数。
    t_train = int(frames) if frames else (pe.TW + pe.T_OUT)
    ds = tslv.TransolverDataset(test_file, data_dir, initial_step=pe.TW,
                                t_train=t_train, test_ratio=0.0, if_test=False)
    grid = torch.tensor(ds.grid, dtype=torch.float32).to(device)
    data = torch.tensor(ds.data, dtype=torch.float32).to(device)
    n, T, H, W, C = data.shape

    model = tslv.Transolver2D(
        space_dim=2, n_layers=m["n_layers"], n_hidden=m["n_hidden"],
        dropout=m.get("dropout", 0.0), n_head=m["n_head"],
        Time_Input=m.get("Time_Input", False), act=m.get("act", "gelu"),
        mlp_ratio=m.get("mlp_ratio", 1), fun_dim=C * pe.TW, out_dim=C,
        slice_num=m.get("slice_num", 64), ref=m.get("ref", 8),
        unified_pos=m.get("unified_pos", False), H=H, W=W).to(device)

    epoch = pe.load_checkpoint(model, run_dir, which=which, device=device)
    model = model.to(device).eval()
    return Loaded(
        run_dir=run_dir, exp_id=os.path.basename(os.path.abspath(run_dir)),
        model_id="Transolver", family="transolver",
        subset=subset_of_exp_id(os.path.basename(run_dir)),
        seed=seed_of_exp_id(os.path.basename(run_dir)), ckpt_epoch=epoch,
        model=model, device=device, args=args,
        ctx={"dataset": ds, "grid": grid, "data": data, "n_var": C},
    )


def _load_gnot(run_dir, device, which, frames, ds_file=None, ds_dir=None):
    """GNOT —— 图留 CPU，**模型与张量照常上 GPU**。

    ⚠ 本函数曾经把整个 ``device`` 降级成 CPU，理由是"dgl 2.1.0 是 CPU-only
    编译"。**那个理由是错的**：dgl 只是不能把 **dgl 图**搬到 GPU，而训练脚本
    （``GNOT-master/plume1/train.py``）从来只搬 ``u_p`` / ``inputs``，从不搬 ``g``
    —— 所以 GNOT 的 train 与 evaluate 一直在 GPU 上成功跑。降级的后果不是"保守"，
    而是把成本表里的 GNOT 时延记成了 CPU 墙钟（489 ms 量级），与其余四行的 GPU
    数字不可比，属于凭空造出来的差异。

    正确做法与 ``predict_external.predict_gnot`` 一致：``g`` 不搬，其余搬到
    ``device``。两处是**副本关系**，改一处必须同步另一处（见 ``_load_fno`` 上方的
    登记注释）；2026-09-23 修此处时那边已经是新写法。
    """
    import predict_external as pe

    if device.type != "cpu":
        print("    (dgl 图留 CPU，模型与张量在 {}；与训练脚本一致)".format(device))

    gnot_root = os.path.join(REPO, "GNOT-master")
    gnot_plume = os.path.join(gnot_root, "plume1")
    pe.isolate_paths(gnot_root, gnot_plume)
    sys.path.insert(0, gnot_root)
    sys.path.insert(0, gnot_plume)
    from models.mmgpt import GNOT                    # noqa: E402
    import dataset as gds                            # noqa: E402
    from utils import MultipleTensors                # noqa: E402

    args = _override_dataset(pe.load_config(run_dir), ds_file, ds_dir)
    m = args["model"]

    cls = getattr(gds, "Plume1Dataset", None)
    if cls is None:
        cands = [n for n in dir(gds)
                 if isinstance(getattr(gds, n), type) and "Dataset" in n]
        if not cands:
            raise RuntimeError("在 GNOT dataset.py 里找不到 Dataset 类")
        cls = getattr(gds, cands[0])

    test_file, data_dir = pe._resolve_data(args)
    ds = cls(test_file, data_dir, time_window=pe.TW,
             total_time_steps=int(frames) if frames else pe.TW + pe.T_OUT + 1, if_test=False, test_ratio=0.0)

    model = GNOT(trunk_size=2 + 1, branch_sizes=[pe.TW * 6],
                 space_dim=2, output_size=pe.TW * 6,
                 n_layers=m["n_layers"], n_hidden=m["n_hidden"], n_head=m["n_head"],
                 n_experts=m["n_experts"], n_inner=m["n_inner"], mlp_layers=m["mlp_layers"],
                 attn_type=m["attn_type"], act=m["act"], ffn_dropout=m["ffn_dropout"],
                 attn_dropout=m["attn_dropout"],
                 horiz_fourier_dim=m["horiz_fourier_dim"]).to(device)

    epoch = pe.load_checkpoint(model, run_dir, which=which, device=device)
    model = model.eval()
    return Loaded(
        run_dir=run_dir, exp_id=os.path.basename(os.path.abspath(run_dir)),
        model_id="GNOT", family="gnot",
        subset=subset_of_exp_id(os.path.basename(run_dir)),
        seed=seed_of_exp_id(os.path.basename(run_dir)), ckpt_epoch=epoch,
        model=model, device=device, args=args,
        ctx={"dataset": ds, "MultipleTensors": MultipleTensors, "nx": ds.nx, "ny": ds.ny},
    )


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------

def load_run(run_dir, device, which="best", frames=None, t_span=None,
             ds_file=None, ds_dir=None):
    """装载一个 run 的模型。

    Args:
        run_dir: ``E0*/runs/<exp_id>/``。
        device: 目标设备。GNOT 会被强制改到 CPU。
        which: ``best`` 或 ``latest``。
        frames: **覆盖** ``resolution_t``（全时程帧数）。GNN 家族必须给 ——
            见 ``_load_gnn`` 的说明。``None`` 表示保持配置原值（E01/E02 口径）。
            外部家族用它设 ``t_train``（Transolver 会据此分配 data 数组，
            传小了**会截断数据**；FNO 只影响 ``__getitem__``）。
        t_span: **覆盖** ``temporal_domain``，形如 ``(0.0, 55.24)``。
        ds_file: **覆盖** ``dataset.file_name``（**train** 文件名，
            ``_resolve_data`` 据此推出 test 文件）。外部家族长程滚动必须给 ——
            配置指向 ``*_first21*``（21 帧训练口径），长程会因帧数不足而
            形状不匹配。权威文件名见 ``E03/data_full.SUBSET_SPEC``。
        ds_dir: **覆盖** ``dataset.saved_folder``。与 ``ds_file`` 配套。

    Raises:
        RuntimeError: 同一进程里尝试装载不同家族时。
    """
    global _LOADED_FAMILY

    run_dir = os.path.abspath(run_dir)
    exp_id = os.path.basename(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError("run 目录不存在: {}".format(run_dir))

    # 先探 model_id —— 用 run 里的配置副本，避免为了读一个字段就装载整个模型
    import yaml
    with open(find_config(run_dir), "r", encoding="utf8") as f:
        model_id = yaml.safe_load(f)["model_id"]
    family = family_of(model_id)

    if _LOADED_FAMILY is not None and _LOADED_FAMILY != family:
        raise RuntimeError(
            "同一进程已装载过 {!r} 家族，不能再装 {!r} —— "
            "isolate_paths 会摘掉 REPO 路径，破坏 vendor 的导入；"
            "请一个进程只装一个家族（外部模型之间也要换进程）。".format(
                _LOADED_FAMILY, family))
    _LOADED_FAMILY = family

    if family in GNN_FAMILIES:
        return _load_gnn(run_dir, device, which, frames, t_span)
    if family == "fno":
        return _load_fno(run_dir, device, which, frames, ds_file, ds_dir)
    if family == "transolver":
        return _load_transolver(run_dir, device, which, frames, ds_file, ds_dir)
    if family == "gnot":
        return _load_gnot(run_dir, device, which, frames, ds_file, ds_dir)
    raise NotImplementedError("未实现的家族: {}".format(family))


def reset_family_guard():
    """清掉"已装载家族"的记录。**仅供测试脚本**在同一进程里连续验证多个家族时使用 ——
    正常流程不要调，它会掩盖本该被拦住的跨界装载。"""
    global _LOADED_FAMILY
    _LOADED_FAMILY = None
