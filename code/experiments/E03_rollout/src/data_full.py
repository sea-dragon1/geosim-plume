#!/usr/bin/env python3
# coding=utf-8
"""全时程数据接入（E03 专用）。

E01/E02 用的是 ``*_first21.hdf5``（前 21 帧）—— 那是为了让上游那条
``assert nt - tw - tw*unrolled > tw`` 成立而截出来的训练口径（见
``data/plume1d_corrected/make_first20_data.py``）。**21 帧只能支撑 1 个窗口**，
长程滚动必须改用全时程文件（131 帧 → 12 个窗口）。

**为什么不复用 ``MPNNDatasetSingle``**：它把 ``total_time_steps`` 作用到
``self.t[idx, :total_time_steps]`` 上 —— 传 21 会把时间轴截到 21 列，而
``compute_cumulative_dt`` 里有 ``clamp(max=t.shape[1]-1)``，于是**后续所有窗口的
dt 都会被 clamp 成同一个值，模型退化成"输出=输入复制"而且不报错**；
传 ``-1`` 也等价于砍掉最后一帧。所以这里自己按轨迹分块读，
顺带把「97 条训练轨迹 3.7 GB 全量读入」降成「只读需要的 rows」。

用法::

    fd = read_split("plume1d", "test")            # 12 条 × 131 帧
    fd = read_split("plume1d", "train", rows=range(87, 97))   # 只要验证集那 10 条
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
E03_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E03_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
for _p in (COMMON,):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import numpy as np      # noqa: E402
import torch            # noqa: E402
import h5py             # noqa: E402

import eval_metrics as em   # noqa: E402
import paths                # noqa: E402

# --------------------------------------------------------------------------
# 子集规格
# --------------------------------------------------------------------------

DATA_ROOT = paths.DATA   # GEOSIM_DATA 可覆盖；默认 geosim-plume/data/

SUBSET_SPEC = {
    "plume1d": dict(
        data_dir=os.path.join(DATA_ROOT, "plume1d_corrected") + "/",
        train_file="plume1d_train_ds_128x128_phys1e6.hdf5",
        test_file="plume1d_test_ds_128x128_phys1e6.hdf5",
        n_traj_train=97, n_frame_train=97,
        n_traj_test=12, n_frame_test=131,
        # t 已经是 Myr（phys1e6 标尺），不需要换算
        t_divisor=1.0,
        # 训练口径（first21）覆盖到的物理时间上限 —— 越过它才叫 temporal extrapolation
        t_train=17.2433,
        # 验证集 = 训练池里切出的后 10%，[start, stop)
        #   plume1d: int(97*0.9)=87 → 索引 87..96
        val_rows=(87, 97),
    ),
    "plume2": dict(
        data_dir=os.path.join(DATA_ROOT, "plume2") + "/",
        # ⚠ 用修正版（保守重映射 + 正确坐标）；旧的 *_norm.hdf5 已从工作区移除
        train_file="plume2d_train_ds_128x128_norm_corrected.hdf5",
        test_file="plume2d_test_ds_128x128_norm_corrected.hdf5",
        n_traj_train=103, n_frame_train=131,
        n_traj_test=25, n_frame_test=131,
        # ⚠ 修正版的 t **已经是 Myr**（实测 t.max = 67.991），不再需要换算。
        #   旧版 _norm 的 t 是年、才要 ÷1e6 —— 那份已被取代。
        t_divisor=1.0,
        t_train=12.9913,
        #   plume2: int(103*0.9)=92 → 索引 92..102
        val_rows=(92, 103),
    ),
}

# 说明：plume1d 的 train 与 test 帧数**不同**（97 vs 131）—— 前者是训练池、
# 后者是独立模拟且跑得更久。验证集因此只有 8 个窗口，比测试集（12）短。
# 这一点会传导到 τ_c 的可用时域，REPORT 里必须写明。

SPLITS = ("train", "test")
TW = 10          # time_window，与所有训练配置一致


def spec_of(subset: str) -> dict:
    if subset not in SUBSET_SPEC:
        raise KeyError("未知子集 {!r}，可选: {}".format(subset, sorted(SUBSET_SPEC)))
    return SUBSET_SPEC[subset]


def n_traj_of(subset: str, split: str) -> int:
    return int(spec_of(subset)["n_traj_{}".format(split)])


def n_frame_of(subset: str, split: str) -> int:
    return int(spec_of(subset)["n_frame_{}".format(split)])


def data_path(subset: str, split: str) -> str:
    s = spec_of(subset)
    return os.path.join(s["data_dir"], s["{}_file".format(split)])


def val_rows_of(subset: str) -> range:
    lo, hi = spec_of(subset)["val_rows"]
    return range(int(lo), int(hi))


def train_rows_of(subset: str) -> range:
    """训练行 = 除验证集以外的全部（阈值里的 [lo_c, hi_c] 只能用它，避免泄漏）。"""
    lo = spec_of(subset)["val_rows"][0]
    return range(0, int(lo))


# --------------------------------------------------------------------------
# 窗口算术
# --------------------------------------------------------------------------

def n_windows(n_frame: int, tw: int = TW) -> int:
    """给定帧数最多能做多少次窗口推进。

    初始窗口吃帧 ``[0, tw)``；第 k 次调用（k 从 0 数）的输入是
    ``[tw*k, tw*(k+1))``、输出是 ``[tw*(k+1), tw*(k+2))``。
    所以需要 ``tw*(k+2) <= n_frame``，即 ``k <= (n_frame - 2*tw)/tw``。
    """
    return (int(n_frame) - 2 * tw) // tw + 1


def window_slice(k: int, tw: int = TW):
    """第 k 次窗口推进的 ``(输入切片, 输出切片)``（沿帧维）。"""
    if k < 0:
        raise ValueError("k 必须 >= 0，收到 {}".format(k))
    return slice(tw * k, tw * (k + 1)), slice(tw * (k + 1), tw * (k + 2))


def first_predicted_frame(tw: int = TW) -> int:
    """第一个**被预测**的帧号。

    ⚠ 这条不是废话：E03 的固定端点 {2,5,8,12} Myr 若按**绝对时刻**读，
    plume1d 的 2/5/8 Myr 全部落在真值输入窗（帧 0–9）内，根本没有对应的预测
    → ``n_valid = 0``。所以端点主口径改用 lead（自窗口最后输入帧起算的时效）。
    """
    return tw


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class FullData:
    """一个 split 的全时程数据。

    ``u`` 是 ``(n, T, H, W, C)``，通道顺序固定为 ``eval_metrics.FIELD_ORDER``
    （ro, pr, tk, nu, vx, vy）—— 与训练配置里的读取顺序一致，
    **不是** hdf5 里按字母序存储的顺序。
    """

    u: torch.Tensor          # (n, T, H, W, C) float32
    t: torch.Tensor          # (n, T) float64，单位 Myr
    coords: torch.Tensor     # (H*W, 2)，与训练管线同序（meshgrid 'ij' + ravel）
    x: torch.Tensor          # (H,)
    y: torch.Tensor          # (W,)
    subset: str
    split: str
    rows: list
    meta: dict = field(default_factory=dict)

    @property
    def n_traj(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_frame(self) -> int:
        return int(self.u.shape[1])

    @property
    def shape_hw(self):
        return int(self.u.shape[2]), int(self.u.shape[3])

    def flat(self) -> torch.Tensor:
        """``(n, T, H*W, C)`` —— GNN 的 ``create_data`` 需要的布局。

        展平顺序与 ``vendor/utils_tubor.py`` 的
        ``flatten(data, start_dim=1, end_dim=-2)`` 一致：节点扁平索引
        ``= i*ny + j``，对应坐标 ``(x[i], y[j])``。
        """
        n, T, H, W, C = self.u.shape
        return self.u.reshape(n, T, H * W, C)

    def subset_rows(self, rows: Iterable[int]) -> "FullData":
        idx = torch.as_tensor(list(rows), dtype=torch.long)
        return FullData(
            u=self.u[idx], t=self.t[idx], coords=self.coords, x=self.x, y=self.y,
            subset=self.subset, split=self.split, rows=[self.rows[i] for i in idx.tolist()],
            meta=dict(self.meta, parent_rows=self.rows),
        )

    def window_time(self, k: int, traj: int = 0) -> dict:
        """第 k 次窗口推进的时间戳（物理时刻，Myr）。"""
        inp, out = window_slice(k)
        t_row = self.t[traj]
        return {
            "t_in_first": float(t_row[inp.start]),
            "t_in_last": float(t_row[inp.stop - 1]),
            "t_out_first": float(t_row[out.start]),
            "t_out_last": float(t_row[out.stop - 1]),
        }


# --------------------------------------------------------------------------
# 读盘
# --------------------------------------------------------------------------

def _read_coords(f) -> tuple:
    # ⚠ **必须是 float32**，与训练管线一致。数据文件里的 x/y 就是 float32，
    #   而 `MPNNDatasetSingle` 直接 `torch.from_numpy(f["x-coordinate"])` 不做提升。
    #   这里若转成 float64，`x_pos` 会带着 float64 进模型的
    #   `torch.cat([u_flat(f32), x_pos(f64), t_pos(f32), variables(f32)])` ——
    #   cat 会**类型提升成 float64**，随后撞上 float32 的 Linear 权重，
    #   报 "mat1 and mat2 must have the same dtype, but got Double and Float"。
    #   实测于 E04 的成本测量（mfgno/mpnn 两个家族都因此失败）。
    x = np.asarray(f["x-coordinate"][:], dtype=np.float32)
    y = np.asarray(f["y-coordinate"][:], dtype=np.float32)
    # ⚠ 与四条训练管线一致：torch.meshgrid 默认 'ij'（FNO 虽没显式写，但 torch 默认就是 ij）
    X, Y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y), indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    return x, y, coords


def read_split(subset: str, split: str, rows: Optional[Iterable[int]] = None,
               verbose: bool = True) -> FullData:
    """读一个 split 的全时程数据，可选只读部分轨迹。

    Args:
        rows: 要读的轨迹索引。``None`` 表示全部。
            **验证集与训练集极值统计都应该传 rows**，否则会把 19 GB 级别的
            数据全读一遍（训练文件更大）。

    ⚠ ``t`` 会除以 ``spec["t_divisor"]``（plume2 是 1e6）—— 这是全项目里
    **唯一**做这个换算的地方。换出来的单位是 Myr，与 plume1d 对齐。
    """
    if split not in SPLITS:
        raise ValueError("split 必须是 {} 之一，收到 {!r}".format(SPLITS, split))
    s = spec_of(subset)
    path = data_path(subset, split)
    if not os.path.exists(path):
        raise FileNotFoundError("数据文件不存在: {}".format(path))

    n_total = n_traj_of(subset, split)
    rows = list(range(n_total)) if rows is None else sorted(set(int(r) for r in rows))
    bad = [r for r in rows if r < 0 or r >= n_total]
    if bad:
        raise IndexError("轨迹索引越界（{}/{}）: {}".format(subset, n_total, bad[:5]))

    fields = list(em.FIELD_ORDER)
    with h5py.File(path, "r") as f:
        n_frame = f[fields[0]].shape[1]
        expect = n_frame_of(subset, split)
        if n_frame != expect:
            raise ValueError(
                "{} {} 的帧数是 {}，与 SUBSET_SPEC 声明的 {} 不符 —— "
                "规格表过时了，请先更新它".format(subset, split, n_frame, expect))

        x, y, coords = _read_coords(f)
        n, H, W = len(rows), f[fields[0]].shape[2], f[fields[0]].shape[3]
        u = np.empty((n, n_frame, H, W, len(fields)), dtype=np.float32)
        # 逐轨迹分块读：h5py 允许按第一维切片，比整文件读省得多
        for j, r in enumerate(rows):
            for c, name in enumerate(fields):
                u[j, ..., c] = f[name][r]           # (T, H, W)
        t_raw = np.asarray(f["t-coordinate"][:], dtype=np.float64)[rows]   # (n, T)

    t = t_raw / float(s["t_divisor"])

    # ── 体检：时间轴必须逐轨迹严格递增、首帧为 0、量级合理 ──────────────
    if not np.all(np.diff(t, axis=1) > 0):
        raise ValueError("{} {} 的时间轴非严格递增".format(subset, split))
    if not np.allclose(t[:, 0], 0.0, atol=1e-9):
        raise ValueError("{} {} 的首帧时间不为 0".format(subset, split))
    if t.max() > 200.0:
        raise ValueError(
            "{} {} 换算后 t.max()={:.4g}，超出 Myr 的合理量级 —— "
            "t_divisor 写错了吗？".format(subset, split, t.max()))

    fd = FullData(
        u=torch.from_numpy(u), t=torch.from_numpy(t),
        coords=coords, x=torch.from_numpy(x), y=torch.from_numpy(y),
        subset=subset, split=split, rows=rows,
        meta={
            "path": path, "n_frame": n_frame,
            "t_divisor": s["t_divisor"], "t_train": s["t_train"],
            "n_windows": n_windows(n_frame),
            "fields": fields,
        },
    )
    if verbose:
        print("    {} {}: {} 条 × {} 帧 → {} 个窗口，t ∈ [{:.3f}, {:.3f}] Myr".format(
            subset, split, fd.n_traj, fd.n_frame, fd.meta["n_windows"],
            float(fd.t.min()), float(fd.t.max())))
    return fd


def load_initial_window(fd: FullData, k: int = 0, tw: int = TW) -> torch.Tensor:
    """第 k 次推进的**输入窗口**（真值），形状 ``(n, tw, H*W, C)``。

    自治 rollout 只在 ``k=0`` 用它；``k>=1`` 的输入必须来自上一步的预测。
    """
    inp, _ = window_slice(k, tw)
    flat = fd.flat()[:, inp, :, :]          # (n, tw, HW, C)
    return flat.contiguous()


def load_labels(fd: FullData, k: int, tw: int = TW) -> torch.Tensor:
    """第 k 次推进的**真值标签**，形状 ``(n, tw, H*W, C)``。

    只用于评分；**绝不能**把它当作 ``k>=1`` 的输入（那是 teacher forcing，
    属 teacher 模式，不是自治）。
    """
    _, out = window_slice(k, tw)
    return fd.flat()[:, out, :, :].contiguous()


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def _selftest():
    """``python -B data_full.py`` —— 不读全量，只验算术与一个子集的 test split。"""
    print("窗口算术:")
    for nf in (21, 97, 131):
        print("  n_frame={:<4} → {} 个窗口".format(nf, n_windows(nf)))
    # 21 帧恰好能做 **1** 个窗口（k=0：输入帧 0-9 → 输出帧 10-19）——
    # 这正是 E01/E02 的单窗口口径。KNOWN_ISSUES K6 说的是上游那个
    # `range(2*tw, nt-tw+1, tw)` 循环为空（即**没有第二次**推进），两者不矛盾：
    # 初始那一次照常发生，滚动那一次不发生。
    assert n_windows(21) == 1, n_windows(21)
    # 全时程 131 帧 → 12 个窗口（k=0..11，最后一个用帧 110-129）
    assert n_windows(131) == 12, n_windows(131)
    # 训练池只有 97 帧 → 8 个窗口，比测试集少 4 个（验证集的可用时域更短）
    assert n_windows(97) == 8, n_windows(97)

    print("\n窗口切片:")
    for k in range(3):
        inp, out = window_slice(k)
        print("  k={}: 输入帧 [{}, {})  输出帧 [{}, {})".format(
            k, inp.start, inp.stop, out.start, out.stop))

    print("\n读一个真实 split（只 2 条轨迹）:")
    fd = read_split("plume1d", "test", rows=[0, 1])
    print("  u={} t={} coords={}".format(tuple(fd.u.shape), tuple(fd.t.shape),
                                         tuple(fd.coords.shape)))
    print("  第 0 次窗口时间:", fd.window_time(0))
    print("  坐标序检查: coords[0]={} coords[1]={} coords[-1]={}".format(
        fd.coords[0].tolist(), fd.coords[1].tolist(), fd.coords[-1].tolist()))
    print("\ndata_full 自检通过")


if __name__ == "__main__":
    _selftest()
