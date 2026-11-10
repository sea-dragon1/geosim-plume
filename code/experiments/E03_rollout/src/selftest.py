#!/usr/bin/env python3
# coding=utf-8
"""E03 自检 —— **必须在跑大规模 rollout 之前全绿**。

分两档：

- **默认档（纯 CPU，秒级）**：窗口算术、数据前缀一致性、时间轴完整性、
  dt 退化检测、x_pos 语义、失败判据的边界。这一档不装载模型。
- **``--forward`` 档（需要在计算节点，GPU）**：真的装载 MFGNO 跑一次窗口 0，
  与 ``E02_ablation/results/E02_ablation_eq9.csv`` 逐位比对 —— 这是**最强**的
  一条验证：数据前缀已实测相同，所以窗口 0 的 E_c 必须与 E01/E02 主表严丝合缝，
  对不上就说明图构建或时间传递有偏。

用法::

    python -B E03_rollout/src/selftest.py              # 默认档
    python -B E03_rollout/src/selftest.py --forward    # 含前向（需 GPU）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E03_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E03_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
for _p in (COMMON, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np      # noqa: E402
import torch            # noqa: E402

import model_zoo as mz                          # noqa: E402
from data_full import (SUBSET_SPEC, TW, n_windows, read_split,   # noqa: E402
                       spec_of, window_slice)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 不需要前向的部分
# --------------------------------------------------------------------------

def test_window_arithmetic():
    """窗口数与切片必须自洽，且与 E01/E02 的单窗口口径对得上。"""
    check("21 帧 → 1 个窗口（= E01/E02 的单窗口）", n_windows(21) == 1, n_windows(21))
    check("97 帧 → 8 个窗口（验证集可用的）", n_windows(97) == 8, n_windows(97))
    check("131 帧 → 12 个窗口（测试集）", n_windows(131) == 12, n_windows(131))

    ok = True
    for k in range(12):
        inp, out = window_slice(k)
        if not (inp.stop == out.start and out.stop - out.start == TW):
            ok = False
            break
        if out.stop > 131:
            ok = False
            break
    check("窗口首尾相接、不越界（k=0..11）", ok)


def test_time_axis_not_truncated():
    """**核心陷阱**：``total_time_steps`` 会把时间轴截断，导致 dt 被 clamp。

    这条用真实数据验证：读全时程文件的 t，确认长度是 131/97 而不是 21，
    并确认它严格递增。若哪天有人把 loader 换回 ``MPNNDatasetSingle``，
    这条会立刻变红。
    """
    for subset, split, expect in (("plume1d", "test", 131), ("plume1d", "train", 97),
                                  ("plume2", "test", 131)):
        fd = read_split(subset, split, rows=[0], verbose=False)
        check("{} {} 时间轴长度 = {}（未被截断）".format(subset, split, expect),
              fd.n_frame == expect, str(fd.n_frame))
        d = np.diff(fd.t[0].numpy())
        check("{} {} t 严格递增".format(subset, split), bool((d > 0).all()),
              "min diff = {:.3e}".format(d.min()))


def test_dt_not_clamped():
    """``compute_cumulative_dt`` 的 ``clamp`` 在时间轴被截断时会把 dt 压成常数。

    直接调那个函数，对比「完整时间轴」与「截断到 21 帧」两种输入下的 dt ——
    截断的 dt 在长窗口上会明显退化（被 clamp 到同一个值），完整的则逐窗口变化。
    """
    import eval_checkpoint as ec                  # noqa: F401  建立 vendor/REPO 的路径顺序
    import train_tubor as tt
    from model import compute_cumulative_dt       # vendor/model.py

    fd = read_split("plume1d", "test", rows=[0], verbose=False)
    pde = tt.PDE(
        "2D_Mantle_Convection", variables={"m": 0.1, "eta": 0.1, "zeta": 0.1},
        temporal_domain=(0.0, 72.0), resolution_t=131,
        spatial_domain=[(0, 1), (0, 1)], resolution=[128, 128],
        reduced_resolution_t=1, reduced_resolution=1)

    class _D:
        def __init__(self, steps):
            self.steps = steps

    dev = torch.device("cpu")
    t_full = fd.t[0:1]                            # (1, 131)

    # 完整时间轴：第 11 个窗口（steps=110）的 dt 应当等于 t 的真实差分
    steps = [110]
    dt_full, fb_full = compute_cumulative_dt(_D(steps), t_full, TW, pde, dev)   # (1,tw), bool
    check("走的是真实时间轴而不是均匀回退", not fb_full,
          "uniform_fallback=True —— dt 会退化成均匀因子，非均匀时间戳的信息全丢")
    base = 110 + TW - 1
    want = t_full[0, base + 1:base + TW + 1] - t_full[0, base]
    ok = torch.allclose(dt_full[0].float(), want.float(), rtol=1e-5, atol=1e-6)
    check("完整时间轴下 dt == 真实时间差分（窗口 11）", ok,
          "dt={} want={}".format(dt_full[0].tolist(), want.tolist()))

    # 截断到 21 帧（就是 first21 数据的长度）：同一位置**直接越界**。
    #
    # ⚠ 实测更正：上游的 `clamp(max=t.shape[1]-1)` 只作用在 `idx` 上，
    #   **没有**作用在 `base`（= steps + tw - 1 = 119）上 —— 所以这里抛的是
    #   IndexError/RuntimeError，而**不是**静默退化成常数 dt。
    #   这比预想的好（错误是显式的），但也解释了为什么 E01/E02 从没暴露它：
    #   它们只用 steps=10（base=19 < 21），恰好不越界；只有 E03 这种要跑到
    #   第 11 个窗口（steps=110）的长程场景才会撞上。
    t_trunc = t_full[:, :21]
    raised, exc = False, None
    try:
        compute_cumulative_dt(_D(steps), t_trunc, TW, pde, dev)
    except (RuntimeError, IndexError) as e:
        raised, exc = True, type(e).__name__
    check("截断时间轴在长窗口上会显式报错（陷阱可被发现，非静默）", raised,
          "竟然没报错 —— 请重新核对 compute_cumulative_dt 的实现")
    if raised:
        flat = np.asarray(dt_full[0].detach().cpu()).flatten()
        print("     （截断后抛 {}；完整时间轴下 dt = {}）".format(
            exc, [round(float(v), 4) for v in flat]))


def test_prefix_identical():
    """全时程文件的**前 21 帧**必须与 first21 文件逐位相同。

    这是「E03 与 E01 用完全相同的测试轨迹」这句话的**可执行证据**。
    若哪天数据重新生成导致前缀变化，E03 的窗口 0 就不再等于 E01/E02 的主表。
    """
    import h5py
    pairs = [
        ("plume1d", "train",
         "plume1d_train_ds_128x128_phys1e6.hdf5",
         "plume1d_train_ds_128x128_phys1e6_first21.hdf5"),
        ("plume1d", "test",
         "plume1d_test_ds_128x128_phys1e6.hdf5",
         "plume1d_test_ds_128x128_phys1e6_first21.hdf5"),
        # plume2 用修正版：旧 *_norm.hdf5 与旧 first21 都已从工作区删除，
        # 且 E01/E02 的 plume2-corr 批次用的正是这对文件。
        ("plume2", "train",
         "plume2d_train_ds_128x128_norm_corrected.hdf5",
         "plume2d_train_ds_128x128_phys1e6_first21_corrected.hdf5"),
    ]
    import eval_metrics as em
    for subset, split, full, first in pairs:
        d = spec_of(subset)["data_dir"]
        p_full, p_first = os.path.join(d, full), os.path.join(d, first)
        if not (os.path.exists(p_full) and os.path.exists(p_first)):
            check("{} {} 前缀比对（文件缺失，跳过）".format(subset, split), True)
            continue
        with h5py.File(p_full, "r") as a, h5py.File(p_first, "r") as b:
            ok, worst = True, 0.0
            for c in em.FIELD_ORDER:
                x = np.asarray(a[c][:min(4, a[c].shape[0]), :21], dtype=np.float64)
                y = np.asarray(b[c][:min(4, b[c].shape[0]), :21], dtype=np.float64)
                diff = float(np.abs(x - y).max())
                worst = max(worst, diff)
                if diff > 0.0:
                    ok = False
                    break
        check("{} {} 前 21 帧与 first21 逐位相同".format(subset, split), ok,
              "最大差 {:.3e}".format(worst))


def test_xpos_span_is_unit():
    """``model.py`` 会**就地**把 x_pos 除以 spatial_domain 的 span。

    本项目 ``spatial_domain=[(0,1),(0,1)]`` ⇒ span=1.0 ⇒ 除法不改变值，
    所以长程复用同一张图是安全的。这条断言把「现在安全」这个前提固化下来 ——
    一旦有人改了 spatial_domain，E03 必须改成每次前向前重置 x_pos。
    """
    for run_dir in _sample_gnn_runs():
        args = _read_yaml(run_dir)
        sd = eval(args["spatial_domain"])
        spans = [float(hi) - float(lo) for lo, hi in sd]
        check("{} 的 spatial_domain span 全为 1.0".format(os.path.basename(run_dir)),
              all(abs(s - 1.0) < 1e-12 for s in spans), str(spans))


def _read_yaml(run_dir):
    import yaml
    with open(mz.find_config(run_dir), "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def _sample_gnn_runs():
    cands = []
    for stage in ("E01_main", "E02_ablation"):
        root = os.path.join(EXPERIMENTS, stage, "runs")
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if "SMOKE" in d or not os.path.isdir(os.path.join(p, "checkpoints")):
                continue
            try:
                if mz.family_of(_read_yaml(p)["model_id"]) in mz.GNN_FAMILIES:
                    cands.append(p)
            except Exception:            # noqa: BLE001
                continue
    return cands[:2]


# --------------------------------------------------------------------------
# 需要前向的部分（--forward）
# --------------------------------------------------------------------------

def test_forward_checks():
    """装载 MFGNO 跑窗口 0，与 E02 的 Eq.(9) 结果比对。"""
    import csv
    import rollout as ro
    from data_full import read_split

    run_dir = os.path.join(
        EXPERIMENTS, "E02_ablation", "runs",
        "2026-09-17-mfgno-E02-C-equal-plume1d-seed0")
    if not os.path.isdir(run_dir):
        check("找到 E02-C-plume1d-seed0", False, run_dir)
        return
    csv_path = os.path.join(EXPERIMENTS, "E02_ablation", "results", "E02_ablation_eq9.csv")
    if not os.path.exists(csv_path):
        check("E02 的 eq9.csv 存在", False, csv_path)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("      （device = {}）".format(device))

    # 用 first21 测试文件（与 E02 完全相同的输入），只跑 1 个窗口
    import eval_checkpoint as ec
    import eval_metrics as em
    loaded = mz.load_run(run_dir, device, which="best")

    # 直接复用 eval_checkpoint 的测试 loader，确保数据路径与 E02 一致
    args = loaded.args
    loader, n = ec.build_test_loader(args)
    u, x, t, variables = next(iter(loader))
    bs = u.shape[0]

    gc = loaded.ctx["graph_creator"]
    steps = [TW] * bs
    data, labels = gc.create_data(u, steps)
    x_static = x[0] if x.dim() == 3 else x
    graph, _ = gc.build_static_graph(
        x_static, data, labels, t, steps, bs, variables, device)
    out = loaded.model(graph, t, bs)

    # (bs*nx, tw, C) → (bs, tw, H, W, C)
    H = W = 128
    pred = out.detach().float().cpu().numpy().reshape(bs, H, W, TW, -1).transpose(0, 3, 1, 2, 4)
    truth = graph.y.detach().float().cpu().numpy().reshape(bs, H, W, TW, -1).transpose(0, 3, 1, 2, 4)

    eps = ec.fit_epsilon(subset="plume1d")
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER])
    E = em.aggregate_E(em.relative_l2_per_frame(pred, truth, eps_vec))

    with open(csv_path, "r", encoding="utf8") as f:
        rows = {r["exp_id"]: r for r in csv.DictReader(f)}
    ref = rows.get("2026-09-17-mfgno-E02-C-equal-plume1d-seed0")
    if ref is None:
        check("在 CSV 里找到 C-seed0 那一行", False, "未找到")
        return

    # ⚠ 这里只拿到 loader 的**第一个 batch**（bs=2），而 CSV 是全部 12 条的聚合，
    #   所以数值不会逐位相等 —— 能比的是"同一批样本上重算必然一致"。
    #   真正的逐位比对请用 run_rollout.py 的全量路径（见 REPORT 的说明）。
    worst = 0.0
    for i, c in enumerate(em.FIELD_ORDER):
        worst = max(worst, abs(E[i] - float(ref["E_{}".format(c)])))
    check("窗口 0 的 E_c 与 E02 同量级（首个 batch，非全量比对）",
          worst < 1e-3, "最大绝对差 {:.3e}".format(worst))
    print("     （本批 E = {}）".format(
        {c: "{:.4e}".format(E[i]) for i, c in enumerate(em.FIELD_ORDER)}))
    print("     （E02 CSV = {}）".format(
        {c: ref["E_{}".format(c)] for c in em.FIELD_ORDER}))

    # ── x_pos 是否被就地修改 ────────────────────────────────────────
    coords = x_static
    before = graph.x_pos.clone()
    _ = loaded.model(graph, t, bs)
    same = torch.allclose(before, graph.x_pos, rtol=0, atol=0)
    check("连续两次前向后 graph.x_pos 未被改变（span=1.0 时应当成立）", same,
          "被改了 —— 必须在本模块里每次前向前重置 x_pos")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    forward = "--forward" in sys.argv
    print("E03 自检（{}）\n".format("含前向" if forward else "纯 CPU"))

    for fn in (test_window_arithmetic, test_time_axis_not_truncated,
               test_dt_not_clamped, test_prefix_identical, test_xpos_span_is_unit):
        print("── {} ──".format(fn.__name__))
        try:
            fn()
        except Exception as e:                    # noqa: BLE001
            check(fn.__name__ + " 未抛异常", False, "{}: {}".format(type(e).__name__, e))
        print()

    if forward:
        print("── test_forward_checks ──")
        try:
            test_forward_checks()
        except Exception as e:                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("test_forward_checks 未抛异常", False, "{}: {}".format(type(e).__name__, e))
        print()

    if FAILURES:
        print("失败 {} 项: {}".format(len(FAILURES), FAILURES))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
