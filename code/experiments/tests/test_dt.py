"""真实时间轴 dt 的单元测试。

本项目把残差积分的 dt 从"均匀因子"改成了"相对输入窗口最后一帧的真实时间差"
（见 ``protocol/time.md``）。这是**改变模型定义**的改动，必须有守护：

  1. dt 的数值语义正确（相对最后一帧、逐样本、非均匀）
  2. 缺 t / steps 时退回均匀并**显式报告**，不静默
  3. 扩展到节点维后与解码输出对齐
  4. 模型前向在真实 dt 下可跑通且输出有限

运行： ``python -B tests/test_dt.py``（CPU，秒级）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "..", "common", "vendor")
sys.path.insert(0, VENDOR)

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from model import compute_cumulative_dt, expand_dt_to_nodes, MFGNO, MPNN  # noqa: E402
from utils_tubor import PDE  # noqa: E402

FAILURES = []
TW = 10


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


class FakePDE:
    tmin, tmax, resolution_t = 0.0, 16.7433, 21


def make_t(bs=2, nt=30):
    """模拟真实时间轴：起动段 0.55/0.75、主体 1.0、尾段 0.5（Myr）。"""
    t = torch.zeros(bs, nt)
    base = [0.0, 0.5547, 1.3035]
    for i in range(3):
        t[:, i] = base[i]
    for i in range(3, nt - 4):
        t[:, i] = t[:, i - 1] + 1.0
    for i in range(nt - 4, nt):
        t[:, i] = t[:, i - 1] + 0.5
    return t


def make_data(steps, n_nodes=20, tw=TW, nt=30):
    g = torch.Generator().manual_seed(0)
    return Data(
        x=torch.randn(n_nodes, tw, 6, generator=g),
        x_pos=torch.rand(n_nodes, 2, generator=g),
        t_pos=torch.rand(n_nodes, generator=g),
        variables=torch.rand(n_nodes, 3, generator=g),
        edge_index=torch.randint(0, n_nodes, (2, 60), generator=g),
        batch=torch.zeros(n_nodes, dtype=torch.long),
        steps=steps,
    )


def test_dt_is_relative_to_last_input_frame():
    """dt[b, q] 必须等于 t[b, steps+tw-1+q] - t[b, steps+tw-1]。"""
    t = make_t(bs=2, nt=30)
    steps = [5, 7]
    data = make_data(steps)
    dt, uniform = compute_cumulative_dt(data, t, TW, FakePDE(), torch.device("cpu"))

    check("非均匀路径被使用", uniform is False)
    check("dt 形状为 per-sample [bs, tw]", tuple(dt.shape) == (2, TW), str(tuple(dt.shape)))

    for b, s in enumerate(steps):
        base = s + TW - 1
        expected = torch.stack([t[b, base + q] - t[b, base] for q in range(1, TW + 1)])
        check("样本 {} 的 dt 与手工计算一致".format(b),
              torch.allclose(dt[b], expected, atol=1e-5),
              "got {} expected {}".format(dt[b].tolist(), expected.tolist()))


def test_dt_reflects_nonuniform_intervals():
    """dt 必须反映真实的非均匀间隔，而不是均匀因子。

    注意窗口位置：dt 只覆盖 ``base+1 .. base+tw``，若窗口整段落在均匀区间内，
    差分自然是常数。这里取 steps=16（base=25），使 dt 跨越 nt-4 处的
    1.0 → 0.5 分界，才能真正检验非均匀性。
    """
    # 手工构造：索引 <30 处 dt=1.0，之后 dt=0.5。
    # steps=15 -> base=24 -> dt 覆盖索引 25..34，**跨越**分界 30。
    nt = 45
    t = torch.zeros(1, nt)
    for i in range(1, nt):
        t[0, i] = t[0, i - 1] + (1.0 if i < 30 else 0.5)
    data = make_data([15], nt=nt)
    dt, _ = compute_cumulative_dt(data, t, TW, FakePDE(), torch.device("cpu"))
    diffs = torch.diff(dt[0], prepend=torch.zeros(1))
    check("dt 的差分非均匀（窗口跨越 1.0/0.5 分界）",
          float(diffs.max() - diffs.min()) > 1e-3,
          "diff range [{}, {}]".format(float(diffs.min()), float(diffs.max())))

    # 与均匀因子对比：应当显著不同
    unif = torch.cumsum(torch.ones(TW) * (FakePDE().tmax - 0.0) / FakePDE().resolution_t, dim=0)
    check("dt 与均匀因子明显不同", float((dt[0] - unif).abs().max()) > 0.1)


def test_dt_is_per_sample():
    """不同样本的时间轴不同时，dt 也应当不同。"""
    t = make_t(bs=2, nt=30)
    t[1, :] = t[1, :] * 2.0            # 第二个样本时间轴翻倍
    data = make_data([5, 5])
    dt, _ = compute_cumulative_dt(data, t, TW, FakePDE(), torch.device("cpu"))
    check("两样本的 dt 不同", not torch.allclose(dt[0], dt[1]))


def test_uniform_fallback_is_reported():
    """缺 t 或缺 steps 时退回均匀因子，且**返回标志位**（不静默）。"""
    t = make_t(bs=1, nt=30)
    pde = FakePDE()

    dt_no_t, uni1 = compute_cumulative_dt(make_data([5]), None, TW, pde, torch.device("cpu"))
    check("缺 t 时报告回退", uni1 is True)
    check("缺 t 时形状为 [1, tw]", tuple(dt_no_t.shape) == (1, TW))

    d = make_data([5])
    del d.steps
    dt_no_steps, uni2 = compute_cumulative_dt(d, t, TW, pde, torch.device("cpu"))
    check("缺 steps 时报告回退", uni2 is True)

    expected = torch.cumsum(torch.ones(1, TW) * (pde.tmax - pde.tmin) / pde.resolution_t, dim=1)
    check("回退值与均匀因子一致", torch.allclose(dt_no_t, expected, atol=1e-6))


def test_expand_to_nodes():
    """per-sample 的 dt 要按 batch 索引扩到节点维。"""
    t = make_t(bs=2, nt=30)
    dt, _ = compute_cumulative_dt(make_data([5, 5]), t, TW, FakePDE(), torch.device("cpu"))

    # 均匀广播
    e1 = expand_dt_to_nodes(torch.ones(1, TW), 6, None)
    check("均匀 dt 广播到节点维", tuple(e1.shape) == (6, TW))

    # 真实 dt 按 batch 索引
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    e2 = expand_dt_to_nodes(dt, 6, batch)
    check("真实 dt 扩展到节点维", tuple(e2.shape) == (6, TW))
    check("节点 0-2 与样本 0 一致", torch.allclose(e2[:3], dt[0].expand(3, -1)))
    check("节点 3-5 与样本 1 一致", torch.allclose(e2[3:], dt[1].expand(3, -1)))

    try:
        expand_dt_to_nodes(dt, 6, None)
        check("真实 dt 缺 batch 应报错", False)
    except ValueError:
        check("真实 dt 缺 batch 应报错", True)


def test_model_forward_with_real_dt():
    """MFGNO / MPNN 在真实 dt 下前向可跑通且输出有限。"""
    pde = PDE(name="2D_Mantle_Convection",
              variables={"m": 0.1, "eta": 0.1, "zeta": 0.1},
              temporal_domain=(0, 16.7433), resolution_t=21,
              spatial_domain=[(0, 1), (0, 1)], resolution=[128, 128])
    t = make_t(bs=1, nt=30)
    data = make_data([5])

    c = MFGNO(pde, time_window=TW, hidden_features=32, hidden_layers=2,
                    num_physics_vars=6, eq_variables={"m": 0.1, "eta": 0.1, "zeta": 0.1})
    c.eval()
    with torch.no_grad():
        out = c(data, t, 1)
    check("MFGNO 输出形状", tuple(out.shape) == (20, TW, 6), str(tuple(out.shape)))
    check("MFGNO 输出有限", torch.isfinite(out).all())

    # MPNN 的 output_mlp 是按 (hidden, tw) 组合硬编码 Conv1d kernel 的
    # （见 CLAUDE.md 的「time_window-dependent output_mlp」），只能用 128/tw=10。
    m = MPNN(pde, time_window=TW, hidden_features=128, hidden_layers=2,
             eq_variables={"m": 0.1, "eta": 0.1, "zeta": 0.1})
    m.eval()
    with torch.no_grad():
        out_m = m(data, t, 0, 1)
    check("MPNN 输出形状", tuple(out_m.shape) == (20, TW), str(tuple(out_m.shape)))
    check("MPNN 输出有限", torch.isfinite(out_m).all())


def test_real_dt_changes_output_vs_uniform():
    """真实 dt 与均匀 dt 必须给出不同结果 —— 否则改动没生效。"""
    pde = PDE(name="2D_Mantle_Convection",
              variables={"m": 0.1, "eta": 0.1, "zeta": 0.1},
              temporal_domain=(0, 16.7433), resolution_t=21,
              spatial_domain=[(0, 1), (0, 1)], resolution=[128, 128])
    t = make_t(bs=1, nt=30)
    data = make_data([5])

    m = MFGNO(pde, time_window=TW, hidden_features=32, hidden_layers=2,
                    num_physics_vars=6, eq_variables={"m": 0.1, "eta": 0.1, "zeta": 0.1})
    m.eval()
    with torch.no_grad():
        out_real = m(data, t, 1)
        out_uniform = m(data, None, 1)     # t=None -> 均匀回退

    check("真实 dt 与均匀 dt 输出不同",
          not torch.allclose(out_real, out_uniform, atol=1e-6),
          "max diff = {}".format(float((out_real - out_uniform).abs().max())))


def main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if FAILURES:
        print("失败 {} 项: {}".format(len(FAILURES), FAILURES))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
