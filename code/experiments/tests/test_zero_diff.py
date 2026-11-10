"""B 变体（zero-diff）的结构验证。

这是 E02 消融可信度的**前提**：若 B 与 C 参数量不同，或"置零"没有真正生效，
那么 Δ = E_B − E_C 就不能归因于显式差分路径。交接文档要求在正式运行前
检查"B/C 参数量一致、差分张量确实为零、前向有限"。

验证三件事：
  1. B/C 的参数量**逐一相同**（含 state_dict 的键与形状）
  2. 送入 message_net 的差分槽在 B 中**恒为零**、在 C 中非平凡
  3. 两者的前向都能跑通且输出有限

运行： ``python -B tests/test_zero_diff.py``（CPU，秒级）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "..", "common", "vendor")
sys.path.insert(0, VENDOR)

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from model import MFGNO  # noqa: E402
from utils_tubor import PDE  # noqa: E402

FAILURES = []

# 与历史 large 配置一致的结构超参
HIDDEN = 128
LAYERS = 4
TW = 10
NV = 6
POS_DIM = 2
N_EQ = 4

# 消息拼接顺序: [x_i, x_j, pos_diff, var_diff, variables_i]
DIFF_SLOT_START = 2 * HIDDEN + POS_DIM
DIFF_SLOT_END = DIFF_SLOT_START + TW * NV


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# 与历史配置一致：三个方程/参数变量，对应 Data.variables 的 3 维
EQ_VARS = {"m": 0.1, "eta": 0.1, "zeta": 0.1}


def make_pde():
    return PDE(name="2D_Mantle_Convection",
               variables={"m": 0.1, "eta": 0.1, "zeta": 0.1},
               temporal_domain=(0, 8.7433),
               resolution_t=21,
               spatial_domain=[(0, 1), (0, 1)],
               resolution=[128, 128])


def make_graph(n_nodes=40, n_edges=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    return Data(
        x=torch.randn(n_nodes, TW, NV, generator=g),
        x_pos=torch.rand(n_nodes, POS_DIM, generator=g),
        t_pos=torch.rand(n_nodes, generator=g) * 8.7433,
        variables=torch.rand(n_nodes, 3, generator=g),
        edge_index=edge_index,
        batch=torch.zeros(n_nodes, dtype=torch.long),
    )


def test_parameter_counts_identical():
    """B 与 C 的参数量必须逐一相同。"""
    pde = make_pde()
    c = MFGNO(pde, time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                    num_physics_vars=NV, eq_variables={"m": 0.1, "eta": 0.1, "zeta": 0.1},
                    zero_var_diff=False)
    b = MFGNO(pde, time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                    num_physics_vars=NV, eq_variables={"m": 0.1, "eta": 0.1, "zeta": 0.1},
                    zero_var_diff=True)

    n_c = sum(p.numel() for p in c.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    check("B/C 总参数量相同", n_c == n_b, "C={} B={}".format(n_c, n_b))

    sd_c, sd_b = c.state_dict(), b.state_dict()
    check("state_dict 键集合相同", set(sd_c) == set(sd_b))
    shape_mismatch = [k for k in sd_c if sd_c[k].shape != sd_b[k].shape]
    check("state_dict 形状逐一相同", not shape_mismatch, str(shape_mismatch[:3]))

    # 同步权重后两者张量值应完全相同 —— 证明唯一差异是置零开关本身
    b.load_state_dict(c.state_dict())
    check("同步权重后张量值相同（唯一差异是开关）",
          all(torch.equal(sd_c[k], b.state_dict()[k]) for k in sd_c))


def test_message_net_input_dim_unchanged():
    """置零不得改变 message_net 的输入维度（否则就是删了参数）。"""
    pde = make_pde()
    b = MFGNO(pde, time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                    num_physics_vars=NV, eq_variables=EQ_VARS, zero_var_diff=True)
    first_linear = b.gnn_layers[0].message_net[0]
    expected = 2 * HIDDEN + POS_DIM + TW * NV + N_EQ
    check("message_net[0] 输入维 = msg_input_dim",
          first_linear.in_features == expected,
          "got {} expected {}".format(first_linear.in_features, expected))
    check("message_net[0] 权重形状保留差分列",
          first_linear.weight.shape == (HIDDEN, expected),
          str(tuple(first_linear.weight.shape)))


def _capture_diff_slot(model, graph):
    """hook message_net 的第一个 Linear，捕获其输入张量的差分槽。"""
    captured = []

    def hook(module, args):
        captured.append(args[0][:, DIFF_SLOT_START:DIFF_SLOT_END].detach().clone())

    handles = [layer.message_net[0].register_forward_pre_hook(hook)
               for layer in model.gnn_layers]
    try:
        model.eval()
        with torch.no_grad():
            model(graph, torch.tensor([1.0]), 1)
    finally:
        for h in handles:
            h.remove()
    return captured


def test_diff_slot_is_zero_in_b_and_not_in_c():
    """B 的差分槽恒为零；C 的差分槽非平凡。"""
    pde = make_pde()
    kwargs = dict(time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                  num_physics_vars=NV, eq_variables=EQ_VARS)
    c = MFGNO(pde, zero_var_diff=False, **kwargs)
    b = MFGNO(pde, zero_var_diff=True, **kwargs)
    # 参数同步，保证两者唯一差异就是置零开关
    b.load_state_dict(c.state_dict())

    graph = make_graph()
    cap_c = _capture_diff_slot(c, graph)
    cap_b = _capture_diff_slot(b, graph)

    check("捕获到每一层的消息输入", len(cap_c) == LAYERS and len(cap_b) == LAYERS,
          "C={} B={}".format(len(cap_c), len(cap_b)))

    b_all_zero = all(t.abs().max().item() == 0.0 for t in cap_b)
    check("B 的差分槽恒为零", b_all_zero,
          "max per layer: {}".format([float(t.abs().max()) for t in cap_b]))

    c_nonzero = all(t.abs().max().item() > 0 for t in cap_c)
    check("C 的差分槽非平凡", c_nonzero,
          "max per layer: {}".format([float(t.abs().max()) for t in cap_c]))


def test_flag_propagates_to_all_layers():
    pde = make_pde()
    kwargs = dict(time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                  num_physics_vars=NV, eq_variables=EQ_VARS)
    b = MFGNO(pde, zero_var_diff=True, **kwargs)
    c = MFGNO(pde, zero_var_diff=False, **kwargs)
    check("B 的每一层都置位", all(l.zero_var_diff for l in b.gnn_layers))
    check("C 的每一层都未置位", not any(l.zero_var_diff for l in c.gnn_layers))


def test_forward_is_finite_and_output_shape():
    pde = make_pde()
    kwargs = dict(time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                  num_physics_vars=NV, eq_variables=EQ_VARS)
    graph = make_graph()
    for flag in (False, True):
        m = MFGNO(pde, zero_var_diff=flag, **kwargs)
        m.eval()
        with torch.no_grad():
            out = m(graph, torch.tensor([1.0]), 1)
        check("zero_var_diff={} 输出形状 ({}, tw, 6)".format(flag, graph.x.shape[0]),
              out.shape == (graph.x.shape[0], TW, NV), str(tuple(out.shape)))
        check("zero_var_diff={} 输出有限".format(flag), torch.isfinite(out).all())


def test_b_and_c_outputs_differ():
    """置零必须真的改变行为 —— 否则开关没起作用。"""
    pde = make_pde()
    kwargs = dict(time_window=TW, hidden_features=HIDDEN, hidden_layers=LAYERS,
                  num_physics_vars=NV, eq_variables=EQ_VARS)
    c = MFGNO(pde, zero_var_diff=False, **kwargs)
    b = MFGNO(pde, zero_var_diff=True, **kwargs)
    b.load_state_dict(c.state_dict())
    graph = make_graph()
    c.eval(); b.eval()
    with torch.no_grad():
        out_c = c(graph, torch.tensor([1.0]), 1)
        out_b = b(graph, torch.tensor([1.0]), 1)
    check("B 与 C 输出确实不同", not torch.allclose(out_c, out_b, atol=1e-9),
          "max diff = {}".format(float((out_c - out_b).abs().max())))


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
