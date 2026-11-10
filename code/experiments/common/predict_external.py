#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一推理层：FNO / GNOT / Transolver 在**协议测试集**上推理。

为什么需要它（见 ``../KNOWN_ISSUES.md`` 的 K4 / K4c）
----------------------------------------------------
这三个脚本各自带推理入口（``if_training: False`` 的测试模式），但**无法**用配置调到
协议口径，因为：

1. **``get_dataloader`` 无条件构造 ``train_loader``** ⇒ ``train_data`` 不能为空
   ⇒ ``test_ratio`` 必须给 train 留至少 1 条 ⇒ val 最多 n−1 条；
2. **切片方向相反**：GNN 系 / GNOT / Transolver 是 ``test_idx = int(n*(1-ratio))`` 且取
   ``data[test_idx:]``（后段），FNO 是 ``int(n*ratio)`` 且取 ``data[:test_idx]``（前段）
   ⇒ 同一 ratio 下两者取到的是**不同**的 n−1 条；
3. 三者结果还各自展平/裁剪（GNOT 把「轨迹 × 窗口起点」展平成样本，FNO 输出整条序列）。

本脚本绕开前两条：**只复用各脚本的 dataset 类做数据加载与格式转换**
（取 ``if_test=False, test_ratio=0.0``，此时三者都返回**全部 n 条**），
**索引与前向循环由本脚本控制**，因此三条都能拿到完整的 12 条测试轨迹。

统一口径（与 ``eval_checkpoint.py`` 对 GNN 系的做法一致）
--------------------------------------------------------
- 数据：``plume1d_test_ds_128x128_phys1e6_first21.hdf5`` 的 12 条轨迹
- 窗口：**step=10** —— 输入帧 0–9，预测帧 10–19
- 输出：``[n_traj, 10, 128, 128, 6]`` 的 ``pred`` 与 ``truth``，写到
  ``<run_dir>/eval/predictions.hdf5``，随后可直接用 ``eval_metrics`` 按 Eq.(9) 打分

用法
----
::

    PY=python

    $PY -B common/predict_external.py --runs <run_dir> [<run_dir> ...]
    $PY -B common/predict_external.py --stage E01_main --subset plume1d

需要 GPU（或慢速 CPU）。在计算节点上跑。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, ".."))
# 发布包：外部基线代码在 code/vendor/（不再是仓库根）
VENDOR = paths.VENDOR

if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import h5py  # noqa: E402
import yaml  # noqa: E402

import eval_metrics as em  # noqa: E402
import paths  # noqa: E402

# ⚠ 这两个只作为**回退**保留。测试文件与目录一律由 run 自己的配置推导
#   （见 `_resolve_data`）—— 硬编码会让 `--subset plume2` 跑到 plume1d 的数据上，
#   而且因为两边文件形状相近，未必会立刻报错。
TEST_FILE = "plume1d_test_ds_128x128_phys1e6_first21.hdf5"
DATA_DIR = paths.DATA_DIRS["plume1d"]


def _resolve_data(args):
    """从 run 配置推导 ``(test_file, data_dir)``。

    训练配置里存的是 **train** 文件（`*_train_ds_*`），测试文件是同族的
    `*_test_ds_*`，直接替换即可 —— 这样两个子集自动适配，不需要维护映射表。
    """
    ds = args.get("dataset") or {}
    fname = os.path.basename(str(ds.get("file_name", "")))
    folder = ds.get("saved_folder")
    if not fname or not folder:
        print("    (配置里缺 dataset.file_name / saved_folder，回退到 plume1d 默认值)")
        return TEST_FILE, DATA_DIR
    test_file = fname.replace("_train_ds_", "_test_ds_")
    if test_file == fname:
        raise ValueError(
            "无法从 {!r} 推出测试文件（文件名缺 `_train_ds_` 片段）".format(fname))
    return test_file, folder
TW = 10          # time_window
T_OUT = 10       # 预测帧数（帧 10–19）


# --------------------------------------------------------------------------
# 公共
# --------------------------------------------------------------------------
def _model_of(run_name):
    for p in run_name.split("-"):
        if p.lower() in ("fno", "gnot", "transolver"):
            return {"fno": "fno", "gnot": "GNOT", "transolver": "Transolver"}[p.lower()]
    return None


def load_config(run_dir):
    c = sorted(glob.glob(os.path.join(run_dir, "*.yaml")))
    if not c:
        raise FileNotFoundError("run 目录无配置: {}".format(run_dir))
    with open(c[0], "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, run_dir, which="best", device="cpu"):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*-{}.pt".format(which))))
    if not cks:
        raise FileNotFoundError("找不到 *-{}.pt: {}".format(which, run_dir))
    ck = torch.load(cks[0], map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    return ck.get("epoch")


def isolate_paths(*keep):
    """只保留指定脚本的目录，并清掉同名模块。

    ⚠ 这三个脚本**不能共存于同一进程**：FNO 与 GNOT 各有一个 ``utils.py``
    （后者提供 ``MultipleTensors``），Transolver 有一个 ``train.py``。
    若同时把它们加进 ``sys.path``，先导入的那个会遮蔽后一个 ——
    实测报错为 ``cannot import name 'MultipleTensors' from 'utils' (…/fno/utils.py)``。
    所以在切换模型前必须隔离路径并清掉已导入的同名模块。
    """
    keep = set(os.path.abspath(p) for p in keep)
    for p in list(sys.path):
        ap = os.path.abspath(p) if p else p
        if ap.startswith(VENDOR) and ap not in keep and ap not in (HERE, EXPERIMENTS):
            sys.path.remove(p)
    for mod in ("utils", "train", "dataset", "models", "model"):
        sys.modules.pop(mod, None)


def save_preds(run_dir, pred, truth):
    """pred/truth: (N, T, H, W, C) numpy。

    优先写 hdf5；在 Lustre 上偶发 ``OSError: [Errno 7] ... 'Argument list too long'``
    （实测 GNOT 那条链会触发，因为它跑了 dgl），此时回退到 ``.npz``，
    两者内容等价，``eval_metrics`` 读取时按扩展名分支即可。
    """
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.hdf5")
    try:
        # 分块写：chunks 让 HDF5 不必一次性分配大缓冲，降低触发概率
        with h5py.File(path, "w") as f:
            for c in em.FIELD_ORDER:
                i = em.FIELD_ORDER.index(c)
                f.create_dataset(c, data=pred[..., i].astype(np.float32),
                                 chunks=(1, pred.shape[1], pred.shape[2], pred.shape[3]))
            for c in em.FIELD_ORDER:
                i = em.FIELD_ORDER.index(c)
                f.create_dataset("truth_" + c, data=truth[..., i].astype(np.float32),
                                 chunks=(1, truth.shape[1], truth.shape[2], truth.shape[3]))
        return path
    except OSError as e:
        print("    hdf5 写入失败（{}），回退到 npz".format(str(e)[:60]))
        npath = os.path.join(out_dir, "predictions.npz")
        np.savez_compressed(npath,
                            **{c: pred[..., i].astype(np.float32)
                               for i, c in enumerate(em.FIELD_ORDER)},
                            **{"truth_" + c: truth[..., i].astype(np.float32)
                               for i, c in enumerate(em.FIELD_ORDER)})
        return npath


# --------------------------------------------------------------------------
# FNO
# --------------------------------------------------------------------------
def predict_fno(run_dir, device):
    fno_dir = os.path.join(VENDOR, "fno")
    isolate_paths(fno_dir)
    sys.path.insert(0, fno_dir)
    from utils import FNODatasetSingle            # noqa: E402
    from model.fno import FNO2d                   # noqa: E402

    args = load_config(run_dir)
    m = args["model"]
    model = FNO2d(num_channels=m["num_channels"], width=m["width"],
                  modes1=m["modes"], modes2=m["modes"],
                  initial_step=args["initial_step"]).to(device)

    test_file, data_dir = _resolve_data(args)
    print("    数据: {}/{}".format(data_dir, test_file))
    ds = FNODatasetSingle(filename=test_file, saved_folder=data_dir,
                          initial_step=TW, t_train=TW + T_OUT,
                          reduced_resolution=1, reduced_resolution_t=1, reduced_batch=1,
                          if_test=False, test_ratio=0.0,
                          pde_name=args["pde_name"])
    grid = ds.grid.to(device)
    if grid.dim() == 3:                     # (H, W, 2) → (1, H, W, 2)
        grid = grid.unsqueeze(0)
    data = ds.data                          # (N, H, W, T, C)
    n, H, Wd, T, C = data.shape
    print("    FNO dataset: {} 条 × {} 帧".format(n, T))
    load_checkpoint(model, run_dir, device=device)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(n):
            y = data[i].unsqueeze(0).to(device)             # (1,H,W,T,C)
            x = y[..., :TW, :].clone()                      # 输入窗
            pred = y[..., :TW, :].clone()                   # 前 TW 帧取真值
            for t in range(TW, TW + T_OUT):
                inp = x.reshape(1, H, Wd, -1)
                out = model(inp, grid)                      # (1,H,W,1,C)
                pred = torch.cat((pred, out), dim=-2)
                x = torch.cat((x[..., 1:, :], out), dim=-2)
            # pred 是 (1,H,W,T,C)：取预测帧后转成 (T,H,W,C)，与 eval_metrics 的
            # [n_traj, n_frame, X, Y, C] 对齐
            p = pred[0, ..., TW:TW + T_OUT, :].permute(2, 0, 1, 3)
            t = y[0, ..., TW:TW + T_OUT, :].permute(2, 0, 1, 3)
            preds.append(p.cpu().numpy())
            truths.append(t.cpu().numpy())
    return np.stack(preds), np.stack(truths)                 # (N,T,H,W,C)


# --------------------------------------------------------------------------
# Transolver
# --------------------------------------------------------------------------
def predict_transolver(run_dir, device):
    tslv_dir = os.path.join(VENDOR, "Transolver")
    isolate_paths(tslv_dir)
    sys.path.insert(0, tslv_dir)
    import train as tslv                             # noqa: E402

    args = load_config(run_dir)
    m = args["model"]

    test_file, data_dir = _resolve_data(args)
    print("    数据: {}/{}".format(data_dir, test_file))
    ds = tslv.TransolverDataset(test_file, data_dir, initial_step=TW,
                                t_train=TW + T_OUT, test_ratio=0.0, if_test=False)
    grid = torch.tensor(ds.grid, dtype=torch.float32).to(device)     # (N, 2)
    # ⚠ data 也必须搬到 device：模型内部 forward 会 cat(x=grid, fx=当前窗)，
    # 两者设备不一致会报 "Expected all tensors to be on the same device"。
    data = torch.tensor(ds.data, dtype=torch.float32).to(device)      # (n, T, H, W, C)
    n, T, H, W, C = data.shape
    N = H * W
    print("    Transolver dataset: {} 条 × {} 帧".format(n, T))

    # out_dim 是**单帧**的通道数（模型每次前向只出一帧，滚动 T_OUT 次），
    # 不是 T_OUT —— 见 train.py:387 的实例化。
    model = tslv.Transolver2D(
        space_dim=2, n_layers=m["n_layers"], n_hidden=m["n_hidden"],
        dropout=m.get("dropout", 0.0), n_head=m["n_head"],
        Time_Input=m.get("Time_Input", False), act=m.get("act", "gelu"),
        mlp_ratio=m.get("mlp_ratio", 1), fun_dim=C * TW, out_dim=C,
        slice_num=m.get("slice_num", 64), ref=m.get("ref", 8),
        unified_pos=m.get("unified_pos", False), H=H, W=W).to(device)
    load_checkpoint(model, run_dir, device=device)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(n):
            s = data[i]                                              # (T,H,W,C)
            cur = s[:TW].permute(1, 2, 0, 3).reshape(N, -1)          # (N, TW*C)
            g = grid.unsqueeze(0)                                    # (1,N,2)
            cur = cur.unsqueeze(0)                                   # (1,N,TW*C)
            outs = []
            for t in range(T_OUT):
                out = model(g, cur)                                  # (1,N,C)
                outs.append(out)
                cur = torch.cat([cur[:, :, C:], out], dim=-1)
            pred = torch.stack(outs, dim=2)                          # (1,N,T_OUT,C)
            pred = pred.reshape(H, W, T_OUT, C).permute(2, 0, 1, 3)   # (T_OUT,H,W,C)
            truth = s[TW:TW + T_OUT]                                  # (T_OUT,H,W,C)
            preds.append(pred.cpu().numpy())
            truths.append(truth.cpu().numpy())
    return np.stack(preds), np.stack(truths)


# --------------------------------------------------------------------------
# GNOT
# --------------------------------------------------------------------------
def predict_gnot(run_dir, device):
    # ⚠ 集群装的 dgl 2.1.0 是 CPU-only 编译，**图**不能 ``.to('cuda')``：
    #   "Check failed: allow_missing: Device API cuda is not enabled"。
    # 但这**不妨碍模型在 GPU 上跑** —— 训练脚本 plume1/train.py 就是只搬
    # ``u_p`` / ``inputs`` / ``ndata``，**从不搬 ``g``**，训练日志里的
    # "Using device: cuda:0" 可证。
    # 曾经的写法是把整个模型降到 CPU 来绕开，那既慢又不必要（GNOT 的
    # cost 时延因此记成 489 ms 的 CPU 数字，与其余行不可比）。
    # 现在的做法与训练一致：dgl 图留在 CPU 当拓扑用，张量和模型上 GPU。
    if device.type != "cpu":
        print("    (dgl 图留在 CPU，模型与张量仍在 {}；与训练脚本一致)".format(device))

    # GNOT-master/ 提供 utils.py（MultipleTensors）与 models/；
    # GNOT-master/plume1/ 提供 dataset.py 与 train.py —— 两者都要在 path 上。
    gnot_root = os.path.join(VENDOR, "GNOT-master")
    gnot_plume = os.path.join(gnot_root, "plume1")
    isolate_paths(gnot_root, gnot_plume)
    sys.path.insert(0, gnot_root)
    sys.path.insert(0, gnot_plume)
    from models.mmgpt import GNOT                    # noqa: E402
    import dataset as gds                            # noqa: E402
    from utils import MultipleTensors                # noqa: E402

    args = load_config(run_dir)
    m = args["model"]

    # dataset.py 里的类名在不同版本可能是 Plume1Dataset / Dataset …
    # ⚠ 它**没有**继承 torch.utils.data.Dataset，`_indices` 也是__init__ 里才赋值的实例属性，
    # 所以不能靠 issubclass 或类属性去找，只能按名字匹配。
    cls = getattr(gds, "Plume1Dataset", None)
    if cls is None:
        cands = [n for n in dir(gds)
                 if isinstance(getattr(gds, n), type) and "Dataset" in n]
        if not cands:
            raise RuntimeError("在 GNOT dataset.py 里找不到 Dataset 类")
        cls = getattr(gds, cands[0])
    print("    GNOT dataset 类: {}".format(cls.__name__))

    test_file, data_dir = _resolve_data(args)
    print("    数据: {}/{}".format(data_dir, test_file))
    ds = cls(test_file, data_dir, time_window=TW, total_time_steps=TW + T_OUT + 1,
             if_test=False, test_ratio=0.0)
    # _indices = [(s, step) for s in 轨迹 for step in valid_steps] → 取 step==TW 的
    idxs = [i for i, (_s, step) in enumerate(ds._indices) if step == TW]
    print("    GNOT dataset: {} 条轨迹 × {} 个 step → 取 step={} 的 {} 个".format(
        len(ds.data), len(ds.valid_steps), TW, len(idxs)))

    # trunk_size = 空间维(2) + step index(1) = 3 —— 见 train.py:277-281。
    # 只写 2 会让 trunk_mlp 变成 Linear(2,128) 而输入是 16384×3，报
    # "mat1 and mat2 shapes cannot be multiplied (16384x3 and 2x128)"。
    model = GNOT(trunk_size=2 + 1, branch_sizes=[TW * len(em.FIELD_ORDER)],
                 space_dim=2, output_size=TW * len(em.FIELD_ORDER),
                 n_layers=m["n_layers"], n_hidden=m["n_hidden"], n_head=m["n_head"],
                 n_experts=m["n_experts"], n_inner=m["n_inner"], mlp_layers=m["mlp_layers"],
                 attn_type=m["attn_type"], act=m["act"], ffn_dropout=m["ffn_dropout"],
                 attn_dropout=m["attn_dropout"], horiz_fourier_dim=m["horiz_fourier_dim"],
                 ).to(device)
    load_checkpoint(model, run_dir, device=device)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in idxs:
            g, u_p, inputs = ds[i]
            # ⚠ 不要把 ``g`` 搬到 GPU：dgl 是 CPU-only 编译，`g.to('cuda')` 会抛
            #   "Device API cuda is not enabled"。图只提供拓扑，留在 CPU 即可 ——
            #   训练脚本 plume1/train.py 正是这么做的（它只搬 u_p/inputs/ndata）。
            u_p = u_p.to(device)
            # ⚠ branch 输入要 **3 维**：mmgpt.py:140 的 `_, T2, _ = y[i].size()`
            # 期望 (batch, n_nodes, branch_dim)。走它的 dataloader 时 batch 维由
            # collate 补上，这里手动构造必须自己 unsqueeze(0)。
            inputs = MultipleTensors([t.unsqueeze(0).to(device) for t in inputs])
            # 输出已是 (n_nodes, TW*C)：forward 内部 dgl.unbatch 后按
            # batch_num_nodes 拼回，**没有 batch 维**，不能再取 [0]。
            out = model(g, u_p, inputs)
            # 还原：(n_nodes, TW*C) → (TW, nx, ny, C)
            pred = out.reshape(ds.nx, ds.ny, TW, len(em.FIELD_ORDER)).permute(2, 0, 1, 3)
            truth = g.ndata["y"].reshape(ds.nx, ds.ny, TW, len(em.FIELD_ORDER)).permute(2, 0, 1, 3)
            preds.append(pred.cpu().numpy())
            truths.append(truth.cpu().numpy())
    return np.stack(preds), np.stack(truths)


PREDICTORS = {"fno": predict_fno, "Transolver": predict_transolver, "GNOT": predict_gnot}


def main():
    ap = argparse.ArgumentParser(description="FNO/GNOT/Transolver 的协议口径推理")
    ap.add_argument("--stage", default="E01_main")
    ap.add_argument("--subset", default="plume1d")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    if a.runs:
        runs = [os.path.abspath(r) for r in a.runs]
    else:
        runs = []
        for d in sorted(glob.glob(os.path.join(EXPERIMENTS, a.stage, "runs", "*"))):
            name = os.path.basename(d)
            if "SMOKE" in name or (a.subset and a.subset not in name):
                continue
            if os.path.isdir(os.path.join(d, "checkpoints")):
                runs.append(d)

    runs = [r for r in runs if _model_of(os.path.basename(r)) in PREDICTORS]
    if not runs:
        print("没有找到可推理的外部模型 run")
        return 1
    print("待推理 {} 个 run".format(len(runs)))

    for d in runs:
        mid = _model_of(os.path.basename(d))
        print("\n[{}] {}".format(mid, os.path.basename(d)))
        try:
            pred, truth = PREDICTORS[mid](d, device)
            path = save_preds(d, pred, truth)
            print("    pred {}  →  {}".format(pred.shape, path))
        except Exception as e:  # noqa: BLE001
            import traceback
            print("    [失败] {}: {}".format(type(e).__name__, e))
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
