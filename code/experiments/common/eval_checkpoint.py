#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按论文 Eq.(9) 对已训练的 checkpoint 做**协议口径**评估。

为什么需要这个脚本（见 ``../KNOWN_ISSUES.md`` 的 K2）
------------------------------------------------------
训练自带的 ``test_loop`` 与论文协议有三处不符：

1. **评估数据**用 ``val_loader`` —— 即从训练文件里切出的后 10%（T87–T96），
   而不是 ``protocol/splits.yaml`` 冻结的**独立测试集**（T_test0–11，12 条）；
2. **只保存最后一个 batch**：上游把 ``pred_all.append(...)`` 注释掉了，
   ``output/*.hdf5`` 的 shape 是 ``(2, 10, 128, 128, 6)``，不是全部测试样本；
3. **指标定义**用仓库根 ``metrics.py`` 的 ``L2RE``（把空间与时间维一起 reshape 求范数），
   不是 Eq.(9) 的「逐帧相对 L² → 先对帧平均 → 再对轨迹平均」。

本脚本按协议重做这三件事：独立测试集、完整累积、Eq.(9) 两级平均。

用法
----
::

    PY=python

    # 评估某个 stage 已经跑完的全部 run
    $PY -B common/eval_checkpoint.py --stage E02 --subset plume1d

    # 只评估指定的 run 目录
    $PY -B common/eval_checkpoint.py --runs E02_ablation/runs/<exp_id> ...

    # 额外输出 B/C 的逐场 Δ_c（正值表示 C 更好）
    $PY -B common/eval_checkpoint.py --stage E02 --subset plume1d --compare

产物
----
- 每个 run：``<run_dir>/eval/<exp_id>_eq9.json``
- 汇总：``E0X_*/results/<stage>_eq9.csv``

设备
----
默认用 GPU（若可用）。登录节点无 GPU 时自动退回 CPU，但会慢很多
（图构建是瓶颈）；正式评估建议提交到计算节点。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, ".."))
VENDOR = os.path.join(HERE, "vendor")

# 顺序与 train_tubor.py 一致：`metrics.py` 所在目录必须**最前**
# （发布包里它是 code/vendor，不再是仓库根；common/eval_metrics.py 刻意改名以免遮蔽）。
for _p in (HERE, VENDOR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import paths  # noqa: E402
REPO = paths.VENDOR
if REPO in sys.path:
    sys.path.remove(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import h5py  # noqa: E402

import eval_metrics as em  # noqa: E402
import train_tubor as tt  # noqa: E402
from utils_tubor import MPNNDatasetSingle, to_PDEBench_format  # noqa: E402

# 两个子集的**训练集**文件（测试集文件名由它推导，见 ``_derive_sibling``）。
#
# ⚠ 不要硬编码单个子集：plume2 的测试文件前缀是 `plume2d_`（多了个 d），
#   目录也不同。写死 plume1d 会让 plume2 的评估去 data/plume2/ 找
#   plume1d 的文件 —— 好在是 FileNotFoundError（不是静默用错数据），
#   但那时几十小时的训练已经跑完，白等一场。
# ⚠ 文件名必须带 `_corrected`：旧的无后缀版（均匀池化降采样 + linspace 坐标）
#   已从工作区删除，写旧名会直接 FileNotFoundError。2026-09-23 修 —— 与
#   `splits.yaml`/`manifest.yaml` 的路径失效是**同一类**问题（改数据版本时漏改）。
SUBSET_TRAIN_FILE = {
    "plume1d": "plume1d_train_ds_128x128_phys1e6_first21.hdf5",
    "plume2": "plume2d_train_ds_128x128_phys1e6_first21_corrected.hdf5",
}
# 保留末尾分隔符：下游有的地方按字符串拼接
SUBSET_DATA_DIR = {k: v + os.sep for k, v in paths.DATA_DIRS.items()}

DEFAULT_TEST_FILE = "plume1d_test_ds_128x128_phys1e6_first21.hdf5"
DEFAULT_DATA_DIR = SUBSET_DATA_DIR["plume1d"]


def subset_of_exp_id(exp_id):
    """从 exp_id 判断子集。判定不出时返回 None，由调用方决定是报错还是回退。"""
    for s in SUBSET_TRAIN_FILE:
        if "-{}-".format(s) in exp_id or exp_id.endswith("-" + s):
            return s
    return None


def _derive_sibling(args, want):
    """把配置里的 ``dataset.file_name`` 换成同族的另一半（train ↔ test）。

    这样评估脚本对两个子集**自动适配** —— `plume1d_train_ds_*` 与
    `plume2d_train_ds_*` 都能推出各自的测试文件，不需要维护两张表。
    """
    fname = os.path.basename(str(args["dataset"]["file_name"]))
    other = "test" if want == "train" else "train"
    out = fname.replace("_{}_ds_".format(other), "_{}_ds_".format(want))
    if out == fname:
        raise ValueError(
            "无法从 {!r} 推出 {} 文件（文件名缺 `_{}_ds_` 片段）".format(fname, want, other))
    return out


# --------------------------------------------------------------------------
# 配置与模型
# --------------------------------------------------------------------------
def find_config(run_dir):
    """run 目录里的配置副本由 train_tubor 在训练开始时写入。"""
    cands = sorted(glob.glob(os.path.join(run_dir, "*.yaml")))
    if not cands:
        raise FileNotFoundError("run 目录里找不到配置 yaml: {}".format(run_dir))
    return cands[0]


def load_args(run_dir):
    cfg = find_config(run_dir)
    with open(cfg, "r", encoding="utf8") as f:
        args = yaml.safe_load(f)
    args["root_path"] = os.path.dirname(os.path.abspath(run_dir))
    args["exp_id"] = os.path.basename(os.path.abspath(run_dir))
    return args


def build_env(args, device):
    """构造 PDE / 模型 / 图构建器（与 train_tubor.main 一致）。"""
    pde = tt.PDE(
        args["pde_name"],
        variables=args["variables"],
        temporal_domain=eval(args["temporal_domain"]),
        resolution_t=args["resolution_t"],
        spatial_domain=eval(args["spatial_domain"]),
        resolution=args["resolution"],
        reduced_resolution_t=args["dataset"]["reduced_resolution_t"],
        reduced_resolution=args["dataset"]["reduced_resolution"],
    )
    args["pde"] = pde
    graph_creator = tt.GraphCreator(
        pde,
        neighbors=args["neighbors"],
        time_window=args["time_window"],
    ).to(device)
    model = tt.get_model(args, pde)
    return model, graph_creator


def load_checkpoint(model, run_dir, which="best", device="cpu"):
    path = os.path.join(run_dir, "checkpoints", "{}-{}.pt".format(
        os.path.basename(os.path.abspath(run_dir)), which))
    if not os.path.exists(path):
        raise FileNotFoundError("checkpoint 不存在: {}".format(path))
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    return ck.get("epoch"), ck.get("loss")


def build_test_loader(args, test_file=None, data_dir=None, batch_size=None):
    """独立测试集 loader。

    ``test_ratio=0.0`` + ``if_test=False`` ⇒ 切片 ``data[:n]`` 即**全部**样本
    （见 utils_tubor.MPNNDatasetSingle 的 ``test_idx = int(n*(1-ratio))``）。
    """
    # 测试文件由**该 run 自己的配置**推导，不硬编码子集（见 _derive_sibling）
    test_file = test_file or _derive_sibling(args, "test")
    data_dir = data_dir or args["dataset"].get("saved_folder", DEFAULT_DATA_DIR)
    ds = MPNNDatasetSingle(
        test_file,
        data_dir,
        reduced_resolution=args["dataset"]["reduced_resolution"],
        reduced_resolution_t=args["dataset"]["reduced_resolution_t"],
        reduced_batch=args["dataset"]["reduced_batch"],
        test_ratio=0.0,
        if_test=False,
        variables=args["variables"],
        pde_name=args["pde_name"],
        total_time_steps=args.get("resolution_t", -1),
    )
    bs = batch_size or args["dataloader"]["batch_size"]
    return torch.utils.data.DataLoader(ds, shuffle=False, batch_size=bs), len(ds)


# --------------------------------------------------------------------------
# ε_c：按 protocol/metrics.md，在**训练集**上拟合
# --------------------------------------------------------------------------
def fit_epsilon(train_file=None, data_dir=None, subset=None):
    """``ε_c = 1e-6 × sqrt(mean(truth²))``，在训练集上按通道拟合。

    protocol/metrics.md 规定 ε 的具体数值由 E00 拟合后写入
    ``protocol/epsilon.json``；该文件目前尚不存在，因此这里**现算**并如实
    标注来源，不改写 protocol/。
    """
    # ⚠ ε 必须在**各自子集**的训练集上拟合 —— 两个子集的场值量级不同，
    #   拿 plume1d 的 ε 去评 plume2 是错的口径（即便触发次数为 0 也不能马虎）。
    #   显式给了 train_file 时允许 subset 为 None（调用方直接从 run 的配置读出了
    #   文件名，不依赖子集名做查表），此时不再查 SUBSET_TRAIN_FILE。
    if not train_file:
        subset = subset or "plume1d"
        if subset not in SUBSET_TRAIN_FILE:
            raise ValueError("未知子集 {!r}，可选: {}"
                             .format(subset, sorted(SUBSET_TRAIN_FILE)))
        train_file = SUBSET_TRAIN_FILE[subset]
    if not data_dir:
        data_dir = SUBSET_DATA_DIR.get(subset, DEFAULT_DATA_DIR)
    path = os.path.join(data_dir, train_file)
    vals = {}
    with h5py.File(path, "r") as f:
        for c in em.FIELD_ORDER:
            a = np.asarray(f[c], dtype=np.float64)
            vals[c] = float(np.sqrt((a ** 2).mean()) * 1e-6)
    return vals


# --------------------------------------------------------------------------
# 推理：完整累积全部测试样本
# --------------------------------------------------------------------------
def _forward(model, args, graph, t, batch_size, num_outputs):
    if args["model_id"] == "MFGNO":
        return model(graph, t, batch_size)
    out = torch.empty_like(graph.y)
    for i in range(num_outputs):
        out[..., i] = model[i](graph, t, i, batch_size)
    return out


@torch.no_grad()
def predict_all(args, model, loader, graph_creator, device):
    """推理并**累积全部样本**，返回 ``(pred, truth)``，形状 ``[N, T, X, Y, C]``。

    ⚠ **实际只有单个窗口，不是滚动**（见 KNOWN_ISSUES 的 K6）：
    下面那个 ``range(2*tw, nt-tw+1, tw)`` 在本项目的 ``nt=21`` 下是**空迭代**
    （``range(20, 12, 10)``），所以只会做一次前向 ——
    输入帧 0–9、输出帧 10–19，**帧 20 不参与评估**。

    保留该循环是为了与 vendor 的 ``val_loop`` **逐行同构**（口径必须一致，
    否则重算结果与训练时打印的数字对不上）。要 `nt >= 31` 才会真正滚动一次；
    长程 rollout 属 E03 的范围，届时需另写实现并用全时程数据。
    """
    model.eval()
    preds_all, truth_all = [], []
    tw, nt = graph_creator.tw, graph_creator.nt

    for u, x, t, variables in loader:
        bs = u.shape[0]
        num_outputs = u.shape[-1]
        steps = [tw] * bs
        data, labels = graph_creator.create_data(u, steps)
        x_static = x[0] if x.dim() == 3 else x
        graph, _ = graph_creator.build_static_graph(
            x_static, data, labels, t, steps, bs, variables, device)

        targets_list = [graph.y]
        preds_list = []

        out = _forward(model, args, graph, t, bs, num_outputs)
        preds_list.append(out)

        for step in range(2 * tw, nt - tw + 1, tw):
            steps = [step] * bs
            _, labels = graph_creator.create_data(u, steps)
            graph = graph_creator.update_graph_data(graph, out, labels, steps, t, device)
            targets_list.append(graph.y)
            out = _forward(model, args, graph, t, bs, num_outputs)
            preds_list.append(out)

        pred = torch.cat(preds_list, dim=-2)
        target = torch.cat(targets_list, dim=-2)
        pred = to_PDEBench_format(pred, bs, graph_creator.pde)
        target = to_PDEBench_format(target, bs, graph_creator.pde)
        preds_all.append(pred.detach().cpu().numpy())
        truth_all.append(target.detach().cpu().numpy())

    pred = np.concatenate(preds_all, axis=0)      # (N, H, W, T, C)
    truth = np.concatenate(truth_all, axis=0)
    return pred.transpose(0, 3, 1, 2, 4), truth.transpose(0, 3, 1, 2, 4)


# --------------------------------------------------------------------------
# Eq.(9)
# --------------------------------------------------------------------------
def evaluate_eq9(pred, truth, eps):
    """返回逐场 ``E_c``、逐帧 ``e``、ε 触发计数。"""
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER], dtype=np.float64)
    sse, energy = em.weighted_squares(pred, truth)
    e = em.relative_l2_per_frame(pred, truth, eps_vec)      # [N, T, C]
    E = em.aggregate_E(e)                                    # [C]
    rmse = em.aggregate_E(em.rmse_per_frame(pred, truth))    # [C]
    return {
        "E_per_field": {c: float(E[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "rmse_per_field": {c: float(rmse[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "second_order": {c: float(E[i] ** 2) for i, c in enumerate(em.FIELD_ORDER)},
        "epsilon_triggered": em.epsilon_triggered(sse, energy, eps_vec),
        "n_traj": int(pred.shape[0]),
        "n_frame": int(pred.shape[1]),
    }


def _merge_into_existing(csv_path, new_rows):
    """把本次结果并入已有汇总表：同 ``exp_id`` 的旧行被替换，其余保留。

    这样"跑一个子集"不会抹掉另一个子集的结果 —— 见 ``main()`` 里的说明。
    旧行只带 CSV 的标量列，缺 ``E_per_field`` 这类嵌套结构，因此这里**直接搬运
    字典**：``write_csv`` 读的是 ``E_per_field[c]``，所以要把 CSV 的平列还原回去。
    """
    if not os.path.exists(csv_path):
        return list(new_rows)
    old = {}
    try:
        with open(csv_path, newline="", encoding="utf8") as f:
            for r in csv.DictReader(f):
                old[r["exp_id"]] = dict(
                    exp_id=r["exp_id"], model_id=r["model_id"],
                    variant=r["variant"], subset=r["subset"],
                    seed=int(r["seed"]) if r["seed"] not in ("", "None") else None,
                    checkpoint_epoch=int(r["checkpoint_epoch"])
                    if r["checkpoint_epoch"] not in ("", "None") else None,
                    n_test_samples=int(r["n_test_samples"])
                    if r["n_test_samples"] not in ("", "None") else None,
                    E_per_field={c: float(r["E_" + c]) for c in em.FIELD_ORDER},
                    rmse_per_field={c: float(r["rmse_" + c]) for c in em.FIELD_ORDER},
                )
    except Exception:                       # 旧表损坏时不要连累本次结果
        return list(new_rows)
    for r in new_rows:
        old[r["exp_id"]] = r
    out = list(old.values())
    out.sort(key=lambda r: (r.get("subset") or "", r.get("variant") or "",
                            r.get("seed") if r.get("seed") is not None else -1,
                            r.get("exp_id") or ""))
    return out


def fit_epsilon_for_run(args):
    """按**该 run 自己的配置**拟合 ε —— 与测试文件的解析方式对齐。

    ⚠ 不要用命令行 ``--subset`` 决定 ε：跑 plume2 的 run 时若 ``--subset`` 取了
    默认的 ``plume1d``，就会拿 plume1d 的 ε 去评 plume2（两个子集场值量级不同）。
    实测踩过 —— 判定口径以**每个 run 的配置**为准，与 ``_derive_sibling`` 同源。

    run 的配置里 ``dataset.file_name`` 本身就是**训练**文件，直接用，不必反推
    （``_derive_sibling(args, "train")`` 在配置已是 train 时会抛错）。
    """
    ds = args["dataset"]
    return fit_epsilon(
        train_file=os.path.basename(str(ds["file_name"])),
        data_dir=ds.get("saved_folder"),
        subset=subset_of_exp_id(args["exp_id"]),
    )


def eval_one(run_dir, device, eps=None, test_file=None, data_dir=None, which="best"):
    args = load_args(run_dir)
    # eps=None 时按本 run 自己的子集拟合（推荐；显式传入仅用于测试/复现）
    eps_subset = subset_of_exp_id(args["exp_id"])
    if eps is None:
        eps = fit_epsilon_for_run(args)
    model, graph_creator = build_env(args, device)
    epoch, ck_loss = load_checkpoint(model, run_dir, which=which, device=device)
    model = model.to(device)

    # 先解析出实际用的测试文件，后面要写进结果里存档
    test_file = test_file or _derive_sibling(args, "test")
    loader, n = build_test_loader(args, test_file=test_file, data_dir=data_dir)
    pred, truth = predict_all(args, model, loader, graph_creator, device)
    res = evaluate_eq9(pred, truth, eps)
    res.update({
        "exp_id": args["exp_id"],
        "model_id": args["model_id"],
        "variant": args.get("stage_variant") or _variant_of(args["exp_id"]),
        "subset": subset_of_exp_id(args["exp_id"]) or "unknown",
        "seed": args.get("seed"),
        "checkpoint_epoch": epoch,
        "checkpoint_val_loss": ck_loss,
        "n_test_samples": n,
        "test_file": test_file,
        "epsilon_source": "fitted_on_train__protocol/epsilon.json 尚未生成",
        # 记下 ε 用的是哪个子集的训练文件 —— 与 test_file 一样要能事后审计，
        # 否则"ε 与数据是否同源"这件事没有痕迹（2026-09-24 之前就是这样错的）。
        "epsilon_subset": eps_subset,
        "epsilon_train_file": os.path.basename(str(args["dataset"]["file_name"])),
    })
    return res, pred, truth


def _variant_of(exp_id):
    for v in ("-A-", "-B-", "-C-"):
        if v in exp_id:
            return v.strip("-")
    return None


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def collect_runs(stage, subset, runs):
    if runs:
        return [os.path.abspath(r) for r in runs]
    pat = os.path.join(EXPERIMENTS, stage, "runs", "*")
    out = []
    for d in sorted(glob.glob(pat)):
        if not os.path.isdir(os.path.join(d, "checkpoints")):
            continue
        if "SMOKE" in os.path.basename(d):
            continue
        if subset and subset not in os.path.basename(d):
            continue
        out.append(d)
    return out


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = (["exp_id", "model_id", "variant", "subset", "seed",
             "checkpoint_epoch", "n_test_samples"]
            + ["E_" + c for c in em.FIELD_ORDER]
            + ["rmse_" + c for c in em.FIELD_ORDER])
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [r.get("exp_id"), r.get("model_id"), r.get("variant"),
                 r.get("subset"), r.get("seed"), r.get("checkpoint_epoch"),
                 r.get("n_test_samples")]
                + ["{:.6e}".format(r["E_per_field"][c]) for c in em.FIELD_ORDER]
                + ["{:.6e}".format(r["rmse_per_field"][c]) for c in em.FIELD_ORDER]
            )
    return path


def report_compare(rows):
    """按 (subset, seed) 配对，输出 B/C 的逐场 Δ_c = E_B − E_C（正值 favor C）。"""
    by = {}
    for r in rows:
        if r.get("variant") in ("B", "C"):
            by.setdefault((r["subset"], r["seed"]), {})[r["variant"]] = r
    if not by:
        print("\n（没有可配对的 B/C 结果）")
        return
    print("\n== Δ_c = E_B − E_C（正值表示 C 更好）==")
    header = "{:<10} {:>5}".format("subset", "seed") + "".join(
        "{:>12}".format(c) for c in em.FIELD_ORDER)
    print(header)
    print("-" * len(header))
    for (subset, seed), d in sorted(by.items(), key=lambda kv: str(kv[0])):
        if "B" not in d or "C" not in d:
            continue
        cells = "".join("{:>12.4e}".format(
            d["B"]["E_per_field"][c] - d["C"]["E_per_field"][c]) for c in em.FIELD_ORDER)
        print("{:<10} {:>5}".format(subset, seed) + cells)


def main():
    ap = argparse.ArgumentParser(description="按 Eq.(9) 评估 checkpoint")
    ap.add_argument("--stage", default="E02_ablation")
    ap.add_argument("--subset", default="plume1d")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--compare", action="store_true", help="额外输出 B/C 的 Δ_c")
    ap.add_argument("--device", default=None)
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--test-file", default=None)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    # ⚠ ε **不再**由 `--subset` 决定，改为每个 run 按自己的配置拟合（见
    #   `fit_epsilon_for_run`）。`--subset` 现在只用于筛选 run 目录。
    #   旧行为：跑 plume2 的 run 时不带 `--subset` 就会拿 plume1d 的 ε。
    print("ε_c（按**每个 run 自己的训练集**拟合，见 protocol/metrics.md）")

    runs = collect_runs(a.stage, a.subset, a.runs)
    if not runs:
        print("没有找到可评估的 run（需要 checkpoints/ 目录）")
        return 1
    print("待评估 {} 个 run".format(len(runs)))

    rows = []
    for d in runs:
        try:
            res, _, _ = eval_one(d, device, eps=None,      # None → 按本 run 自己的子集拟合
                                 test_file=a.test_file, data_dir=a.data_dir, which=a.which)
        except Exception as e:  # noqa: BLE001
            print("  [跳过] {}: {}: {}".format(os.path.basename(d), type(e).__name__, e))
            continue
        rows.append(res)
        out_dir = os.path.join(d, "eval")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}_eq9.json".format(res["exp_id"])),
                  "w", encoding="utf8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        line = "  {:<48} ".format(res["exp_id"]) + " ".join(
            "{:>10.4e}".format(res["E_per_field"][c]) for c in em.FIELD_ORDER)
        print(line)

    if not rows:
        return 1
    # ⚠ 汇总文件是「该 stage 的**全部**结果」，不是「本次跑的那批」。
    #   直接 `write_csv(rows)` 会用本次的子集覆盖掉别的子集 —— 实测踩过：
    #   跑 plume2 的 B/C 评估把 plume1d 的 9 行 A/B/C 结果冲掉了
    #   （幸好逐 run 的 eval/*.json 都在，得以重建）。
    #   所以这里读旧表、**只替换本次涉及的那部分**，其余原样保留。
    csv_path = os.path.join(EXPERIMENTS, a.stage, "results",
                            "{}_eq9.csv".format(a.stage))
    merged = _merge_into_existing(csv_path, rows)
    write_csv(merged, csv_path)
    print("\n写出:", csv_path,
          "（本次 {} 行；文件共 {} 行，旧行已保留）".format(len(rows), len(merged)))
    print("字段顺序:", " ".join(em.FIELD_ORDER))
    if a.compare:
        report_compare(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
