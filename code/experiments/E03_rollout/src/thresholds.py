#!/usr/bin/env python3
# coding=utf-8
"""E03 阈值与ε的**预先计算**（"运行前冻结"的落地）。

``PLAN.md`` 与 ``MATH.md`` 把三样东西当成"已有点"来写，但**全仓库从没算过**：

===============  ==================================================  ==========================
量                定义                                                来源
===============  ==================================================  ==========================
``ε_c``          逐场范数稳定项                                       ``eval_checkpoint.fit_epsilon``（训练集）
``τ_c``          逐场 L2RE 的 99% 分位                                **验证集**轨迹（plume1d 87–96 / plume2 92–102）
``[lo_c, hi_c]`` 范围违例的上下界                                    **全时程训练行**的实测极值 + 余量
===============  ==================================================  ==========================

本模块把它们算出来并**落盘**（``results/tau_c.json`` / ``range_limits.csv``）——
"预先固定"的语义要求这些数字在跑 rollout **之前**就存在、且事后可核对，
不能只活在内存里。

几条刻意的设计
--------------
1. **τ_c 的粒度是 (子集, 模型, 种子, 场)**。""验证集 L2RE"" 本质上是**该模型自己**的
   能力刻画；用一个跨模型的全局阈值会让弱模型的失败率恒为 100%、强模型恒为 0%。
   同时另存一份 ``tau_c_global`` 作灵敏度参考。
2. **τ_c 用逐帧 ``e[n,q,c]`` 而不是逐轨迹 ``E_c``**：plume1d 验证集只有 10 条轨迹 ×
   10 帧 = 100 个逐帧样本（逐轨迹只有 10 个），99% 分位在 10 个样本上基本等于最大值，
   方差极大。``tau_c.json`` 里同时记 ``p50/p95/max`` 让读者看得见阈值离最大值多近。
3. **``[lo_c, hi_c]`` 用全时程训练行，不用 first21**：rollout 会走到 t≈72 Myr，
   而 first21 只覆盖 0→17.24 Myr —— 拿它当包络判定晚期状态必然误报。
   用全时程但**只取训练行**（``rows < 87`` / ``< 92``），既不越界也不泄漏验证集。
4. **必须留余量**（``nu`` 除外）：实测**测试集的参考真值本身就超界**
   （plume1d ``tk``、plume2 ``ro``/``pr``），零余量会把真值判成失败。

用法::

    python -B thresholds.py --subset plume1d --run <run_dir>       # 单个模型
    python -B thresholds.py --subset plume1d --range-only          # 只算范围界（与模型无关）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
E03_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E03_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
for _p in (COMMON, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np      # noqa: E402
import torch            # noqa: E402

import eval_metrics as em                    # noqa: E402
import model_zoo as mz                       # noqa: E402
import paths                                 # noqa: E402
from data_full import read_split, spec_of, train_rows_of, val_rows_of   # noqa: E402
# ⚠ 必须在这里（模块级）导入，**不能**延迟到函数内。
#   外部模型装载会走 ``predict_external.isolate_paths``，它把**所有**
#   ``startswith(REPO)`` 且不在豁免名单里的 ``sys.path`` 项删掉 ——
#   ``E03_rollout/src`` 正好会被删（它不在 keep 里，也不等于 HERE/EXPERIMENTS）。
#   延迟 import 的后果实测是 ``ModuleNotFoundError: No module named 'rollout'``，
#   而且**只在外部模型上出现**（GNN 装载不走 isolate_paths），很难猜。
#   ``run_rollout.py`` 一直是模块级导入，所以它没这个问题。
import rollout as ro                         # noqa: E402
# ⚠ 同样必须模块级：``eval_checkpoint`` 会连带导入 vendor 的 ``train_tubor``，
#   后者 ``from model import MPNN, ...`` —— 这些都在 REPO 下，外部模型装载时
#   已被 ``isolate_paths`` 从 ``sys.path`` 摘掉，延迟导入必炸
#   （实测 ``ImportError: cannot import name 'MPNN' from 'model'``）。
#   **顺带修掉 ``run_rollout.py`` 的同类问题**：它在模块顶部就 ``import thresholds``，
#   于是 ``eval_checkpoint`` 会被一起带进 ``sys.modules``，其 ``run_one`` 里
#   那处延迟导入就不会再找路径了。
import eval_checkpoint as ec                 # noqa: E402

RESULTS = os.path.join(E03_ROOT, "results")
TAU_PATH = os.path.join(RESULTS, "tau_c.json")
RANGE_PATH = os.path.join(RESULTS, "range_limits.csv")
META_PATH = os.path.join(RESULTS, "thresholds_meta.json")

QUANTILE = 0.99
"""失败阈值取验证集逐场 L2RE 的第 99 百分位（用户 2026-09-21 决定）。

与"连续 3 个窗口超阈才算失败"配合：单窗口超阈概率约 1%，
连续三次约 1e-6 —— 正常的长程误差增长几乎不可能触发，只有真发散才会。
"""

MARGIN_FRAC = 0.05
"""范围界的相对余量（预注册主值）。

实测测试集真值超出训练极值的幅度是 0.006%–0.15%，5% 远大于它、
又远小于真正的数值发散。``nu`` 例外取 0（见 ``compute_range_limits``）。
"""


# --------------------------------------------------------------------------
# ε_c
# --------------------------------------------------------------------------

def fit_eps(subset: str) -> dict:
    """逐场 ε_c —— 在**训练集**上拟合（``ε_c = 1e-6 · sqrt(mean(truth²))``）。

    直接复用 ``eval_checkpoint.fit_epsilon``，保持与 E01/E02 的口径一致。
    ``protocol/epsilon.json`` 至今不存在（该函数自己也会现算并如实标注来源）。
    """
    return ec.fit_epsilon(subset=subset)   # ec 已在本模块顶部导入，见那里的说明


# --------------------------------------------------------------------------
# τ_c：验证集单窗口 L2RE 的分位数
# --------------------------------------------------------------------------

def val_single_window_errors(loaded, subset: str, device) -> np.ndarray:
    """在**验证集**轨迹上跑单窗口推理，返回逐帧逐场误差 ``e[n_val, tw, C]``。

    ⚠ 验证集用的是**训练文件**的后 10%（plume1d 索引 87–96、plume2 92–102），
    不是测试集 —— 这是 ``MATH.md`` 的硬要求（阈值不能用测试集定）。

    跑 ``k_max=1``（单窗口）是刻意的：τ_c 描述的是"该模型在它熟悉的时域里
    的正常误差水平"，不是长程误差。长程误差是要被它判定的对象。
    """
    rows = val_rows_of(subset)
    n_frame = spec_of(subset)["n_frame_train"]
    # 用训练文件的**全时程**（不是 first21）—— 验证集的可用窗口数因此是 8 个而不是 1 个，
    # 但这里只要第 1 个窗口，读全时程与读 first21 的前 21 帧等价（已由 selftest 验证）。
    fd = read_split(subset, "train", rows=rows, verbose=False)

    if loaded.family in mz.GNN_FAMILIES:
        # ⚠ 必须覆盖 resolution_t 为真实帧数：GraphCreator.nt 取自 pde.resolution_t
        res = ro.rollout_gnn(loaded, fd, mode="teacher", k_max=1, verbose=False)
    elif loaded.family == "gnot":
        res = ro.rollout_gnot(loaded, fd, mode="teacher", k_max=1, verbose=False)
        np.testing.assert_array_equal(res.truth, fd.u[:, 10:20].numpy())
        np.testing.assert_array_equal(res.t_out, fd.t[:, 10:20].numpy())
    else:
        # The loader defaults to TEST data. Explicitly supply the frozen
        # validation rows from the TRAIN file, including their own time axis.
        old_data = loaded.ctx["data"]
        old_time = loaded.ctx.get("time_axis")
        loaded.ctx["data"] = (fd.u.permute(0, 2, 3, 1, 4)
                              if loaded.family == "fno" else fd.u)
        loaded.ctx["time_axis"] = fd.t.numpy()
        try:
            res = ro.rollout_external(loaded, k_max=1, mode="teacher", verbose=False)
            np.testing.assert_array_equal(res.truth, fd.u[:, 10:20].numpy())
            np.testing.assert_array_equal(res.t_out, fd.t[:, 10:20].numpy())
        finally:
            loaded.ctx["data"] = old_data
            if old_time is None:
                loaded.ctx.pop("time_axis", None)
            else:
                loaded.ctx["time_axis"] = old_time
    assert res.n_traj == len(rows), (res.n_traj, len(rows))

    eps = fit_eps(subset)
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER])

    # NaN/Inf 必须先处理：eval_metrics 的 _as_float 遇到非有限值直接 raise
    pred, n_bad = _sanitize(res.pred, res.truth)
    if n_bad:
        raise ValueError("Nonfinite validation prediction: cannot fit failure thresholds")
        print("      ⚠ 验证集预测里有 {} 个非有限值（已按同位置真值填补）".format(n_bad))

    e = em.relative_l2_per_frame(pred, res.truth, eps_vec)      # (n_val, tw, C)
    return e, eps


def _sanitize(pred: np.ndarray, truth: np.ndarray):
    """把非有限预测替换为**同位置真值**并计数。

    这是声明式的填补规则：该帧的误差因此记为 0（而不是 NaN 传播，
    也不是直接抛异常）。计数会被记入结果，失败判据另有专门的 NaN/Inf 分支，
    所以填补不会掩盖数值失败。
    """
    bad = ~np.isfinite(pred)
    n_bad = int(bad.sum())
    if n_bad:
        pred = np.where(bad, truth, pred)
    return pred, n_bad


def compute_tau_c(e_val: np.ndarray, q: float = QUANTILE) -> dict:
    """逐场 τ_c + 足够的上下文，让读者能判断阈值离最大值多近。"""
    out = {}
    n_val, n_frame, n_var = e_val.shape
    for c_i, c in enumerate(em.FIELD_ORDER):
        vals = e_val[:, :, c_i].reshape(-1)                     # 逐帧逐轨迹
        vals = vals[np.isfinite(vals)]
        out[c] = {
            "tau_c": float(np.quantile(vals, q, method="linear")),
            "n_values": int(vals.size),
            "quantile": q,
            "p50": float(np.quantile(vals, 0.50)),
            "p95": float(np.quantile(vals, 0.95)),
            "max": float(vals.max()) if vals.size else float("nan"),
        }
    # 跨场汇总（灵敏度参考）：把所有场的逐帧值堆一起再取分位
    allv = e_val.reshape(-1)
    allv = allv[np.isfinite(allv)]
    out["_global"] = {
        "tau_c_global": float(np.quantile(allv, q, method="linear")),
        "n_values": int(allv.size),
        "p50": float(np.quantile(allv, 0.50)),
        "p95": float(np.quantile(allv, 0.95)),
        "max": float(allv.max()) if allv.size else float("nan"),
    }
    return out


# --------------------------------------------------------------------------
# [lo_c, hi_c]：范围违例界
# --------------------------------------------------------------------------

def _scan_extremes(subset: str, split: str, rows):
    """流式扫一个 split 的逐场极值 —— **不把整个 split 载入内存**。

    ⚠ 为什么不用 ``read_split``：plume1d 的训练池是 97×97×128×128×6 的 float32
    ≈ 3.7 GB，plume2 是 5.3 GB。``read_split`` 会一次性读进来，在登录节点上
    直接把进程撑爆（实测：无输出、无报错、2m38s 后被 OOM 杀掉）。
    极值统计只需要逐轨迹扫一遍，内存 O(1 条轨迹) 就够。
    """
    import h5py
    s = spec_of(subset)
    path = os.path.join(s["data_dir"], s["{}_file".format(split)])
    rows = list(rows)
    out = {}
    with h5py.File(path, "r") as f:
        for c in em.FIELD_ORDER:
            lo, hi, n_bad = np.inf, -np.inf, 0
            for r in rows:
                a = np.asarray(f[c][r])              # (T, H, W) —— 单条轨迹
                n_bad += int((~np.isfinite(a)).sum())
                lo = min(lo, float(a.min()))
                hi = max(hi, float(a.max()))
            out[c] = {"lo": lo, "hi": hi, "n_nonfinite": n_bad}
    return out, len(rows)


def compute_range_limits(subset: str, margin_frac: float = MARGIN_FRAC) -> dict:
    """逐场 ``[lo_c, hi_c]``，来自**全时程训练行**的实测极值 + 相对余量。

    ``nu`` 是例外：它的物理硬边界是 ``log10(ν)/24 ∈ [0.75, 1.0]``（实测精确成立，
    六场里唯一有解析边界的），所以取**零余量** —— 加余量反而会让"黏度越界"
    这个本该敏感的检查失去意义。

    同时记录验证集/测试集的实测极值（**仅记录，不参与阈值**），
    这样 REPORT 里能直接展示"测试集真值本身超界多少"。
    """
    names = list(em.FIELD_ORDER)
    tr, n_tr = _scan_extremes(subset, "train", train_rows_of(subset))
    va, n_va = _scan_extremes(subset, "train", val_rows_of(subset))
    te, n_te = _scan_extremes(subset, "test", range(spec_of(subset)["n_traj_test"]))

    out = {}
    for c in names:
        raw_lo, raw_hi = tr[c]["lo"], tr[c]["hi"]
        margin = 0.0 if c == "nu" else margin_frac * (raw_hi - raw_lo)
        out[c] = {
            "raw_lo": raw_lo, "raw_hi": raw_hi,
            "lo_c": raw_lo - margin, "hi_c": raw_hi + margin,
            "margin": margin,
            "margin_rule": "zero (physical hard bound [0.75, 1.0])" if c == "nu"
                           else "{:.0%} of train range".format(margin_frac),
            "val_lo": va[c]["lo"], "val_hi": va[c]["hi"],
            "test_lo": te[c]["lo"], "test_hi": te[c]["hi"],
            "n_train_rows": n_tr, "n_val_rows": n_va, "n_test_rows": n_te,
            "n_nonfinite_train": tr[c]["n_nonfinite"],
        }
        # 训练集极值本身是否被 val/test 突破 —— 这正是"零余量会把真值判成失败"的证据
        out[c]["test_exceeds_train"] = bool(
            out[c]["test_lo"] < raw_lo or out[c]["test_hi"] > raw_hi)
    return out


def write_range_limits(limits: dict, subset: str, margin_frac: float = MARGIN_FRAC):
    os.makedirs(RESULTS, exist_ok=True)
    header = ["subset", "field", "raw_lo", "raw_hi", "lo_c", "hi_c", "margin", "margin_rule",
              "val_lo", "val_hi", "test_lo", "test_hi", "test_exceeds_train",
              "n_train_rows", "n_val_rows", "n_test_rows"]
    lines = [",".join(header)]
    for c in em.FIELD_ORDER:
        d = limits[c]
        lines.append(",".join(str(x) for x in [
            subset, c, d["raw_lo"], d["raw_hi"], d["lo_c"], d["hi_c"], d["margin"],
            '"{}"'.format(d["margin_rule"]), d["val_lo"], d["val_hi"],
            d["test_lo"], d["test_hi"], int(d["test_exceeds_train"]),
            d["n_train_rows"], d["n_val_rows"], d["n_test_rows"]]))
    with open(RANGE_PATH if subset == "plume1d" else RANGE_PATH.replace(".csv", "_{}.csv".format(subset)),
              "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    return RANGE_PATH


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------

def _load_tau() -> dict:
    if os.path.exists(TAU_PATH):
        with open(TAU_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    return {"_meta": {"created": datetime.now().isoformat(timespec="seconds"),
                      "quantile": QUANTILE, "granularity": "subset/model/seed/field"}}


def _save_tau(d: dict):
    os.makedirs(RESULTS, exist_ok=True)
    with open(TAU_PATH, "w", encoding="utf8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print("  写出:", TAU_PATH)


def _display_path(path: str) -> str:
    """包内路径写成相对路径，避免 provenance 带机器绝对路径。"""
    root = os.path.abspath(paths.GEOSIM_ROOT)
    p = os.path.abspath(path)
    if p == root or p.startswith(root + os.sep):
        return os.path.relpath(p, root)
    return path


def _save_meta(entry: dict):
    entry = dict(entry)
    if "run_dir" in entry:
        entry["run_dir"] = _display_path(entry["run_dir"])
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf8") as f:
            meta = json.load(f)
    runs = meta.setdefault("runs", [])
    runs = [r for r in runs if r["key"] != entry["key"]]
    runs.append(entry)
    meta["runs"] = runs
    meta["generated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(META_PATH, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_one(run_dir: str, subset: str, which: str = "best", device=None):
    """算一个 run 的 τ_c 并合并进 ``tau_c.json``。"""
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("── {} ──".format(os.path.basename(run_dir)))
    loaded = mz.load_run(run_dir, device, which=which,
                         frames=spec_of(subset)["n_frame_train"])
    print("   {}  ckpt_epoch={}".format(loaded.describe(), loaded.ckpt_epoch))

    e_val, eps = val_single_window_errors(loaded, subset, device)
    tau = compute_tau_c(e_val)

    key = "{}|{}|{}".format(subset, loaded.exp_id, which)
    store = _load_tau()
    store[key] = {
        "subset": subset,
        "exp_id": loaded.exp_id,
        "model_id": loaded.model_id,
        "family": loaded.family,
        "seed": loaded.seed,
        "which": which,
        "ckpt_epoch": loaded.ckpt_epoch,
        "eps_c": {c: eps[c] for c in em.FIELD_ORDER},
        "per_field": {c: tau[c] for c in em.FIELD_ORDER},
        "global": tau["_global"],
        "source": "validation rows of the TRAIN file (splits.yaml val)",
        "source_rows": [int(val_rows_of(subset).start), int(val_rows_of(subset).stop)],
    }
    _save_tau(store)
    _save_meta({"key": key, "run_dir": run_dir, "which": which,
                "ckpt_epoch": loaded.ckpt_epoch,
                "n_val_traj": int(e_val.shape[0]), "n_frame": int(e_val.shape[1])})

    print("   τ_c:", " ".join(
        "{}={:.3e}".format(c, tau[c]["tau_c"]) for c in em.FIELD_ORDER))
    print("   全局 τ_c = {:.3e}（n={}，p95={:.3e}，max={:.3e}）".format(
        tau["_global"]["tau_c_global"], tau["_global"]["n_values"],
        tau["_global"]["p95"], tau["_global"]["max"]))
    return key


def main():
    ap = argparse.ArgumentParser(description="E03 阈值预计算（ε / τ_c / 范围界）")
    ap.add_argument("--subset", required=True, choices=["plume1d", "plume2"])
    ap.add_argument("--run", default=None, help="run 目录；不给则只算与模型无关的范围界")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--range-only", action="store_true")
    a = ap.parse_args()

    print("E03 阈值预计算 —— subset={} quantile={}".format(a.subset, QUANTILE))
    limits = compute_range_limits(a.subset)
    p = write_range_limits(limits, a.subset)
    print("范围界已写出:", p)
    for c in em.FIELD_ORDER:
        d = limits[c]
        flag = "  ⚠ 测试集真值超界" if d["test_exceeds_train"] else ""
        print("  {:<3} [{:.6f}, {:.6f}]  余量={:.4g}{}".format(
            c, d["lo_c"], d["hi_c"], d["margin"], flag))

    if a.range_only or not a.run:
        print("\n（未给 --run，只算了范围界）")
        return 0

    dev = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("\ndevice:", dev)
    build_one(os.path.abspath(a.run), a.subset, which=a.which, device=dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
