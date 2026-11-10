#!/usr/bin/env python3
# coding=utf-8
"""算持久性（persistence）基线在 **10→10 窗口协议**下的逐场 L2RE。

**为什么单独一个脚本**：主表第一行是 Persistence，但它不是任何 run 的产物，
没有 checkpoint、没有 predictions.hdf5 —— 现有两条评分链（``eval_checkpoint.py``
读 run、``score_predictions.py`` 读已落盘预测）都覆盖不到它。

**协议口径**（``protocol/metrics.md``）：

- 多步窗口预测（10→10）：基线 = **最后一个输入帧** $t=9$，对全部输出帧恒定。
- 自治 rollout 才是锚在首帧 $t=0$；两者**不可混用**（见 ``eval_metrics.BASELINE_ANCHOR``）。
- **persistence 不伪造训练种子方差** → 输出单行，表格里不写 ± SD。

指标实现**完全复用** ``eval_metrics``（与两类模型同一条路径），ε 复用
``eval_checkpoint.fit_epsilon`` —— 口径不一致会让主表不可比。

用法::

    python -B score_persistence.py --subset plume1d
    python -B score_persistence.py --subset plume1d --out <path.csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import numpy as np      # noqa: E402

import eval_metrics as em          # noqa: E402
import eval_checkpoint as ec       # noqa: E402
import paths                       # noqa: E402

# 各子集的测试集文件（与 eval_checkpoint 的 SUBSET_* 保持一致；这里是全时程，
# 但只用到前 20 帧 —— first21 是它的前 21 帧切片，逐位相同）
SUBSET_TEST = {
    "plume1d": (paths.DATA_DIRS["plume1d"] + os.sep,
                "plume1d_test_ds_128x128_phys1e6_first21.hdf5"),
    "plume2": (paths.DATA_DIRS["plume2"] + os.sep,
               "plume2d_test_ds_128x128_phys1e6_first21_corrected.hdf5"),
}

TW = 10          # 输入窗帧数
T_OUT = 10       # 输出窗帧数


def load_output_window(folder, fname):
    """读测试集的 **输入窗+输出窗**（帧 0–19），返回 ``[n, T_out, X, Y, C]`` 与输入锚点。

    Returns:
        (anchor, truth_out)：anchor ``[n, 1, X, Y, C]``（帧 9），
        truth_out ``[n, 10, X, Y, C]``（帧 10–19）
    """
    import h5py
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError("测试集不存在: {}".format(path))
    with h5py.File(path, "r") as f:
        missing = [c for c in em.FIELD_ORDER if c not in f]
        if missing:
            raise KeyError("{} 缺字段 {}".format(fname, missing))
        # [n, T, H, W] -> [n, T, H, W, C]，只取前 20 帧
        data = np.stack([np.asarray(f[c][:, :TW + T_OUT], dtype=np.float32)
                         for c in em.FIELD_ORDER], axis=-1)
    n, T, X, Y, C = data.shape
    if T < TW + T_OUT:
        raise ValueError("测试集只有 {} 帧，不足 {}".format(T, TW + T_OUT))
    anchor = data[:, TW - 1:TW]            # 帧 9 —— 输入窗最后一帧
    truth_out = data[:, TW:TW + T_OUT]     # 帧 10–19
    return anchor, truth_out


def score(subset):
    folder, fname = SUBSET_TEST[subset]
    anchor, truth = load_output_window(folder, fname)

    # 与论文式一致：把锚点帧广播到全部输出帧（**不是**逐帧读真值）
    pred = np.broadcast_to(anchor, truth.shape).copy()

    eps = ec.fit_epsilon(subset=subset)
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER])

    sse, energy = em.weighted_squares(pred, truth)
    E = em.aggregate_E(em.relative_l2_per_frame(pred, truth, eps_vec))
    rmse = em.aggregate_E(em.rmse_per_frame(pred, truth))
    trig = em.epsilon_triggered(sse, energy, eps_vec)
    skill_self = np.zeros(len(em.FIELD_ORDER))   # 基线对自身 skill ≡ 0（定义）

    return {
        "model": "Persistence",
        "subset": subset,
        "anchor": "last_input_frame (t=9)",
        "n_traj": int(truth.shape[0]),
        "n_frame": int(truth.shape[1]),
        "source_file": fname,
        "E_per_field": {c: float(E[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "rmse_per_field": {c: float(rmse[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "skill_per_field": {c: float(skill_self[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "epsilon_triggered": trig,
        "epsilon_used": {c: float(eps[c]) for c in em.FIELD_ORDER},
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser(description="persistence 基线的逐场 L2RE（10→10 窗口）")
    ap.add_argument("--subset", default="plume1d", choices=["plume1d", "plume2"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    r = score(a.subset)
    print("Persistence 基线（锚点: {}）".format(r["anchor"]))
    print("  数据: {}  ({} 条 × {} 帧)".format(r["source_file"], r["n_traj"], r["n_frame"]))
    print("  " + " ".join("{:>11}".format(c) for c in em.FIELD_ORDER))
    print("  " + " ".join("{:>11.4e}".format(r["E_per_field"][c]) for c in em.FIELD_ORDER))
    print("  RMSE: " + " ".join("{:.4e}".format(r["rmse_per_field"][c])
                                for c in em.FIELD_ORDER))
    print("  ε 触发次数: {}".format(r["epsilon_triggered"]))

    out = a.out or os.path.join(EXPERIMENTS, "E01_main", "results",
                                "persistence_{}_eq9.csv".format(a.subset))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "anchor", "n_traj", "n_frame", "source_file"]
                   + ["E_{}".format(c) for c in em.FIELD_ORDER]
                   + ["rmse_{}".format(c) for c in em.FIELD_ORDER]
                   + ["scored_at"])
        w.writerow([r["model"], r["subset"], r["anchor"], r["n_traj"], r["n_frame"],
                    r["source_file"]]
                   + [r["E_per_field"][c] for c in em.FIELD_ORDER]
                   + [r["rmse_per_field"][c] for c in em.FIELD_ORDER]
                   + [r["scored_at"]])
    print("\n写出:", out)

    js = os.path.splitext(out)[0] + ".json"
    with open(js, "w", encoding="utf8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print("写出:", js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
