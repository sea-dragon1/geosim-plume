#!/usr/bin/env python3
# coding=utf-8
"""E04 空间诊断：固定物理剖面上的**幅值误差与位置偏移分解**。

**为什么要这个诊断**：相对 L2 分不清两件事 ——

- 峰值**算小了**（幅值问题）
- 峰值**移位了**（位置问题）

对地幔对流尤其重要：速度剖面的极值对应上涌/下沉柱的位置，把位置错了和把幅值
算小了在 L2RE 上是同一回事，但物理含义完全不同。

按 ``E04_diagnostics_cost/MATH.md`` 的定义::

    剖面   v_true(y), v_pred(y)     沿固定物理位置 x = x* 取竖直剖面
    幅值   A = ( max_y|v_pred| − max_y|v_true| ) / max_y|v_true|     ← 对形状不敏感
    位置   D = | y*_pred − y*_true |,  y* = argmax_y |v|              ← 对幅值不敏感

**x\\* = 0.5 落在单元界面上**（用户 2026-09-21 定，且实测两子集都精确满足
``x[63] + x[64] = 1.000000``）—— 所以剖面按**预声明的两点平均**取::

    profile = (F[..., 63, :] + F[..., 64, :]) / 2

同时输出只用 63 / 只用 64 的两条作为灵敏度（两者相差约 3.9e-3，域长的 0.4%）。

位置偏移 ``D`` **必须同时给三种单位**：归一化域长、km（×660）、网格步长 dy。
``D/dy < 1`` 是亚网格位移，已经触及 argmax 的量化解像度极限，读者必须看得见这一点。

用法::

    python -B velocity_profile.py --subset plume1d              # 全部可用模型
    python -B velocity_profile.py --subset plume1d --model mfgno
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
E04_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E04_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
for _p in (COMMON, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np      # noqa: E402

import eval_metrics as em                    # noqa: E402

RESULTS = os.path.join(E04_ROOT, "results")
RUNS = os.path.join(E04_ROOT, "runs")

# 剖面位置：域中心。实测 x[63]=0.49608016, x[64]=0.50391984，两点和恰为 1.0
X_STAR = 0.5
I_LO, I_HI = 63, 64

# 物理域尺度（见 Plume_data/paper/figures/ 的 X_KM/Y_KM 约定）
DOMAIN_KM = {"x": 1150.0, "y": 660.0}

# ⚠ 2026-09-25 更正两处默认值，两者原先都与协议/正文不符：
#
# MAIN_FRAME：原先写 19（「协议窗口的末帧」）。但 ``predictions.hdf5`` 存的是
#   **10 帧输出窗**（帧索引 0..9），根本没有 19 这个索引 —— 原值靠下游一句
#   ``frame if frame < pred.shape[1] else shape[1]-1`` 夹住才没越界，
#   属于「靠兜底才不出错」的写法。正文与 ``fig03_manifest.json`` 用的是
#   **窗内第 9 帧**（= 全局帧 19），所以这里应是 9。
MAIN_FRAME = 9

# EXAMPLE_TRAJ：原先写 2，而正文 §5.2 写明示例轨迹是「fixed a priori as
#   trajectory 0」、``fig03_manifest.json`` 也记 ``example_traj: 0``。
EXAMPLE_TRAJ = 0


# --------------------------------------------------------------------------
# 剖面几何
# --------------------------------------------------------------------------

def profile_at_x_star(field, i_lo=I_LO, i_hi=I_HI, frac=0.5):
    """沿 x* 取竖直剖面。

    Args:
        field: ``(..., H, W)`` 或 ``(..., H, W, C)`` —— 只对 H 维取索引。
        frac: ``0.0`` 只用 i_lo、``1.0`` 只用 i_hi、``0.5`` 两点平均（主口径）。

    返回形状与输入相比少一维（H 被消掉）。
    """
    # H 是倒数第二维（数据是 (..., H, W[, C])，H 对应 x、W 对应 y）
    lo = np.take(field, i_lo, axis=-2)
    hi = np.take(field, i_hi, axis=-2)
    return (1.0 - frac) * lo + frac * hi


def y_coords_km(subset: str, W: int = 128):
    """竖直坐标（km）。

    ⚠ 2026-09-25 更正：原先这里对 plume2 写「坐标是 linspace(0,1,128)（节点）」，
    取了 ``lo, hi = 0.0, 1.0`` —— **与实测相反**。两子集的 128×128 坐标是
    **逐位相同**的（`protocol/manifest.yaml` 已记录，四份文件实测 element-wise
    相等），y ∈ [0.003157975, 0.9977078]，既不是 ``linspace(0,1)`` 采样点、
    也不是单元中心（单元中心首值应为 0.5/128 = 0.003906）。
    `VALIDATION.md` 也记过这条坐标说法被修正过，本文件当时漏改。

    另注：真实坐标**非均匀**（等面积单元边界），这里用线性插值只是近似 ——
    权威实现与已知偏差见 ``plot_diagnostics.py`` 里 ``peak_shift_km`` 那段注释。
    """
    # 两子集共用同一组坐标（实测逐位相同）
    lo, hi = 0.003157975, 0.9977078
    return (lo + (hi - lo) * np.arange(W) / (W - 1)) * DOMAIN_KM["y"]


# --------------------------------------------------------------------------
# 幅值 / 位置分解
# --------------------------------------------------------------------------

def amplitude_error(v_pred, v_true):
    """``A = (max|pred| − max|true|) / max|true|``。

    **不是** L2 范数比 —— 这个量对剖面的**形状**不敏感，只刻画峰值大小。
    ``max|v_true|`` 为 0 时返回 NaN（不编造数字）。
    """
    mt = float(np.max(np.abs(v_true)))
    if mt <= 0:
        return float("nan")
    return (float(np.max(np.abs(v_pred))) - mt) / mt


def position_error(v_true, y, v_pred):
    """极值点的位置偏移。

    ``y* = argmax_y |v|``；``D = |y*_pred − y*_true|``。
    同时给**带符号**的 Δy（判断峰值是内移还是外移）、质心位置差 ``D_c``
    （对平坦/多峰剖面更稳），以及三种单位的 D。

    ⚠ ``argmax`` 只取**全局**最大 —— 剖面上若有多个局部极大，这个量只反映最强那个。
    质心量 ``D_c`` 是为此准备的补充。
    """
    at, ap = np.abs(v_true), np.abs(v_pred)
    i_t, i_p = int(np.argmax(at)), int(np.argmax(ap))
    y_t, y_p = float(y[i_t]), float(y[i_p])
    d_km = abs(y_p - y_t)
    dy = float(np.mean(np.diff(y)))
    return {
        "y_true_km": y_t, "y_pred_km": y_p,
        "D_km": d_km,
        "D_signed_km": y_p - y_t,
        "D_norm": d_km / DOMAIN_KM["y"],
        "D_over_dy": d_km / dy if dy > 0 else float("nan"),
        "maxabs_true": float(at[i_t]), "maxabs_pred": float(ap[i_p]),
        "centroid_true_km": centroid_position(v_true, y),
        "centroid_pred_km": centroid_position(v_pred, y),
    }


def centroid_position(v, y):
    """``y_c = Σ y|v| / Σ|v|`` —— 对平坦或多峰剖面比 argmax 稳。"""
    w = np.abs(v)
    s = float(w.sum())
    if s <= 0:
        return float("nan")
    return float((y * w).sum() / s)


def profile_rel_l2(v_pred, v_true):
    """剖面自身的相对 L2 —— 正是 A/D 要与之区分的那个量，一并报出来对照。"""
    den = float(np.linalg.norm(v_true))
    if den <= 0:
        return float("nan")
    return float(np.linalg.norm(v_pred - v_true) / den)


# --------------------------------------------------------------------------
# 读预测
# --------------------------------------------------------------------------

def load_predictions(exp_id: str):
    """读 ``runs/<exp_id>/predictions.hdf5``（predict_test.py 的产物）。

    返回 ``(pred, truth)``，形状 ``(n_traj, T, H, W, C)``。
    """
    import h5py
    path = os.path.join(RUNS, exp_id, "predictions.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "没有预测文件: {}\n请先跑 predict_test.py --model <m> --subset <s>".format(path))
    with h5py.File(path, "r") as f:
        fields = [c for c in em.FIELD_ORDER if c in f]
        n, T, H, W = f[fields[0]].shape
        pred = np.stack([np.asarray(f[c]) for c in fields], axis=-1)
        truth = np.stack([np.asarray(f["truth_{}".format(c)]) for c in fields], axis=-1)
    return pred, truth


def available_runs(subset: str):
    """E04 自己推理过、且属于该子集的 run。"""
    out = []
    if not os.path.isdir(RUNS):
        return out
    for d in sorted(os.listdir(RUNS)):
        if subset not in d:
            continue
        if os.path.exists(os.path.join(RUNS, d, "predictions.hdf5")):
            out.append(d)
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def summarize(pred, truth, subset, exp_id, components=("vy", "vx"), frame=MAIN_FRAME):
    """对**全部**测试轨迹汇总 A / D，逐轨迹一行。

    ``PLAN.md`` 要求"对全部测试轨迹汇总，不能只展示一张好看的图" ——
    所以这里逐轨迹算，示例只是其中被指名的那一条。
    """
    y = y_coords_km(subset)
    n = pred.shape[0]
    rows = []
    for c_name in components:
        c = em.FIELD_ORDER.index(c_name)
        for i in range(n):
            t = frame if frame < pred.shape[1] else pred.shape[1] - 1
            vp = profile_at_x_star(pred[i, t, :, :, c])
            vt = profile_at_x_star(truth[i, t, :, :, c])
            pos = position_error(vt, y, vp)
            # 灵敏度：只用 63 / 只用 64
            a_lo = amplitude_error(profile_at_x_star(pred[i, t, :, :, c], frac=0.0),
                                   profile_at_x_star(truth[i, t, :, :, c], frac=0.0))
            a_hi = amplitude_error(profile_at_x_star(pred[i, t, :, :, c], frac=1.0),
                                   profile_at_x_star(truth[i, t, :, :, c], frac=1.0))
            rows.append({
                "exp_id": exp_id, "subset": subset, "trajectory": i,
                "frame": t, "component": c_name,
                "A": amplitude_error(vp, vt),
                "A_only63": a_lo, "A_only64": a_hi,
                "profile_rel_l2": profile_rel_l2(vp, vt),
                **pos,
                "dc_km": abs(pos["centroid_pred_km"] - pos["centroid_true_km"]),
            })
    return rows


def write_csv(rows: list, subset: str):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "profile_errors_{}.csv".format(subset))
    if not rows:
        return path
    header = list(rows[0].keys())
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return path


def pick_failure_case(all_rows, subset):
    """失败案例：**按规则**取窗口 0 的 ``E_vx`` 最大者。

    不是手工挑的 —— 规则在跑之前就写死在这里，图注里要写明"按规则(最大 E_vx)选取"。
    """
    cands = [r for r in all_rows if r["component"] == "vx"]
    if not cands:
        return None
    return max(cands, key=lambda r: r["profile_rel_l2"])["trajectory"]


def main():
    ap = argparse.ArgumentParser(description="E04 速度剖面诊断（幅值/位置分解）")
    ap.add_argument("--subset", required=True, choices=["plume1d", "plume2"])
    ap.add_argument("--model", default=None, help="只做一个模型（默认全部可用的）")
    ap.add_argument("--frame", type=int, default=MAIN_FRAME)
    a = ap.parse_args()

    runs = available_runs(a.subset)
    if a.model:
        runs = [d for d in runs if "-{}-".format(a.model) in d or a.model in d]
    if not runs:
        print("没有可用的预测（先跑 predict_test.py）。runs 目录: {}".format(RUNS))
        return 1

    print("E04 剖面诊断 —— subset={} x*={} (索引 {}/{})".format(
        a.subset, X_STAR, I_LO, I_HI))
    all_rows = []
    for d in runs:
        pred, truth = load_predictions(d)
        rows = summarize(pred, truth, a.subset, d, frame=a.frame)
        all_rows.extend(rows)
        vx = [r for r in rows if r["component"] == "vx"]
        A = np.array([r["A"] for r in vx]); D = np.array([r["D_km"] for r in vx])
        print("  {:<52} vx: A={:+.3f}±{:.3f}  D={:.1f}±{:.1f} km  (D/dy 中位 {:.2f})".format(
            d, np.nanmean(A), np.nanstd(A), np.nanmean(D), np.nanstd(D),
            np.nanmedian([r["D_over_dy"] for r in vx])))

    path = write_csv(all_rows, a.subset)
    print("\n  写出:", path)

    # 摘要：按模型聚合，并指出示例与失败案例
    summary = {}
    for r in all_rows:
        key = (r["exp_id"], r["component"])
        summary.setdefault(key, []).append(r)
    out = {"subset": a.subset, "x_star": X_STAR, "indices": [I_LO, I_HI],
           "frame": a.frame, "n_traj": len({r["trajectory"] for r in all_rows}),
           "example_trajectory_fixed": EXAMPLE_TRAJ,
           "failure_rule": "max profile_rel_l2 on vx (pre-declared, not hand-picked)",
           "failure_trajectory": pick_failure_case(all_rows, a.subset),
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "per_run": {}}
    for (exp_id, comp), rs in sorted(summary.items()):
        A = np.array([r["A"] for r in rs]); D = np.array([r["D_km"] for r in rs])
        out["per_run"]["{}|{}".format(exp_id, comp)] = {
            "A_mean": float(np.nanmean(A)), "A_sd": float(np.nanstd(A)),
            "D_km_mean": float(np.nanmean(D)), "D_km_sd": float(np.nanstd(D)),
            "D_over_dy_median": float(np.nanmedian([r["D_over_dy"] for r in rs])),
            "profile_rel_l2_mean": float(np.nanmean([r["profile_rel_l2"] for r in rs])),
            "n_traj": len(rs),
        }
    jpath = os.path.join(RESULTS, "profile_summary_{}.json".format(a.subset))
    with open(jpath, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("  摘要:", jpath)
    print("  失败案例（按规则取）: 轨迹 {}".format(out["failure_trajectory"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
