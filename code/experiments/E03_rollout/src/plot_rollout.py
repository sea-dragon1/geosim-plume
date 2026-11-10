#!/usr/bin/env python3
# coding=utf-8
"""E03 出图：长程误差曲线（论文图 F04 的数据来源）。

固定的画图规范（两处独立来源一致）：``ro/pr/nu → viridis``、``tk → hot``、
``vx/vy → RdBu_r``；模型固定配色 FNO ``#C44E52`` / GNOT ``#55A860`` /
MPNN ``#56B4E9`` / MFGNO ``#3D2670`` / Transolver 补 ``#CCB974``。

**两条不确定性必须分开画**（``PLAN.md:76`` 硬要求），且**不叠加在同一个带里**
（叠起来会让人误以为那是同一个来源的方差）：

- **种子变异**：同模型不同种子的曲线之离散 —— 深色带。只有 MFGNO(C) 与
  MPNN(A) 有 3 个种子，且**仅 plume1d**；FNO / Transolver 是单种子，图例里标明。
- **轨迹不确定性**：同模型同种子下跨 12/25 条测试轨迹的离散 —— 浅色带。

横轴是**物理时间（Myr）**而不是帧号 —— 原模拟的时间间隔非均匀，帧号会掩盖这一点。
``T_train`` 用虚线标出：越过它才叫 temporal extrapolation。

⚠ 各轨迹的时间戳非均匀且互不相同，「同一物理时刻的有效轨迹数」随 t 变化。
本脚本取所有轨迹的**共同支持区间**（起点最大值 → 终点最小值）并在其中插值，
区间外的信息不编造；每个 panel 的实际支持区间记进 manifest。

用法::

    python -B plot_rollout.py                       # 两个子集都画
    python -B plot_rollout.py --install             # 另存一份给论文图目录
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
E03_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E03_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
for _p in (COMMON, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                      # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
# ⚠ 不设这一行，PDF 里全是 Type 3 字体，部分阅读器/投稿系统打不开（同 make_fig01.py）
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt         # noqa: E402

import eval_metrics as em               # noqa: E402
from data_full import spec_of           # noqa: E402
import paths                            # noqa: E402

RESULTS = os.path.join(E03_ROOT, "results")
FIGDIR = os.path.join(E03_ROOT, "figures")

CMAP = {"ro": "viridis", "pr": "viridis", "tk": "hot",
        "nu": "viridis", "vx": "RdBu_r", "vy": "RdBu_r"}
MODEL_COLORS = {
    "MFGNO": "#3D2670", "MPNN": "#56B4E9",
    "fno": "#C44E52", "Transolver": "#CCB974", "GNOT": "#55A860",
}
MODEL_LABEL = {"MFGNO": "MFGNO", "MPNN": "Independent MPNN",
               "fno": "FNO", "Transolver": "Transolver", "GNOT": "GNOT"}
# ⚠ 粘度符号必须用 \eta，**不是** \nu：全篇表格与正文统一写 $\log_{10}\eta$
#   （tables/04_ablation.tex、sections/03_benchmark.tex、08_appendix.tex、
#    make_tables.py:343、make_fig01.py:84 全用 \eta）。
#   这里曾写成 \nu，导致 rollout 图与其余所有材料不一致。
FLABEL = {"ro": r"$\rho$", "pr": r"$P$", "tk": r"$T$",
          "nu": r"$\log_{10}\eta$", "vx": r"$v_x$", "vy": r"$v_y$"}

# ⚠ 不启用中文标签：集群上的 matplotlib 没有配中文字体，会退化成方块。
plt.rcParams.update({
    "font.size": 8, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------
# 读 CSV → 按 (subset, model, seed, mode) 组织，逐轨迹保留
# --------------------------------------------------------------------------

def load_curves(path=None):
    path = path or os.path.join(RESULTS, "rollout_curves.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("还没有曲线数据: {}".format(path))

    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, "r", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["metric"] != "l2re":
                continue
            key = (r["subset"], r["model"], str(r["seed"]), r["mode"])
            acc[key][r["field"]][int(r["trajectory"])].append(
                (float(r["physical_time"]), float(r["value"])))

    out = {}
    for key, fields in acc.items():
        out[key] = {}
        for field, trajs in fields.items():
            d = {}
            for traj, lst in trajs.items():
                lst.sort(key=lambda x: x[0])
                d[traj] = (np.array([x[0] for x in lst]),
                           np.array([x[1] for x in lst]))
            out[key][field] = d
    return out


def _interp_common(traj_dict, n_grid=200):
    """插到共同支持区间上的时间网格。

    返回 ``(grid, mat, n_traj)``；没有任何共同区间时返回 ``(None, None, 0)``。
    """
    t_lo = max(float(t[0]) for t, _ in traj_dict.values())
    t_hi = min(float(t[-1]) for t, _ in traj_dict.values())
    if not (t_hi > t_lo):
        return None, None, 0
    grid = np.linspace(t_lo, t_hi, n_grid)
    mat = np.array([np.interp(grid, t, e) for t, e in traj_dict.values()])
    return grid, mat, mat.shape[0]


# --------------------------------------------------------------------------
# 画一个子集
# --------------------------------------------------------------------------

def plot_subset(organized, subset: str, out_pdf: str, modes=None):
    t_train = float(spec_of(subset)["t_train"])
    models = sorted({k[1] for k in organized})
    modes = modes or sorted({k[3] for k in organized})
    if not models or not modes:
        print("  跳过 {}（无数据）".format(subset))
        return None

    n_row, n_col = len(em.FIELD_ORDER), len(modes)
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.6 * n_col, 1.6 * n_row),
                             sharex=True, squeeze=False)
    manifest = {"subset": subset, "t_train_Myr": t_train, "panels": [],
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "cmap_spec": CMAP,
                "note": "浅带 = 跨轨迹 SD（用第一个种子）；深带 = 跨种子 SD；"
                        "两者来源不同，不叠加"}

    for c_i, field in enumerate(em.FIELD_ORDER):
        for m_j, mode in enumerate(modes):
            ax = axes[c_i][m_j]
            for model in models:
                seeds = sorted({k[2] for k in organized
                                if k[1] == model and k[3] == mode})
                if not seeds:
                    continue
                color = MODEL_COLORS.get(model, "#888888")
                seed_curves, grid_ref, n_band = [], None, 0
                for sd in seeds:
                    traj_dict = organized.get((subset, model, sd, mode), {}).get(field)
                    if not traj_dict:
                        continue
                    grid, mat, n = _interp_common(traj_dict)
                    if grid is None:
                        continue
                    grid_ref, n_band = grid, max(n_band, n)
                    seed_curves.append(mat.mean(axis=0))
                    if sd == seeds[0] and mat.shape[0] > 1:
                        mu, sdv = mat.mean(axis=0), mat.std(axis=0, ddof=1)
                        ax.fill_between(grid, mu - sdv, mu + sdv,
                                        color=color, alpha=0.13, linewidth=0)
                if not seed_curves:
                    continue
                arr = np.array(seed_curves)
                mu = arr.mean(axis=0)
                if arr.shape[0] > 1:
                    sdv = arr.std(axis=0, ddof=1)
                    ax.fill_between(grid_ref, mu - sdv, mu + sdv,
                                    color=color, alpha=0.34, linewidth=0)
                ax.plot(grid_ref, mu, color=color, linewidth=1.1,
                        label="{} ({} seed{})".format(
                            MODEL_LABEL.get(model, model), len(seeds),
                            "" if len(seeds) == 1 else "s"))

                manifest["panels"].append({
                    "field": field, "mode": mode, "model": model, "label":
                        MODEL_LABEL.get(model, model),
                    "seeds": seeds, "n_traj_band": int(n_band),
                    "color": color, "t_lo": float(grid_ref[0]),
                    "t_hi": float(grid_ref[-1]),
                })

            ax.axvline(t_train, color="k", linestyle="--", linewidth=0.7)
            ax.set_yscale("log")
            if c_i == 0:
                ax.set_title("{} — {}".format(mode, subset), fontsize=8)
            if m_j == 0:
                ax.set_ylabel(FLABEL[field], fontsize=9)
            if c_i == n_row - 1:
                ax.set_xlabel("physical time (Myr)")
            if c_i == 0 and m_j == 0:
                ax.legend(fontsize=5.2, loc="lower right", framealpha=0.9)

    fig.suptitle("E03 long-horizon rollout — {}\n"
                 "dashed = $T_{{train}}$ ({:.2f} Myr); beyond it is temporal extrapolation"
                 .format(subset, t_train), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_pdf)
    plt.close(fig)

    with open(out_pdf.replace(".pdf", ".manifest.json"), "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("  写出: {}  (+ manifest)".format(out_pdf))
    return out_pdf


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E03 长程曲线出图")
    ap.add_argument("--subset", action="append", default=None,
                    choices=["plume1d", "plume2"])
    ap.add_argument("--curves", default=None)
    ap.add_argument("--mode", action="append", default=None, choices=["auto", "teacher"])
    ap.add_argument("--install", action="store_true",
                    help="另存一份到 iclr2027/figures/fig04_rollout.pdf")
    a = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    rows = load_curves(a.curves)
    if not rows:
        print("CSV 里没有 l2re 数据")
        return 1
    print("读入 {} 组 (subset, model, seed, mode)".format(len(rows)))

    subsets = a.subset or sorted({k[0] for k in rows})
    made = []
    for subset in subsets:
        sub_org = {k: v for k, v in rows.items() if k[0] == subset}
        if not sub_org:
            print("  跳过 {}（无数据）".format(subset))
            continue
        out = os.path.join(FIGDIR, "rollout_curves_{}.pdf".format(subset))
        p = plot_subset(sub_org, subset, out, modes=a.mode)
        if p:
            made.append(p)

    if not made:
        print("没有产出任何图")
        return 1
    if len(made) == 1:
        shutil.copy(made[0], os.path.join(FIGDIR, "rollout_curves.pdf"))
        print("  单子集，已复制为 rollout_curves.pdf")

    if a.install:
        dst_dir = os.path.join(paths.RESULTS, "figures")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "fig04_rollout.pdf")
        shutil.copy(made[0], dst)
        print("  已安装到论文图目录:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
