#!/usr/bin/env python3
# coding=utf-8
"""出论文用的 rollout 图：正文 1×4 + 附录 3×4。

作者 2026-09-26 定的两项改动：

1. **横轴改成 lead time，原点 = 最后一个输入帧 t[9]**。
   旧图横轴写的是 absolute physical time（``plot_rollout.py`` 的
   ``_interp_common`` 直接在绝对时间上取共同区间），与"lead time"的说法不符。
   起点取 ``t[9]``（**不是**首预测帧 ``t[10]``）是为了与论文冻结的 lead 口径
   一致 —— 两者恰好差一个时间步（plume1d 恒为 1.0 Myr），用 ``t[10]`` 会让
   读者在图 Δt=12 处读到论文 lead 13 的数。

   横轴从 0 起是用 ``set_xlim(left=0)`` 做的，**没有**加 Δt=0 的零误差锚点：
   那个点在 log 纵轴上不可绘，会被钳到轴底画成垂直下坠线。
   所以实际画出的曲线从首预测帧（plume1d 是 Δt=1.0 Myr）开始。

2. **正文只留 4 个场、只留均值曲线**：ρ / T / v_x / v_y，仅 GeoSim-Plume1。
   依据（作者定）：P 的 rollout 行为与 ρ 相近，log₁₀η 与 T 相近，
   故 ρ/T 各代表一类 scalar 行为；速度两个分量都必须留，因为 velocity
   是全文最重要的 difficulty。完整六场 + 两个子集 + uncertainty bands 进附录。

⚠ 训练边界虚线 = **逐轨迹**的 ``t[19] - t[9]``，**不是**全局标量
``spec_of()["t_train"]`` 减起点。s=11 的训练窗口 dt=0，没有有效梯度；
有效训练只覆盖 s=10（输入 0–9，目标 10–19），实测 plume1d 逐轨迹
恒为 **8.000 Myr**。plume2 异质（范围见 manifest），画中位数。

需要读该子集**测试文件**的 ``t-coordinate``（取 t[9]/t[10]/t[19]/t[20]），
所以会碰 hdf5 —— 数据很小（(12,131) / (25,131)），不需要 GPU，
但仍走计算节点。

用法::

    python -B plot_rollout_paper.py            # 出两张图 + manifest
    python -B plot_rollout_paper.py --report   # 只打印正文要引用的数字
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
# ⚠ 不设这一行，PDF 里全是 Type 3 字体，部分阅读器/投稿系统打不开
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.ticker

import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_rollout as pr                                    # noqa: E402
from data_full import data_path                              # noqa: E402

import paths                                                 # noqa: E402

# 发布包：E03 的最终产物在 results/metrics/E03_rollout/formal_all5/，
# 不再保留原始 campaign 的 formal_all5_20260925/merged/ 目录。
DEFAULT_CSV = os.path.join(paths.RESULTS, "metrics", "E03_rollout",
                           "formal_all5", "rollout_curves.csv")
OUTDIR = os.path.join(paths.RESULTS, "figures")

# 正文 1×4：ρ 代表一类 scalar，T 代表另一类；速度两分量全留（作者 2026-09-26 定）
MAIN_FIELDS = ("ro", "tk", "vx", "vy")
ALL_FIELDS = ("ro", "pr", "tk", "nu", "vx", "vy")
MODE = "auto"
MAIN_SUBSET = "plume1d"
SUBSET_LABEL = {"plume1d": "GeoSim-Plume1", "plume2": "GeoSim-Plume2"}

# 图注/表头里用的模型顺序（与图例一致）
MODEL_ORDER = ["fno", "GNOT", "MPNN", "Transolver", "MFGNO"]


# --------------------------------------------------------------------------
# 读 + 重定基
# --------------------------------------------------------------------------

def load_time_bounds(subset):
    """从该子集的**测试**文件读逐轨迹的 t[9] / t[10] / t[19] / t[20]。

    ⚠ 这些值**CSV 里都没有**：``rollout_curves.csv`` 只含被预测的帧
    （绝对索引 10..129）的物理时间。所以必须回数据文件取。

      ``t9``  : 最后一个**输入**帧 —— 论文 lead time 的口径（协议已冻结）
      ``t10`` : 第一个**预测**帧 —— 仅用于自检 CSV 的轨迹索引与文件行号对应
      ``t19`` : 有效训练窗（输入 0–9，目标 10–19）的最后一个目标帧
      ``t20`` : 21 帧加载范围的上限，仅用于核对，不作为有效训练支持
    """
    p = data_path(subset, "test")
    with h5py.File(p, "r") as f:
        t = np.asarray(f["t-coordinate"][:], dtype=np.float64)
    if t.shape[1] <= 20:
        raise ValueError("%s 只有 %d 帧，取不到 t[20]" % (p, t.shape[1]))
    return {"t9": t[:, 9], "t10": t[:, 10],
            "t19": t[:, 19], "t20": t[:, 20], "file": p}


def load_rebased(path, bounds):
    """返回 {(subset, model, seed, mode): {field: {traj: (dt, err)}}}。

    Δt 原点 = :func:`load_time_bounds` 给出的 ``t[9]``（**最后一个输入帧**），
    与论文 lead time 的口径一致。这是作者 2026-09-26 拍板的修正：早先用
    ``t[10]``（首预测帧）作原点，导致图与表相差恰好一个时间步（plume1d 是
    1.0 Myr），读者在图 Δt=12 处读到的值其实对应论文的 lead 13。

    每条序列前置一个 **Δt=0、误差为 0 的锚点**：该处模型尚未预测，
    误差按定义为 0。这样曲线仍从 0 开始，但横轴口径与表一致。
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, "r", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["metric"] != "l2re":
                continue
            key = (r["subset"], r["model"], str(r["seed"]), r["mode"])
            raw[key][r["field"]][int(r["trajectory"])].append(
                (float(r["physical_time"]), float(r["value"])))

    out = {}
    for key, fields in raw.items():
        subset = key[0]
        b = bounds[subset]
        out[key] = {}
        for field, trajs in fields.items():
            d = {}
            for traj, lst in trajs.items():
                lst.sort(key=lambda x: x[0])
                # 自检：CSV 的首帧必须就是数据文件的 t[10]。
                # 不成立就说明 CSV 的 trajectory 索引不是文件行号，后面的
                # t[9]/t[20] 会张冠李戴，而且是静默的。
                if abs(lst[0][0] - b["t10"][traj]) > 1e-6:
                    raise ValueError(
                        "轨迹 %d 的首帧 %.6f != 文件 t[10] %.6f —— "
                        "CSV 轨迹索引与文件行号不对应" % (traj, lst[0][0], b["t10"][traj]))
                dt = np.array([x[0] for x in lst]) - b["t9"][traj]
                err = np.array([x[1] for x in lst])
                # ⚠ **不加 Δt=0 锚点**。试过了：误差为 0 的点在 log 纵轴上
                #   不可绘（log 0 无定义），matplotlib 会把它钳到轴底，
                #   画出一条**垂直下坠线**，看起来像"误差先崩到零再爆炸"。
                #   横轴从 0 起改用 ``_style_axis`` 里的 ``set_xlim(left=0)``
                #   实现，效果一样但没有假数据点。
                d[traj] = (dt, err)
            out[key][field] = d
    return out


def curve_of_frames(organized, subset, field, models=None):
    """按**帧序号**对齐求均值 —— 每个点都是实测值，不做时间插值。

    ⚠ 为什么不用 ``pr._interp_common`` 的时间插值：各轨迹的物理时间轴
    非均匀且互不相同（实测 traj0 是 1.0 Myr 一步、到第 6 帧变成 0.5 Myr），
    把 12 条折线插到同一时间网格再平均，每越过任意一条轨迹的节点就换一次
    斜率（12 × 120 = 1440 个节点），画出来是锯齿。**单条轨迹本身是光滑的**
    （实测 traj0 / MFGNO / ro：6.01e-5 单调升到 8.31e-4，无振荡），
    所以锯齿纯属重采样伪影，不是数据性质。

    按帧序号对齐则每个 (轨迹, 帧) 都是 CSV 里的实测误差，横坐标取该帧序号
    上各轨迹 Δt 的均值。一次 rollout 对每个模型都是 120 帧，帧序号是公共索引。
    """
    models = models or MODEL_ORDER
    res = {}
    for model in models:
        seeds = sorted({k[2] for k in organized
                        if k[0] == subset and k[1] == model and k[3] == MODE})
        if not seeds:
            continue
        per_seed, ref_dt, traj_sd, n_traj = [], None, None, 0
        for seed in seeds:
            key = (subset, model, seed, MODE)
            if field not in organized.get(key, {}):
                continue
            trajs = organized[key][field]
            lens = {len(t) for t, _ in trajs.values()}
            if len(lens) != 1:
                raise ValueError(
                    "%s 各轨迹帧数不一致 %s —— 帧序号对齐不成立" % (key, sorted(lens)))
            errs = np.array([e for _, e in trajs.values()])
            dts = np.array([t for t, _ in trajs.values()])
            dt_mean = dts.mean(axis=0)
            if ref_dt is None:
                ref_dt = dt_mean
                traj_sd = errs.std(axis=0, ddof=1) if errs.shape[0] > 1 \
                    else np.zeros(errs.shape[1])
                n_traj = errs.shape[0]
            per_seed.append(errs.mean(axis=0))
        if not per_seed:
            continue
        arr = np.asarray(per_seed)
        res[model] = dict(
            grid=ref_dt, mean=arr.mean(axis=0),
            seed_sd=(arr.std(axis=0, ddof=1) if len(per_seed) > 1 else None),
            traj_sd=traj_sd, n_seeds=len(per_seed), n_traj=n_traj)
    return res




def dt_train_of(bounds, subset):
    """有效训练支持在 Δt 轴上的分布（每条轨迹一个值）。

    s=11 的训练窗口 dt=0，对模型没有有效梯度；有效训练只覆盖
    s=10（输入 0–9，目标 10–19）。因此这里是逐轨迹的
    ``t[19] - t[9]``，不是 ``t[20] - t[9]``。
    """
    b = bounds[subset]
    return np.sort(b["t19"] - b["t9"])


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------

def _style_axis(ax, show_ylabel, show_xlabel):
    ax.set_yscale("log")
    # 横轴钉在 0 起（数据从 Δt=1.0 开始）。用 xlim 而不是加一个误差为 0 的
    # 锚点：后者在 log 轴上会被钳到轴底画成垂直下坠线。见 load_rebased 注释。
    ax.set_xlim(left=0.0)
    ax.grid(True, which="major", alpha=0.25, lw=0.4)
    ax.grid(True, which="minor", alpha=0.10, lw=0.3)
    ax.tick_params(labelsize=7)
    if show_ylabel:
        ax.set_ylabel(r"Relative $L^2$ error", fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Physical lead time (Myr)", fontsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))


def _fill_band(ax, g, mu, sd, color, alpha):
    """只画有正下界的 band；不把无定义/非正的下界画成 1e-12 的假下坠。"""
    lo = mu - sd
    hi = mu + sd
    keep = np.isfinite(lo) & np.isfinite(hi) & (lo > 1e-8)
    if np.any(keep):
        ax.fill_between(g, np.maximum(lo, 1e-8), hi, where=keep,
                        color=color, alpha=alpha, linewidth=0,
                        interpolate=False)


def draw_full(curves_by_subset, dt_train_by_subset, out_pdf):
    """附录 3×4：六场 × 两子集，带 seed 与轨迹两条 uncertainty band。

    与 ``plot_all5.py`` 的差别只有横轴（Δt 而非绝对时间）与字体设置。
    色带沿用原约定：浅色 = 首个种子的轨迹 SD，深色 = 种子 SD，不叠加。
    """
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.6), squeeze=False)
    for si, subset in enumerate(("plume1d", "plume2")):
        curves = curves_by_subset[subset]
        for fi, field in enumerate(ALL_FIELDS):
            ax = axes[fi // 2][si * 2 + fi % 2]
            for model in MODEL_ORDER:
                c = curves.get(field, {}).get(model)
                if c is None:
                    continue
                col = pr.MODEL_COLORS[model]
                g, mu = c["grid"], c["mean"]
                if c["traj_sd"] is not None:
                    _fill_band(ax, g, mu, c["traj_sd"], col, 0.09)
                if c["seed_sd"] is not None:
                    _fill_band(ax, g, mu, c["seed_sd"], col, 0.24)
                ax.plot(g, mu, lw=1.1, color=col, label=pr.MODEL_LABEL[model])
            ax.axvline(float(np.median(dt_train_by_subset[subset])),
                       color="black", ls="--", lw=0.7)
            ax.set_title("{} / {}".format(SUBSET_LABEL[subset], pr.FLABEL[field]),
                         fontsize=8)
            _style_axis(ax, show_ylabel=(fi % 2 == 0), show_xlabel=True)
    # ⚠ 图例**放底部**，不要放顶部：旧版把 legend 与 suptitle 都摆在上方，
    #   两者印在同一行，文字互相盖住（实测 "AGNOTformous-rollout..."）。
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("Autonomous rollout: 12 windows / 120 frames", y=0.996)
    fig.tight_layout(rect=(0, 0.028, 1, 0.972))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


# --------------------------------------------------------------------------
# 打印正文要引用的数字（避免凭印象写结论）
# --------------------------------------------------------------------------

def report(curves_by_subset, dt_train_by_subset):
    print("\n" + "=" * 76)
    print("有效训练支持（Δt 轴，= 逐轨迹 t[19] - t[9]）")
    for sub, v in dt_train_by_subset.items():
        print("  %-10s n=%d  中位=%.3f  范围=[%.3f, %.3f] Myr  %s"
              % (sub, v.size, np.median(v), v.min(), v.max(),
                 # 容差放到 1e-4：t 是 float32 存的，逐轨迹恒定的量仍有末位差
                 "（常数）" if np.ptp(v) < 1e-4 else "（逐轨迹不同）"))

    # 写正文要用的判定：某个 lead 是否落在该轨迹自己的训练时域内。
    # 判据等价于 t[9] + lead > t[19]，即 lead > (t[19] - t[9])。
    print()
    print("各 lead 是否在训练时域内（逐轨迹判定，口径 = 论文的 t[9] 起点）")
    print("  %-10s %-8s %-14s %s" % ("subset", "lead", "界外轨迹数", "判定"))
    for sub, v in dt_train_by_subset.items():
        for lead in (2.0, 5.0, 8.0, 12.0):
            over = int((lead > v).sum())
            verdict = ("全部界内" if over == 0 else
                       "**全部界外**" if over == v.size else
                       "%d/%d 界外" % (over, v.size))
            print("  %-10s %-8.1f %-14s %s"
                  % (sub, lead, "%d / %d" % (over, v.size), verdict))

    print("\n公共网格点数（= 逐轨迹帧数的中位数，自适应，避免虚构分辨率）")
    for sub in ("plume1d", "plume2"):
        for field in ALL_FIELDS:
            c = curves_by_subset[sub].get(field, {})
            if c:
                print("  %-10s %-4s %s"
                      % (sub, field, {m: len(v["grid"]) for m, v in c.items()}))
                break

    for sub in ("plume1d", "plume2"):
        curves = curves_by_subset[sub]
        print("\n" + "=" * 76)
        print("子集 %s —— 逐场在若干 lead time 上的均值（seed 均值）" % sub)
        for field in ALL_FIELDS:
            if field not in curves:
                continue
            print("\n  [%s]" % field)
            print("    %-18s %10s %10s %10s %10s %10s"
                  % ("model", "dt=2", "dt=5", "dt=8", "dt=12", "dt=40"))
            for model in MODEL_ORDER:
                c = curves.get(field, {}).get(model)
                if c is None:
                    continue
                g, mu = c["grid"], c["mean"]
                row = []
                for lead in (2.0, 5.0, 8.0, 12.0, 40.0):
                    if g[0] <= lead <= g[-1]:
                        row.append("%10.4g" % float(np.interp(lead, g, mu)))
                    else:
                        row.append("%10s" % "-")
                print("    %-18s %s" % (pr.MODEL_LABEL[model], " ".join(row)))
            # 末段（最后一个网格点）用于判断"长时间后是否反转"
            print("    %-18s %s" % ("(last dt=%.1f Myr)" % g[-1],
                                    " ".join("%s=%.4g" % (pr.MODEL_LABEL[m].split()[0],
                                                          float(curves[field][m]["mean"][-1]))
                                             for m in MODEL_ORDER
                                             if m in curves.get(field, {}))))
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser(description="出论文用 rollout 图")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--report", action="store_true", help="只打印数字，不出图")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        print("!! 找不到曲线数据:", a.csv, file=sys.stderr)
        return 2
    print("读:", a.csv)
    bounds = {s: load_time_bounds(s) for s in ("plume1d", "plume2")}
    for s, b in bounds.items():
        print("  %s : %s" % (s, os.path.basename(b["file"])))
        print("     t[9]  = [%.4f, %.4f]   ← 图的 Δt 原点（最后输入帧）"
              % (b["t9"].min(), b["t9"].max()))
        print("     t[19] = [%.4f, %.4f]   ← 有效训练支持上限"
              % (b["t19"].min(), b["t19"].max()))
    organized = load_rebased(a.csv, bounds)
    print("轨迹数:", len(bounds["plume1d"]["t9"]) + len(bounds["plume2"]["t9"]))

    dt_train = {s: dt_train_of(bounds, s) for s in ("plume1d", "plume2")}
    curves_by_subset = {
        s: {f: curve_of_frames(organized, s, f) for f in ALL_FIELDS}
        for s in ("plume1d", "plume2")
    }

    if a.report:
        report(curves_by_subset, dt_train)
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    p_main = os.path.join(OUTDIR, "fig04_rollout_main.pdf")
    p_full = os.path.join(OUTDIR, "fig04_rollout_full.pdf")

    draw_main(curves_by_subset[MAIN_SUBSET], dt_train[MAIN_SUBSET], p_main)
    print("写出:", p_main)

    draw_full(curves_by_subset, dt_train, p_full)
    print("写出:", p_full)

    # provenance 用包内相对路径，避免把机器绝对路径写进随包 manifest。
    root = os.path.abspath(paths.GEOSIM_ROOT)
    abs_csv = os.path.abspath(a.csv)
    source_csv = (os.path.relpath(abs_csv, root)
                  if abs_csv == root or abs_csv.startswith(root + os.sep)
                  else a.csv)

    manifest = {
        "figures": [os.path.basename(p_main), os.path.basename(p_full)],
        "source_csv": source_csv,
        "aggregation": "frame-index aligned: every plotted point is a raw per-frame value, no time interpolation (time-interpolation over 12 mismatched non-uniform axes produced a resampling-artefact sawtooth)",
        "x_axis": "dt = physical_time - t[9] of the same trajectory, where t[9] is "
                  "the LAST OBSERVED INPUT frame; this matches the paper's lead-time "
                  "convention. The x-axis is pinned to start at 0 via xlim; the drawn "
                  "curves begin at the first predicted frame (dt = 1.0 Myr on "
                  "GeoSim-Plume1). A zero-error anchor at dt=0 is deliberately NOT "
                  "added: it is not drawable on a log axis and renders as a spurious "
                  "vertical drop to the axis floor",
        "why_t9": "the first version used t[10] (first predicted frame) as the origin. "
                  "That differs from the frozen protocol by exactly one time step "
                  "(1.0 Myr on GeoSim-Plume1, 0.098-1.0 on GeoSim-Plume2), so a reader "
                  "taking a value off the figure at dt=12 was reading the paper's "
                  "lead-13 number. Author ruling 2026-09-26: match the protocol.",
        "main_panels": list(MAIN_FIELDS),
        "main_subset": SUBSET_LABEL[MAIN_SUBSET],
        "main_bands": "none (mean curves only)",
        "full_panels": "6 fields x 2 subsets, seed and trajectory bands kept",
        "mode": MODE,
        "training_boundary": {
            s: {"median_dt_Myr": float(np.median(v)),
                "min_dt_Myr": float(v.min()), "max_dt_Myr": float(v.max()),
                "n_trajectories": int(v.size),
                "constant": bool(np.ptp(v) < 1e-4)}
            for s, v in dt_train.items()},
        "boundary_note": "computed per trajectory as t[19] - t[9], the last "
                         "target frame of the effective s=10 window. The s=11 boundary "
                         "window has dt=0 and contributes no gradient. The dashed line is "
                         "the median; on GeoSim-Plume1 it is exactly 8.000 Myr for every "
                         "trajectory, on GeoSim-Plume2 it varies and the spread is recorded "
                         "here.",
        "model_colors": pr.MODEL_COLORS,
        "fonttype_setting": 42,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    mp = os.path.join(OUTDIR, "fig04_rollout_manifest.json")
    with open(mp, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("写出:", mp)
    report(curves_by_subset, dt_train)
    return 0


def draw_main(curves_by_field, dt_train, out_pdf):
    """正文 1×4：四个场、只有均值曲线、无 uncertainty band。

    ⚠ 去掉 shading 是**作者 2026-09-26 定的**：旧的轨迹/种子色带在 log 纵轴下
    有些直接铺到图底，视觉上盖过曲线本身。正文只需讲清趋势，
    完整 uncertainty 进附录版（``draw_full``）。
    """
    fig, axes = plt.subplots(1, len(MAIN_FIELDS), figsize=(7.2, 2.05))
    for i, field in enumerate(MAIN_FIELDS):
        ax = axes[i]
        for model in MODEL_ORDER:
            c = curves_by_field.get(field, {}).get(model)
            if c is None:
                continue
            ax.plot(c["grid"], c["mean"], lw=1.15,
                    color=pr.MODEL_COLORS[model], label=pr.MODEL_LABEL[model])
        ax.axvline(float(np.median(dt_train)), color="black", ls="--", lw=0.7)
        ax.set_title(pr.FLABEL[field], fontsize=9, pad=3)
        _style_axis(ax, show_ylabel=(i == 0), show_xlabel=(i == 0))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=max(len(h), 1), frameon=False,
               fontsize=6.5, handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    sys.exit(main())
