#!/usr/bin/env python3
# coding=utf-8
"""FA01 —— 参数覆盖与时间覆盖图 ``figures/figA01_parameter_coverage.pdf``。

按 ``figures/README.md`` 对 FA01 的要求：**真实参数坐标、train/val/test 划分、
轨迹帧数及物理时长**。

数据来源
--------
**本脚本不解析 `init.t3c`。** 参数只有一份来源：
``experiments/protocol/trajectory_parameters.csv``（由
``protocol/export_trajectory_parameters.py`` 生成，来源与坑见
``protocol/PARAMETERS.md``）。图上画的就是那张表里的点。

这么做的原因很实际：参数的提取逻辑只要出现第二份，迟早会有一份悄悄取错列
（最容易踩的是把 `/Plume_bottom` 当成 `T_plume` —— 它恒为 pipe+100 K，
会超出声明的 [2000, 2150]）。

**采样可复现**：plume-2D 的采样脚本在库
（`data/plume1d_corrected/i2vis_source/00_sobol_sample.py`），
`qmc.Sobol(d=10, scramble=True, seed=42).random(128)` 再线性映射到
T∈[2000,2150] K、H∈[160,220] km；`--check` 会重算并断言与 `init.t3c` 一致。

用法::

    python3 -B figures/make_figA01.py            # 出图 + manifest
    python3 -B figures/make_figA01.py --check    # 只打印参数统计
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
# ⚠ 不设这一行，PDF 里全是 Type 3 字体，部分阅读器/投稿系统打不开（同 make_fig01.py）
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

# 发布包：向上四级是 geosim-plume/
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
EXPERIMENTS = PAPER   # 发布包：figures/ 的上一级就是 experiments/

OUT_PDF = os.path.join(HERE, "figA01_parameter_coverage.pdf")
OUT_MANIFEST = os.path.join(HERE, "figA01_manifest.json")

# 逐轨迹参数表由 protocol/ 统一产生 —— 本脚本**只读**，不再自己解析 init.t3c。
# 单一来源的理由见 protocol/PARAMETERS.md：参数的提取逻辑一旦有两份，
# 迟早会有一份悄悄取错列（比如把 /Plume_bottom 当成 T_plume）。
PROTO_CSV = os.path.join(EXPERIMENTS, "protocol", "trajectory_parameters.csv")

# 划分色（与 F01/F02 的模型色不冲突；train 用 MFGNO 主色系）
C = {"train": "#2C6FBB", "val": "#E8A33D", "test": "#B5442E"}

# 有效训练口径：输入帧 0–9、目标帧 10–19，共 20 个有效帧。
# s=11 的 dt=0，没有有效梯度，因此第 21 帧（frame 20）只用于时间轴/边界，
# 不作为训练支持。这里取各子集 t[:, 19] 的最大值（Myr）。
TRAIN_WINDOW = {"plume1d": 16.7433, "plume2": 12.4913}


# --------------------------------------------------------------------------
# 读参数表
# --------------------------------------------------------------------------

def load_params():
    """读 ``protocol/trajectory_parameters.csv``，整理成绘图需要的三块。"""
    if not os.path.exists(PROTO_CSV):
        raise FileNotFoundError(
            "缺 {} —— 先跑 protocol/export_trajectory_parameters.py".format(PROTO_CSV))
    with open(PROTO_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    def sel(sub, sp):
        return [r for r in rows if r["subset"] == sub and r["split"] == sp]

    def T_of(rs):
        return np.array([float(r["T_plume_K"]) for r in rs])

    def H_of(rs):
        return np.array([float(r["H_lith_km"]) for r in rs])

    def t_of(rs):
        return np.array([float(r["t_end_Myr"]) for r in rs])

    p1_tr, p1_va, p1_te = (sel("plume1d", s) for s in ("train", "val", "test"))

    # ⚠ plume2 的 val 在 CSV 里**没有标记**：PARAMETERS.md 只记「test=25 / train=103」。
    #   但 splits.yaml 的 val 段写明它由训练配置的 test_ratio=0.1 自动切出：
    #       data[int(103*0.9):] = 索引 92–102（11 条）
    #   而 `train.ids` 是升序表，其位置 92–102 = [117..127]，与 CSV 的末 11 行
    #   逐一相同 —— 所以按行序切即可还原，**不必**引入对 data_full.py 的新依赖。
    #   切完之后 (a)(b) 两个面板口径一致（都报拆分后的计数）。
    p2_pool = sel("plume2", "train")
    n_tr2 = int(len(p2_pool) * 0.9)
    p2_tr, p2_va = p2_pool[:n_tr2], p2_pool[n_tr2:]
    p2_te = sel("plume2", "test")

    p1d = dict(train=T_of(p1_tr), val=T_of(p1_va), test=T_of(p1_te),
               pool_T=T_of(p1_tr + p1_va), test_T=T_of(p1_te),
               n_pool=len(p1_tr) + len(p1_va))
    p2d = dict(train_T=T_of(p2_tr), train_H=H_of(p2_tr),
               val_T=T_of(p2_va), val_H=H_of(p2_va),
               test_T=T_of(p2_te), test_H=H_of(p2_te),
               train_ids=[r["traj_id"] for r in p2_tr],
               val_ids=[r["traj_id"] for r in p2_va],
               test_ids=[r["traj_id"] for r in p2_te])
    # ⚠ (c) 面板按 **train+val** 合并，不分开画三组（作者 2026-09-26 定）。
    #   实测（figures/_probe_val_span.py，读同一个协议 CSV）：
    #     1-param  train [54.41, 55.24] 中位 54.91 | val [54.41, 55.24] 中位 55.03
    #     2-param  train [51.62, 67.99] 中位 66.87 | val [54.61, 67.88] 中位 66.52
    #   → 两组覆盖**几乎完全重合**（中位差 0.13 / 0.36 Myr）。分成两行会画出
    #     两条几乎重合的带，读者无法区分，纯属冗余。
    #   而且 (c) 画的是"时间覆盖"这个**数据文件本身的属性**（train 与 val 来自
    #   同一个文件 plume1d/2d_train_*），不是"某条轨迹属于哪个划分"，
    #   所以按池子 97 / 103 合并才是对的口径。
    #   ⚠ 但 val 不能凭空消失：(a)(b) 两个面板仍按 train/val/test 三分显示，
    #     且 (b) 的 val 是橙色空心圆。
    dur = {}
    for sub, sp, rs in (("plume1d", "train+val", p1_tr + p1_va),
                        ("plume1d", "test", p1_te),
                        ("plume2", "train+val", p2_tr + p2_va),
                        ("plume2", "test", p2_te)):
        dur[(sub, sp)] = dict(t_end=t_of(rs), n_frame=int(rs[0]["n_frame"]),
                              n_traj=len(rs))
    return p1d, p2d, dur, rows


# 绘图
# --------------------------------------------------------------------------

def draw(p1d, p2d, dur):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.85))
    # ⚠ wspace 0.34 -> 0.62：标签从 "…train" 改成 "…train+val" 后变长，
    #   原间距不够 —— 高清渲染实测 "1-param train+val"/"2-param train+val"
    #   的左侧跨过了 (b) 的右边框（边框 x=1140，标签自 x=1107 起）。
    #   量得标签宽 208 px + 右侧 pad 23 px = 需 231 px，而 wspace=0.46 只给 198 px。
    #   0.62 给 246 px，留 15 px 余量。
    #   ⚠ 不用缩字号解决：本图在论文里按 width=\linewidth 排，已被缩到约 0.77 倍，
    #     5.6pt 到纸上只剩约 4.3pt，再缩就不可读了。
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.30, wspace=0.62)

    # ── (a) plume-1D：1 维参数的分布 ──────────────────────────────────
    ax = axs[0]
    rows = [("train", p1d["train"], 2), ("val", p1d["val"], 1),
            ("test", p1d["test"], 0)]
    for name, v, y in rows:
        ax.plot(v, np.full(v.shape, y), "|", ms=11, mew=1.3, color=C[name],
                label="{} (n={})".format(name, v.size))
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["test", "val", "train"], fontsize=7)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel(r"plume temperature $T$ (K)", fontsize=7)
    ax.set_title("(a) GeoSim-Plume1: 1-param", fontsize=7.6, pad=5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.4, loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.30), handletextpad=0.2, columnspacing=0.7)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)
    ax.text(0.5, -0.42,
            "train+val: one LHS pool (n=100, 97 kept)\n"
            "test: an independent LHS draw (n=12)",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.3,
            color="#555555", linespacing=1.4)

    # ── (b) plume-2D：2 维参数散点 ────────────────────────────────────
    ax = axs[1]
    ax.scatter(p2d["train_T"], p2d["train_H"], s=9, c=C["train"], alpha=0.75,
               linewidths=0, label="train (n={})".format(len(p2d["train_T"])))
    # val：与 (a) 面板同色系的中空圆，口径与 splits.yaml 的 val 段一致
    ax.scatter(p2d["val_T"], p2d["val_H"], s=16, facecolors="none",
               edgecolors=C["val"], linewidths=0.9, marker="o",
               label="val (n={})".format(len(p2d["val_T"])))
    ax.scatter(p2d["test_T"], p2d["test_H"], s=22, facecolors="none",
               edgecolors=C["test"], linewidths=0.9, marker="s",
               label="test (n={})".format(len(p2d["test_T"])))
    ax.set_xlabel(r"plume temperature $T$ (K)", fontsize=7)
    ax.set_ylabel(r"lithosphere thickness $H$ (km)", fontsize=7)
    ax.set_title("(b) GeoSim-Plume2: 2-param", fontsize=7.6, pad=5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.4, loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.30), handletextpad=0.2, columnspacing=0.7)
    ax.grid(ls=":", lw=0.4, alpha=0.6)
    allT = np.concatenate([p2d["train_T"], p2d["val_T"], p2d["test_T"]])
    allH = np.concatenate([p2d["train_H"], p2d["val_H"], p2d["test_H"]])
    ax.text(0.5, -0.42,
            r"$T$: {:.1f} $\pm$ {:.1f} K, $H$: {:.1f} $\pm$ {:.1f} km (N={})"
            .format(allT.mean(), allT.std(), allH.mean(), allH.std(), allT.size)
            + "\nSobol d=10, 8 dims reserved; seed=42",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.3,
            color="#555555", linespacing=1.4)

    # ── (c) 时间覆盖：逐轨迹从 0 到 t_end ────────────────────────────
    # 轨迹数（97）远多于可分辨的行高，所以逐条画细线 + 小抖动：这样既给出
    # 每条模拟的终点，也能一眼看出**样本内的离散度**（plume2 的 t_end 跨 19 Myr，
    # plume1d 只有 0.8 Myr）。行标签里带上"轨迹数 × 帧数"，避免在图内塞文字。
    ax = axs[2]
    # ⚠ 标签写 "train+val"，对应 dur 里的合并键（见 load_params 的注释）：
    #   分开画会得到两条几乎重合的带
    order = [("plume1d", "train+val", 3), ("plume1d", "test", 2),
             ("plume2", "train+val", 1), ("plume2", "test", 0)]
    # ⚠ **按 y 存标签，不要按循环顺序 append 后再喂 set_yticklabels**。
    #   原实现是 `ylabels.append(...)`，而循环顺序是 y=[3,2,1,0]、
    #   `set_yticklabels` 却按 tick=[0,1,2,3] 铺 —— **四行标签整体上下颠倒**，
    #   实测顶部画的是 1-param train(87 条、止于 55 Myr) 却标成 "2-param test"。
    #   改成 dict 后，取值顺序显式写死，不可能再错位。
    ylabels = {}
    for sub, sp, y in order:
        d = dur[(sub, sp)]
        te = np.sort(d["t_end"])
        yy = y + (np.linspace(-0.24, 0.24, te.size) if te.size > 1 else 0.0)
        # "train+val" 用 train 的颜色（合并池 = 训练文件本身）
        ax.hlines(yy, 0.0, te, color=C[{("train+val"): "train"}.get(sp, sp)],
                  lw=0.5, alpha=0.75)
        # 两行写：长标签会溢出到坐标区内（已踩过），换成窄标签
        ylabels[y] = "{}\n{} traj x {}".format(
            # ⚠ 写 1-param/2-param，**不写 1-D/2-D**：后者会被误读成空间维数，
            #   而两个子集都是二维空间模拟，区别只在**变参数个数**（作者 2026-09-26 定）。
            {"plume1d": "1-param", "plume2": "2-param"}[sub]
            + {"train+val": " train+val", "test": " test"}[sp],
            te.size, d["n_frame"])
    # 有效训练支持为前 20 帧 → 物理时间上限（虚线只在两组内部有区别）
    #
    # ⚠ 必须用 **vlines 分段画**，不能用 axvline。
    #   原实现是 `for y, lim in ((2.5, w1), (0.5, w2)): ax.axvline(lim, ...)` ——
    #   `y` 解包了却**从未使用**，而 axvline 是贯穿整个轴的：既丢掉了"这条线只管
    #   对应那两行"的原意，又让它从顶部的说明文字里穿过去（作者 2026-09-26 指出重叠）。
    #   分段后：1-param 的线只覆 y=2,3 两行，2-param 的只覆 y=0,1 两行。
    w1, w2 = TRAIN_WINDOW["plume1d"], TRAIN_WINDOW["plume2"]
    for lim, y0, y1 in ((w1, 1.70, 3.30), (w2, -0.30, 1.30)):
        ax.vlines(lim, y0, y1, color="#333333", lw=0.8, ls="--", zorder=5)
    # ⚠ 这段说明**必须放在带状区上方**。原实现放在 (26.0, 2.05)、四行带长箭头，
    #   正好压在 1-param test 那一行的轨迹线上（作者 2026-09-26 指出重叠）。
    #   现在压成两行、去掉箭头，并给 ylim 顶部留出空档（见下方 set_ylim）。
    #   两个数值与两条虚线的 x 位置一一对应，读者按数值即可对上，不需要箭头指。
    ax.text(-0.5, 3.85,
            "effective training window (first 20 frames)\n"
            "1-param: {:.2f} Myr     2-param: {:.2f} Myr".format(w1, w2),
            fontsize=5.2, color="#333333", linespacing=1.5,
            ha="left", va="center")
    ax.set_yticks([0, 1, 2, 3])
    # 显式按 tick 顺序取，与上面 order 的书写顺序解耦（见 ylabels 的注释）
    ax.set_yticklabels([ylabels[y] for y in (0, 1, 2, 3)],
                       fontsize=5.6, linespacing=1.3)
    # 顶部留出注释行的空档：最上面那条带 jitter 后到 3.24，注释两行中心在 3.72，
    # 半高约 0.36 → 下半缘 3.36，与带顶留 0.12 的余量。改 ylim 前先算一遍，
    # 别凭感觉调（这个面板已经因为"调数字躲重叠"踩过一次）。
    ax.set_ylim(-0.65, 4.30)
    ax.set_xlim(-2, 76)
    ax.set_xlabel("physical time from $t=0$ (Myr)", fontsize=7)
    ax.set_title("(c) per-trajectory time coverage", fontsize=7.6, pad=5)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)

    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只打印参数统计")
    a = ap.parse_args()

    p1d, p2d, dur, rows = load_params()

    print("数据源: {}（{} 行）".format(
        os.path.relpath(PROTO_CSV, PAPER), len(rows)))
    print("plume1d: train {} val {} test {}".format(
        p1d["train"].size, p1d["val"].size, p1d["test"].size))
    print("  pool T: {:.2f} ± {:.2f}  [{:.2f}, {:.2f}]".format(
        p1d["pool_T"].mean(), p1d["pool_T"].std(),
        p1d["pool_T"].min(), p1d["pool_T"].max()))
    print("  test T: {:.2f} ± {:.2f}  [{:.2f}, {:.2f}]".format(
        p1d["test_T"].mean(), p1d["test_T"].std(),
        p1d["test_T"].min(), p1d["test_T"].max()))
    allT = np.concatenate([p2d["train_T"], p2d["test_T"]])
    allH = np.concatenate([p2d["train_H"], p2d["test_H"]])
    print("plume2 : train {} test {}".format(p2d["train_T"].size,
                                             p2d["test_T"].size))
    print("  T: {:.2f} ± {:.2f}  [{:.2f}, {:.2f}]".format(
        allT.mean(), allT.std(), allT.min(), allT.max()))
    print("  H: {:.2f} ± {:.2f}  [{:.2f}, {:.2f}]".format(
        allH.mean(), allH.std(), allH.min(), allH.max()))
    for k in sorted(dur):
        d = dur[k]
        print("  {:16s} {:>4} traj x {:>3} frames, t_end {:.2f}–{:.2f} Myr".format(
            "{} {}".format(*k), d["n_traj"], d["n_frame"],
            d["t_end"].min(), d["t_end"].max()))
    if a.check:
        return

    draw(p1d, p2d, dur)
    manifest = {
        "figure": "FA01",
        "file": os.path.basename(OUT_PDF),
        "role": "real parameter coordinates, split assignment, frame count and "
                "physical duration per trajectory",
        "data_source": {
            "table": os.path.relpath(PROTO_CSV, PAPER),
            "provenance": "experiments/protocol/PARAMETERS.md",
            "generator": "experiments/protocol/export_trajectory_parameters.py",
            "note": "本图不自行解析 init.t3c —— 参数只有一份来源，避免提取逻辑"
                    "分叉后取错列",
            "sampling_reproducible": True,
            "sampling_script": "data/plume1d_corrected/i2vis_source/"
                               "00_sobol_sample.py (Sobol, seed=42, d=10, scrambled)",
        },
        "plume1d": {
            "n_train": int(p1d["train"].size), "n_val": int(p1d["val"].size),
            "n_test": int(p1d["test"].size),
            "pool_T_mean_K": float(p1d["pool_T"].mean()),
            "pool_T_std_K": float(p1d["pool_T"].std()),
            "test_is_independent_draw": True,
        },
        "plume2": {
            "n_train": int(p2d["train_T"].size), "n_val": int(p2d["val_T"].size),
            "n_test": int(p2d["test_T"].size),
            "pool_size": int(p2d["train_T"].size + p2d["val_T"].size),
            "val_source": "splits.yaml: data[int(103*0.9):] = rows 92-102 (11 traj); "
                          "the protocol CSV carries no val label for plume2",
            "T_mean_K": float(allT.mean()), "T_std_K": float(allT.std()),
            "H_mean_km": float(allH.mean()), "H_std_km": float(allH.std()),
            "train_ids": p2d["train_ids"], "val_ids": p2d["val_ids"],
            "test_ids": p2d["test_ids"],
        },
        "time_coverage": {
            "{} {}".format(*k): {
                "n_traj": int(v["n_traj"]), "n_frame": int(v["n_frame"]),
                "t_end_min_Myr": float(v["t_end"].min()),
                "t_end_max_Myr": float(v["t_end"].max()),
            } for k, v in dur.items()
        },
        "training_window_Myr": dict(TRAIN_WINDOW),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\n→ {}".format(OUT_PDF))
    print("→ {}".format(OUT_MANIFEST))


if __name__ == "__main__":
    main()
