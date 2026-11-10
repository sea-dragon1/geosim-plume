#!/usr/bin/env python3
# coding=utf-8
"""生成 ``figures/fig01_benchmark_overview.pdf`` —— GeoSim-Plume 数据集总览。

**只画真值场，不涉及任何模型预测**，所以不需要 GPU，也不受推理管线影响。

本图只用一个子集（\pone）的**一条预先声明的轨迹**，六个观测通道齐出。

为什么要这样改（2026-09-25 作者定）：
  · **不要两个子集** —— 两行六列挤在 7.2 英寸里，每格再挂一条竖 colorbar，
    刻度文字会压到相邻面板。而且总览图不需要靠并排比较来传达信息。
  · **不要 frame 0** —— 初始帧的速度场**恒为零**（实测非零占比 0.0%），
    `vx`/`vy` 两格是纯空白，占了 1/3 的画幅却没有任何信息。
    默认改用**第一个预测窗的末帧（帧 19）**：协议上有确定含义
    （10→10 窗口的终点），且速度场已充分发展。
  · **不要底排的时间 protocol 条** —— 那段在正文与图注里已经讲清楚了，
    在这里只是重复，还把图撑高。
  · **不要底部脚注小字行** —— 通道全名、坐标单位、归一化口径统一写进
    LaTeX 图注，图内不再重复一遍（也不再占一行高度）。
  · **轴标 0–1，不标 km**（作者 2026-09-26 定）—— 实测依据（2026-09-26，
    脚本 ``figures/_check_coords.py``，CPU 作业 3589011/3589017/3589023/3589034）：
      ‣ 128×128 的 ``x/y-coordinate`` 各 128 个值，而场轴长度也是 128
        → 与场一一对应，只能是**单元中心**（边界要有 129 个）。
      ‣ 它们在索引方向**完全均匀**（相邻间隔 max/min = 1.0000，与 linspace
        的最大偏差 4e-8 = float32 舍入），所以场形**没有**被拉伸 ——
        用线性 extent 是安全的。
      ‣ 但实测范围是 x ∈ [2.508, 1147.492] km、y ∈ [2.084, 658.487] km，
        按中心还原外边界得 x ∈ [−2.000, 1152.000]、y ∈ [−0.500, 661.071] km，
        **比标称域 1150 × 660 宽 4.0 / 1.57 km**（0.35% / 0.24%）。
      ‣ 剖面配准（原生场按「归一化 × 域长」内插到 128 坐标上）给出
        y 轴 r = 0.99958、反演最佳 scale x = 1.0000 —— 说明乘域长这个
        线性映射本身成立，**错的是端点/跨度，不是比例**。
      ‣ 结论：km 口径「差一点点但不对」，而这点误差（最坏 ±2.5 km）与正文
        F03 剖面位移结论（"at most 5.2 km"）同量级 —— 标 0–1 不承担这个风险，
        域的物理尺寸由 LaTeX 图注给出。
  · **colorbar 横放**在每个面板下方，从根本上消除刻度与邻格的挤压。

**色标**（``figures/README.md`` 冻结）：``ro/pr/nu → viridis``、
``tk → hot``、``vx/vy → RdBu_r``（``TwoSlopeNorm(vcenter=0)``）。
``imshow(field.T, extent=(0, 1, 1, 0), origin='upper')``，**再加**
``ax.set_aspect(660/1150)`` 把面板钉成物理域的 1150:660（**不能用**
``aspect='equal'`` —— 单位方格下那会变成正方形，见 ``ASPECT`` 处的注释）。

**轴用归一化 0–1，不用 km**（作者 2026-09-26 定）。理由是实测出来的，列在下方
「为什么要这样改」的倒数第二条，脚本在 ``figures/_check_coords.py``。

⚠ **这里的场是归一化值**（发布口径），不是物理量 —— 图注与 manifest 都写明，
避免把归一化幅值误读成 kg/m^3 或 K。

⚠ ``pdf.fonttype = 42``（TrueType）**必须设**：matplotlib 默认是 3，
产出的 PDF 里全是 **Type 3** 字体，部分阅读器与投稿系统会打不开或渲染成空白。
本仓库另有六张图也踩过同一个坑（见 ``figures/README.md``）。

用法::

    python -B make_fig01.py
    python -B make_fig01.py --frame 19 --traj 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import TwoSlopeNorm

# ⚠ 见文件头：不设这一行，产出的 PDF 会全是 Type 3 字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# 发布包：figures/ 在 code/experiments/ 下，向上四级是 geosim-plume/
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = ["ro", "pr", "tk", "nu", "vx", "vy"]
# 一行六格很窄，标题只放符号；通道全名在图注里给。
LABELS = {"ro": r"$\rho$", "pr": r"$P$", "tk": r"$T$",
          "nu": r"$\log_{10}\eta$", "vx": r"$v_x$", "vy": r"$v_y$"}
CMAP = {"ro": "viridis", "pr": "viridis", "tk": "hot", "nu": "viridis",
        "vx": "RdBu_r", "vy": "RdBu_r"}
DIVERGING = ("vx", "vy")
DOMAIN_KM = (1150.0, 660.0)

# ⚠ 面板必须按物理域 1150:660 画（作者 2026-09-26 定），**不能**用 aspect="equal"。
#   本图的 extent 是 (0,1,1,0)（归一化单位方格），equal 会把它变成**正方形**——
#   而 [0,1]² 的 1.0 在 x 上是 1150 km、在 y 上只有 660 km，画成方的等于宣称
#   两个方向尺度相同。matplotlib 里 aspect 是「y 单位与 x 单位的显示比」，
#   单位方格下 框宽/框高 = 1/aspect，所以要 1150/660 就得 aspect = 660/1150。
#   这同时让本图与 fig03/fig03c（已按 km 的 extent + aspect="equal" 画成 1.742）
#   一致 —— 同一个温度场出现在两张图里，形状必须一样。
ASPECT = DOMAIN_KM[1] / DOMAIN_KM[0]

SUBSET = "plume1d"
# ⚠ 数据集命名（作者 2026-09-25 定）：**GeoSim-Plume1** / **GeoSim-Plume2**，
#   与论文宏 `\dataset` / `\pone` 已统一（paper_macros.tex），不再有大小写分歧。
#   但 `SUBSET` 与 `DATA_PATH` 用的是**磁盘上的旧标签/文件名**，不随命名走。
SUBSET_LABEL = "GeoSim-Plume1"
DATA_PATH = REPO + "/data/plume1d_corrected/plume1d_test_ds_128x128_phys1e6_first21.hdf5"

# 第一个预测窗的末帧。帧 0-9 是输入窗、10-19 是输出窗，所以 19 是它的终点。
# **不要改回 0** —— 那一帧速度恒为零（见文件头）。
DEFAULT_FRAME = 19


def load_one(path, traj, frame):
    with h5py.File(path, "r") as f:
        missing = [c for c in FIELDS if c not in f]
        if missing:
            raise KeyError("{} 缺字段 {}".format(path, missing))
        fields = {c: np.asarray(f[c][traj, frame], dtype=np.float32) for c in FIELDS}
        t = np.asarray(f["t-coordinate"][traj], dtype=np.float64) if "t-coordinate" in f else None
        x = np.asarray(f["x-coordinate"], dtype=np.float64)
        y = np.asarray(f["y-coordinate"], dtype=np.float64)
    return fields, t, x, y


def _unit_ticks(ax, show_x, show_y):
    """标**归一化单位方格**（0 与 1），不换算成 km。

    ⚠ **作者 2026-09-26 定：这张图的轴就用 0–1，不用物理 km。**
    理由是实测出来的（见下方「实测」注释）：128×128 文件的坐标数组
    并非「域长 1150 × 660 km」的严格线性刻度，把它乘域长标成 km 是在
    断言一个不成立的口径。归一化方格是这个文件自己的坐标定义，直接标 0–1
    才是诚实的；域的真实尺寸 1150 × 660 km 写在 LaTeX 图注里，不在轴上。

    ⚠ **这里刻意不设 ``set_xlabel`` / ``set_ylabel``**：横 colorbar 挂在面板
    下方，而轴标签默认也贴在轴下缘，两者会印在同一处；把 ``labelpad`` 调大去
    躲它又会撞进 colorbar 里（实测两轮都重叠）—— 靠调数字让两个独立定位的
    元素错开本身就是脆的，换个画幅尺寸就会再犯。
    单位**只在 LaTeX 图注里写一次**，图内不依赖任何对齐。
    """
    # ⚠ 一行六格时每格只有约 1.0 英寸宽，**只标两端**。
    #   标三个数时相邻两格的末值与首值会连成一个连续刻度串（实测）。
    if show_x:
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "1"], fontsize=5)
        ax.tick_params(axis="x", pad=1, length=2)
    else:
        ax.set_xticks([])
    if show_y:
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["0", "1"], fontsize=5)
        ax.tick_params(axis="y", pad=1, length=2)
    else:
        ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description="生成 fig01_benchmark_overview.pdf")
    ap.add_argument("--traj", type=int, default=0,
                    help="预先声明的示例轨迹（默认 0；不按误差挑选）")
    ap.add_argument("--frame", type=int, default=DEFAULT_FRAME,
                    help="帧序号。默认 %d = 第一个预测窗(10-19)的末帧；"
                         "**不要用 0**，那一帧速度恒为零、速度场是空白" % DEFAULT_FRAME)
    a = ap.parse_args()

    fields, t, x, y = load_one(DATA_PATH, a.traj, a.frame)

    # **一行六格**（作者 2026-09-25 定）。横 colorbar 放各面板下方：
    # 单行没有「下一行」会被压到，而竖排 colorbar 在 1.0 英寸宽的格子里
    # 必然把刻度文字挤到右邻格（这正是原先那版重叠的来源）。
    # ⚠ 高度 1.60（原 2.3）是配合下面的 set_aspect 算出来的，两个数必须一起改。
    #   set_aspect 会把面板框收缩到 1.742，槽位若仍按 2.3 的高度留，
    #   面板上方/下方会空掉一大片。探针实测 figH=1.60 时面板 0.937 x 0.538 in，
    #   正好填满列宽，槽位余量刚好够放横 colorbar 及其刻度。
    fig = plt.figure(figsize=(7.2, 1.60))
    gs = fig.add_gridspec(1, 6, wspace=0.26,
                          left=0.045, right=0.995, top=0.80, bottom=0.30)

    for i, c in enumerate(FIELDS):
        ax = fig.add_subplot(gs[0, i])
        v = fields[c]
        if c in DIVERGING:
            m = float(np.abs(v).max()) or 1.0
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)
            im = ax.imshow(v.T, extent=(0, 1, 1, 0),
                           origin="upper", aspect="auto", cmap=CMAP[c], norm=norm)
            ax.set_aspect(ASPECT)
        else:
            im = ax.imshow(v.T, extent=(0, 1, 1, 0),
                           origin="upper", aspect="auto", cmap=CMAP[c])
            ax.set_aspect(ASPECT)
        _unit_ticks(ax, show_x=True, show_y=(i == 0))
        ax.set_title(LABELS[c], fontsize=8, pad=2)
        # 横 colorbar 挂在下方；pad 要让开 x 刻度标签（0/1）那一行，
        # 否则 colorbar 的刻度数会和它们印在一起（2×3 那版实测踩过）。
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.075, pad=0.20, aspect=13)
        cb.ax.tick_params(labelsize=5, pad=1, length=1.5)
        cb.outline.set_linewidth(0.4)
        cb.locator = matplotlib.ticker.MaxNLocator(nbins=3)
        cb.update_ticks()

    t_str = ("{:.2f}".format(t[a.frame])) if t is not None else "?"
    fig.suptitle(
        r"{}, six observed fields — test trajectory {}, frame {} ($t={}$ Myr)"
        .format(SUBSET_LABEL, a.traj, a.frame, t_str), fontsize=9, y=0.955)
    # ⚠ **不画脚注行**（作者 2026-09-25 定）。原先那行小字承担两件事：
    #   坐标单位与场值量纲 —— 现在**全部搬进 LaTeX 图注**（见 01_introduction.tex
    #   的 fig:overview），由正文列出六个通道名与「归一化单位、km 坐标」。
    #   两者只留一处，避免图里图外重复同一句话。
    #
    #   图高不会因此变高：gridspec 的 top/bottom 一个数字都没动，面板几何与
    #   改前逐位相同；腾出的底部空白由 ``bbox_inches="tight"`` 自动裁掉，
    #   产出 PDF 只会**更矮**。

    out = os.path.join(HERE, "fig01_benchmark_overview.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("写出:", out)

    manifest = {
        "figure": "fig01_benchmark_overview.pdf",
        "subset": SUBSET,
        "file": os.path.relpath(DATA_PATH, REPO),
        "traj": a.traj, "frame": a.frame,
        "t_Myr": float(t[a.frame]) if t is not None else None,
        "n_frames_in_file": int(t.shape[0]) if t is not None else None,
        "t_range_Myr": [float(t.min()), float(t.max())] if t is not None else None,
        "frame_choice_rule": "last frame of the first prediction window (10-19); "
                             "frame 0 is excluded because its velocity field is identically zero",
        "fields": FIELDS, "units": "normalised (published convention)",
        "colormap": CMAP,
        "axes": "normalised 0-1 on both axes (NOT km); extent=(0,1,1,0)",
        "physical_domain_km_not_plotted": DOMAIN_KM,
        "coordinate_range_normalised": {
            "x": [float(np.min(x)), float(np.max(x))],
            "y": [float(np.min(y)), float(np.max(y))],
            "note": "measured from the file itself; this is why the km axis was "
                    "dropped — the array does not span [0,1] exactly",
        },
        "why_not_km": "the 128x128 coordinate arrays are cell centres whose span "
                      "is 0.35%/0.24% wider than the nominal domain; plotting them "
                      "as km would assert a calibration that does not hold exactly",
        "colorbar": "horizontal, below each panel",
        "footer": "none — channel names, coordinate unit and normalisation "
                  "convention live in the LaTeX caption (fig:overview)",
        "fonttype_setting": 42,
        "field_range_normalised": {c: [float(fields[c].min()), float(fields[c].max())]
                                   for c in FIELDS},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    mp = os.path.join(HERE, "fig01_manifest.json")
    with open(mp, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("写出:", mp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
