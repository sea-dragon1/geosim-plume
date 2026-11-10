#!/usr/bin/env python3
# coding=utf-8
"""E04 空间诊断出图：``fig03_field_predictions.pdf``（正文）与速度剖面图。

**图的数据纪律**（``E04_diagnostics_cost/PLAN.md`` 与 ``figures/README.md``）：

- 模型间**同一轨迹、同一时刻、同一单位、同一色标**；
- 轨迹与时刻**预先声明**（这里：``--traj`` 与 ``--frame`` 的默认值），
  不按模型的误差事后挑选；
- **同时展示失败案例**（误差最大的轨迹，规则写死在代码里并记入 manifest）；
- 剖面位置**预先声明**（``x* = 0.5``，即横向中点），幅值误差与位置偏移**分开**报告；
- 每个 panel 的来源（data version / traj / frame / checkpoint）写入 manifest，
  **不手工修改图中数值**。

**色标**（``figures/README.md``）：``ro/pr/nu → viridis``、``tk → hot``、
``vx/vy → RdBu_r``（``TwoSlopeNorm(vcenter=0)``）；误差图**沿用对应物理场的 cmap**。
速度取模 $|v|=\sqrt{v_x^2+v_y^2}$（无符号，用 viridis），因为 $v_x/v_y$ 的发散色标
不适合画幅值。

用法::

    python -B plot_diagnostics.py --subset plume1d
    python -B plot_diagnostics.py --subset plume1d --traj 0 --frame 9
"""
from __future__ import annotations

import argparse
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

import numpy as np                              # noqa: E402

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
# ⚠ 不设这一行，PDF 里全是 Type 3 字体，部分阅读器/投稿系统打不开（同 make_fig01.py）
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt                 # noqa: E402

import eval_metrics as em                       # noqa: E402
import paths                                    # noqa: E402

FIGDIR = os.path.join(E04_ROOT, "figures")
RESULTS = os.path.join(E04_ROOT, "results")

# 物理域（km）—— 与数据一致：x ∈ [0, 1150]，y ∈ [0, 660]
DOMAIN_KM = (1150.0, 660.0)

# 模型色（figures/README.md 冻结）
COLORS = {"FNO": "#C44E52", "GNOT": "#55A860", "Independent MPNN": "#56B4E9",
          "Transolver": "#CCB974", "MFGNO": "#3D2670", "Persistence": "#888888"}

# 主表口径的 run（plume1d/plume2 由 --subset 决定）
MODEL_RUNS = {
    "FNO":            ("E01_main",     "2026-09-17-fno-E01-{subset}-seed0"),
    "GNOT":           ("E01_main",     "2026-09-17-gnot-E01-{subset}-seed0"),
    "Transolver":     ("E01_main",     "2026-09-17-transolver-E01-{subset}-seed0"),
    "Independent MPNN": ("E02_ablation", "2026-09-17-mpnn-E02-A-{subset}-seed0"),
    "MFGNO":          ("E02_ablation", "2026-09-17-mfgno-E02-C-equal-{subset}-seed0"),
}
MODEL_ORDER = ["Truth", "FNO", "GNOT", "Independent MPNN", "Transolver", "MFGNO"]

IDX = {c: i for i, c in enumerate(em.FIELD_ORDER)}     # ro0 pr1 tk2 nu3 vx4 vy5


# --------------------------------------------------------------------- 读数据
def load_pred(stage, exp_id):
    import h5py
    run = os.path.join(EXPERIMENTS, stage, "runs", exp_id)
    for name in ("predictions.hdf5", "predictions.npz"):
        p = os.path.join(run, "eval", name)
        if not os.path.exists(p):
            continue
        if name.endswith(".hdf5"):
            with h5py.File(p, "r") as f:
                pred = np.stack([np.asarray(f[c], dtype=np.float32)
                                 for c in em.FIELD_ORDER], axis=-1)
                truth = np.stack([np.asarray(f["truth_" + c], dtype=np.float32)
                                  for c in em.FIELD_ORDER], axis=-1)
        else:
            d = np.load(p)
            pred = np.stack([d[c] for c in em.FIELD_ORDER], axis=-1)
            truth = np.stack([d["truth_" + c] for c in em.FIELD_ORDER], axis=-1)
        return pred, truth, os.path.relpath(p, EXPERIMENTS)
    return None, None, None


def gather(subset):
    """返回 ``{模型: (pred, truth, 相对路径)}``，缺的模型不在字典里。"""
    out = {}
    for disp in MODEL_ORDER[1:]:
        stage, tmpl = MODEL_RUNS[disp]
        exp_id = tmpl.format(subset=subset)
        if not os.path.isdir(os.path.join(EXPERIMENTS, stage, "runs", exp_id)):
            continue
        pred, truth, rel = load_pred(stage, exp_id)
        if pred is None:
            print("  [跳过] {} —— 还没有 predictions.hdf5（先跑 eval_per_traj --dump-fields）"
                  .format(disp))
            continue
        out[disp] = (pred, truth, rel)
    return out


# --------------------------------------------------------------------- 出图
FIELD_CMAP = {"tk": "hot", "speed": "viridis"}
FIELD_LABEL = {"tk": r"$T$", "speed": r"$|v|$"}


def _field_of(arr, which):
    """``which ∈ {tk, speed}`` → 二维场。

    速度取模 $|v|=\\sqrt{v_x^2+v_y^2}$：$v_x/v_y$ 是有符号量、用发散色标，
    幅值用它会把零点位置误读成结构，所以幅值单独用 sequential 色标。
    """
    if which == "tk":
        return arr[..., IDX["tk"]]
    vx, vy = arr[..., IDX["vx"]], arr[..., IDX["vy"]]
    return np.sqrt(vx ** 2 + vy ** 2)


def plot_f03(data, subset, traj, frame, out_pdf):
    """温度与速度场：真值 / 各模型预测 / 误差，同一轨迹同一帧。"""
    rows = [("tk", "pred"), ("tk", "err"), ("speed", "pred"), ("speed", "err")]
    # 只画**实际有预测**的模型：缺一个不能让整张图崩，但要能从 manifest 看出来
    cols = [m for m in MODEL_ORDER if m == "Truth" or m in data]
    ncol = len(cols)
    # ⚠ 高度从 6.2 降到 3.40，是配合下面的 aspect="equal" 一起算出来的，
    #   两个数**必须一起改**（见 imshow 处的注释）。
    #   面板宽 ~0.907 in（由 figW/ncol 决定，与 ncol 无关）→ 域要 1.742 就得
    #   0.521 in 高；探针实测 figH=3.40 时面板 0.921 x 0.529，正好填满槽位。
    #   若只改 aspect 不降高度，面板会缩到 0.907 x 0.521 而槽位仍有 1.316 高，
    #   行间留下大片空白。
    fig, axes = plt.subplots(4, ncol, figsize=(1.55 * ncol, 3.40))

    truth = next(iter(data.values()))[1]
    for r, (which, kind) in enumerate(rows):
        # 统一色标：预测/真值一组，误差一组，均取自**同行全部模型**的极值
        vals = []
        for disp in cols:
            arr = truth if disp == "Truth" else data[disp][0]
            vals.append(_field_of(arr[traj, frame], which))
        if kind == "pred":
            lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
            vmin, vmax = lo, hi
        else:
            errs = []
            for disp in cols[1:]:
                e = np.abs(_field_of(data[disp][0][traj, frame], which)
                           - _field_of(truth[traj, frame], which))
                errs.append(e)
            vmin, vmax = 0.0, max(e.max() for e in errs)
        cmap = FIELD_CMAP[which]

        for c, disp in enumerate(cols):
            ax = axes[r, c]
            if kind == "pred":
                v = vals[c]
            elif disp == "Truth":
                v = None
            else:
                v = np.abs(_field_of(data[disp][0][traj, frame], which)
                           - _field_of(truth[traj, frame], which))
            if v is None:
                ax.axis("off")
                continue
            # ⚠ aspect="equal"，**不是** "auto"（作者 2026-09-26 定）。
            #   "auto" 的含义是"填满坐标框"，画出来的长宽比 = 坐标框的长宽比，
            #   与物理域的 1150/660 = 1.742 毫无关系。实测旧版坐标框是
            #   0.907 x 1.316 in（w/h = 0.690）→ 场被**纵向拉伸 2.53 倍**：
            #   x 约 7.8 km/px、y 约 3.1 km/px，柱头这类各向同性结构会变成
            #   竖长椭圆，图上任何角度/长宽比都不可信。
            #   "equal" 让 1 km 在 x 与 y 上占同样像素，自动画成横宽矩形。
            #   ⚠ 冻结规范 figures/README.md 里写的也是 "auto"，那里一并要改。
            im = ax.imshow(v.T, extent=(0, DOMAIN_KM[0], DOMAIN_KM[1], 0),
                           origin="upper", aspect="equal", cmap=cmap,
                           vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(disp, fontsize=8)
            if c == 0:
                lbl = (r"$T$" if which == "tk" else r"$|v|$")
                lbl += "  truth" if kind == "pred" else "  $|$err$|$"
                ax.set_ylabel(lbl + "\n" + (r"$y$ [km]" if r % 2 else ""), fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    # ⚠ 写 "(common 10->10 protocol)"，不写 "(frozen protocol)"（作者 2026-09-26 定）：
    #   "frozen" 会被读成"模型权重被冻结/未训练"，而这里指的是**推理协议**——
    #   所有模型共用同一个 10→10 单窗协议。
    fig.suptitle(r"%s, trajectory %d, predicted frame %d (common 10$\rightarrow$10 protocol)"
                 % ({"plume1d": "GeoSim-Plume1", "plume2": "GeoSim-Plume2"}[subset], traj, frame),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("  写出:", out_pdf)


def plot_profile(data, subset, traj, frame, out_pdf, x_star=0.5):
    """横向中点 ``x* = 0.5`` 的垂向剖面：``v_x`` 与 ``T`` vs ``y``，并报位置偏移。

    剖面位置**预先声明**（``x* = 0.5``，落在 128 网格的 63/64 之间 → 两点平均），
    不按模型误差事后挑选。
    """
    truth = next(iter(data.values()))[1]
    n = truth.shape[2]
    # x* = 0.5 恰落在 63/64 界面上 → 取两点平均（PLAN 的实测结论）
    i0, i1 = n // 2 - 1, n // 2
    ys = np.linspace(0, DOMAIN_KM[1], n)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, which in zip(axes, ("tk", "speed")):
        t = _field_of(truth[traj, frame], which)[i0:i1 + 1].mean(axis=0)
        ax.plot(t, ys, "k-", lw=1.8, label="Truth", zorder=5)
        for disp in MODEL_ORDER[1:]:
            if disp not in data:
                continue
            p = _field_of(data[disp][0][traj, frame], which)[i0:i1 + 1].mean(axis=0)
            ax.plot(p, ys, lw=1.1, color=COLORS[disp], label=disp)
        ax.set_xlabel(r"$T$" if which == "tk" else r"$|v|$", fontsize=8)
        ax.set_ylabel(r"$y$ [km]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, loc="best")
    fig.suptitle(r"Vertical profile at $x^\ast=0.5$ (traj %d, frame %d)" % (traj, frame),
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("  写出:", out_pdf)


def profile_errors(data, frame, trajs=None, x_star=0.5):
    """逐模型逐轨迹：剖面幅值误差与**位置偏移**（两者必须分开报）。

    ``trajs=None`` 表示**全部测试轨迹** —— ``PLAN.md`` 要求剖面诊断必须
    "对全部测试轨迹汇总，不能只展示一张好看的图"，所以默认就是全量；
    单条轨迹的行只是其中的一部分，不是替代。
    """
    truth = next(iter(data.values()))[1]
    n = truth.shape[2]
    N = truth.shape[0]
    if trajs is None:
        trajs = list(range(N))
    i0, i1 = n // 2 - 1, n // 2
    # ⚠⚠ 这里的 ys 是**均匀假设**，不是文件里的真实坐标 —— 它决定
    #   ``peak_shift_km`` 的换算，而论文 §5.2 引用的 5.2 / 99 / 104 km
    #   全部由它乘出来。三个候选步长互不相等：
    #       linspace(0,660,128) → 660/127 = 5.197 km  ← 当前用的
    #       全域均匀铺开         → 660/128 = 5.156 km
    #       实测坐标间距         → (0.997708-0.003158)/127×660 = 5.169 km ← 最接近真值
    #   128 网格的 y ∈ [0.003158, 0.997708] 是等面积**单元边界**、非均匀，
    #   所以「位移 = 索引差 × 常数」本身只是近似：真实位移取决于峰值落在
    #   轴的哪一段。正确做法是读数据文件的 ``y-coordinate`` 换算成 km；
    #   但那会改变已写入论文与 ``results/profile_errors_summary.csv`` 的数字，
    #   必须**连同 CSV 一起重跑**才有意义（需 ``E04/runs/*/eval/predictions.hdf5``，
    #   该目录当前为空）。故此处**不改数值**，只把口径写明，避免下一个人照抄。
    #
    #   另注：当前所有位移都是 5.1969 km 的整数倍，即**量化到网格**的结果 ——
    #   5.2 km 实为「1 格」。论文里应同时给出网格步长，否则读者看不出这一点。
    ys = np.linspace(0, DOMAIN_KM[1], n)
    rows = []
    for traj in trajs:
        for which in ("tk", "speed"):
            t = _field_of(truth[traj, frame], which)[i0:i1 + 1].mean(axis=0)
            t_peak = ys[int(np.argmax(np.abs(t)))]
            t_amp = float(np.max(np.abs(t)))
            for disp in MODEL_ORDER[1:]:
                if disp not in data:
                    continue
                p = _field_of(data[disp][0][traj, frame], which)[i0:i1 + 1].mean(axis=0)
                amp = float(np.max(np.abs(p)))
                shift = abs(ys[int(np.argmax(np.abs(p)))] - t_peak)
                rows.append({"traj": traj, "field": which, "model": disp,
                             "amp_pred": amp, "amp_true": t_amp,
                             "amp_rel_err": (amp - t_amp) / t_amp,
                             "peak_shift_km": shift})
    return rows


def summarize_profile(rows):
    """把逐轨迹剖面误差压成逐模型汇总：中位数与 [min, max]，不藏分布。"""
    out = {}
    for r in rows:
        key = (r["field"], r["model"])
        out.setdefault(key, []).append(r)
    lines = []
    for (field, model), rs in sorted(out.items()):
        amp = np.array([x["amp_rel_err"] for x in rs])
        sh = np.array([x["peak_shift_km"] for x in rs])
        lines.append({
            "field": field, "model": model, "n_traj": len(rs),
            "amp_rel_err_median": float(np.median(amp)),
            "amp_rel_err_min": float(amp.min()), "amp_rel_err_max": float(amp.max()),
            "peak_shift_km_median": float(np.median(sh)),
            "peak_shift_km_min": float(sh.min()), "peak_shift_km_max": float(sh.max()),
        })
    return lines


def main():
    ap = argparse.ArgumentParser(description="E04 空间诊断出图")
    ap.add_argument("--subset", default="plume1d", choices=["plume1d", "plume2"])
    ap.add_argument("--traj", type=int, default=0, help="预先声明的示例轨迹（默认 0）")
    ap.add_argument("--frame", type=int, default=9,
                    help="输出窗内的帧序号，0..9，默认 9 = 预测第 10 步（误差最大处）")
    ap.add_argument("--install", action="store_true",
                    help="把 F03（三个 PDF + manifest）复制到 results/figures/")
    a = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    data = gather(a.subset)
    if not data:
        print("没有任何模型的预测 —— 先在 GPU 上跑 eval_per_traj.py --dump-fields")
        return 1
    n_traj = next(iter(data.values()))[0].shape[0]
    print("收集到 {} 个模型；测试轨迹 {} 条".format(len(data), n_traj))

    # 失败案例：按**预先写死的规则**选取（主表口径 E_vx 最大的轨迹）
    truth = next(iter(data.values()))[1]
    eps = {c: 1e-12 for c in em.FIELD_ORDER}      # 只用于排序，不参与报告
    worst = {}
    for disp, (pred, tr, _) in data.items():
        e = em.relative_l2_per_frame(pred, tr, np.array([eps[c] for c in em.FIELD_ORDER]))
        worst[disp] = int(np.argmax(e.mean(axis=1)[:, IDX["vx"]]))
    fail_traj = max(set(worst.values()), key=list(worst.values()).count)
    print("失败案例轨迹（规则: 多数模型 E_vx 最大的轨迹）: {}".format(fail_traj))

    f03 = os.path.join(FIGDIR, "fig03_field_predictions.pdf")
    plot_f03(data, a.subset, a.traj, a.frame, f03)
    plot_profile(data, a.subset, a.traj, a.frame,
                 os.path.join(FIGDIR, "fig03b_profile.pdf"))
    plot_f03(data, a.subset, fail_traj, a.frame,
             os.path.join(FIGDIR, "fig03c_failure_case.pdf"))

    # 全部测试轨迹（不是只算示例那条）—— PLAN 的硬要求
    rows = profile_errors(data, a.frame)
    import csv
    p = os.path.join(RESULTS, "profile_errors.csv")
    with open(p, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  写出: {}  ({} 条轨迹 × {} 个模型 × 2 个场)".format(
        p, len({r["traj"] for r in rows}), len(data), 2))

    summ = summarize_profile(rows)
    ps = os.path.join(RESULTS, "profile_errors_summary.csv")
    with open(ps, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print("  写出:", ps)
    print("  ── 剖面汇总（中位数 [min, max]，全部 {} 条轨迹）──".format(len({r['traj'] for r in rows})))
    for s in summ:
        print("     %-5s %-18s amp %+7.1f%% [%+7.1f, %+7.1f]   peak shift %5.1f km [%.1f, %.1f]"
              % (s["field"], s["model"], 100 * s["amp_rel_err_median"],
                 100 * s["amp_rel_err_min"], 100 * s["amp_rel_err_max"],
                 s["peak_shift_km_median"], s["peak_shift_km_min"], s["peak_shift_km_max"]))

    manifest = {
        "figure": "fig03_field_predictions.pdf",
        "subset": a.subset,
        "example_traj": a.traj, "failure_traj": fail_traj, "frame_in_window": a.frame,
        "frame_semantics": "index into the 10-frame output window (0..9)",
        "profile_x_star": 0.5,
        "color_scale": {"tk": "hot", "speed(|vx,vy|)": "viridis", "shared_per_row": True},
        "model_runs": {d: data[d][2] for d in data},
        "selection_rule": "example traj fixed a priori; failure traj = modal argmax of E_vx",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    mp = os.path.join(FIGDIR, "fig03_manifest.json")
    with open(mp, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("  写出:", mp)

    if a.install:
        import shutil
        # ⚠ 必须拷**四份**：这张「图」其实是三个 PDF 加一份 manifest。
        #   原先只拷第一张，结果剖面图与失败案例图长期没进正文 ——
        #   §5.2 的正文却写着 F03「pairs … with a velocity profile diagnostic」，
        #   于是正文在描述一张读者看不到的图（2026-09-25 才发现）。
        #   manifest 同理：它一直存在，只是没被安装，被误判成「F03 缺 manifest」。
        figdir = os.path.join(paths.RESULTS, "figures")
        for name in ("fig03_field_predictions.pdf", "fig03b_profile.pdf",
                     "fig03c_failure_case.pdf", "fig03_manifest.json"):
            src = os.path.join(FIGDIR, name)
            if not os.path.exists(src):
                print("  ⚠ 缺 {}，跳过（正文对应 \\paperfigure 会显示占位框）".format(name))
                continue
            dst = os.path.join(figdir, name)
            shutil.copyfile(src, dst)
            print("  安装到:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
