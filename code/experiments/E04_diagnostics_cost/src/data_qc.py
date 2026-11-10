#!/usr/bin/env python3
# coding=utf-8
"""E04 数据质检 —— 六项检查 + 一张"重映射信息损失"表（纯 CPU）。

对应 ``E04_diagnostics_cost/PLAN.md`` 第三节的六项：无效值 / 异常范围 /
时间单调性 / 坐标一致性 / 六场对齐 / 重映射误差。

**重映射那一项为什么不按 PLAN 的字面写法做**
------------------------------------------------
PLAN 写的是逐场误差 ``‖F_128 − R·F_401‖``。实测该量恒为 float32 舍入
（vx/vy 恰为 0.0）—— **因为 128×128 文件本身就是 R 作用在 401×309 上的产物**，
误差是"同一个算子算了两遍"的浮点差，不含任何信息。所以这里改成一张
"重映射信息损失"表（``--remap`` 段）：

  - **面积积分守恒**：``∫(R·F) dA_tgt`` vs ``∫F dA_src``。守恒式重映射应当
    ~1e-7（float32）。**这一项才是真正能抓轴序/权重错误的检查** —— 若把 x/y
    权重用反，面积元对不上，误差直接跳到 O(1)。
  - **二阶矩比**：``Σ F x²/Σ F`` 目标/源，量化低通（网格粗化）带来的空间展宽。
  - **动能比**：``½Σ(vx²+vy²) dA`` 目标/源。⚠ **必须取速度非零的帧**，帧 0 速度为 0。
  - **通量乘积 vx·vy 的守恒性**：MATH 点名的非线性项，正是"统一权重不保证通量
    乘积守恒"那句话的量化。同样必须用速度非零的帧。
  - **复现残差**：``‖(R·F)/c − 已发布 128 文件‖ / ‖已发布‖``，``c`` 从数据最小二乘
    拟合（不硬编码常数，避免常数表与文件版本不一致时误判）。

判定口径写在 ``results/data_qc.csv`` 的 ``verdict`` 列，不在这里下结论。

用法::

    python3 -B E04_diagnostics_cost/src/data_qc.py                 # 两子集
    python3 -B E04_diagnostics_cost/src/data_qc.py --subset plume1d
    python3 -B E04_diagnostics_cost/src/data_qc.py --remap-rows 3  # 源文件只读 3 条
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E04 = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E04)
E03_SRC = os.path.join(EXPERIMENTS, "E03_rollout", "src")
COMMON = os.path.join(EXPERIMENTS, "common")

for _p in (COMMON, E03_SRC, HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import numpy as np          # noqa: E402
import h5py                 # noqa: E402

import eval_metrics as em   # noqa: E402
import paths                # noqa: E402

# 发布包：数据根由 paths 统一解析（GEOSIM_DATA 可覆盖；默认 geosim-plume/data/）
DATA_ROOT = paths.DATA

# 六场的物理量纲（only 用于报告列名，不做换算）
FIELD_UNIT = {
    "ro": "kg/m^3", "pr": "Pa", "tk": "K", "nu": "log10(Pa*s)",
    "vx": "m/s", "vy": "m/s",
}

# 重映射源 → 目标（两子集共用同一套权重，网格相同）
REMAP_NPZ = os.path.join(DATA_ROOT, "plume1d_corrected",
                         "remap_weights_401x309_to_128x128.npz")


# --------------------------------------------------------------------------
# 子集规格（与 data_full.SUBSET_SPEC 对齐，但这里是**质检口径**）
# --------------------------------------------------------------------------

def spec_of(subset):
    """`(raw_401_path, ds_128_path, x401_name, x128_name)`。

    ⚠ 401 源文件是**原始物理量**（见其 ``raw_physical_values`` attr），
    128 文件是**归一化**的。两者不可直接相减 —— 这正是 PLAN 那条检查退化的
    深层原因，不只是浮点问题。
    """
    d = os.path.abspath(DATA_ROOT)
    if subset == "plume1d":
        base = os.path.join(d, "plume1d_corrected")
        return dict(
            raw=os.path.join(base, "plume1d_train_401x309.hdf5"),
            ds=os.path.join(base, "plume1d_train_ds_128x128_phys1e6.hdf5"),
            n_traj=97, n_frame=97,
        )
    if subset == "plume2":
        base = os.path.join(d, "plume2")
        return dict(
            raw=os.path.join(base, "plume2d_train_401x309.hdf5"),
            ds=os.path.join(base, "plume2d_train_ds_128x128_norm_corrected.hdf5"),
            n_traj=103, n_frame=131,
        )
    raise KeyError("未知子集 {!r}".format(subset))


def _open(path):
    if not os.path.exists(path):
        raise FileNotFoundError("文件不存在: {}".format(path))
    return h5py.File(path, "r")


# --------------------------------------------------------------------------
# 检查 1/2：无效值与异常范围
# --------------------------------------------------------------------------

def check_values_and_range(path, fields, max_traj=None, label=""):
    """扫一遍（可按轨迹上限抽样），返回逐场的 NaN/Inf 计数与极值。

    ⚠ 抽样是**已知局限**：``n_traj_scanned`` 会一起写进 CSV，不要把它当成全量结论。
    """
    rows = []
    with _open(path) as f:
        n_traj = f[fields[0]].shape[0]
        n_traj_scan = n_traj if max_traj is None else min(n_traj, int(max_traj))
        n_frame = f[fields[0]].shape[1]
        acc = {c: dict(nan=0, inf=0, vmin=np.inf, vmax=-np.inf) for c in fields}
        for c in fields:
            for r in range(n_traj_scan):
                a = np.asarray(f[c][r], dtype=np.float64)      # (T, H, W)
                acc[c]["nan"] += int(np.isnan(a).sum())
                fin = np.isfinite(a)
                acc[c]["inf"] += int((~fin).sum()) - int(np.isnan(a).sum())
                if fin.any():
                    acc[c]["vmin"] = min(acc[c]["vmin"], float(a[fin].min()))
                    acc[c]["vmax"] = max(acc[c]["vmax"], float(a[fin].max()))
        t = np.asarray(f["t-coordinate"][:n_traj_scan], dtype=np.float64)

    # 时间单调性（检查 3）
    dt = np.diff(t, axis=1)
    mono_ok = bool(np.all(dt > 0))
    rows.append(dict(
        subset=label, check="invalid_values", field="ALL",
        value=sum(acc[c]["nan"] for c in fields), aux=sum(acc[c]["inf"] for c in fields),
        verdict="pass" if all(acc[c]["nan"] == 0 and acc[c]["inf"] == 0 for c in fields)
                else "FAIL",
        note="value=NaN 计数, aux=Inf 计数, 扫了 {}/{} 条".format(n_traj_scan, n_traj),
    ))
    for c in fields:
        rows.append(dict(
            subset=label, check="range", field=c,
            value=acc[c]["vmin"], aux=acc[c]["vmax"],
            verdict="",
            note="{} ; 扫了 {}/{} 条".format(FIELD_UNIT.get(c, ""), n_traj_scan, n_traj),
        ))
    rows.append(dict(
        subset=label, check="time_monotonic", field="t",
        value=float(dt.min()) if dt.size else float("nan"),
        aux=float(dt.max()) if dt.size else float("nan"),
        verdict="pass" if mono_ok else "FAIL",
        note="value=min dt, aux=max dt (单位随文件); 首帧 t={:.6g}".format(
            float(t[0, 0]) if t.size else float("nan")),
    ))
    rows.append(dict(
        subset=label, check="time_first_frame_zero", field="t",
        value=float(np.abs(t[:, 0]).max()) if t.size else float("nan"),
        aux=0.0,
        verdict="pass" if t.size and np.allclose(t[:, 0], 0.0, atol=1e-9) else "FAIL",
        note="value=max|t[:,0]|",
    ))
    return rows, dict(n_traj=n_traj, n_frame=n_frame, scanned=n_traj_scan)


# --------------------------------------------------------------------------
# 检查 4：坐标一致性（本项目踩过的最贵的坑）
# --------------------------------------------------------------------------

def check_coords(path, fields, label, n_sample=3):
    """坐标方向与展平顺序是否与训练代码一致。

    训练侧的约定（``vendor/utils_tubor.py`` 的 ``flatten(..., start_dim=1,
    end_dim=-2)`` 配 ``torch.meshgrid(x, y, indexing='ij')``）：
        节点扁平索引 ``k = i*ny + j``  ↦  坐标 ``(x[i], y[j])``
    把它写错（例如推理时用 ``np.meshgrid`` 默认的 ``'xy'``）会让每个节点的 x/y
    对调，等价于空间场转置：标量场误差 +30%，**矢量场放大 8–20 倍**。

    这里做的是**物理性**检查：x 镜像下 ``vx`` 应变号、``vy`` 应不变；两者若
    互换就说明 x/y 被对调了。它不依赖"哪端是地表"这类外部真值，是可证伪的。
    （展平顺序 ``k = i*ny + j`` 的**结构性**核对放在 ``check_field_alignment``
    里做：场的内层两维必须与 x/y 轴长一一对应。）
    """
    rows = []
    with _open(path) as f:
        x = np.asarray(f["x-coordinate"][:], dtype=np.float64)
        y = np.asarray(f["y-coordinate"][:], dtype=np.float64)
        nx, ny = len(x), len(y)
        r = min(n_sample, f[fields[0]].shape[0])
        tk = np.asarray(f["tk"][r - 1], dtype=np.float64)
        vx = np.asarray(f["vx"][r - 1], dtype=np.float64)
        vy = np.asarray(f["vy"][r - 1], dtype=np.float64)

    # ⚠ 场的空间两维是 (nx, ny) —— **第 0 轴是 x**，不是行/列意义上的 y。
    #   所以"沿 x 镜像" = ``[::-1, :]``，"沿 y 镜像" = ``[:, ::-1]``。
    #   写反了这条判定会落在近零相关上、靠符号侥幸 pass，失去判别力（已踩过）。
    if tuple(vx.shape[1:]) != (nx, ny):
        rows.append(dict(
            subset=label, check="coord_flatten_order", field="-",
            value=float(nx), aux=float(ny), verdict="FAIL",
            note="场的空间维 {} 与 (nx,ny)=({},{}) 不符 —— 展平顺序这条不能验，"
                 "后续镜像检查也不可信".format(tuple(vx.shape[1:]), nx, ny),
        ))
        return rows

    tmid = vx.shape[0] // 2            # 帧 0 速度恒为 0（见 MATH），取中段
    vxs, vys = vx[tmid], vy[tmid]

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else float("nan")

    mirror_x = lambda A: A[::-1, :]            # noqa: E731
    mirror_y = lambda A: A[:, ::-1]            # noqa: E731
    rx_vx, rx_vy = corr(vxs, mirror_x(vxs)), corr(vys, mirror_x(vys))
    ry_vx, ry_vy = corr(vxs, mirror_y(vxs)), corr(vys, mirror_y(vys))

    # 判据用**强阈值**而不是符号：真值约 (−0.84, +0.92)，用错轴的对照组约 ±0.005。
    # 符号判据在对照组上也可能偶然成立，等于没有判别力。
    THR = 0.3
    coords_ok = (rx_vx < -THR) and (rx_vy > THR)
    rows.append(dict(
        subset=label, check="coord_flatten_order", field="-",
        value=float(nx), aux=float(ny), verdict="pass",
        note="场的空间两维 (nx,ny)=({},{}) 与 x/y 轴长一致；节点 k=i*ny+j ↦ (x[i],y[j])，"
             "与训练 flatten+meshgrid 'ij' 同序".format(nx, ny),
    ))
    rows.append(dict(
        subset=label, check="coord_x_mirror_symmetry", field="vx,vy",
        value=rx_vx, aux=rx_vy,
        verdict="pass" if coords_ok else "FAIL",
        note="value=corr(vx, mirror_x vx) 应<−{t}；aux=corr(vy, mirror_x vy) 应>+{t}。"
             "轴写反时这两个量会掉到 ~0（实测对照组 ±0.005），故用强阈值而非符号判据"
             .format(t=THR),
    ))
    rows.append(dict(
        subset=label, check="coord_y_mirror_symmetry", field="vx,vy",
        value=ry_vx, aux=ry_vy, verdict="",
        note="对照项：沿 y 镜像的相关。若把 x/y 弄反，x/y 两行会互换 —— 这就是"
             "这条检查要抓的失效模式",
    ))
    # 竖直温度剖面：沿 x 平均后剩 (ny,)，只记录不下结论（需要外部真值才知道朝向）
    prof = tk[tmid].mean(axis=0)       # (ny,)
    rows.append(dict(
        subset=label, check="coord_tk_vertical_profile", field="tk",
        value=float(prof[0]), aux=float(prof[-1]), verdict="",
        note="value=tk 沿 x 平均后在 y 索引 0 处, aux=在 y 索引 -1 处；"
             "用于人工核对「哪端是地表」，脚本不自行判定",
    ))
    return rows


# --------------------------------------------------------------------------
# 检查 5：六场对齐
# --------------------------------------------------------------------------

def check_field_alignment(path, fields, label):
    rows = []
    with _open(path) as f:
        shapes = {c: tuple(f[c].shape) for c in fields}
        tshape = tuple(f["t-coordinate"].shape)
        xshape = tuple(f["x-coordinate"].shape)
        yshape = tuple(f["y-coordinate"].shape)
    nx_ax, ny_ax = int(xshape[0]), int(yshape[0])

    uniq = set(shapes.values())
    ok_shape = len(uniq) == 1
    ok_t = ok_shape and tshape == (shapes[fields[0]][0], shapes[fields[0]][1])
    rows.append(dict(
        subset=label, check="field_alignment", field="ALL",
        value=len(uniq), aux=1.0,
        verdict="pass" if (ok_shape and ok_t) else "FAIL",
        note="value=不同 shape 的个数（应为 1）；六场 {} + t {} + x {} + y {}".format(
            sorted(uniq), tshape, xshape, yshape),
    ))
    n_traj, n_frame, ny, nx = shapes[fields[0]][0], shapes[fields[0]][1], \
        shapes[fields[0]][2], shapes[fields[0]][3]
    rows.append(dict(
        subset=label, check="field_alignment_xy", field="ALL",
        value=float(nx), aux=float(ny),
        verdict="pass" if (nx == nx_ax and ny == ny_ax) else "FAIL",
        note="场的内层两维 (ny,nx)=({},{}) 必须与 x/y 轴长一致".format(ny, nx),
    ))
    return rows


# --------------------------------------------------------------------------
# 检查 6：重映射信息损失
# --------------------------------------------------------------------------

def check_remap(subset, rows_src, label, ds_path, max_traj=2):
    """守恒式重映射的守恒量 + 复现残差。见模块 docstring 的说明。"""
    out = []
    npz = os.path.abspath(REMAP_NPZ)
    if not os.path.exists(npz):
        out.append(dict(subset=label, check="remap", field="ALL", value=float("nan"),
                        aux=float("nan"), verdict="SKIP",
                        note="缺权重文件 {}".format(npz)))
        return out
    w = np.load(npz)
    wx = np.asarray(w["w_x"], dtype=np.float64)      # (128, 401)
    wy = np.asarray(w["w_y"], dtype=np.float64)      # (128, 309)
    xb_s, yb_s = np.asarray(w["xb_src"]), np.asarray(w["yb_src"])
    xb_t, yb_t = np.asarray(w["xb_tgt"]), np.asarray(w["yb_tgt"])
    ax_s = np.diff(xb_s)[:, None] * np.diff(yb_s)[None, :]   # (401, 309) 源面积元
    ax_t = np.diff(xb_t)[:, None] * np.diff(yb_t)[None, :]   # (128, 128) 目标面积元

    def R(F):                                        # (401,309) -> (128,128)
        return wx @ F @ wy.T

    spec = spec_of(subset)
    src = os.path.abspath(spec["raw"])
    if not os.path.exists(src):
        out.append(dict(subset=label, check="remap", field="ALL", value=float("nan"),
                        aux=float("nan"), verdict="SKIP",
                        note="缺 401 源文件 {}".format(src)))
        return out

    with _open(src) as fs, _open(ds_path) as ft:
        n_s = min(fs["tk"].shape[0], max_traj)
        n_t = ft["tk"].shape[0]
        n_f = min(fs["tk"].shape[1], ft["tk"].shape[1])
        # 速度非零的帧：帧 0 恒为 0，取中后段
        f_idx = min(n_f - 1, max(1, int(n_f * 0.6)))
        for c in em.FIELD_ORDER:
            # 只读**单帧**：整条轨迹读下来是 131×401×309×4 ≈ 65 MB/场，
            # 而 plume2 的 401 源文件有 42 GB，没必要为了一帧付这个代价。
            F = np.asarray(fs[c][0, f_idx], dtype=np.float64)     # (401,309)
            G = R(F)
            I_src = float((F * ax_s).sum())
            I_tgt = float((G * ax_t).sum())
            # ⚠ 分母用 ∫|F|dA 而不是 |∫F dA|：vx/vy 的积分因流入=流出而 ≈0
            #   （实测 ~1e-14），拿它当分母会把舍入噪声放大成 1e-6 的假偏差。
            scale = float((np.abs(F) * ax_s).sum())
            rel = abs(I_tgt - I_src) / max(scale, 1e-30)
            out.append(dict(
                subset=label, check="remap_area_integral", field=c,
                value=I_src, aux=rel,
                verdict="pass" if rel < 1e-5 else "FAIL",
                note="aux=|∫(R·F)dA − ∫F dA| / ∫|F|dA；守恒式重映射应 ~1e-7(float32)。"
                     ">1e-5 说明权重或轴序用错",
            ))

        # 二阶矩比（空间展宽）
        xs_c = 0.5 * (xb_s[:-1] + xb_s[1:])
        for c in ("tk", "nu"):
            F = np.asarray(fs[c][0, f_idx], dtype=np.float64)
            G = R(F)
            m_src = float((F * ax_s * xs_c[:, None] ** 2).sum() / max((F * ax_s).sum(), 1e-30))
            xs_t = 0.5 * (xb_t[:-1] + xb_t[1:])
            m_tgt = float((G * ax_t * xs_t[:, None] ** 2).sum() / max((G * ax_t).sum(), 1e-30))
            out.append(dict(
                subset=label, check="remap_second_moment_ratio", field=c,
                value=m_tgt / m_src if m_src else float("nan"), aux=m_src,
                verdict="",
                note="value=二阶矩 目标/源，量化网格粗化的低通展宽",
            ))

        # 动能比 —— 必须用速度非零帧
        vxs = np.asarray(fs["vx"][0, f_idx], dtype=np.float64)
        vys = np.asarray(fs["vy"][0, f_idx], dtype=np.float64)
        ke_src = float((0.5 * (vxs ** 2 + vys ** 2) * ax_s).sum())
        vxg, vyg = R(vxs), R(vys)
        ke_tgt = float((0.5 * (vxg ** 2 + vyg ** 2) * ax_t).sum())
        out.append(dict(
            subset=label, check="remap_kinetic_energy_ratio", field="vx,vy",
            value=ke_tgt / ke_src if ke_src else float("nan"),
            aux=float(np.abs(vxs).max()),
            verdict="pass" if abs(vxs).max() > 0 else "FAIL",
            note="帧 {} 上取值；aux=|vx|max，若为 0 则该帧不可用于此项".format(f_idx),
        ))

        # 通量乘积 vx*vy 的守恒性 —— MATH 点名的项
        pfx_src = float((vxs * vys * ax_s).sum())
        pfx_tgt = float((vxg * vyg * ax_t).sum())
        out.append(dict(
            subset=label, check="remap_flux_product", field="vx*vy",
            value=pfx_src, aux=pfx_tgt,
            verdict="",
            note="value=∫(vx*vy)dA 源, aux=目标；二者不等正是"
                 "「统一权重不保证非线性通量乘积守恒」的量化",
        ))

        # 复现残差：已发布 128 文件 == (R·F)*scale ?
        cname = "tk"
        F = np.asarray(fs[cname][0, f_idx], dtype=np.float64)
        G = R(F)
        H = np.asarray(ft[cname][0, f_idx], dtype=np.float64)
        denom = float((G * G).sum())
        s = float((G * H).sum() / denom) if denom > 0 else float("nan")
        resid = float(np.sqrt(((s * G - H) ** 2).sum() / max((H ** 2).sum(), 1e-30)))
        out.append(dict(
            subset=label, check="remap_reproduction", field=cname,
            value=s, aux=resid,
            verdict="pass" if resid < 1e-4 else "FAIL",
            note="value=最小二乘标尺 s，aux=相对残差；已发布 128 文件应≈s*(R·F)。"
                 "这里比较的是**归一化后**的量，s 吸收了 max-normalization 常数",
        ))
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def run(subset, max_traj, remap_rows, do_remap):
    spec = spec_of(subset)
    ds = os.path.abspath(spec["ds"])
    rows = []
    rows += check_values_and_range(ds, em.FIELD_ORDER, max_traj=max_traj, label=subset)[0]
    rows += check_coords(ds, em.FIELD_ORDER, subset)
    rows += check_field_alignment(ds, em.FIELD_ORDER, subset)
    if do_remap:
        rows += check_remap(subset, None, subset, ds, max_traj=remap_rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="both",
                    choices=["plume1d", "plume2", "both"])
    ap.add_argument("--max-traj", type=int, default=8,
                    help="128 文件扫描的轨迹上限；0 表示全量")
    ap.add_argument("--remap-rows", type=int, default=1,
                    help="401 源文件读取的轨迹数（文件很大，别调高）")
    ap.add_argument("--no-remap", action="store_true", help="跳过重映射段")
    a = ap.parse_args()

    subsets = ["plume1d", "plume2"] if a.subset == "both" else [a.subset]
    out_dir = os.path.join(E04, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "data_qc.csv")

    all_rows = []
    for s in subsets:
        print("══ {} ══".format(s))
        r = run(s, None if a.max_traj == 0 else a.max_traj, a.remap_rows,
                not a.no_remap)
        all_rows += r
        n_fail = sum(1 for x in r if x["verdict"] == "FAIL")
        print("    {} 项，FAIL {} 项".format(len(r), n_fail))
        for x in r:
            if x["verdict"] == "FAIL":
                print("    !! {} {} → value={} aux={}".format(
                    x["check"], x["field"], x["value"], x["aux"]))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "check", "field", "value",
                                          "aux", "verdict", "note"])
        w.writeheader()
        for x in all_rows:
            w.writerow(x)
    print("\n→ {}".format(out_csv))
    print("列说明：value/aux 的含义随 check 变化，逐行写在 note 里（不做统一化，"
          "免得丢掉量纲）")


if __name__ == "__main__":
    main()
