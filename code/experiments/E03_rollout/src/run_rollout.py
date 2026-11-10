#!/usr/bin/env python3
# coding=utf-8
"""E03 驱动：跑长程 rolling、判定失败、写三张 CSV。

产物（列名沿用 ``PLAN.md`` 的规定，另加了几列有据可查的上下文）::

    results/rollout_curves.csv    逐帧逐场误差 vs 物理时间
    results/rollout_summary.csv   固定端点的汇总（三种读法都算）
    results/failures.csv          全部失败记录（含"未触发失败"的完整轨迹）

**端点为何有三种读法**：``PLAN.md`` 定的端点是 ``{2, 5, 8, 12} Myr``。
按窗口递推，第一个被预测的帧是**帧 10** —— 实测 plume1d 的 ``t[:,10]`` 落在
8.93–9.74 Myr，所以 **2/5/8 Myr 全部落在真值输入窗内**（``n_valid = 0/12``），
那一时刻根本没有预测可比。三种读法：

======================  ==========================================================
``abs_user``            用户原定的 {2,5,8,12} 绝对物理时刻；未覆盖的写 NaN
``abs_protocol``        由本协议推导的四个**已覆盖**时刻（见 ``protocol_endpoints``）
``lead``                {2,5,8,12} 读作**预报时效**（自窗口最后输入帧起算）
======================  ==========================================================

用户 2026-09-21 定 **``lead`` 为主口径**（唯一能同时保住四个值的解释），
另两套仍会算进 CSV 供核对。

自变量与记账
------------
``n_window``（窗口推进次数）跨模型同义；``n_frame = 10 × n_window``；
``n_forward``（**真实网络前向次数**）在 FNO / Transolver 上是 ``n_window × 10``。
三个量都进 CSV，因为"单次输出 10 帧 ≠ 10 次自治更新"。

用法::

    # 自治 + teacher 对照，全 12 个窗口
    python -B run_rollout.py --subset plume1d --model mfgno --run <run_dir> \\
        --mode auto --mode teacher

    # 小样验证
    python -B run_rollout.py --subset plume1d --model mfgno --run <run_dir> --k-max 2
"""
from __future__ import annotations

import argparse
import csv
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
from data_full import TW, n_windows, read_split, spec_of, val_rows_of   # noqa: E402
import rollout as ro                         # noqa: E402
import thresholds as th                      # noqa: E402

RESULTS = os.path.join(E03_ROOT, "results")
CACHE = os.path.join(E03_ROOT, "runs")

# ⚠ 三个 CSV 是 **append 且无去重** 的，所以自检/小样跑出来的行会与正式结果
#   混在同一个文件里（本项目已在 eval_checkpoint / score_predictions 上踩过
#   两次同类问题）。``E03_CSV_DIR`` 只重定向**输出**，输入（tau_c.json /
#   range_limits*.csv）仍走 RESULTS —— 否则自检会因为找不到阈值而失败。
CSV_DIR = os.environ.get("E03_CSV_DIR") or RESULTS

CONSECUTIVE_WINDOWS = 3
""""连续 3 个输出窗口超阈才算失败"（``MATH.md`` 规定）。"""


# --------------------------------------------------------------------------
# 加载阈值
# --------------------------------------------------------------------------

def load_tau(subset: str, exp_id: str, which: str = "best") -> dict:
    """取该 run 的 τ_c；没有就报错（**不允许**静默回退到别的 run 的阈值）。"""
    path = os.path.join(RESULTS, "tau_c.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "tau_c.json 不存在 —— τ_c 必须**先于** rollout 算好（"
            "python -B thresholds.py --subset {} --run <run_dir>）".format(subset))
    with open(path, "r", encoding="utf8") as f:
        store = json.load(f)
    key = "{}|{}|{}".format(subset, exp_id, which)
    if key not in store:
        raise KeyError(
            "tau_c.json 里没有 {} 这一项 —— 该 run 的阈值还没算。"
            "已登记: {}".format(key, [k for k in store if not k.startswith("_")]))
    return store[key]


def load_limits(subset: str) -> dict:
    """范围界。只有 plume1d 会写进主 CSV，plume2 落在 ``_plume2.csv`` 后缀那份。"""
    path = os.path.join(RESULTS, "range_limits.csv")
    if subset != "plume1d":
        alt = path.replace(".csv", "_{}.csv".format(subset))
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(
            "range_limits.csv 不存在 —— 先跑 thresholds.py --range-only")
    out = {}
    with open(path, "r", encoding="utf8") as f:
        for r in csv.DictReader(f):
            out[r["field"]] = {
                "lo_c": float(r["lo_c"]), "hi_c": float(r["hi_c"]),
                "margin": float(r["margin"]),
            }
    return out


# --------------------------------------------------------------------------
# 失败判定
# --------------------------------------------------------------------------

def evaluate_failures(e: np.ndarray, truth: np.ndarray, pred: np.ndarray,
                      nonfinite: np.ndarray, tau: dict, limits: dict,
                      t_out: np.ndarray, traj_ids: list) -> list:
    """逐轨迹逐场判定三类失败，**全部记录**（含未触发的）。

    ``e`` 是 ``(n, K*10, C)`` 逐帧逐场 L2RE。三类判据（``MATH.md``）::

        numerical : 出现 NaN / Inf
        over_tau  : 连续 3 个**输出窗口**的窗口均值 > τ_c
        range     : pred 越出 [lo_c, hi_c]

    ⚠ ``over_tau`` 按**窗口**（每 10 帧）而不是按帧判 —— ``MATH.md`` 的原话是
    "连续 3 个输出窗口"，窗口是模型一次调用的产物，按帧判会把同一窗口内
    10 个高度相关的帧当成 10 次独立证据。
    """
    n, n_frame, n_var = e.shape
    k_win = n_frame // TW
    rows = []

    for i in range(n):
        for c_i, c in enumerate(em.FIELD_ORDER):
            rec = {
                "trajectory": traj_ids[i], "field": c,
                "fail_kind": "none", "first_fail_time": "", "first_fail_window": "",
                "n_consecutive": 0, "tau_c": tau["per_field"][c]["tau_c"],
                "value": "", "lo_c": limits[c]["lo_c"], "hi_c": limits[c]["hi_c"],
                "n_invalid": int(nonfinite[i].sum()),
            }

            # ① 数值失败（不需要阈值）
            bad_frames = np.where(nonfinite[i] > 0)[0]
            if bad_frames.size:
                rec.update(fail_kind="numerical",
                           first_fail_time=float(t_out[i, bad_frames[0]]),
                           first_fail_window=int(bad_frames[0] // TW),
                           value=float(nonfinite[i, bad_frames[0]]))
                rows.append(rec)
                continue

            # ② 超阈失败：连续 >=3 个**窗口**的窗口均值超阈
            win_mean = e[i, :, c_i].reshape(k_win, TW).mean(axis=1)
            thr = tau["per_field"][c]["tau_c"]
            over = win_mean > thr
            run_len, hit = 0, None
            for k in range(k_win):
                run_len = run_len + 1 if over[k] else 0
                if run_len >= CONSECUTIVE_WINDOWS:
                    hit = k - CONSECUTIVE_WINDOWS + 1
                    break
            if hit is not None:
                rec.update(fail_kind="over_tau",
                           first_fail_time=float(t_out[i, hit * TW]),
                           first_fail_window=int(hit),
                           n_consecutive=int(run_len),
                           value=float(win_mean[hit]))
                rows.append(rec)
                continue

            # ③ 范围违例
            p = pred[i, :, :, :, c_i]
            p_lo, p_hi = float(p.min()), float(p.max())
            if p_lo < limits[c]["lo_c"] or p_hi > limits[c]["hi_c"]:
                bad = np.where((p < limits[c]["lo_c"]) | (p > limits[c]["hi_c"]))[0]
                rec.update(fail_kind="range",
                           first_fail_time=float(t_out[i, bad[0]]),
                           first_fail_window=int(bad[0] // TW),
                           value=p_lo if p_lo < limits[c]["lo_c"] else p_hi)
            rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# 端点
# --------------------------------------------------------------------------

USER_ENDPOINTS = (2.0, 5.0, 8.0, 12.0)   # PLAN.md 定的


def protocol_endpoints(subset: str) -> list:
    """由本协议推导的四个**已覆盖**绝对时刻。"""
    s = spec_of(subset)
    t_train = float(s["t_train"])
    return [
        ("frame10_median", "t(帧10) 的中位数", None),   # 运行时按数据填
        ("t_train", "训练时域上限 {:.4g} Myr".format(t_train), t_train),
        ("t30", "30 Myr", 30.0),
        ("t50", "50 Myr", 50.0),
    ]


def _mean_at(t_out: np.ndarray, e: np.ndarray, t_star: float,
             tol_frac: float = 0.05) -> tuple:
    """Linear interpolation at an absolute time; never extrapolate.
    tol_frac is retained for API compatibility but does not widen the target.
    """
    from endpoint_summary import interpolate_mean
    curves = {i: list(zip(t_out[i], e[i])) for i in range(t_out.shape[0])}
    return interpolate_mean(curves, {i: float(t_star) for i in curves})


def _lead_at(t_out: np.ndarray, e: np.ndarray, lead: float,
             k_win: int, tw: int = TW, t_anchor=None) -> tuple:
    """Forecast lead measured from the LAST observed frame, not window length."""
    from endpoint_summary import interpolate_mean
    if t_anchor is None or len(t_anchor) != t_out.shape[0]:
        raise ValueError("Actual last-input timestamps are required for lead metrics")
    curves = {i: list(zip(t_out[i], e[i])) for i in range(t_out.shape[0])}
    return interpolate_mean(curves, {i: float(t_anchor[i]) + lead for i in curves})


# --------------------------------------------------------------------------
# 写 CSV
# --------------------------------------------------------------------------

def _ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)     # ≤--csv-dir 时与 RESULTS 不同


def write_curves(rows: list, subset: str, append: bool = True):
    path = os.path.join(CSV_DIR, "rollout_curves.csv")
    header = ["run_id", "model", "variant", "subset", "seed", "mode", "level",
              "trajectory", "window", "frame", "physical_time", "field",
              "metric", "value", "n_valid", "tau_c", "over_tau_c"]
    exists = os.path.exists(path)
    with open(path, "a" if (append and exists) else "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        if not (append and exists):
            w.writerow(header)
        w.writerows(rows)
    return path


def write_summary(rows: list, subset: str, append: bool = True):
    path = os.path.join(CSV_DIR, "rollout_summary.csv")
    header = ["run_id", "model", "subset", "seed", "mode", "endpoint_kind",
              "endpoint_label", "physical_time", "field", "metric", "mean",
              "sd_across_traj", "n_valid", "covered"]
    exists = os.path.exists(path)
    with open(path, "a" if (append and exists) else "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        if not (append and exists):
            w.writerow(header)
        w.writerows(rows)
    return path


def write_failures(rows: list, subset: str, append: bool = True):
    path = os.path.join(CSV_DIR, "failures.csv")
    header = ["run_id", "model", "subset", "seed", "mode", "trajectory", "field",
              "fail_kind", "first_fail_time", "first_fail_window", "n_consecutive",
              "tau_c", "value", "lo_c", "hi_c", "n_invalid"]
    exists = os.path.exists(path)
    with open(path, "a" if (append and exists) else "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        if not (append and exists):
            w.writerow(header)
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def run_one(run_dir: str, subset: str, mode: str, which: str, k_max: int,
            device, allow_live: bool, dump: bool, traj_max: int = None):
    import eval_checkpoint as ec

    exp_id = os.path.basename(os.path.abspath(run_dir))

    if mz.checkpoint_is_live(run_dir, which=which) and not allow_live:
        raise RuntimeError(
            "{} 的 {} checkpoint 在最近 15 分钟内被写过 —— 很可能还在训练，"
            "此时读到的 'best' 会变，结果不可复现。等作业结束，"
            "或显式 --allow-live 承担这个风险。".format(exp_id, which))

    tau = load_tau(subset, exp_id, which)
    limits = load_limits(subset)
    eps_vec = np.array([tau["eps_c"][c] for c in em.FIELD_ORDER])

    # 先看 family 再装载：GNN 需要把 resolution_t 覆盖成全时程帧数
    # （GraphCreator.nt = pde.resolution_t，配置里是 21，长程会切错窗口），
    # 而外部模型的 dataset 自己在装载时就按帧数读全量，不需要覆盖。
    import yaml
    with open(mz.find_config(run_dir), "r", encoding="utf8") as _f:
        _family = mz.family_of(yaml.safe_load(_f)["model_id"])

    if _family in mz.GNN_FAMILIES or _family == "gnot":
        n_frame = spec_of(subset)["n_frame_test"]
        # ⚠ 只覆盖 frames。temporal_domain 不必动 —— 它只喂给均匀回退分支，
        #   而有真实时间轴时 dt 走 compute_cumulative_dt 的真实差分（selftest 已验证）。
        loaded = mz.load_run(run_dir, device, which=which, frames=n_frame)
        print("   {}  device={}  resolution_t→{}".format(
            loaded.describe(), device, n_frame))
        fd = read_split(subset, "test", verbose=True)
        engine = ro.rollout_gnot if _family == "gnot" else ro.rollout_gnn
        res = engine(loaded, fd, mode=mode, k_max=k_max, traj_max=traj_max)
    else:
        s = spec_of(subset)
        # ⚠ 外部基线的配置指向 ``*_first21*.hdf5`` —— 那是**训练口径**（21 帧），
        #   在 21 帧轴上滚动只能切出 11 帧真值，评分时形状不匹配（已实测）。
        #   长程必须换到全时程文件，且文件名**不能裁字符串猜**（plume2 的两份
        #   不同名），所以从 SUBSET_SPEC 取权威值 —— 与 GNN 分支同一份数据。
        loaded = mz.load_run(run_dir, device, which=which,
                             frames=s["n_frame_test"],
                             ds_file=s["train_file"], ds_dir=s["data_dir"])
        print("   {}  device={}  全时程 {} 帧".format(
            loaded.describe(), device, s["n_frame_test"]))
        fd = None
        res = ro.rollout_external(loaded, k_max=k_max, mode=mode, traj_max=traj_max)

    # 非有限值先记数、再填补 —— eval_metrics 的 _as_float 遇到非有限值会直接 raise
    pred, n_bad = th._sanitize(res.pred, res.truth)
    if n_bad:
        print("   ⚠ 非有限预测 {} 个（记入 failures 的 numerical 分支）".format(n_bad))

    e = em.relative_l2_per_frame(pred, res.truth, eps_vec)     # (n, K*tw, C)
    rmse = em.rmse_per_frame(pred, res.truth)

    variant = ""
    for tag in ("-B-", "-C-", "-A-"):
        if tag in exp_id:
            variant = tag.strip("-")
            break

    run_id = "{}|{}|{}".format(exp_id, mode, which)

    # ── 曲线 CSV ───────────────────────────────────────────────
    k_win = res.n_window
    curves = []
    for i in range(res.n_traj):
        for k in range(k_win):
            for q in range(TW):
                fr = k * TW + q
                for c_i, c in enumerate(em.FIELD_ORDER):
                    curves.append([
                        run_id, loaded.model_id, variant, subset, loaded.seed, mode,
                        "traj", res.traj_ids[i], k, fr, float(res.t_out[i, fr]), c,
                        "l2re", float(e[i, fr, c_i]), res.n_traj,
                        tau["per_field"][c]["tau_c"],
                        int(e[i, fr, c_i] > tau["per_field"][c]["tau_c"])])
                    curves.append([
                        run_id, loaded.model_id, variant, subset, loaded.seed, mode,
                        "traj", res.traj_ids[i], k, fr, float(res.t_out[i, fr]), c,
                        "rmse", float(rmse[i, fr, c_i]), res.n_traj,
                        tau["per_field"][c]["tau_c"], ""])
    p1 = write_curves(curves, subset)

    # ── 汇总 CSV：三种端点读法 ─────────────────────────────────
    summary = []
    t_anchor = (fd.t[:res.n_traj, TW-1].numpy() if fd is not None
                else ro._t_axis_of(loaded)[:res.n_traj, TW-1])
    t10_med = float(np.median(res.t_out[:, 0]))    # 第一个被预测的帧（帧 10）
    eps_sets = [("lead", "lead={}Myr".format(int(v)), v) for v in USER_ENDPOINTS]
    eps_sets += [("abs_user", "abs={}Myr".format(int(v)), v) for v in USER_ENDPOINTS]
    eps_sets += [("abs_protocol", lbl, v if v is not None else t10_med)
                 for _, lbl, v in protocol_endpoints(subset)]

    for kind, label, tgt in eps_sets:
        for c_i, c in enumerate(em.FIELD_ORDER):
            if kind == "lead":
                m, s, nv = _lead_at(res.t_out, e[:, :, c_i], tgt, k_win, t_anchor=t_anchor)
            else:
                m, s, nv = _mean_at(res.t_out, e[:, :, c_i], tgt)
            summary.append([
                run_id, loaded.model_id, subset, loaded.seed, mode, kind, label,
                tgt, c, "l2re", m, s, nv, int(nv > 0)])
    p2 = write_summary(summary, subset)

    # ── 失败 CSV ───────────────────────────────────────────────
    fail = evaluate_failures(e, res.truth, pred, res.nonfinite, tau, limits,
                             res.t_out, res.traj_ids)
    frows = [[run_id, loaded.model_id, subset, loaded.seed, mode,
              r["trajectory"], r["field"], r["fail_kind"], r["first_fail_time"],
              r["first_fail_window"], r["n_consecutive"], r["tau_c"], r["value"],
              r["lo_c"], r["hi_c"], r["n_invalid"]] for r in fail]
    p3 = write_failures(frows, subset)

    n_fail = sum(1 for r in fail if r["fail_kind"] != "none")
    kinds = {}
    for r in fail:
        kinds[r["fail_kind"]] = kinds.get(r["fail_kind"], 0) + 1

    print("   n_window={} n_frame={} n_forward={} 轨迹={} 场={}".format(
        res.n_window, res.n_frame, res.n_forward, res.n_traj, len(em.FIELD_ORDER)))
    print("   失败判定: {}".format(kinds))
    print("   写出: {} / {} / {}".format(
        os.path.basename(p1), os.path.basename(p2), os.path.basename(p3)))
    print("   E_c（末窗口均值）= {}".format(
        " ".join("{:.3e}".format(float(e[:, -TW:, c_i].mean()))
                 for c_i in range(len(em.FIELD_ORDER)))))

    if dump:
        out = os.path.join(CACHE, subset, exp_id)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(
            os.path.join(out, "rollout_{}_{}.npz".format(mode, which)),
            pred=pred, truth=res.truth, t_out=res.t_out, e=e, nonfinite=res.nonfinite,
            traj_ids=np.array(res.traj_ids), n_window=res.n_window,
            n_forward=res.n_forward)
        print("   缓存: {}/rollout_{}_{}.npz".format(out, mode, which))

    return {"exp_id": exp_id, "mode": mode, "n_window": res.n_window,
            "n_forward": res.n_forward, "failures": kinds}


def main():
    ap = argparse.ArgumentParser(description="E03 长程滚动驱动")
    ap.add_argument("--subset", required=True, choices=["plume1d", "plume2"])
    ap.add_argument("--run", required=True, help="run 目录")
    ap.add_argument("--model", default=None, help="仅用于日志标注")
    ap.add_argument("--mode", action="append", default=None,
                    choices=["auto", "teacher"],
                    help="可重复；默认只跑 auto")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-live", action="store_true",
                    help="允许读取仍在训练的 checkpoint（结果不可复现）")
    ap.add_argument("--dump", action="store_true", help="落盘预测数组（占空间）")
    ap.add_argument("--traj-max", type=int, default=None,
                    help="只跑前 N 条轨迹。**路径自检用**：配 --k-max 全量，"
                         "在 GPU 上 1 分钟跑完 4 条家族路径的全部形状/设备分支，"
                         "正式结果不要带这个参数")
    ap.add_argument("--csv-dir", default=None,
                    help="把三张 CSV 写到别处（自检用）。输入仍走 results/ ——"
                         "否则会找不到 tau_c.json。也可用环境变量 E03_CSV_DIR")
    a = ap.parse_args()

    if a.csv_dir:
        global CSV_DIR
        CSV_DIR = os.path.abspath(a.csv_dir)
        if a.traj_max is None:
            print("⚠ --csv-dir 给了但没给 --traj-max：确认这不是误把自检"
                  "当正式跑", file=sys.stderr)

    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    modes = a.mode or ["auto"]
    _ensure_dirs()

    k_max = a.k_max
    if k_max is None:
        k_max = n_windows(spec_of(a.subset)["n_frame_test"])

    print("E03 rollout —— subset={} modes={} k_max={}".format(a.subset, modes, k_max))
    print("device:", device)
    if a.traj_max is not None:
        print("⚠ --traj-max={} —— 只跑前 {} 条轨迹，仅供路径自检，"
              "不是正式结果".format(a.traj_max, a.traj_max))

    out = []
    for mode in modes:
        print("\n[{}]".format(mode))
        out.append(run_one(os.path.abspath(a.run), a.subset, mode, a.which,
                           k_max, device, a.allow_live, a.dump, a.traj_max))

    print("\n完成 {} 个模式".format(len(out)))
    for r in out:
        print("  {:<8} K={:<3} 前向={:<5} 失败={}".format(
            r["mode"], r["n_window"], r["n_forward"], r["failures"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
