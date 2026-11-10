#!/usr/bin/env python3
# coding=utf-8
"""把已落盘的预测按论文 Eq.(9) 打分。

**为什么需要它**：外部三模型（FNO / GNOT / Transolver）的推理走
``predict_external.py``，它只**产出预测数组**（``eval/predictions.hdf5``），
不负责打分；而 ``eval_checkpoint.py`` 是"自己推理 + 打分"揉在一起的，
读不了别人产出的预测。

所以补齐这条链的最后一环：**读预测 → 按 Eq.(9) 算逐场指标 → 写 CSV**。

⚠ 两者的**指标口径必须完全一致**（都是 ``eval_metrics`` 的同一套实现、
同样的 ε 拟合），否则主表里的 GNN 行与外部模型行不可比。

用法::

    python -B score_predictions.py --stage E01_main --subset plume1d
    python -B score_predictions.py --stage E01_main --subset plume1d --runs <dir> [...]
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
for _p in (HERE,):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import numpy as np      # noqa: E402

import eval_metrics as em          # noqa: E402
import eval_checkpoint as ec       # noqa: E402  （复用它的 ε 拟合，保证口径一致）


def load_predictions(run_dir):
    """读 ``<run>/eval/predictions.hdf5``。

    形状 ``[n_traj, T, H, W, C]`` —— 与 ``eval_metrics`` 的约定一致。
    """
    import h5py
    for name in ("predictions.hdf5", "predictions.npz"):
        p = os.path.join(run_dir, "eval", name)
        if not os.path.exists(p):
            continue
        if name.endswith(".hdf5"):
            with h5py.File(p, "r") as f:
                fields = [c for c in em.FIELD_ORDER if c in f]
                if not fields:
                    raise KeyError("predictions.hdf5 里没有六个场: {}".format(p))
                pred = np.stack([np.asarray(f[c], dtype=np.float32) for c in fields], axis=-1)
                truth = np.stack([np.asarray(f["truth_{}".format(c)], dtype=np.float32)
                                  for c in fields], axis=-1)
            return pred, truth, os.path.basename(p)
        d = np.load(p)
        fields = [c for c in em.FIELD_ORDER if c in d]
        pred = np.stack([d[c] for c in fields], axis=-1)
        truth = np.stack([d["truth_{}".format(c)] for c in fields], axis=-1)
        return pred, truth, os.path.basename(p)
    raise FileNotFoundError("没有预测文件: {}/eval/predictions.{{hdf5,npz}}".format(run_dir))


def score_one(run_dir, subset, eps_vec):
    pred, truth, src = load_predictions(run_dir)

    # 非有限值：先记数、再按同位置真值填补 —— eval_metrics 遇到 NaN 会直接 raise
    bad = ~np.isfinite(pred)
    n_bad = int(bad.sum())
    if n_bad:
        pred = np.where(bad, truth, pred)

    sse, energy = em.weighted_squares(pred, truth)
    E = em.aggregate_E(em.relative_l2_per_frame(pred, truth, eps_vec))
    rmse = em.aggregate_E(em.rmse_per_frame(pred, truth))
    trig = em.epsilon_triggered(sse, energy, eps_vec)

    exp_id = os.path.basename(os.path.abspath(run_dir))
    return {
        "exp_id": exp_id,
        "subset": subset,
        "n_traj": int(pred.shape[0]),
        "n_frame": int(pred.shape[1]),
        "pred_file": src,
        "E_per_field": {c: float(E[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "rmse_per_field": {c: float(rmse[i]) for i, c in enumerate(em.FIELD_ORDER)},
        "epsilon_triggered": trig,
        "n_nonfinite": n_bad,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    }


def _merge_into_existing(out_path, new_rows):
    """把本次结果并入已有汇总表：同 ``exp_id`` 的旧行被替换，其余保留。

    旧行只带平铺的 CSV 列，缺 ``E_per_field`` 这类嵌套结构，所以这里把它还原成
    与 ``rows`` 同形的字典 —— 这样"跑一个子集"不会抹掉另一个子集的结果。
    """
    if not os.path.exists(out_path):
        return list(new_rows)
    old = {}
    try:
        with open(out_path, newline="", encoding="utf8") as f:
            for r in csv.DictReader(f):
                old[r["exp_id"]] = dict(
                    exp_id=r["exp_id"], subset=r["subset"],
                    n_traj=int(r["n_traj"]), n_frame=int(r["n_frame"]),
                    pred_file=r["pred_file"],
                    E_per_field={c: float(r["E_" + c]) for c in em.FIELD_ORDER},
                    rmse_per_field={c: float(r["rmse_" + c]) for c in em.FIELD_ORDER},
                    n_nonfinite=r["n_nonfinite"], scored_at=r["scored_at"],
                )
    except Exception:                   # 旧表损坏时不要连累本次结果
        return list(new_rows)
    for r in new_rows:
        old[r["exp_id"]] = r
    out = list(old.values())
    out.sort(key=lambda r: (r.get("subset") or "", r.get("exp_id") or ""))
    return out


def main():
    ap = argparse.ArgumentParser(description="对已落盘的预测按 Eq.(9) 打分")
    ap.add_argument("--stage", default="E01_main")
    ap.add_argument("--subset", default="plume1d", choices=["plume1d", "plume2"])
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="CSV 输出路径（默认 <stage>/results/）")
    a = ap.parse_args()

    import glob
    if a.runs:
        runs = [os.path.abspath(r) for r in a.runs]
    else:
        runs = [d for d in sorted(glob.glob(
                    os.path.join(EXPERIMENTS, a.stage, "runs", "*")))
                if "SMOKE" not in os.path.basename(d)
                and a.subset in os.path.basename(d)
                and os.path.exists(os.path.join(d, "eval", "predictions.hdf5"))]
    if not runs:
        print("没有找到带 predictions.hdf5 的 run（先跑 predict_external.py）")
        return 1

    # ε 按**子集**拟合，与 eval_checkpoint 同一套 —— 保证两类模型的指标可比
    eps = ec.fit_epsilon(subset=a.subset)
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER])
    print("ε_c（{} 训练集拟合）: {}".format(
        a.subset, {k: "{:.3e}".format(v) for k, v in eps.items()}))

    rows = []
    print("\n{:<52} {}".format("run", " ".join("{:>11}".format(c) for c in em.FIELD_ORDER)))
    for d in runs:
        try:
            r = score_one(d, a.subset, eps_vec)
        except Exception as e:                       # noqa: BLE001
            print("  [跳过] {}: {}: {}".format(os.path.basename(d), type(e).__name__, e))
            continue
        rows.append(r)
        out_dir = os.path.join(d, "eval")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}_eq9.json".format(r["exp_id"])),
                  "w", encoding="utf8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print("{:<52} {}".format(
            r["exp_id"],
            " ".join("{:>11.4e}".format(r["E_per_field"][c]) for c in em.FIELD_ORDER)))
        if r["n_nonfinite"]:
            print("    ⚠ 非有限预测 {} 个（已按同位置真值填补）".format(r["n_nonfinite"]))

    if not rows:
        return 1
    out = a.out or os.path.join(EXPERIMENTS, a.stage, "results",
                                "{}_eq9_external.csv".format(a.stage))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    header = (["exp_id", "subset", "n_traj", "n_frame", "pred_file"]
              + ["E_{}".format(c) for c in em.FIELD_ORDER]
              + ["rmse_{}".format(c) for c in em.FIELD_ORDER]
              + ["n_nonfinite", "scored_at"])
    # ⚠ 汇总文件是「该 stage 的**全部**外部模型结果」，不是「本次跑的那批」。
    #   直接覆盖会**抹掉另一个子集** —— 实测踩过：`--subset plume2` 把 plume1d 的
    #   9 行（3 模型 × 3 种子）整个冲掉了。与 `eval_checkpoint.py` 的
    #   `_merge_into_existing` 是同一类问题、同一解法：同 exp_id 替换、其余保留。
    merged = _merge_into_existing(out, rows)
    with open(out, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in merged:
            w.writerow([r["exp_id"], r["subset"], r["n_traj"], r["n_frame"], r["pred_file"]]
                       + [r["E_per_field"][c] for c in em.FIELD_ORDER]
                       + [r["rmse_per_field"][c] for c in em.FIELD_ORDER]
                       + [r["n_nonfinite"], r["scored_at"]])
    print("\n写出:", out,
          "（本次 {} 行；文件共 {} 行，旧行已保留）".format(len(rows), len(merged)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
