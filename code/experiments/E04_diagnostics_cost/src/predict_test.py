#!/usr/bin/env python3
# coding=utf-8
"""E04 的单窗口协议推理 —— **把预测数组落盘**，供剖面诊断与六场图使用。

**为什么必须重新推理**（实测过，不是想当然）：``E0*/runs/*/output/pred.hdf5``
**不能**直接拿来用 ——

1. **只有最后一个 batch**：上游把 ``pred_all.append(...)`` 注释掉了，
   ``output/pred.hdf5`` 的实际形状是 ``(2, 10, 128, 128, 6)``（E02 全部 run；
   E01 另有 (9,20)/(10,20)/(20,10)/(10,10) 四种），**不是**全部测试轨迹；
2. **来自 val_loader**：那是训练文件里切出的后 10%（轨迹 87–96），
   **不是** ``protocol/splits.yaml`` 冻结的独立测试集；
3. 形状跨模型不统一，做不了 ``PLAN.md`` 要求的"模型间同一轨迹、时刻、单位"对比。

所以 E04 自己推一遍，产出形状统一的 ``(n_traj, 10, 128, 128, 6)``。

副产品是**顺带算一遍 Eq.(9)** —— 一物三用：① 交叉核对 E02 的 C/A 数字；
② 补上外部三模型（含 GNOT）的 ``E_vx``/``E_vy`` 填 ``tables/06_cost.tex``；
③ 给空间诊断提供与主表同口径的样本。

⚠ **一个进程只装一个模型**（``--model`` 单值）。FNO / GNOT / Transolver 各有
``utils.py`` / ``train.py``，且 ``isolate_paths`` 会摘掉 ``REPO`` 路径而 GNN 家族
依赖它 —— 由 ``run_diag.sh`` 串行调用本脚本 5 次来规避。

用法::

    python -B predict_test.py --model mfgno --subset plume1d
    python -B predict_test.py --model gnot  --subset plume1d --which latest
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

import numpy as np      # noqa: E402
import torch            # noqa: E402

import eval_metrics as em                    # noqa: E402
import model_zoo as mz                       # noqa: E402

RESULTS = os.path.join(E04_ROOT, "results")
RUNS = os.path.join(E04_ROOT, "runs")

# 模型名 → (stage, run 目录名模板)
# E01 只有三个外部基线；MFGNO / MPNN 复用 E02（见 EXPERIMENTS.md 的「复用」一节）
# (stage, run 目录名里那段与模型相关的 token)；日期前缀由 glob 匹配，不写死。
MODEL_RUNS = {
    "mfgno": ("E02_ablation", "-mfgno-E02-C-equal-"),
    "mpnn": ("E02_ablation", "-mpnn-E02-A-"),
    "fno": ("E01_main", "-fno-E01-"),
    "transolver": ("E01_main", "-transolver-E01-"),
    "gnot": ("E01_main", "-gnot-E01-"),
}

# 配置名里代表该子集的片段。
# ⚠ plume2 换过数据（2026-09-22 起用修正版），配置/run 名里多了 `-corr`。
#   原先 MODEL_RUNS 直接拿子集名拼模板，于是对 plume2 拼出的是**换代前**
#   的目录名 `...-plume2-seed0` —— 五个模型全部 FileNotFoundError，
#   等于本脚本对 plume2 一天都没跑通过。同理日期前缀也随换代而变
#   （plume1d 09-17、plume2 09-22），所以这里改用 glob 匹配。
SUBSET_TAG = {"plume1d": "plume1d", "plume2": "plume2-corr"}


def run_dir_of(model: str, subset: str, stage_override: str = None) -> str:
    """定位 run 目录。**glob 匹配、要求唯一命中**。

    不拼死名字：日期前缀随子集换代而变，写死会出现「plume2 找不到」而
    只有跑起来才发现。命中 0 个或 >1 个都直接报错 —— 后者比前者危险
    （静默用到错的那个 run）。
    """
    import glob
    stage, token = MODEL_RUNS[model]
    stage = stage_override or stage
    pat = os.path.join(EXPERIMENTS, stage, "runs",
                       "*{}{}-seed0".format(token, SUBSET_TAG[subset]))
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise FileNotFoundError(
            "run 目录应唯一匹配，实得 {} 个：{}\n  模式：{}".format(
                len(hits), [os.path.basename(h) for h in hits], pat))
    return hits[0]


# --------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------

def predict_gnn(run_dir, device):
    """GNN 系：复用 ``eval_checkpoint`` 的全链路（与 E01/E02 口径逐行同构）。"""
    import eval_checkpoint as ec

    args = ec.load_args(run_dir)
    model, graph_creator = ec.build_env(args, device)
    epoch, _ = ec.load_checkpoint(model, run_dir, which="best", device=device)
    model = model.to(device).eval()

    loader, n = ec.build_test_loader(args)
    pred, truth = ec.predict_all(args, model, loader, graph_creator, device)
    return pred, truth, epoch, args["exp_id"], "best"


def predict_external(run_dir, device, model_id):
    """外部三模型：走 ``predict_external`` 的既有实现（返回数组，不落盘）。"""
    import predict_external as pe

    fn = {"fno": pe.predict_fno, "Transolver": pe.predict_transolver,
          "GNOT": pe.predict_gnot}[model_id]
    pred, truth = fn(run_dir, device)
    return pred, truth, None, os.path.basename(os.path.abspath(run_dir)), "best"


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------

def save_predictions(pred, truth, out_dir, exp_id, meta: dict):
    """写 ``<out_dir>/<exp_id>/predictions.hdf5``。

    ⚠ **不要**写到 run 目录的 ``eval/predictions.hdf5`` —— 那是
    ``predict_external.save_preds`` 的地盘，会覆盖 E01/E02 的既有产物。
    E04 的产出放在 ``E04_diagnostics_cost/runs/`` 下，与训练 run 目录分开。
    """
    import h5py
    d = os.path.join(out_dir, exp_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "predictions.hdf5")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with h5py.File(tmp, "w") as f:
        # pred/truth 形状 (n_traj, T, H, W, C)，逐场按名字存
        for c_i, c in enumerate(em.FIELD_ORDER):
            f.create_dataset(c, data=pred[..., c_i].astype(np.float32),
                             compression="gzip", compression_opts=4)
            f.create_dataset("truth_{}".format(c),
                             data=truth[..., c_i].astype(np.float32),
                             compression="gzip", compression_opts=4)
        for k, v in meta.items():
            f.attrs[k] = v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def evaluate_eq9(pred, truth, subset):
    """按 Eq.(9) 算逐场指标 —— 口径与 ``eval_checkpoint.evaluate_eq9`` 完全一致。"""
    import eval_checkpoint as ec
    eps = ec.fit_epsilon(subset=subset)
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER])

    bad = ~np.isfinite(pred)
    n_bad = int(bad.sum())
    if n_bad:
        pred = np.where(bad, truth, pred)      # 声明式填补，计数另记

    sse, energy = em.weighted_squares(pred, truth)
    E = em.aggregate_E(em.relative_l2_per_frame(pred, truth, eps_vec))
    rmse = em.aggregate_E(em.rmse_per_frame(pred, truth))
    trig = em.epsilon_triggered(sse, energy, eps_vec)
    return {c: float(E[i]) for i, c in enumerate(em.FIELD_ORDER)}, \
           {c: float(rmse[i]) for i, c in enumerate(em.FIELD_ORDER)}, \
           eps, trig, n_bad


def write_eq9_csv(rows: list, subset: str):
    import csv
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "single_window_eq9_{}.csv".format(subset))
    header = (["exp_id", "model", "subset", "epoch", "n_test_samples"]
              + ["E_{}".format(c) for c in em.FIELD_ORDER]
              + ["rmse_{}".format(c) for c in em.FIELD_ORDER]
              + ["eps_triggered", "n_nonfinite", "generated_at"])
    with open(path, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E04 单窗口协议推理")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_RUNS))
    ap.add_argument("--subset", required=True, choices=["plume1d", "plume2"])
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--run", default=None, help="覆盖默认 run 目录")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-save-preds", action="store_true",
                    help="只算指标，不落盘预测（省空间）")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    run_dir = os.path.abspath(a.run) if a.run else run_dir_of(a.model, a.subset)

    print("E04 推理 —— model={} subset={} device={}".format(a.model, a.subset, a.device))
    print("  run:", run_dir)

    if a.model in ("mfgno", "mpnn"):
        pred, truth, epoch, exp_id, which = predict_gnn(run_dir, device)
        model_id = "MFGNO" if a.model == "mfgno" else "MPNN"
    else:
        mid = {"fno": "fno", "transolver": "Transolver", "gnot": "GNOT"}[a.model]
        pred, truth, epoch, exp_id, which = predict_external(run_dir, device, mid)
        model_id = mid

    print("  pred {}  truth {}".format(pred.shape, truth.shape))

    E, rmse, eps, trig, n_bad = evaluate_eq9(pred, truth, a.subset)
    print("  E_c =", " ".join("{}={:.4e}".format(c, E[c]) for c in em.FIELD_ORDER))
    print("  RMSE =", " ".join("{}={:.3e}".format(c, rmse[c]) for c in em.FIELD_ORDER))
    if n_bad:
        print("  ⚠ 非有限预测 {} 个".format(n_bad))
    if any(v > 0 for v in trig.values()):
        print("  ε 触发:", {k: v for k, v in trig.items() if v > 0})

    meta = {"exp_id": exp_id, "model": a.model, "model_id": model_id,
            "subset": a.subset, "which": which,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_run": run_dir, "eps_c": eps, "E_per_field": E}

    if not a.no_save_preds:
        p = save_predictions(pred, truth, RUNS, exp_id, meta)
        print("  落盘:", p)

    os.makedirs(RESULTS, exist_ok=True)
    jpath = os.path.join(RESULTS, "{}_eq9.json".format(exp_id))
    with open(jpath, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    row = ([exp_id, a.model, a.subset, epoch if epoch is not None else "",
            int(pred.shape[0])]
           + [E[c] for c in em.FIELD_ORDER] + [rmse[c] for c in em.FIELD_ORDER]
           + [json.dumps(trig), n_bad, meta["generated_at"]])
    # 追加式：同一子集多次调用（不同模型）会累积到同一张表
    path = os.path.join(RESULTS, "single_window_eq9_{}.csv".format(a.subset))
    if os.path.exists(path):
        import csv
        with open(path, "r", encoding="utf8") as f:
            existing = list(csv.reader(f))
        header, body = existing[0], [r for r in existing[1:] if r and r[0] != exp_id]
        body.append([str(x) for x in row])
        with open(path, "w", encoding="utf8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(body)
    else:
        write_eq9_csv([row], a.subset)
    print("  指标:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
