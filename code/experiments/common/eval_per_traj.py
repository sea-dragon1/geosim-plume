#!/usr/bin/env python3
# coding=utf-8
"""重跑指定的 run，保存**逐轨迹**逐场指标，供配对 bootstrap 使用。

**为什么需要它**：``eval_checkpoint.py`` 的 ``evaluate_eq9`` 只返回聚合后的
``E_per_field``（先按帧平均、再按轨迹平均），逐轨迹的 ``e[n,q,c]`` 被丢掉了。
而 ``protocol/metrics.md`` 对配对比较（如 $E_B-E_C$）规定：

    先逐轨迹相减，再重采样；不把同轨迹的帧当独立样本。

没有逐轨迹值就无法做这个 —— 用聚合值相减会**低估**配对方差。

**产物**：每个 run 下 ``eval/<exp_id>_pertraj.npz``，含
``e[N, T, C]``（逐帧逐场相对 L2）、``E_traj[N, C]``（先按帧平均）、``fields``、
``n_traj`` / ``n_frame`` / ``exp_id`` / ``seed`` / ``variant``。
数组很小（12×10×6），适合随 run 长期保存。

用法::

    # 单个 run
    python -B eval_per_traj.py --runs <run_dir> [<run_dir> ...]
    # 整个阶段（按子集筛）
    python -B eval_per_traj.py --stage E02_ablation --subset plume1d
    python -B eval_per_traj.py --stage E02_ablation --subset plume1d --device cpu
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import numpy as np      # noqa: E402

import eval_metrics as em          # noqa: E402
import eval_checkpoint as ec       # noqa: E402


def per_traj_of(pred, truth, eps):
    """逐帧逐场 ``e[N, T, C]`` → 逐轨迹 ``E_traj[N, C] = mean_q e``。"""
    eps_vec = np.array([eps[c] for c in em.FIELD_ORDER], dtype=np.float64)
    e = em.relative_l2_per_frame(pred, truth, eps_vec)      # [N, T, C]
    return e, e.mean(axis=1)                                # [N,C], [N,C]


def save_fields(run_dir, pred, truth):
    """把场落盘成 ``eval/predictions.hdf5``，**与 predict_external.save_preds 同格式**。

    这样 E01 的外部模型与 E02 的 GNN 系产出可以走同一套绘图/评分代码
    （``score_predictions.load_predictions`` 直接能读）。六个场各一个 dataset，
    形状 ``[n_traj, T, H, W]``，真值加 ``truth_`` 前缀。

    Lustre 上偶发 ``OSError: Argument list too long`` → 回退 npz，与那边一致。
    """
    import h5py
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.hdf5")
    try:
        with h5py.File(path, "w") as f:
            for c in em.FIELD_ORDER:
                i = em.FIELD_ORDER.index(c)
                f.create_dataset(c, data=pred[..., i].astype(np.float32),
                                 chunks=(1, pred.shape[1], pred.shape[2], pred.shape[3]))
                f.create_dataset("truth_" + c, data=truth[..., i].astype(np.float32),
                                 chunks=(1, truth.shape[1], truth.shape[2], truth.shape[3]))
        return path
    except OSError as e:
        print("    hdf5 写入失败（{}），回退到 npz".format(str(e)[:60]))
        npath = os.path.join(out_dir, "predictions.npz")
        np.savez_compressed(
            npath,
            **{c: pred[..., i].astype(np.float32) for i, c in enumerate(em.FIELD_ORDER)},
            **{"truth_" + c: truth[..., i].astype(np.float32)
               for i, c in enumerate(em.FIELD_ORDER)})
        return npath


def run_one(run_dir, device, eps, which="best", force=False, dump_fields=False):
    exp_id = os.path.basename(os.path.abspath(run_dir))
    out = os.path.join(run_dir, "eval", "{}_pertraj.npz".format(exp_id))
    fields_done = os.path.exists(os.path.join(run_dir, "eval", "predictions.hdf5"))
    if os.path.exists(out) and not force and (fields_done or not dump_fields):
        print("  -- 已存在，跳过: {}".format(os.path.basename(out)))
        return out

    t0 = time.time()
    res, pred, truth = ec.eval_one(run_dir, device, eps, which=which)
    e, E_traj = per_traj_of(pred, truth, eps)

    if dump_fields and not fields_done:
        p = save_fields(run_dir, pred, truth)
        print("    场已落盘: {}  ({:.0f} MB)".format(
            os.path.basename(p), os.path.getsize(p) / 1e6))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        e=e.astype(np.float32),
        E_traj=E_traj.astype(np.float64),
        fields=np.array(em.FIELD_ORDER),
        n_traj=np.int32(pred.shape[0]),
        n_frame=np.int32(pred.shape[1]),
        exp_id=np.array(exp_id),
        seed=np.int32(res.get("seed") if res.get("seed") is not None else -1),
        variant=np.array(str(res.get("variant"))),
        model_id=np.array(str(res.get("model_id"))),
        checkpoint_epoch=np.int32(res.get("checkpoint_epoch") or -1),
    )
    dt = time.time() - t0
    # 与聚合口径对一次账：E_traj 的两次平均必须等于 json 里的 E_per_field
    agg = E_traj.mean(axis=0)
    ref = np.array([res["E_per_field"][c] for c in em.FIELD_ORDER])
    rel = np.abs(agg - ref) / np.maximum(np.abs(ref), 1e-300)
    print("  {}  [{:.1f}s]  逐轨迹聚合 vs 聚合口径 最大相对差 {:.2e}".format(
        exp_id, dt, rel.max()))
    if rel.max() > 1e-9:
        raise AssertionError("逐轨迹聚合与聚合口径不一致 —— 实现有问题，不要继续")
    return out


def collect(stage, subset, runs):
    if runs:
        return [os.path.abspath(r) for r in runs]
    pat = os.path.join(EXPERIMENTS, stage, "runs", "*")
    out = []
    for d in sorted(glob.glob(pat)):
        name = os.path.basename(d)
        if "SMOKE" in name or (subset and subset not in name):
            continue
        if os.path.isdir(os.path.join(d, "checkpoints")):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description="保存逐轨迹指标（配对 bootstrap 用）")
    ap.add_argument("--stage", default="E02_ablation")
    ap.add_argument("--subset", default="plume1d", choices=["plume1d", "plume2"])
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--which", default="best")
    ap.add_argument("--force", action="store_true", help="已有产物也重算")
    ap.add_argument("--dump-fields", action="store_true",
                    help="顺带落盘 eval/predictions.hdf5（F03 六场图需要）")
    a = ap.parse_args()

    import torch
    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    runs = collect(a.stage, a.subset, a.runs)
    if not runs:
        print("没有找到 run")
        return 1
    print("待评估 {} 个 run".format(len(runs)))

    eps = ec.fit_epsilon(subset=a.subset)
    print("ε_c: {}".format({k: "{:.3e}".format(v) for k, v in eps.items()}))

    ok = 0
    for d in runs:
        print("[{}]".format(os.path.basename(d)))
        try:
            run_one(d, device, eps, which=a.which, force=a.force,
                    dump_fields=a.dump_fields)
            ok += 1
        except Exception as ex:                       # noqa: BLE001
            import traceback
            print("  [失败] {}: {}".format(type(ex).__name__, ex))
            traceback.print_exc()
    print("\n完成 {}/{}".format(ok, len(runs)))
    return 0 if ok == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
