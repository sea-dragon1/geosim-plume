#!/usr/bin/env python3
# coding=utf-8
"""E04 效率测量：参数量 / 峰值显存 / step 时间 / 推理时延 / 图构建成本 / 总训练时长。

填 ``tables/06_cost.tex`` 用的。那张表的列是
``Params (M) & Peak GiB & Train hours & Infer. ms & E_vx/E_vy``，表注要求
"Specify device, precision, training batch, timing boundary, and graph
construction policy before filling. Inference is batch one."

几条刻意的做法
--------------
1. **登录节点没有 CUDA** —— 脚本在 ``torch.cuda.is_available()`` 为假时直接退出。
   拿 CPU 的"时延"填进 GPU 表是错的，不能靠标注补救。
2. **一个进程一个模型**（``--model`` 单值）：FNO / GNOT / Transolver 各有
   ``utils.py`` / ``train.py``，且 ``isolate_paths`` 会摘掉 REPO 路径而 GNN 家族依赖它。
3. **参数量用 ``Σ numel``**，不是 ``print_model_summary`` —— 后者只统计
   ``requires_grad``（``vendor/utils_visual.py``），与 ``E04/MATH.md`` 的定义不符。
   MPNN 是 ``nn.ModuleList``，6 个子网要**全部**计入。
4. **图构建单列**，且**区分重建与缓存更新** —— 这个区分只存在于
   ``vendor/utils_tubor.py`` 的 ``GraphCreator``（有 ``static_graph`` 缓存），
   ``utils.py`` 那份没有。必须用前者，否则测出来的成本与训练实际不符。
5. **OOM 不隐瞒**：捕获后记 ``oom_note`` 并把 peak 置为 ``None``（表里显示为占位），
   由 ``make_tables.py`` 把原因拼进表注。**不做自动重试** —— 原先这里写的是
   「把 batch 折半重试」，但代码从来没实现过；本轮五个模型都没有 OOM。
6. **总训练时长以 ``sacct`` 的作业墙钟为准**，日志时间戳只作回退 —— 含被 48h
   墙钟杀掉的分段。实测 A 组 plume1d 是 **3 段**（不是原先写的「48h + 4h 两段」）：
   0.06h FAILED + 48.00h TIMEOUT + 4.08h COMPLETED = **52.14h**。
   只按日志里的 start/end 配对会把被杀的 48h 整段丢掉（记成 4.13h）。

用法::

    python -B cost.py --model mfgno --subset plume1d        # 单个模型
    python -B cost.py --aggregate                            # 合并成 cost_table.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
E04_ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.dirname(E04_ROOT)
COMMON = os.path.join(EXPERIMENTS, "common")
# ⚠ E03 的 src 也要在 path 上：本脚本的图构建计时用 ``data_full`` 读全时程轨迹，
#   而那个模块住在 E03_rollout/src 下。漏掉它会让 mfgno/mpnn 在
#   `import data_full` 处直接 ModuleNotFoundError（实测踩过）。
E03_SRC = os.path.join(EXPERIMENTS, "E03_rollout", "src")
for _p in (COMMON, E03_SRC, HERE):
    if _p in sys.path:          # ⚠ 先删再插：写反成 `not in` 会在 import 阶段就抛
        sys.path.remove(_p)     #    ValueError: list.remove(x): x not in list（实测踩过）
    sys.path.insert(0, _p)

import numpy as np      # noqa: E402
import torch            # noqa: E402

import model_zoo as mz                       # noqa: E402
from predict_test import run_dir_of          # noqa: E402

RESULTS = os.path.join(E04_ROOT, "results")
LOGS = os.path.join(EXPERIMENTS, "logs")

MODELS = ["mfgno", "mpnn", "fno", "transolver", "gnot"]

# 各模型训练配置里的真实 batch（用于测"训练 step"而非推理）
TRAIN_BATCH = {"mfgno": 2, "mpnn": 2, "fno": 32, "transolver": 4, "gnot": 2}


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------

def timed(fn, warmup=5, reps=20, device=None):
    """预热 + synchronize + 多次重复（``MATH.md`` 的硬要求）。

    返回 ``(mean_s, std_s)``。缺任何一步（尤其 synchronize）测出来的都是
    "内核入队时间"而不是真实耗时。
    """
    for _ in range(warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.array(ts)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


# --------------------------------------------------------------------------
# 训练时长：从日志累加
# --------------------------------------------------------------------------

def _job_ids_for(exp_id: str):
    """从日志文件名反查该 exp_id 用过的 job id。

    日志是 ``logs/train-<jobid>.out``，内容里带 exp_id —— 这就是 job ↔ exp_id 的
    连接键。**不要**指望 sacct 直接按 exp_id 查：JobName 全是 ``geosim-train``。
    """
    out = []
    for p in sorted(glob.glob(os.path.join(LOGS, "train-*.out"))):
        try:
            if exp_id in open(p, "r", encoding="utf8", errors="ignore").read():
                m = re.search(r"train-(\d+)\.out$", p)
                if m:
                    out.append(m.group(1))
        except OSError:
            continue
    return out


def train_hours_from_logs(exp_id: str):
    """累加该 exp_id 的**作业墙钟**（小时），优先用 ``sacct``。

    ⚠ ``tables/06_cost.tex`` 的表注承诺的是 "accumulated job wall time from
    ``sacct``" —— 所以这里必须真的用 sacct，不能给出一个来源不同的数。早先本函数
    只解析日志里的 ``start:``/``end:``，与表注**口径不符**（虽然两法数值接近）。

    ⚠ 被墙钟杀掉的分段**没有 ``end:``**：SLURM 到时限直接杀 python，
    ``run_train.sh`` 末尾那对 ``echo end`` 执行不到。旧实现用 ``zip(starts, ends)``
    配对，把最长的一段整个丢掉 —— 实测 ``mpnn-E02-A-plume1d-seed0`` 记成 4.13 h，
    真实是 0.06 + **48.00** + 4.09 ≈ **52.1 h**（sacct 给 3544874 = 2-00:00:02
    TIMEOUT，与日志 mtime 估出的 48.00 h 一致到 0.02 h）。

    每个 job 先查 sacct；查不到（记录过期）才回退到日志时间戳，并在
    ``fallback_jobs`` 里报出来 —— 别把回退值当精确值用。
    """
    from datetime import datetime as _dt
    jobs = _job_ids_for(exp_id)
    total, segs, fallback = 0.0, [], []

    for jid in jobs:
        h = _sacct_hours(jid)
        if h is None:
            h = _log_hours(jid)
            fallback.append(jid)
        if h and h > 0:
            total += h
            segs.append(round(h, 2))

    return {"train_hours_total": round(total, 2), "segments": segs,
            "n_segments": len(segs), "jobs": jobs, "fallback_jobs": fallback,
            "source": ("sacct" if not fallback else "sacct（{} 个 job 回退到日志）"
                       .format(len(fallback)))}


def _sacct_hours(job_id: str):
    """``sacct`` 里该 job 的 Elapsed（小时）；查不到返回 None。"""
    import subprocess
    try:
        r = subprocess.run(["sacct", "-j", str(job_id), "-n", "-P",
                            "-o", "Elapsed"],
                           capture_output=True, text=True, timeout=30)
    except Exception:                       # 没有 sacct / 超时
        return None
    for line in r.stdout.splitlines():
        line = line.strip()
        # 形如 ``2-00:00:02`` 或 ``04:05:06``；跳过 ``00:00:00``（.batch/.extern 行）
        if not line or line in ("00:00:00", "Unknown"):
            continue
        try:
            # ⚠ 只有 ≥24h 才有 ``D-HH:MM:SS`` 形式；<24h 是 ``HH:MM:SS``。
            #   早先用 partition("-") 一把梭：没有 "-" 时整串落进 days、tod 为空，
            #   于是**所有不足一天的作业都解析失败并静默回退**（实测踩过：
            #   MPNN 的 48h 段能解析、两个几小时的段全回退了）。
            if "-" in line:
                d, _, tod = line.partition("-")
                days = int(d)
            else:
                days, tod = 0, line
            hh, mm, ss = (int(x) for x in tod.split(":"))
            return days * 24 + hh + mm / 60.0 + ss / 3600.0
        except ValueError:
            continue
    return None


def _log_hours(job_id: str):
    """回退：从 ``logs/train-<jobid>.out`` 的 start/end 标记估时长。

    规则（按标记出现顺序走）：遇 ``start`` 时若已有未闭合段 → 前一段判为**被杀**，
    闭到新 ``start``；遇 ``end`` 闭合当前段；文件尾仍未闭合 → 用该日志的 **mtime**
    闭合（作业死后再无写入）。
    """
    from datetime import datetime as _dt
    p = os.path.join(LOGS, "train-{}.out".format(job_id))
    if not os.path.exists(p):
        return None
    txt = open(p, "r", encoding="utf8", errors="ignore").read()
    marks = [(m.group(1), m.group(2))
             for m in re.finditer(r"^(start|end)\s*:\s*(\S+)", txt, re.M)]
    total, open_start = 0.0, None
    for kind, ts in marks:
        if kind == "start":
            if open_start is not None:
                total += max(0.0, (_dt.fromisoformat(ts) - open_start
                                   ).total_seconds() / 3600.0)
            open_start = _dt.fromisoformat(ts)
        elif open_start is not None:
            total += max(0.0, (_dt.fromisoformat(ts) - open_start
                               ).total_seconds() / 3600.0)
            open_start = None
    if open_start is not None:
        # ⚠ mtime 是 naive 本地时间，start 标记带 +08:00；不 astimezone() 会抛
        #   "can't subtract offset-naive and offset-aware datetimes"。
        mt = _dt.fromtimestamp(os.path.getmtime(p)).astimezone()
        total += max(0.0, (mt - open_start).total_seconds() / 3600.0)
    return total


# --------------------------------------------------------------------------
# 图构建成本
# --------------------------------------------------------------------------

def graph_cost(loaded, subset, device, n_traj=1):
    """图构建成本：**重建** vs **缓存更新**分别计时，并报 E / N / d / L / batch。"""
    import data_full as df

    gc = loaded.ctx.get("graph_creator")
    if gc is None:
        return {}

    fd = df.read_split(subset, "test", rows=range(n_traj), verbose=False)
    tw = int(gc.tw)
    variables = _variables_as_batch(loaded.args)
    u = fd.flat()
    steps = [tw]
    data, labels = gc.create_data(u, steps)
    t = fd.t

    # 重建（含 radius_graph）
    rebuild_s, rebuild_std = timed(
        lambda: gc.build_static_graph(fd.coords, data, labels, t, steps,
                                      n_traj, variables, device),
        warmup=2, reps=5, device=device)
    graph, _ = gc.build_static_graph(fd.coords, data, labels, t, steps,
                                     n_traj, variables, device)

    # 缓存更新（只换节点特征）
    upd_s, upd_std = timed(
        lambda: gc.update_graph_data(graph, graph.x, labels, steps, t, device),
        warmup=3, reps=10, device=device)

    e_edges = int(graph.edge_index.shape[1]) if getattr(graph, "edge_index", None) is not None else -1
    n_nodes = int(graph.x.shape[0])
    d = int(loaded.args.get("model", {}).get("hidden_features", -1))
    L = int(loaded.args.get("model", {}).get("hidden_layer", -1))
    H = tw
    return {
        "graph_rebuild_ms": round(rebuild_s * 1e3, 3),
        "graph_rebuild_std_ms": round(rebuild_std * 1e3, 3),
        "graph_cached_update_ms": round(upd_s * 1e3, 3),
        "graph_cached_update_std_ms": round(upd_std * 1e3, 3),
        "N_nodes": n_nodes, "E_edges": e_edges,
        "mean_degree": round(e_edges / max(n_nodes, 1), 2),
        # ⚠ 键名必须是 graph_batch，**不能叫 batch** —— measure_one 用
        #   ``**gcost`` 把本字典展开进 row，而 row 里已经有一个 "batch"
        #   （真实训练 batch，来自 TRAIN_BATCH）。同名会把训练 batch 覆盖成
        #   这里的 n_traj（默认 1），实测后果：mfgno/mpnn 的 batch 列报 1
        #   （训练其实是 2）、gnot 报 2（配置与表 5 都是 4）。
        "d_width": d, "L_layers": L, "H_window": H, "graph_batch": n_traj,
        # 论文里的复杂度量级，仅作参考（实测值优先）
        "complexity_L_E_d2_6Hd": (L * e_edges * (d * d + 6 * H * d)) if d > 0 else -1,
    }


# --------------------------------------------------------------------------
# 主测量
# --------------------------------------------------------------------------

def _variables_as_batch(args):
    """历史名，转发到 :func:`model_zoo.variables_as_batch`（唯一实现）。

    保留本名只为不改动既有调用点；**新代码请直接用 ``mz.variables_as_batch``**。
    """
    return mz.variables_as_batch(args)


def measure_one(model_name: str, subset: str, device, which: str = "best"):
    print("── {} / {} ──".format(model_name, subset))
    # ⚠ 集群装的 dgl 2.1.0 是 CPU-only 编译，**图**不能上 GPU；但这不妨碍
    #   模型在 GPU 上跑 —— 训练脚本 plume1/train.py 只搬 u_p/inputs/ndata、
    #   从不搬 ``g``，而它的日志是 "Using device: cuda:0"。
    #   所以这里**不做设备降级**：图留 CPU，模型与张量仍按 device 走，
    #   显存与推理时延就都是 GPU 数字，可与其余四行并列。
    if model_name == "gnot" and device.type != "cpu":
        print("   (dgl 图留 CPU，模型与张量在 {}；与训练脚本一致)".format(device))
    run_dir = run_dir_of(model_name, subset)
    loaded = mz.load_run(run_dir, device, which=which)
    exp_id = loaded.exp_id
    print("   {}  device={}".format(loaded.describe(), device))

    params = mz.count_params(loaded.model)

    if loaded.family in mz.GNN_FAMILIES:
        import data_full as df
        fd = df.read_split(subset, "test", rows=range(1), verbose=False)
        n_frame = df.spec_of(subset)["n_frame_test"]
        loaded = mz.load_run(run_dir, device, which=which, frames=n_frame)
        gc = loaded.ctx["graph_creator"]
        tw = int(gc.tw)
        variables = _variables_as_batch(loaded.args)
        steps = [tw]
        data, labels = gc.create_data(fd.flat(), steps)
        graph, _ = gc.build_static_graph(fd.coords, data, labels, fd.t, steps, 1,
                                         variables, device)

        def fwd():
            return (loaded.model(graph, fd.t, 1) if loaded.family == "gnn_joint"
                    else [loaded.model[i](graph, fd.t, i, 1) for i in range(6)])
        infer_s, infer_std = timed(fwd, warmup=3, reps=10, device=device)
        gcost = graph_cost(loaded, subset, device)
        note = ""
    else:
        # 外部模型的单帧输入随家族而异，这里只测"一次前向"的量级
        infer_s, infer_std = _external_forward_time(loaded, device)
        gcost = {}
        note = ("GNOT 走 CPU（dgl 2.1.0 是 CPU-only 编译），"
                "其延迟与其余四行的 GPU 数字不可比" if loaded.family == "gnot" else "")

    # 峰值显存：跑一次完整前向 + 反向（训练步的近似）
    peak_gib = None
    oom_note = ""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _train_step_probe(loaded, device, subset)
            peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_note = "OOM at batch={}".format(TRAIN_BATCH.get(model_name, "?"))
                peak_gib = None
            else:
                raise
    else:
        # 走到这里说明真的没拿到 GPU 数字。GNOT 不该再落到这一支 —— 它的图留 CPU
        # 但模型在 GPU 上跑，显存与其余行可比。若它仍在此处，说明设备选择有误。
        oom_note = ("N/A (ran on CPU; for GNOT this is unexpected — "
                    "the graph stays on CPU but the model should be on GPU)"
                    if model_name == "gnot" else "no CUDA on this node")

    th = train_hours_from_logs(exp_id)
    row = {
        "model": model_name, "model_id": loaded.model_id, "family": loaded.family,
        "subset": subset, "exp_id": exp_id, "ckpt_epoch": loaded.ckpt_epoch,
        "params_M": round(params / 1e6, 3),
        "peak_alloc_GiB": peak_gib,
        "batch": TRAIN_BATCH.get(model_name, None),
        "precision": "fp32",           # 所有配置的 if_amp 都是 False
        "infer_latency_ms": round(infer_s * 1e3, 3),
        "infer_latency_std_ms": round(infer_std * 1e3, 3),
        "device": str(device),
        "train_hours_total": th["train_hours_total"],
        "train_segments": json.dumps(th["segments"]),
        # ⚠ 这几个字段原先算出来就被丢掉，后果是 `tab:cost` 表注承诺的
        #   「accumulated job wall time from sacct」**无法从产物复核**：
        #   读者看不到某个数是 sacct 来的、还是日志时间戳回退来的。
        #   实测五个模型全部走 sacct、无回退，但那需要写进 JSON 才算证据。
        "train_hours_source": th["source"],
        "train_fallback_jobs": json.dumps(th["fallback_jobs"]),
        "train_job_ids": json.dumps(th["jobs"]),
        "train_n_segments": th["n_segments"],
        "gpu_hours": th["train_hours_total"],       # 单卡
        "oom_note": oom_note, "note": note,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **gcost,
    }
    print("   params={:.3f} M  infer={:.1f}±{:.1f} ms  peak={} GiB".format(
        row["params_M"], row["infer_latency_ms"], row["infer_latency_std_ms"],
        row["peak_alloc_GiB"]))
    if gcost:
        print("   图: E={} N={} 度={} | 重建 {:.1f} ms vs 缓存更新 {:.1f} ms".format(
            gcost["E_edges"], gcost["N_nodes"], gcost["mean_degree"],
            gcost["graph_rebuild_ms"], gcost["graph_cached_update_ms"]))
    print("   训练时长: {:.2f} h（{} 段 {}）".format(
        row["train_hours_total"], th["n_segments"], th["segments"]))
    return row


def _external_forward_time(loaded, device):
    """外部模型的一次前向耗时。用各自 dataset 的第一帧做输入。"""
    tw = 10
    if loaded.family == "fno":
        data = loaded.ctx["data"]          # (N,H,W,T,C)
        grid = loaded.ctx["grid"]
        y = data[0].unsqueeze(0).to(device)
        x = y[..., :tw, :].clone()
        H, W = data.shape[1], data.shape[2]

        def fwd():
            return loaded.model(x.reshape(1, H, W, -1), grid)
    elif loaded.family == "transolver":
        data = loaded.ctx["data"]          # (n,T,H,W,C)
        grid = loaded.ctx["grid"]
        n, T, H, W, C = data.shape
        N = H * W
        cur = data[0, :tw].permute(1, 2, 0, 3).reshape(N, -1).unsqueeze(0).to(device)
        g = grid.unsqueeze(0)

        def fwd():
            return loaded.model(g, cur)
    else:                                   # gnot
        ds = loaded.ctx["dataset"]
        MT = loaded.ctx["MultipleTensors"]
        idxs = [i for i, (_s, step) in enumerate(ds._indices) if step == tw]
        g, u_p, inputs = ds[idxs[0]]
        # 图**不搬**（dgl CPU-only 编译）；只搬张量，与 plume1/train.py 一致
        u_p = u_p.to(device)
        inputs = MT([t.unsqueeze(0).to(device) for t in inputs])

        def fwd():
            return loaded.model(g, u_p, inputs)
    return timed(fwd, warmup=2, reps=5, device=device)


def _train_step_probe(loaded, device, subset: str):
    """近似一次训练步：前向 + 反向（用于量峰值显存）。

    ⚠⚠ **两类模型的口径并不相同，同一列里不可跨组比较**：
        GNN 系（mfgno / mpnn）  → 前向 **+ 反向**
        外部基线（fno / gnot / transolver） → **只有前向**（走 _external_forward_time）
    且探针 batch 恒为 1，而真实训练 batch 是 2（GNN）/ 32（FNO）。
    所以这一列是「batch-one 的近似峰值」，不是训练全程峰值显存。

    论文侧已按上述实际口径描述（§5.5 与 ``tab:cost`` 表注都写明分组不可比），
    代码与文字一致。**若要把外部基线也做成前向+反向**（那才能跨组合并），
    必须**连 ``cost_*.json`` 一起重跑** —— 否则 JSON 里的 ``peak_alloc_GiB``
    仍是旧口径的数，而代码已换含义，两边对不上。

    不真的跑 optimizer.step（会改动权重）—— 峰值显存主要由激活决定。

    ⚠ ``subset`` 必须显式传 —— 早先这里写成从 ``loaded.args`` 里取，
    取不到就 fallback 到 "plume1d"，跑 plume2 时会**静默用错数据**。
    """
    model = loaded.model
    model.train()
    try:
        if loaded.family in mz.GNN_FAMILIES:
            import data_full as df
            fd = df.read_split(subset, "test", rows=range(1), verbose=False)
            gc = loaded.ctx["graph_creator"]
            steps = [int(gc.tw)]
            data, labels = gc.create_data(fd.flat(), steps)
            # ⚠ 必须过 mz.variables_as_batch —— 配置里 variables 是 0 维标量，
            #   直接传会让 build_static_graph 的 stack(dim=1) 抛 IndexError。
            graph, _ = gc.build_static_graph(fd.coords, data, labels, fd.t, steps, 1,
                                             mz.variables_as_batch(loaded.args), device)
            out = (model(graph, fd.t, 1) if loaded.family == "gnn_joint"
                   else [model[i](graph, fd.t, i, 1) for i in range(6)])
            loss = sum(o.float().sum() for o in (out if isinstance(out, list) else [out]))
            loss.backward()
        else:
            _external_forward_time(loaded, device)   # 只测前向
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

AGG_KEYS = ["model", "model_id", "params_M", "peak_alloc_GiB", "batch", "precision",
            "infer_latency_ms", "infer_latency_std_ms", "N_nodes", "E_edges",
            "mean_degree", "d_width", "L_layers", "graph_rebuild_ms",
            "graph_cached_update_ms", "train_hours_total", "gpu_hours", "device",
            # 训练时长的**来源**必须进聚合表，否则表注承诺的 sacct 口径不可复核
            "train_hours_source", "train_fallback_jobs", "train_n_segments",
            "oom_note", "note"]


def aggregate():
    rows, seen = [], set()
    for p in sorted(glob.glob(os.path.join(RESULTS, "cost_*.json"))):
        with open(p, "r", encoding="utf8") as f:
            d = json.load(f)
        key = (d["model"], d["subset"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(d)
    if not rows:
        print("没有 cost_*.json —— 先在计算节点上跑 cost.py --model <m>")
        return None
    out = os.path.join(RESULTS, "cost_table.csv")
    with open(out, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("写出:", out)
    print("{:<12} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
        "model", "params(M)", "peak(GiB)", "infer(ms)", "train(h)", "E_edges"))
    for r in rows:
        print("{:<12} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
            r["model"], r["params_M"], str(r["peak_alloc_GiB"]),
            r["infer_latency_ms"], r["train_hours_total"], r.get("E_edges", "-")))
    return out


def main():
    ap = argparse.ArgumentParser(description="E04 效率测量")
    ap.add_argument("--model", choices=MODELS, default=None)
    ap.add_argument("--subset", default="plume1d", choices=["plume1d", "plume2"])
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if a.aggregate:
        return 0 if aggregate() else 1

    if not a.model:
        print("需要 --model（一进程一模型）或 --aggregate")
        return 1

    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("⚠ 本节点没有 CUDA —— 效率测量必须提交到 GPU 计算节点。")
        print("  CPU 的'时延'填进 GPU 表是错的，不能用标注补救。")
        return 2

    print("E04 效率测量 —— model={} subset={} device={}".format(
        a.model, a.subset, device))
    row = measure_one(a.model, a.subset, device, which=a.which)
    path = os.path.join(RESULTS, "cost_{}_{}.json".format(a.model, a.subset))
    with open(path, "w", encoding="utf8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    print("  写出:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
