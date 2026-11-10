#!/usr/bin/env python3
# coding=utf-8
"""收集 ``tables/05_training.tex`` / ``06_cost.tex`` 需要的**可追溯**元数据。

**为什么不手工填**：待填清单第 6 节明确要求"先形成 run manifest，再把数字填进
论文；不要从截图或多个历史 README 手动拼成主表"。所以参数量从 checkpoint 数、
时长从 ``sacct`` 取、超参从 run 的 config 读 —— 全部可重跑复核。

**口径**（写进表的 caption，不藏）：

- **参数量**：``sum(numel)`` over ``model_state_dict``。A 基线是 6 个独立子网，
  这里统计的是**六个网络的合计**（清单点名要求）。
- **训练时长**：``sacct`` 的**作业墙钟累加**，含被 48h 墙钟杀掉的分段与续跑段，
  不是"纯计算时间"。失败即退的段（<0.1h）也如实记入。
- **Peak GiB / Infer ms**：需在 GPU 上实测（``E04_diagnostics_cost/src/cost.py``），
  本脚本**不猜**；已有产物则读，没有就留空。

用法::

    python -B collect_meta.py                 # 收集并写 results/model_meta.json
    python -B collect_meta.py --print
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import paths  # noqa: E402

REPO = paths.GEOSIM_ROOT

# 每个模型取哪个 run 作为代表（plume1d，seed0；A/B/C 取对应变体）
RUNS = [
    ("FNO",              "E01_main",     "2026-09-17-fno-E01-plume1d-seed0", True),
    ("GNOT",             "E01_main",     "2026-09-17-gnot-E01-plume1d-seed0", True),
    ("Transolver",       "E01_main",     "2026-09-17-transolver-E01-plume1d-seed0", True),
    ("Independent MPNN", "E02_ablation", "2026-09-17-mpnn-E02-A-plume1d-seed0", False),
    ("MFGNO",            "E02_ablation", "2026-09-17-mfgno-E02-C-equal-plume1d-seed0", False),
    ("A control",        "E02_ablation", "2026-09-17-mpnn-E02-A-plume1d-seed0", False),
    ("B control",        "E02_ablation", "2026-09-17-mfgno-E02-B-equal-plume1d-seed0", False),
    ("C control",        "E02_ablation", "2026-09-17-mfgno-E02-C-equal-plume1d-seed0", False),
]

# 宽/深的人读描述（从 config 的 model 段提炼；config 本身随 JSON 一起存盘备查）
MANUAL_STRUCT = {
    "FNO":              ("width 48 / 4 Fourier layers, modes 24",
                         "MSE (per-frame)"),
    "GNOT":             ("hidden 128 / 3 layers, 2 experts",
                         "MSE (per-frame)"),
    "Transolver":       ("hidden 256 / 5 layers, 8 heads",
                         "MSE (per-frame)"),
    "Independent MPNN": ("hidden 200 / 6 layers, 6 independent nets",
                         "MSE, equal-field weights"),
    "MFGNO":            ("hidden 512 / 6 layers, joint 6-field",
                         "MSE, equal-field weights"),
    "A control":        ("hidden 200 / 6 layers, 6 independent nets",
                         "MSE, equal-field weights"),
    "B control":        ("hidden 512 / 6 layers, zeroed difference slot",
                         "MSE, equal-field weights"),
    "C control":        ("hidden 512 / 6 layers, full joint",
                         "MSE, equal-field weights"),
}


def _canon(cfg_name):
    p = os.path.basename(cfg_name)
    p = re.sub(r"^RESUME-", "", p)
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", p)


def _hours_from_log(job_id):
    """从 ``logs/train-<jobid>.out`` 的 start/end 标记估算墙钟。

    **回退路径**：sacct 历史不可用（换机器、记录过期）时使用。规则与
    ``E04_diagnostics_cost/src/cost.py::_log_hours`` 一致：
    遇 ``start`` 时若已有未闭合段 → 前一段判为被杀，闭到新 ``start``；
    遇 ``end`` 闭合；文件尾仍未闭合 → 用日志 **mtime** 闭合。
    """
    from datetime import datetime as _dt
    p = os.path.join(EXPERIMENTS, "logs", "train-{}.out".format(job_id))
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
        # mtime 是 naive 本地时间，start 标记带 +08:00；不 astimezone() 会抛
        # "can't subtract offset-naive and offset-aware datetimes"。
        mt = _dt.fromtimestamp(os.path.getmtime(p)).astimezone()
        total += max(0.0, (mt - open_start).total_seconds() / 3600.0)
    return total


def job_hours_by_config():
    """``config 文件名(规范化)`` → 墙钟小时数。

    优先 ``sacct``（论文口径，含 TIMEOUT 段）；sacct 不可用或查不到时，
    回退到日志里的 start/end（见 :func:`_hours_from_log`）。
    """
    j2c = {}
    for f in sorted(glob.glob(os.path.join(EXPERIMENTS, "logs", "train-*.out"))):
        txt = open(f, "r", errors="ignore").read()
        m = re.search(r"config\s*:\s*(\S+)", txt)
        if not m:
            continue
        c = _canon(m.group(1))
        if c.startswith(("SMOKE", "EVAL")):
            continue
        j2c[os.path.basename(f).replace("train-", "").replace(".out", "")] = c
    if not j2c:
        return {}, {}
    try:
        out = subprocess.run(["sacct", "-j", ",".join(sorted(j2c)),
                              "--format=JobID,Elapsed,State", "--parsable2", "--noheader"],
                             capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = ""
    el = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) < 3 or "." in p[0]:
            continue
        el[p[0]] = (p[1], p[2])
    agg, states = {}, {}
    for jid, cfg in j2c.items():
        if jid in el:
            e, st = el[jid]
            h, mm, ss = (e.split(":") + ["0", "0"])[:3]
            hours = (int(h.split("-")[0]) * 24 + int(h.split("-")[1])) if "-" in h else int(h)
            hours += int(mm) / 60 + int(ss) / 3600
        else:
            hours = _hours_from_log(jid)
            st = "LOG"
            if hours is None:
                continue
        agg[cfg] = agg.get(cfg, 0.0) + hours
        states.setdefault(cfg, []).append("{}:{:.2f}h/{}".format(jid, hours, st))
    return agg, states


def count_params(run_dir):
    """从 best checkpoint 数参数量。"""
    import torch
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*-best.pt")))
    if not cks:
        return None, None
    ck = torch.load(cks[0], map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    return total, os.path.basename(cks[0])


def read_config(stage, exp_id):
    c = sorted(glob.glob(os.path.join(EXPERIMENTS, stage, "configs",
                                      "*{}*.yaml".format(exp_id))))
    c = [p for p in c if "SMOKE" not in os.path.basename(p)
         and not os.path.basename(p).startswith("RESUME")]
    if not c:
        return None, None
    import yaml
    with open(c[0], "r", encoding="utf8") as f:
        return yaml.safe_load(f), os.path.basename(c[0])


def main():
    ap = argparse.ArgumentParser(description="收集 05/06 表所需的可追溯元数据")
    ap.add_argument("--print", dest="do_print", action="store_true")
    a = ap.parse_args()

    hours, states = job_hours_by_config()
    meta = {"collected_at": datetime.now().isoformat(timespec="seconds"),
            "wallclock_source": "sacct (job wall time, incl. TIMEOUT segments)",
            "params_source": "sum(numel) over model_state_dict of *-best.pt",
            "models": {}}

    for disp, stage, exp_id, is_external in RUNS:
        run_dir = os.path.join(EXPERIMENTS, stage, "runs", exp_id)
        cfg, cfg_name = read_config(stage, exp_id)
        params, ck_name = (None, None)
        if os.path.isdir(run_dir):
            try:
                params, ck_name = count_params(run_dir)
            except Exception as e:                      # noqa: BLE001
                print("[warn] {} 参数量读取失败: {}".format(disp, e))
        struct, objective = MANUAL_STRUCT.get(disp, ("", ""))
        h = hours.get(_canon(cfg_name)) if cfg_name else None
        meta["models"][disp] = {
            "exp_id": exp_id,
            "stage": stage,
            "external_baseline": is_external,
            "width_depth": struct,
            "objective": objective,
            "params": params,
            "checkpoint": ck_name,
            "train_hours": h,
            "job_segments": states.get(_canon(cfg_name), []) if cfg_name else [],
            "epochs": (cfg or {}).get("epochs"),
            "batch_size": ((cfg or {}).get("dataloader") or {}).get("batch_size"),
            "seed": (cfg or {}).get("seed"),
            "optimizer": (cfg or {}).get("optimizer"),
            "scheduler": (cfg or {}).get("scheduler"),
            "config_file": cfg_name,
        }

    out = os.path.join(EXPERIMENTS, "E04_diagnostics_cost", "results", "model_meta.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("写出:", out)

    if a.do_print:
        print("\n%-18s %14s %9s  %s" % ("model", "params", "hours", "segments"))
        for disp in meta["models"]:
            m = meta["models"][disp]
            print("%-18s %14s %9s  %s" % (
                disp,
                "{:,}".format(m["params"]) if m["params"] else "-",
                "%.2f" % m["train_hours"] if m["train_hours"] else "-",
                ",".join(m["job_segments"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
