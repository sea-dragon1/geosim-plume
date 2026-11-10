#!/usr/bin/env python3
# coding=utf-8
"""打印成本实测所在的软硬件环境（一次性探针，供 APP-COST 引用）。

「实测于什么环境」是成本数字可比性的前提，但 `nvidia-smi`/`scontrol` 在登录节点
看不到计算节点的 GPU 型号，所以用一个最小 GPU 作业把它取回来。
产物：`E04_diagnostics_cost/results/env.json`。

> ⚠ 更正（2026-09-25）：本节原先写「同样的字段也已加入 `cost.py` 的输出，
> 以后每次测量都会自带」—— **不成立**，`cost.py` 的 row 里从来没有 hostname /
> gpu / torch 这些字段，`cost_*.json` 里也没有。
> 所以「某个成本数字是在哪台机器上测的」只能靠**日志**对应：
> 实测 `logs/cost-3578352.out:2` 的 `host : gsn11` 与 `env.json` 的
> `hostname: gsn11` 一致 —— 但这条联系只存在于未发布的日志里，产物本身断链。
> 要把环境写进产物，需在 `cost.py` 的 row 里补字段，并**重跑**才能填上。
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys

import torch


def env_info():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = "{}.{}".format(*cap)
        info["cuda_version"] = torch.version.cuda
        info["gpu_total_mem_GiB"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
    return info


if __name__ == "__main__":
    e = env_info()
    for k, v in e.items():
        print("{:<26} {}".format(k, v))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "env.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        json.dump(e, f, indent=2, ensure_ascii=False)
    print("\n写出:", os.path.normpath(out))
