"""划分冻结的完整性检查。

交接文档 §3 的硬要求：验证集从训练池按轨迹划分，**禁止同一轨迹跨 split**。
本测试只检查 ``protocol/splits.yaml`` 的自洽性（不读数据文件）。

运行： ``python -B tests/test_splits.py``
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(EXPERIMENTS, "common"))

import yaml  # noqa: E402

SPLITS_PATH = os.path.join(EXPERIMENTS, "protocol", "splits.yaml")

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def expand(node, space):
    """把 ``range: [a, b]`` 或 ``ids: [...]`` 展开成带命名空间的 ID 集合。

    命名空间是必要的：plume1d 的 train 与 test 序号**各自从 0 开始**
    （``T0..T96`` 与 ``T_test0..T_test11``），按裸数值判重会误报重叠。
    """
    if "ids" in node and node["ids"] is not None:
        return {(space, i) for i in node["ids"]}
    lo, hi = node["range"]
    return {(space, i) for i in range(lo, hi + 1)}


def test_plume1d_no_leakage():
    cfg = yaml.safe_load(open(SPLITS_PATH, encoding="utf8"))
    p1 = cfg["plume1d"]

    train = expand(p1["train"], p1["train_pool"].get("id_space", "train"))
    val = expand(p1["val"], p1["train_pool"].get("id_space", "train"))
    pool = expand(p1["train_pool"], p1["train_pool"].get("id_space", "train"))
    test = expand(p1["test"], p1["test"].get("id_space", "test"))

    check("1D train ∩ val = ∅", not (train & val), str(sorted(train & val)[:5]))
    check("1D train ∪ val = train_pool", train | val == pool,
          "missing={} extra={}".format(sorted(pool - (train | val))[:5],
                                       sorted((train | val) - pool)[:5]))
    check("1D train_pool ∩ test = ∅（独立模拟，不得重叠）",
          not (pool & test), str(sorted(pool & test)[:5]))
    check("1D val ⊂ train_pool", val <= pool)
    check("1D 计数与声明一致",
          len(train) == p1["train"]["count"]
          and len(val) == p1["val"]["count"]
          and len(test) == p1["test"]["count"]
          and len(pool) == p1["train_pool"]["count"],
          "train={} val={} test={} pool={}".format(
              len(train), len(val), len(test), len(pool)))


def test_plume1d_id_spaces_declared():
    """两条轨道的 ID 命名空间必须显式声明且不同 —— 否则数值重叠会被误判为泄漏。"""
    cfg = yaml.safe_load(open(SPLITS_PATH, encoding="utf8"))
    p1 = cfg["plume1d"]
    train_space = p1["train_pool"].get("id_space")
    test_space = p1["test"].get("id_space")
    check("1D train 声明了 id_space", train_space is not None)
    check("1D test 声明了 id_space", test_space is not None)
    check("1D train/test 使用不同命名空间", train_space != test_space,
          "train={!r} test={!r}".format(train_space, test_space))


def test_plume2_split_consistent():
    cfg = yaml.safe_load(open(SPLITS_PATH, encoding="utf8"))
    p2 = cfg["plume2"]

    train = expand(p2["train"], "plume2")
    test = expand(p2["test"], "plume2")

    check("2D train ∩ test = ∅", not (train & test), str(sorted(train & test)[:5]))
    check("2D 并集 = 全部 128 条", len(train | test) == 128,
          "union={} train={} test={}".format(len(train | test), len(train), len(test)))
    check("2D 计数与声明一致",
          len(train) == p2["train"]["count"]
          and len(test) == p2["test"]["count"])


def test_plume2_is_frozen():
    """2D 划分已于 2026-09-21 独立核验：t-coordinate 匹配 + `default_rng(42)` 重现，
    两条证据给的 ids 完全一致。此测试守护该结论 —— 若划分被改动或退回未冻结状态，
    说明协议被动过，必须重新核验后再用。"""
    import numpy as np
    cfg = yaml.safe_load(open(SPLITS_PATH, encoding="utf8"))
    p2 = cfg["plume2"]
    check("2D 划分状态为 frozen", p2["status"] == "frozen", p2["status"])
    # 核验结论的守护：ids 必须仍能由 seed=42 重现
    rep = set(np.random.default_rng(42).permutation(128)[:103].tolist())
    check("2D train ids 仍可由 default_rng(42) 重现", rep == set(p2["train"]["ids"]),
          "交集 {}/103".format(len(rep & set(p2["train"]["ids"]))))
    check("2D val 不写死 id 列表（由配置的 test_ratio=0.1 自动切出）",
          p2["val"]["ids"] is None)


def test_plume1d_is_frozen():
    cfg = yaml.safe_load(open(SPLITS_PATH, encoding="utf8"))
    check("1D 划分状态为 frozen", cfg["plume1d"]["status"] == "frozen",
          cfg["plume1d"]["status"])


def main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if FAILURES:
        print("失败 {} 项: {}".format(len(FAILURES), FAILURES))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
