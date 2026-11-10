#!/usr/bin/env python3
# coding=utf-8
"""导出**逐轨迹参数表** → ``protocol/trajectory_parameters.csv``。

这是全项目里**唯一**从原始算例提取物理参数的地方。出图（FA01）、附录表、
复现包都应当读这份 CSV，而不是各写一遍提取逻辑 —— 参数是论文里最容易被
"顺手重算一遍"然后悄悄算错的东西。

参数来源与坑（详见 ``PARAMETERS.md``）
--------------------------------------
逐算例的 I2VIS 输入 ``init.t3c`` 完整保留在仓库里，参数在固定行上：

| 行(1-indexed) | 列(awk $n) | 含义 |
|---|---|---|
| 240 | 10 | ``T_plume``（``/Plume_pipe``，采样值，K） |
| 202 | 6  | ``H_lith``（``/MantleLithospheric_mantle_8``，**米**） |

⚠ 取行 240 而**不是**行 242（``/Plume_bottom``）：bottom 恒等于 pipe+100 K，
会超出论文声明的 [2000, 2150] K；只有 pipe 值落在区间内。

**采样可复现**（两个子集都是）：

- plume-2D：``data/plume1d_corrected/i2vis_source/00_sobol_sample.py`` ——
  ``qmc.Sobol(d=10, scramble=True, seed=42).random(128)``，再线性映射到
  ``[2000,2150] K`` 与 ``[160,220] km``。``--check`` 会重算这 128 条并与
  ``init.t3c`` 的实测值比对（scipy 1.10.1 下最大偏差 0.005，即 2 位小数舍入）。
- plume-1D：``LatinHypercube(d=1, seed=42)``，``T = 1600 + 200·s``。

用法::

    python3 -B protocol/export_trajectory_parameters.py            # 写 CSV
    python3 -B protocol/export_trajectory_parameters.py --check    # 只校验，不写
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
COMMON = os.path.join(EXPERIMENTS, "common")
REPO = os.path.abspath(os.path.join(EXPERIMENTS, "..", "..", ".."))

if COMMON in sys.path:
    sys.path.remove(COMMON)
sys.path.insert(0, COMMON)

import numpy as np          # noqa: E402
import h5py                 # noqa: E402
import yaml                 # noqa: E402

OUT_CSV = os.path.join(HERE, "trajectory_parameters.csv")

ROW_T, COL_T = 240, 10         # /Plume_pipe_(hot_conduit_40-560km)
ROW_H, COL_H = 202, 6          # /MantleLithospheric_mantle_8（单位：米）

DATA = os.path.join(REPO, "data")
P1D_POOL = os.path.join(DATA, "results/result/T")           # T0..T96
P1D_TEST = os.path.join(DATA, "results/result/T_test")      # T_test0..T_test11
P2D_RUNS = os.path.join(DATA, "results/results")            # run0..run127

H5 = {
    ("plume1d", "train"): os.path.join(
        DATA, "plume1d_corrected", "plume1d_train_ds_128x128_phys1e6.hdf5"),
    ("plume1d", "test"): os.path.join(
        DATA, "plume1d_corrected", "plume1d_test_ds_128x128_phys1e6.hdf5"),
    ("plume2", "train"): os.path.join(
        DATA, "plume2", "plume2d_train_ds_128x128_norm_corrected.hdf5"),
    ("plume2", "test"): os.path.join(
        DATA, "plume2", "plume2d_test_ds_128x128_norm_corrected.hdf5"),
}

# 论文里声明的统计量 —— 抽出来做一次回归检查，防止哪天提取列取错了而无人察觉
EXPECTED = {
    "plume2_T_mean": 2075.0, "plume2_T_std": 43.3, "plume2_N": 128,
    "plume2_H_mean": 190.0, "plume2_H_std": 17.3,
}


def read_init(path):
    """从 ``init.t3c`` 读 ``(T_plume_K, H_lith_km | None)``。"""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    if len(lines) < ROW_T:
        raise ValueError("{} 只有 {} 行，读不到第 {} 行".format(path, len(lines), ROW_T))
    parts = lines[ROW_T - 1].split()
    if len(parts) < COL_T:
        raise ValueError("{} 第 {} 行只有 {} 列".format(path, ROW_T, len(parts)))
    T = float(parts[COL_T - 1])
    H = None
    if ROW_H <= len(lines):
        h = lines[ROW_H - 1].split()
        if len(h) >= COL_H and h[COL_H - 1].startswith("m"):
            H = float(h[COL_H - 1][1:]) / 1000.0        # 米 → 千米
    return T, H


def read_splits():
    with open(os.path.join(HERE, "splits.yaml"), "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def read_time(subset, split):
    p = H5[(subset, split)]
    with h5py.File(p, "r") as f:
        t = np.asarray(f["t-coordinate"][:], dtype=np.float64)
    return t[:, -1], t.shape[1], os.path.relpath(p, REPO)


def build():
    s = read_splits()
    rows = []

    # ── plume-1D：train_pool = T0..T96（h5 train 行 i ↔ T{i}），test 独立命名空间 ──
    p1 = s["plume1d"]
    lo, hi = p1["train_pool"]["range"]
    vlo, vhi = p1["val"]["range"]
    te, nf, path = read_time("plume1d", "train")
    for i in range(lo, hi + 1):                    # h5 train 行 = i（顺序一致）
        T, _ = read_init(os.path.join(P1D_POOL, "T{}".format(i), "init.t3c"))
        rows.append(dict(
            subset="plume1d", split="val" if vlo <= i <= vhi else "train",
            traj_id="T{}".format(i), h5_row=i,
            T_plume_K="{:.2f}".format(T), H_lith_km="",
            n_frame=nf, t_end_Myr="{:.4f}".format(float(te[i])),
            source_file=os.path.relpath(
                os.path.join(P1D_POOL, "T{}".format(i), "init.t3c"), REPO),
        ))
    te, nf, path = read_time("plume1d", "test")
    for j in range(p1["test"]["count"]):
        T, _ = read_init(os.path.join(P1D_TEST, "T_test{}".format(j), "init.t3c"))
        rows.append(dict(
            subset="plume1d", split="test", traj_id="T_test{}".format(j), h5_row=j,
            T_plume_K="{:.2f}".format(T), H_lith_km="",
            n_frame=nf, t_end_Myr="{:.4f}".format(float(te[j])),
            source_file=os.path.relpath(
                os.path.join(P1D_TEST, "T_test{}".format(j), "init.t3c"), REPO),
        ))

    # ── plume-2D：h5 行 k ↔ run{splits.ids[k]}（终止时刻已独立验证） ──
    p2 = s["plume2"]
    for split in ("train", "test"):
        ids = list(p2[split]["ids"])
        te, nf, path = read_time("plume2", split)
        if len(ids) != te.size:
            raise ValueError("plume2 {} 的 ids 数 {} != h5 轨迹数 {}".format(
                split, len(ids), te.size))
        for k, i in enumerate(ids):
            T, H = read_init(os.path.join(P2D_RUNS, "run{}".format(i), "init.t3c"))
            if H is None:
                raise ValueError("run{} 没读到 H_lith".format(i))
            rows.append(dict(
                subset="plume2", split=split, traj_id="run{}".format(i), h5_row=k,
                T_plume_K="{:.2f}".format(T), H_lith_km="{:.3f}".format(H),
                n_frame=nf, t_end_Myr="{:.4f}".format(float(te[k])),
                source_file=os.path.relpath(
                    os.path.join(P2D_RUNS, "run{}".format(i), "init.t3c"), REPO),
            ))
    return rows


def check(rows):
    """回归检查：与论文声明的统计量比对，并验守恒性质。"""
    ok = True
    p2 = [r for r in rows if r["subset"] == "plume2"]
    T = np.array([float(r["T_plume_K"]) for r in p2])
    H = np.array([float(r["H_lith_km"]) for r in p2])

    checks = [
        ("plume2 N", T.size, EXPECTED["plume2_N"], 0),
        ("plume2 T mean", T.mean(), EXPECTED["plume2_T_mean"], 0.05),
        ("plume2 T std", T.std(), EXPECTED["plume2_T_std"], 0.05),
        ("plume2 H mean", H.mean(), EXPECTED["plume2_H_mean"], 0.05),
        ("plume2 H std", H.std(), EXPECTED["plume2_H_std"], 0.05),
    ]
    for name, got, want, tol in checks:
        good = abs(got - want) <= max(tol, 1e-9)
        ok &= good
        print("  [{}] {:<16} {:.4f}  (期望 {})".format(
            "ok  " if good else "FAIL", name, got, want))

    # 声明范围：T ∈ [2000, 2150]、H ∈ [160, 220]
    inr = (T.min() >= 2000 - 1e-6 and T.max() <= 2150 + 1e-6
           and H.min() >= 160 - 1e-6 and H.max() <= 220 + 1e-6)
    ok &= inr
    print("  [{}] 全部落在声明区间内  T[{:.2f},{:.2f}] H[{:.2f},{:.2f}]".format(
        "ok  " if inr else "FAIL", T.min(), T.max(), H.min(), H.max()))

    # 划分不重不漏
    for sub in ("plume1d", "plume2"):
        ids = {}
        for r in rows:
            if r["subset"] == sub:
                ids.setdefault(r["split"], []).append(r["traj_id"])
        flat = [x for v in ids.values() for x in v]
        uniq = len(flat) == len(set(flat))
        ok &= uniq
        print("  [{}] {} 划分无重复  {}".format(
            "ok  " if uniq else "FAIL", sub,
            " / ".join("{}={}".format(k, len(v)) for k, v in sorted(ids.items()))))
    return ok


SOBOL_SCRIPT = os.path.join(
    REPO, "data/plume1d_corrected/i2vis_source/00_sobol_sample.py")


def check_sobol_roundtrip(rows, tol=0.01):
    """用保留的采样脚本重算 128 条参数，与 init.t3c 的实测值比对。

    ``init.t3c`` 里的值是 ``round(..., 2)`` 后的，所以容差取 0.01 而不是 0。
    这条断言把"采样可复现"从一句声明变成每次跑都验的事实。
    """
    from scipy.stats import qmc
    import scipy
    if not os.path.exists(SOBOL_SCRIPT):
        print("  [SKIP] 找不到 {}，跳过 Sobol 复现校验".format(SOBOL_SCRIPT))
        return True

    # 与 00_sobol_sample.py 的 generate_base_samples / map_to_physics 逐字一致：
    #   qmc.Sobol(d=10, scramble=True, seed=42).random(n=128)   （existing_count=0）
    #   T = 2000 + p1*150,  H = 160 + p2*60
    X = qmc.Sobol(d=10, scramble=True, seed=42).random(n=128)
    T_pred = 2000 + X[:, 0] * 150.0
    H_pred = 160 + X[:, 1] * 60.0

    by_id = {int(r["traj_id"][3:]): r for r in rows
             if r["subset"] == "plume2"}
    if len(by_id) != 128:
        print("  [FAIL] plume2 应有 128 条，实得 {}".format(len(by_id)))
        return False
    T_act = np.array([float(by_id[i]["T_plume_K"]) for i in range(128)])
    H_act = np.array([float(by_id[i]["H_lith_km"]) for i in range(128)])

    dT, dH = np.abs(T_pred - T_act).max(), np.abs(H_pred - H_act).max()
    ok = dT <= tol and dH <= tol
    print("  [{}] Sobol 复现（scipy {}）  最大偏差  T {:.4f} K / H {:.4f} km".format(
        "ok  " if ok else "FAIL", scipy.__version__, dT, dH))
    if not ok:
        print("       ⚠ 偏差超过 {:.3f} —— scipy 版本变了？采样脚本被改过？"
              .format(tol))
    return ok


def check_lhs_roundtrip(rows, tol=0.005):
    """用 create_T_test.py 的写法重算 plume-1D 温度，与 init.t3c 比对。

    脚本里 LOWER=1600 / UPPER=1800 / seed=42，train pool 用 n=100、
    test 用 n=12（**同一 seed、不同 n ⇒ 两次独立抽样**，不是一次抽样的切分）。
    这里只看前 97 条（pool 实际保留数）。
    """
    from scipy.stats import qmc
    import scipy
    lo, hi = 1600.0, 1800.0

    def draw(n):
        return np.round(lo + (hi - lo) * qmc.LatinHypercube(d=1, seed=42)
                        .random(n=n).flatten(), 2)

    d1 = [r for r in rows if r["subset"] == "plume1d"]
    tr = np.array([float(r["T_plume_K"]) for r in d1 if r["split"] in ("train", "val")])
    te = np.array([float(r["T_plume_K"]) for r in d1 if r["split"] == "test"])
    d_tr = np.abs(draw(100)[:tr.size] - tr).max()
    d_te = np.abs(draw(12) - te).max()
    ok = d_tr <= tol and d_te <= tol
    print("  [{}] LHS 复现（scipy {}）  最大偏差  pool {:.4f} K / test {:.4f} K".format(
        "ok  " if ok else "FAIL", scipy.__version__, d_tr, d_te))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写 CSV")
    ap.add_argument("--no-sobol", action="store_true", help="跳过 Sobol 复现校验")
    ap.add_argument("--no-lhs", action="store_true", help="跳过 plume-1D LHS 复现校验")
    a = ap.parse_args()

    rows = build()
    print("提取 {} 条轨迹".format(len(rows)))
    ok = check(rows)
    if not a.no_lhs:
        ok &= check_lhs_roundtrip(rows)
    if not a.no_sobol:
        ok &= check_sobol_roundtrip(rows)

    if a.check:
        print("\n校验{}通过".format("" if ok else "**未**"))
        sys.exit(0 if ok else 1)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "split", "traj_id", "h5_row",
                                          "T_plume_K", "H_lith_km", "n_frame",
                                          "t_end_Myr", "source_file"])
        w.writeheader()
        w.writerows(rows)
    print("\n→ {}".format(OUT_CSV))


if __name__ == "__main__":
    main()
