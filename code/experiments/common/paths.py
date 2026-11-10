"""发布包统一路径解析。

目录约定::

    geosim-plume/                  <- GEOSIM_ROOT
    ├── code/
    │   ├── experiments/           <- EXPERIMENTS（本文件在 common/ 下）
    │   └── vendor/                <- VENDOR（外部基线 + metrics.py）
    ├── results/                   <- RESULTS
    └── data/                      <- DATA

优先级：

- ``GEOSIM_ROOT`` 环境变量；未设置时从本文件位置推算（``code/experiments/common/`` 向上三级）。
- ``GEOSIM_DATA`` 可单独覆盖数据目录（默认 ``$GEOSIM_ROOT/data``）。

这样同一份代码既能在发布包里跑，也能在开发仓库里跑（开发仓库由
``GEOSIM_ROOT`` 指向仓库根，或把本模块放在对应层级）。
"""
import os

COMMON = os.path.dirname(os.path.abspath(__file__))          # code/experiments/common
EXPERIMENTS = os.path.dirname(COMMON)                        # code/experiments
CODE = os.path.dirname(EXPERIMENTS)                          # code
_PKG = os.path.dirname(CODE)                                 # geosim-plume

GEOSIM_ROOT = os.environ.get("GEOSIM_ROOT") or _PKG
VENDOR = os.path.join(GEOSIM_ROOT, "code", "vendor")
DATA = os.environ.get("GEOSIM_DATA") or os.path.join(GEOSIM_ROOT, "data")
RESULTS = os.path.join(GEOSIM_ROOT, "results")

# 数据子目录（与发布包 data/ 的布局一致）
DATA_DIRS = {
    "plume1d": os.path.join(DATA, "plume1d_corrected"),
    "plume2": os.path.join(DATA, "plume2"),
}


def describe():
    """打印当前解析结果，便于排查路径问题。"""
    return {"GEOSIM_ROOT": GEOSIM_ROOT, "VENDOR": VENDOR, "DATA": DATA,
            "RESULTS": RESULTS}


if __name__ == "__main__":
    for k, v in describe().items():
        print("%-12s %s" % (k, v))
