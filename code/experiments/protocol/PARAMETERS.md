# 逐轨迹物理参数

配套文件：[`trajectory_parameters.csv`](trajectory_parameters.csv)（237 行）。
该 CSV 是发布包中逐轨迹物理参数的权威来源；原始 I2VIS 算例树与数据生成脚本
不在发布包内。

## 1. 字段

| 列 | 含义 |
|---|---|
| `subset` | `plume1d` / `plume2` |
| `split` | `train` / `val` / `test`（以 `splits.yaml` 为准） |
| `traj_id` | 算例目录名（`T{i}` / `T_test{j}` / `run{i}`） |
| `h5_row` | 该轨迹在对应 split 的 HDF5 文件中的行号 |
| `T_plume_K` | 羽流管温度 (K) |
| `H_lith_km` | 岩石圈厚度 (km)；plume-1D 为空（一维参数空间） |
| `n_frame` | 该 split 的帧数 |
| `t_end_Myr` | 轨迹终止时刻 (Myr)，取自 `t-coordinate[:, -1]` |
| `source_file` | 参数读取的 `init.t3c` 路径 |

按 `subset + split + h5_row` 可与 HDF5 中的轨迹一一对应。

## 2. 参数来源

参数从每条算例的 I2VIS 输入文件 `init.t3c` 的固定位置提取，所有算例行号一致：

| 行（1-indexed） | 列 | 区域 | 含义 |
|---|---|---|---|
| 240 | 10 | `/Plume_pipe_(hot_conduit_40-560km)` | `T_plume`（K） |
| 202 | 6 | `/MantleLithospheric_mantle_8` | `H_lith`（米，CSV 中换算为 km） |

`T_plume` 取 `Plume_pipe`，不取 `Plume_bottom`：后者恒为 `pipe + 100 K`，
用作羽流温度会超出声明的 [2000, 2150] K 区间。plume-1D 只有 `T_plume`
一个参数，`H_lith_km` 为空。

CSV 由 `export_trajectory_parameters.py` 从原始 `init.t3c` 导出。原始算例树
未随发布包提供，因此该脚本不能在包内独立重跑；CSV 本身可独立使用。

## 3. 采样设计

**plume-2D（Sobol）**

```python
sampler = qmc.Sobol(d=10, scramble=True, seed=42)   # 2 个参数 + 8 个预留维
samples = sampler.random(n=128)
T_anomaly = 2000 + p1 * (2150 - 2000)               # K
Mantle_H  = 160  + p2 * (220  - 160)                # km
```

`run{i}` 对应 Sobol 第 i 行。在 scipy 1.10.1 下重算，与 CSV 的最大偏差为
T 0.0049 K、H 0.0050 km；偏差来自 `init.t3c` 写入时的两位小数舍入，不是随机性。
复现时建议固定 scipy 版本。

**plume-1D（LHS）**

`LatinHypercube(d=1, seed=42)`，`T = 1600 + 200·s`，区间 [1600, 1800] K，
保留两位小数。抽样 100 条，实际保留 97 条训练轨迹；测试集 12 条使用同一
随机种子、不同样本数独立抽取。

两个子集的参数坐标都有原始 `init.t3c` 作为独立记录。

## 4. HDF5 行号映射

- **plume-1D**：训练 HDF5 行 `i` ↔ `T{i}`（i = 0..96）；测试行 `j` ↔
  `T_test{j}`。两套命名空间各自从 0 开始，数值重叠但属于不同算例。
- **plume-2D**：训练/测试 HDF5 行 `k` ↔ `splits.yaml` 中第 k 个 `run{id}`。
  该映射已用轨迹终止时刻独立核对：HDF5 的 `t-coordinate[i, -1]` 与
  `run{i}` 日志中的结束时间吻合到 6–7 位有效数字。

## 5. 时间范围

| split | 轨迹数 | 帧数 | `t_end` 范围 (Myr) | 前 21 帧覆盖到 |
|---|---|---|---|---|
| plume-1D train | 97 | 97 | 54.41 – 55.24 | 17.24 Myr |
| plume-1D test | 12 | 131 | 71.43 – 72.24 | 17.24 Myr |
| plume-2D train | 103 | 131 | 51.62 – 67.99 | 12.99 Myr |
| plume-2D test | 25 | 131 | 49.04 – 67.92 | 12.92 Myr |

- plume-1D 的 train/test 帧数与总时长不同（97 vs 131 帧，约 55 vs 72 Myr），
  但前 21 帧都覆盖 17.24 Myr，训练窗口在两集间对齐。
- plume-2D 帧数相同，但逐轨迹终止时刻跨度约 19 Myr，且逐样本 `dt` 不同；
  模型按样本计算时间增量，见 `time.md`。

参数覆盖图由 `code/experiments/figures/make_figA01.py` 使用本 CSV 生成。
