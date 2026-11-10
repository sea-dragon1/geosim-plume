# 时间表示协议

本文件定义发布包中物理时间如何进入模型、如何用于残差积分，以及各数据文件的
时间单位。逐轨迹的参数与时间范围见 `PARAMETERS.md`。

## 1. 时间单位与归一化

- 所有数据文件的 `t-coordinate` 以 **Myr** 存储。
- GNN 系模型（`MPNN` / `MFGNO`）**不做时间归一化**：`t` 以 Myr 原值进入
  编码器输入，以及每层消息/更新的条件槽 `[t, m, eta, zeta]`。
- 归一化只作用于六个物理场，不作用于时间轴。

## 2. 残差积分的 dt

模型按残差形式解码：

```text
out[b, q, c] = u_last[b, c] + dt[b, q] · diff[b, q, c]
```

其中 `u_last` 是输入窗口最后一帧在时刻 `t_last` 的状态。因此 `dt` 取相对
`t_last` 的时间增量：

```text
dt[b, q] = t[b, base + q] − t[b, base]
base     = steps[b] + time_window − 1        # 输入窗口最后一帧的索引
q        = 1 .. time_window
```

实现位于 `code/experiments/common/vendor/model.py` 的
`compute_cumulative_dt` / `expand_dt_to_nodes`：

- `dt` 是 per-sample 的 `[batch, time_window]`，不是全局均匀步长；
- 通过 `data.batch` 扩展到节点维 `[batch×nodes, time_window]`；
- 若输入缺少 `t` 或 `data.steps`，退回均匀步长并返回标志位。发布包的
  图构建路径（`build_static_graph` / `update_graph_input` /
  `update_graph_data`）都会设置 `steps`，正常训练与评估不会走回退分支。

### 为什么使用真实时间轴

模拟的时间间隔非均匀，且逐轨迹不同。以一条 21 帧的 plume-1D 轨迹为例：

| 段 | dt |
|---|---|
| 起始两步 | 0.555, 0.749 Myr |
| 中段 13 步 | 1.000 Myr |
| 末段 5 步 | 0.500 Myr |

跨训练集，首步 `dt[0]` 随轨迹变化（约 0.21–0.74 Myr），中段约 0.5–1.0 Myr，
末段恒为 0.5 Myr；不同轨迹的终止时刻也不同。若使用单一均匀步长，会同时丢失
分段结构和逐轨迹差异；总时长不变，因此"平均值"本身并不能反映这一差别。

## 3. 数据文件的时域

| 用途 | 文件 | 时间单位 | 覆盖 |
|---|---|---|---|
| 短窗口训练与评估 | `*_first21*`（21 帧） | Myr | plume-1D：0 → 17.24；plume-2D：0 → 12.99 |
| 长程评估 | 全时程 corrected 文件 | Myr | plume-1D：约 55（train）/ 72（test）；plume-2D：约 49–68 |

两个子集的 `first21` 帧数相同，但物理时长不同（plume-1D 的 `dt` 均值约
0.845 Myr，plume-2D 约 0.471 Myr）。"前 21 帧"在两子集下不是同一段时间，
表格与图注必须分别标注。

发布包不包含原始模拟数据；运行时通过 `GEOSIM_DATA` 指向数据目录
（见 `code/experiments/E01_main/README.md`）。

## 4. 各模型的时间处理

| 模型 | 时间如何进入 | 输出方式 |
|---|---|---|
| MPNN / MFGNO | 显式输入 `t`（Myr，不归一化）；`dt` 按第 2 节计算 | 一次输出 `time_window` 帧 |
| GNOT | 无物理时间输入；使用 step 索引 | 一次输出 `time_window × 6` 后 reshape |
| FNO | 无显式时间输入；`t_train=20` 表示使用帧 0–19（10 输入 + 10 输出） | 逐帧自回归，10 次前向 |
| Transolver | 无显式时间输入；`out_dim` 为单帧通道数 | 逐帧自回归，10 次前向 |

FNO 与 Transolver 的逐帧展开是其训练和推理的固有方式，训练/评估口径一致。
"统一时间协议"指统一输入输出窗口（10→10），不表示所有模型内部都使用
相同的物理时间表示。

## 5. 配置字段与兼容性

- `time_window: 10`、`resolution_t: 21`（first21 数据）、`neighbors: 1`。
- `temporal_domain` 只用于 `dt` 的均匀回退分支；发布配置中填写训练数据的
  实际时间跨度（plume-1D 为 `(0, 17.2433)`，plume-2D 为 `(0, 12.9913)`），
  它不再作为输入缩放因子。
- checkpoint 与本协议绑定。不要与使用均匀 `dt` 或时间归一化的其他权重混用。
