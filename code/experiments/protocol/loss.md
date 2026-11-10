# 训练损失协议

本文件定义 GNN 系模型（`MPNN` / `MFGNO`）的训练目标与 checkpoint 选择规则。
评估指标见 `metrics.md`；两者不是同一个量。

## 1. 逐场相对 L2 损失

发布包使用统一的逐场相对 L2 目标：

```text
L(pred, target; λ) = Σ_{c=1..6} λ_c · mean_batch[ ‖pred_c − target_c‖₂ / (‖target_c‖₂ + ε) ]
```

- 逐场独立求范数，再按 λ 加权求和。
- 范数在 `dim=(1, 2, 3)`（X, Y, T）上计算，保留 batch 维后取均值。
- ε = 1e-8。
- 六个通道顺序固定为 `ro, pr, tk, nu, vx, vy`。

实现见 `code/experiments/common/losses.py` 的 `per_field_l2re_loss`。

## 2. 权重向量

| 预设 | λ = (ro, pr, tk, nu, vx, vy) | 使用范围 |
|---|---|---|
| 等权 | (1/6, 1/6, 1/6, 1/6, 1/6, 1/6) | GNN 系全部实验 |
| 速度加权 | (0.05, 0.05, 0.05, 0.05, 0.4, 0.4) | 代码保留，未用于论文结果 |

"等权"只表示变换后六场系数相同，不表示六个物理量在物理上同尺度。

速度加权的来源是分组均值形式 `0.2·mean(四个标量场) + 0.8·mean(两个速度场)`，
展开成逐场权重即标量场每场 0.05、速度场每场 0.4，两组之和分别为 0.2 与 0.8，
Σλ = 1。不能把 (0.2, 0.2, 0.2, 0.2, 0.8, 0.8) 直接作为逐场权重，那样 Σλ = 2.4，
且会改变标量场与速度场的相对比例。

配置中由 `loss_mode` 选择预设（`equal` / `velocity_weighted`）。论文结果
全部使用 `equal`。

## 3. 训练目标与评估指标分离

- 训练损失：上式（逐场范数比），对每个通道的梯度贡献可控。
- 评估指标：`metrics.md` 的逐帧逐场相对 L2，先帧内再轨迹平均。
- 两者不可互换，报告中不得把训练损失当作评估指标，也不得把其他权重下的
  指标标注为本协议的结果。

## 4. 验证选模规则

- 每个种子按同一个验证指标选择 checkpoint：验证集上的六场等权 L2RE 均值，
  取最小者。plume-1D 的验证集为训练池的后 10% 轨迹（索引 87–96）。
- 不得按测试集表现挑 checkpoint，也不得每场各挑一个最优 checkpoint。
- 选模指标与训练损失解耦：即使训练使用速度加权，选模仍一律使用等权指标。

## 5. 物理损失

发布包的训练目标只包含上面的数据损失，不启用连续性、速度梯度、Stokes 残差
或 I2VIS 物理损失。配置中的相关开关保留为 0，不参与计算。

## 6. 配置对应关系

发布包的 `train_tubor.py` 在 `if_physical_loss: 1` 时执行：

```python
pred = to_PDEBench_format(pred, batch_size, pde)
gt   = to_PDEBench_format(graph.y, batch_size, pde)
loss = per_field_l2re_loss(pred, gt, get_weights(args["loss_mode"]))
```

`loss_mode` 缺省按 `equal` 处理。`if_physical_loss: 0` 时退回逐节点 MSE，
发布配置不使用该路径。
