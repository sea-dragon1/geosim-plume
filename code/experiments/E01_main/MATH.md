# E01 数学定义

> 共通定义（指标、ε、bootstrap）见 `../protocol/metrics.md`，此处只写本阶段特有的部分。

## 张量与字段

内部统一为 `u[n, t, x, y, c]`，通道顺序固定：

```text
c: 0=ro  1=pr  2=tk  3=nu  4=vx  5=vy
```

`nu` 是 **log10(Pa·s)**，其误差不是 Pa·s 黏度的相对误差。

## 单模型误差

对归一化面积权重 $a_i$（均匀网格下 $a_i = 1/N$，在分子分母中约去）：

```text
e[n,q,c] = sqrt(Σ_i a_i (pred - truth)²) / max(sqrt(Σ_i a_i truth²), ε_c)

E[c] = mean_n( mean_q( e[n,q,c] ) )
```

**聚合次序硬性**：先按帧平均，再按轨迹平均。与"整条轨迹取一次范数"不同。

## persistence 基线

10→10 窗口协议下，基线固定为**最后一个输入帧**（$t=9$），对全部输出帧恒定：

```text
baseline[n,q,i,c] = truth[n, 9, i, c]     对所有 q
```

**不是**每步读真值上一帧 —— 那是 teacher forcing。

## 技能分数

```text
SSE_model[n,c] = Σ_{q,i} (pred - truth)²
SSE_base[n,c]  = Σ_{q,i} (baseline - truth)²
skill[n,c]     = 1 − SSE_model[n,c] / SSE_base[n,c]
```

正值表示优于 persistence；分母为零记 **NaN（未定义）**，不得用极小常数伪造大数值。

## 跨种子聚合

| 量 | 定义 |
|---|---|
| 种子均值 | $\bar{E}_c = \text{mean}_{s \in \{0,1,2\}} E_c^{(s)}$ |
| 种子标准差 | $\text{SD}_c = \text{std}_{s} E_c^{(s)}$（样本标准差） |

主表报 $\bar{E}_c \pm \text{SD}_c$。persistence **不伪造**种子方差（只有一行数值）。

## 轨迹级 bootstrap

```text
resamples = rng.integers(0, n_traj, size=(repeats, n_traj))     # 固定 seed
CI(values) = quantile_0.025 / 0.975 of   mean(values[resamples], axis=1)
```

- 所有模型**共用同一组重采样索引**，否则配对比较无效；
- CI 只量化**测试轨迹采样差异**，不含训练随机性、不含调参选择偏差。

## 配对比较

与 persistence 的配对差值：**先逐轨迹相减，再重采样**。

```text
d[n,c] = E_model[n,c] − E_persistence[n,c]
CI(d)  = bootstrap(d)
```

不把同轨迹的不同帧当作独立样本。
