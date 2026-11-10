# E03 数学定义

> 共通指标见 `../protocol/metrics.md`；时间协议见 `../protocol/time.md`。

## rollout 递推

设模型为 $F_\theta$，单次调用输出 $Q$ 帧（$Q$ = `time_window` = 10）。
给定初始真实窗口 $W_0 = U[0:H]$（$H=10$），第 $k$ 次迭代：

```text
自治 rollout:    Ŵ_k     = F_θ( W_{k-1} , t_k )          k = 1, 2, …
                 W_k     = Ŵ_k                            ← 只用预测回填
                 t_{k+1} = t_k + Q 帧                      ← 窗口按帧首尾相接

teacher-forced:  Ŵ_k     = F_θ( W_k^{true} , t_k )        ← 每步用真值
```

$W_k$ 表示"喂给模型的窗口"，$\hat W_k$ 表示模型输出。

**禁止**在自治模式下把任何真实未来状态注入 $W_k$（$k \ge 1$）。
报告时必须写清：

- 总**帧数** $K \cdot Q$
- 总**模型调用次数** $K$（一帧数 ≠ 一调用数）

## 物理时间与训练时域

训练数据覆盖的物理时间上限：

```text
T_train = max(t-coordinate of plume1d_train_ds_128x128_phys1e6_first21.hdf5) ≈ 17.24 Myr
```

- 预测落点 $\le T_{train}$ → 称 **long-horizon rollout**
- 预测落点 $> T_{train}$ → 才可称 **temporal extrapolation**

图中必须标出 $T_{train}$ 的位置。

非均匀时间戳若需公共横轴，用预先声明的插值并限制在共同支持区间内，
报告每个时点的**有效轨迹数** $n_{\text{valid}}(t)$。

## 失败判据（预先固定）

```text
failure(n, q, c) =  nan_inf(pred)
                 OR e[n,q,c] > τ_c             连续 3 个输出窗口
                 OR pred 越出 [lo_c, hi_c]     范围由训练集实测极值确定
```

$\tau_c$ 从**验证集**（轨迹 87–96）的逐场 L2RE 分位数预先确定，不使用测试集。

失败记录进 `results/failures.csv`，**不从均值中默默剔除**。
报告失败率、首次失败时刻、以及被截断的轨迹数。

> 普通的误差增长 **不等于**数值不稳定。只有触发上述判据才算失败。

## 可比性约束

- 使用与 E01 **相同的测试轨迹**与相同的 checkpoint 选择规则；
- 不为长程评测重新挑 checkpoint；
- 模型间排序可在各时刻变化，**不要求** MFGNO 每时刻每场最优。

## 与短窗口结果的关系

短窗口误差是单步（或单窗口）误差，rollout 误差是**复合**误差：

```text
e_rollout(K) ≈ Σ_{k=1..K} (误差累积)  ⊇  e_short
```

因此短窗口更优**不自动**意味着 rollout 更优 —— 这是本阶段要检验的问题本身。
