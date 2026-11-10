# E02 数学定义

> 共通定义（指标、ε、bootstrap）见 `../protocol/metrics.md`；
> 损失与权重向量见 `../protocol/loss.md`。此处只写本阶段特有的部分。

## 三个信息路径的形式化

**C（完整 MFGNO）** —— 第 $\ell$ 层的消息：

```text
m_ij^ℓ = φ_ℓ([ h_i^ℓ , h_j^ℓ , x_i − x_j , u_i − u_j , c_i ])
```

其中 $u_i = \operatorname{vec}(U_i^{\text{hist}}) \in \mathbb{R}^{6H}$，
$H$ = `time_window` = 10，故差分槽为 **60 维**。

**B（无显式差分）** —— 结构完全相同，仅该槽置零：

```text
m_ij^ℓ = φ_ℓ([ h_i^ℓ , h_j^ℓ , x_i − x_j , 0 , c_i ])
```

$\phi_\ell$ 的第一个线性层仍是 $\mathbb{R}^{1090} \to \mathbb{R}^{512}$，
**参数量不变**（差分占 60/1090 列，共 30,720 个权重）。

输入维按 `model_utils.py:92-95` 的公式
`msg_input_dim = 2*hidden + spatial_dim + num_physics_vars*time_window + n_eq_vars`
展开：$2\times512 + 2 + 6\times10 + 4 = 1090$
（`hidden_features: 512`、`spatial_dim: 2`、`num_outputs: 6`、`time_window: 10`；
`n_eq_vars` 见 `model.py:205` 的 `len(eq_variables) + 1` —— 三个 `variables`
`m / eta / zeta` 之外那个 `+1` 是 `t_pos`）。

该值与 `FILL_IN_CHECKLIST.md` 里 F02 记录的 `msg_in=1090` 一致（那一条是从实例化后
的 `MFGNO` 上直接读出来的，比手算可靠）。

**A（独立分场）** —— 场 $c$ 的网络只读该场历史：

```text
h_i^{0,(c)} = E_θ([ \operatorname{vec}(U_{i,c}^{\text{hist}}) , x_i , c_i ])
```

六个网络不共享参数。

### 形式化的局限（必须随结论一起报告）

置零是**输入层面的**操作：B 的隐状态仍可自行编码出场间差分，
因此 B/C 检验的是"**显式提供**差分的价值"，而非"能否进行多场推理"。
B 的 $\phi_\ell$ 仍以零作为输入进入同一个非线性函数，**功能类不等价**于 C。

## 损失（统一形式）

```text
L(pred, target; λ) = Σ_{c=1..6} λ_c · mean_B[ ‖pred_c − target_c‖₂ / (‖target_c‖₂ + ε) ]
```

范数在 $(X, Y, T)$ 上计算、保留 batch 维，$\varepsilon = 10^{-8}$。

| 条件 | λ |
|---|---|
| 等权 | $(1/6, 1/6, 1/6, 1/6, 1/6, 1/6)$ |
| 速度加权 | $(0.05, 0.05, 0.05, 0.05, 0.4, 0.4)$ |

速度加权是历史 `0.2·mean(标量) + 0.8·mean(速度)` 的**逐场展开**，
不是把 0.2/0.8 直接当逐场系数（那样 $\sum\lambda = 2.4$ 且比例会变）。

## 消融增益

```text
Δ_c^(w) = E_B[c, w] − E_C[c, w]
```

- 下标 $c$：通道，$c \in \{\rho, P, T, \log_{10}\eta, v_x, v_y\}$
- 上标 $w$：损失条件，$w \in \{\text{equal}, \text{weighted}\}$
- **正值表示 C 更好**

按通道分别报告 $\Delta_c$，**不只报速度**。

## 配对种子与配对轨迹

两个层次的不确定性分开：

1. **种子层**：$\Delta_c^{(w)}$ 在每个种子上各算一次，报均值 ± 跨种子 SD。
2. **轨迹层**：同种子下先逐轨迹算 $\Delta_c^{(w)}[n]$，再对该向量做 bootstrap。

```text
Δ_c^(w)[n] = E_B^{(s)}[n, c, w] − E_C^{(s)}[n, c, w]
```

只有**相同种子 $s$** 的 B 与 C 才能配对相减。
A 组不参与配对（结构与容量都不同）。

## 容量记录

| 模型 | 参数量 | 相对 C |
|---|---|---|
| A | 9,076,662（6 个子网合计） | 0.86× |
| B | 10,551,362 | 1.000× |
| C | 10,551,362 | 1.000× |

⚠ 上表原先写的是 `A = 3,763,254 / B = C = 720,962 / 5.22×`，**整张表是错的**
（A:C 的大小关系都反了）。现按 `E04_diagnostics_cost/results/model_meta.json` 的
`params` 字段更正，该字段由 `collect_meta.py` 对所选 checkpoint 的 `model_state_dict`
逐个 `numel` 求和得到；按 `model_utils.py` / `model.py` 的结构定义手算结果与之一致
（encoder 593,408 + 6 层 × 1,610,752 + decoder 293,436 + `physics_weight` 6
= 10,551,362，其中 `InstanceNorm` 是 `affine=False`、无仿射参数）。

A 比 B/C **小约 14%**（不是原先写的 5.22 倍大）。差异**如实报告**。A→B 同时改变信息访问与参数共享，
不能拆成单因素因果解释。
