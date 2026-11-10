# E04 数学定义

> 共通指标见 `../protocol/metrics.md`。

## 一、速度剖面的幅值与位置分解

沿固定物理位置 $x = x^\*$（预先声明）取竖直剖面，得两个一维曲线：

```text
v_true(y),  v_pred(y)        y 为竖直坐标（物理单位）
```

**幅值误差**（对形状不敏感）：

```text
A = ( max_y |v_pred| − max_y |v_true| ) / max_y |v_true|
```

**位置误差**（对幅值不敏感）—— 取剖面极值点的位置偏移：

```text
y*_true = argmax_y |v_true(y)|
y*_pred = argmax_y |v_pred(y)|
D = | y*_pred − y*_true |           单位：米（或归一化域长的比例）
```

二者**分开报告**。相对 L2 无法区分"峰值算小了"与"峰值位置移了"。

带符号约定：$v_x$、$v_y$ 的正方向以数据字典为准，图上标注。

**对全部测试轨迹汇总**，不只给单条示例。示例取固定轨迹 ID；
若用中位误差案例，须注明是**事后代表性选取**。

## 二、效率度量

```text
params        = Σ_p numel(p)                    # 参数量（含全部子模块）
peak_alloc    = torch.cuda.max_memory_allocated()   # 峰值 allocated 字节
step_time     = mean( t_i ) ,  std( t_i )       # 预热后多次重复
infer_latency = mean( τ_i ) ,  std( τ_i )       # batch=1，同步后计时
```

计时协议（缺一不可）：

```text
torch.cuda.synchronize()
for _ in range(warmup): model(...)
torch.cuda.synchronize()
for _ in range(reps):
    t0 = time.perf_counter(); model(...); torch.cuda.synchronize()
    record(time.perf_counter() - t0)
```

**图构建成本单列**，并区分：

- 缓存图（只更新节点特征）
- 每次重建图

图消息是显存热点，报告必须包含：

```text
边数 E、节点数 N、特征宽度 d、层数 L、batch
```

（论文中的复杂度量级为 $L\,E\,(d^2 + 6Hd)$，但**实测值优先**。）

## 三、数据质检量

| 量 | 定义 |
|---|---|
| 无效值计数 | 逐场 $\#\{x : \neg \text{isfinite}(x)\}$ |
| 异常范围 | 逐场 $\min / \max$，与训练集参考范围比对 |
| 时间单调性 | $\#\{k : t_{k+1} \le t_k\}$，应为 0 |
| 坐标一致性 | $x/y$ 方向是否与训练代码的 `indexing='ij'` 约定一致 |
| 六场对齐 | 六场共享同一 $(x, y, t)$ 轴的逐元素断言 |
| 重映射误差 | $\| F_{128} - R\,F_{401} \|$，$R$ 为保守重映射算子 |

**守恒式重映射不等于求解器准确性验证**：$R$ 保证的是面积加权积分守恒
（$\sum A\,F$ 不变），不保证非线性通量乘积守恒，也不构成网格收敛性证据。
没有收敛实验就明确写"未评估"。

**已知风险**：`np.meshgrid` 默认 `indexing='xy'`，训练代码用 `'ij'`。
不一致会导致每个节点的 $x/y$ 坐标对调，矢量场误差放大 8–20×。
该项检查必须显式通过。
