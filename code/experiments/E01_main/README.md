# E01 — 六场短窗口主结果：复现说明

> **状态**：已完成（2026-09-25 打分完毕，2026-09-27 进论文并编译）。
> 本文件记录复现步骤、产物位置与论文对应关系。

本阶段训练**三个外部基线**（FNO / GNOT / Transolver），在两个子集上产出论文主表里
它们的行。MFGNO 与 MPNN 两行**不在本目录**——复用 `../E02_ablation/` 的 C / A 组。

---

## 1. 目录结构

```
E01_main/
├── configs/        12 个正式配置
├── runs/           12 个正式 run（best 权重 + predictions + eq9 + yaml）
├── results/        汇总指标 CSV + persistence
├── MATH.md         指标公式
├── README.md       本文件
└── run_all.sh      批量提交训练
```

### 12 个正式 run

| exp_id | 子集 |
|---|---|
| `2026-09-17-fno-E01-plume1d-seed{0,1,2}` | plume1d |
| `2026-09-17-gnot-E01-plume1d-seed{0,1,2}` | plume1d |
| `2026-09-17-transolver-E01-plume1d-seed{0,1,2}` | plume1d |
| `2026-09-22-{fno,gnot,transolver}-E01-plume2-corr-seed0` | plume2 |

每个 run 的结构：`checkpoints/`（best + latest）、`output/`、`log*/`、`eval/`。

---

## 2. 复现步骤

前置：

- `conda activate fno`（Python 3.8 / PyTorch 2.4.1+cu121；见 `environment/README.md`）
- **`GEOSIM_DATA`**：数据根目录，需包含 `plume1d_corrected/` 与 `plume2/`。
  未设置时默认 `geosim-plume/data/`（发布包内为空）。以下脚本依赖它：
  `score_persistence.py`、`score_predictions.py`、`eval_per_traj.py`；
  `eval_checkpoint.py` 用 run 配置里的 `saved_folder`，不受影响。
  ```bash
  export GEOSIM_DATA=/path/to/geosim-data
  ```
- 所有命令在 `geosim-plume/` 下执行。

### ① 训练（GPU，必须 `yhbatch`）

```bash
cd /path/to/geosim-plume

bash code/experiments/E01_main/run_all.sh --only=plume1d-seed0
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed1
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed2
bash code/experiments/E01_main/run_all.sh --with-plume2 --only=plume2
```

- **必须分批**：单用户提交数有上限，实测会在 6–9 之间变化，一次提 9 个会被调度器拒绝。
- `run_all.sh` 默认只提交 plume1d；plume2 需带 `--with-plume2`。
  脚本仍会跳过 `SMOKE-*` / `RESUME-*`，但发布包内已无这类配置。
- 训练入口从配置头部的 `# 训练入口: ...` 注释读取（FNO → `fno/train_new.py`，
  GNOT → `GNOT-master/plume1/train.py`，Transolver → `Transolver/train.py`）。
- 状态查看：`squeue -u $USER`；取消：`yhcancel <jobid>`。
- 脚本**不是幂等的**，重跑会重复提交。

### ② 推理（GPU，必须 `yhbatch`）

```bash
yhbatch code/experiments/common/run_predict.sh --stage E01_main --subset plume1d
yhbatch code/experiments/common/run_predict.sh --stage E01_main --subset plume2
```

`code/experiments/common/predict_external.py` 读 checkpoint，产出
`runs/<exp_id>/eval/predictions.hdf5`：12 个 dataset（6 个预测 + 6 个 `truth_*`），
形状 `(n_traj, 10, 128, 128)`。

### ③ 打分（CPU）

```bash
# 登录节点不跑计算：走通用 CPU 提交脚本（第一个参数须为绝对路径）
yhbatch code/experiments/common/run_cpu.sh \
    "$PWD/code/experiments/common/score_predictions.py" --stage E01_main --subset plume1d
yhbatch code/experiments/common/run_cpu.sh \
    "$PWD/code/experiments/common/score_predictions.py" --stage E01_main --subset plume2
```

产出：
- 每个 run 的 `eval/<exp_id>_eq9.json`
- 汇总 `results/E01_main_eq9_external.csv`

⚠ **两个子集都要跑**。CSV 是合并写：同 `exp_id` 替换、其余保留；只跑一个子集不会
抹掉另一个，但也不会补上它。

### ④ 汇总成论文表

```bash
python -B code/experiments/common/make_tables.py --dry-run   # 只打印，不落盘
```

`make_tables.py` 已与现有版式同步（单表六列、百分比、有效精度），落盘会写到
`results/tables/`；已用发布包内的结果验证六张表逐行一致。

---

## 3. 结果在哪

> 下表路径相对本目录（`code/experiments/E01_main/`）；论文表格与图在包级 `results/`。

| 文件 | 内容 | 行数 |
|---|---|---|
| `results/E01_main_eq9_external.csv` | 12 个 run × 6 场 × (L2RE, RMSE)，含 `n_nonfinite` | 12 |
| `results/persistence_plume1d_eq9.csv` | persistence 基线（plume1d） | 1 |
| `results/persistence_plume2_eq9.csv` | persistence 基线（plume2） | 1 |
| `runs/<exp_id>/eval/predictions.hdf5` | 预测 + 真值场 | 94 MB/run |
| `runs/<exp_id>/eval/<exp_id>_eq9.json` | 单 run 的逐场指标 + ε 触发次数 | — |

**数据口径**：CSV 里是**小数**（相对 L2，如 1.2495）；论文表格是**百分数**（×100，如 122）。

**测试数据**（推理与真值来自同一个文件）：

| 子集 | 文件 | 形状 |
|---|---|---|
| plume1d | `$GEOSIM_DATA/plume1d_corrected/plume1d_test_ds_128x128_phys1e6_first21.hdf5` | 12 × 21 × 128 × 128 |
| plume2 | `$GEOSIM_DATA/plume2/plume2d_test_ds_128x128_phys1e6_first21_corrected.hdf5` | 25 × 21 × 128 × 128 |

配置里写的是 `*_train_ds_*`，`predict_external.py` 按 `_train_ds_` → `_test_ds_`
替换成测试文件（第 74–80 行）。

---

## 4. 与论文的对应

| 论文位置 | 内容 | 本目录来源 |
|---|---|---|
| Tab. `tab:main1`（`sections/05_experiments.tex:60`） | plume1d 六场 L2RE(%)，FNO / GNOT / Transolver 三行 | `E01_main_eq9_external.csv` 的 plume1d 9 行 → 种子均值 ± SD ×100 |
| Tab. `tab:main2`（附录 `08_appendix.tex:700`） | plume2 六场 L2RE(%)，同样三行 | 同 CSV 的 plume2 3 行（单种子，无 SD） |
| Tab. `tab:training`（`05_training.tex`） | 参数量 / epochs / batch / 种子 | 参数与预算来自 `E04` 的 `model_meta.json`；种子列由本目录与 E02 的 CSV 现取 |
| Tab. `tab:cost`（`06_cost.tex`） | 峰值显存 / 单次调用时延 / 训练时长 / $E_{v_x}$·$E_{v_y}$ | 显存与时延来自 `E04` 的 `cost_*.json`；误差列复用 `tab:main1` 的种子均值 |

**不在本目录的行**：
- MFGNO（C 组）与独立 MPNN（A 组）→ `../E02_ablation/results/E02_ablation_eq9.csv`
- Persistence 行 → 本目录的 `persistence_*_eq9.csv`

**论文没有引用本目录的图**——主结果全部以表呈现；E01 不单独出图。

---

## 5. 已知问题

1. 5 处 SD 小数点移位已修正，六张表与源 CSV 核对一致（2026-10-31）。
2. `make_tables.py` 已与现有版式同步，可从 `results/` 重生成。
3. `score_predictions.py` 是 CPU 脚本，按「登录节点不跑计算」的规矩用
   `common/run_cpu.sh` 提交（`run_eval.sh` 包装的是 GPU 上的
   `eval_checkpoint.py`，两者不要混用）：
   ```bash
   yhbatch code/experiments/common/run_cpu.sh \
       "$PWD/code/experiments/common/score_predictions.py" \
       --stage E01_main --subset plume1d
   ```
   `run_cpu.sh` 的第一个参数必须是**绝对路径**。
