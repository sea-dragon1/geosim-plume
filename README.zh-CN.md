# GeoSIM-Plume — 复现说明

本发布包用于复现 GeoSIM 基准论文中两个二维地幔对流子集的实验、表格与图：

- **GeoSim-Plume1**（`plume1d`，单一变化参数：羽流温度）
- **GeoSim-Plume2**（`plume2`，两个变化参数：羽流温度与岩石圈厚度）

## 关联论文

本代码对应论文：

> Hailong Liu, Dinghui Yang, Mengxue Liu, Rui Qi, Meng Zhao, Yihang Tang,
> Jin Wen, and Zhilong Fang (2026). *GeoSim: A Reproducible Benchmark
> Dataset and Multi-Field Graph Neural Operator for Mantle-Plume
> Dynamics.* Manuscript submitted to *Geoscientific Model Development*.

数据集单独归档于 <https://huggingface.co/datasets/irenier/GeoSim>（DOI
`10.57967/hf/11421`）。本包的机器可读引用信息见
[`CITATION.cff`](CITATION.cff)。

英文版：[README.md](README.md)。

## 1. 包内内容

```
geosim-plume/
├── code/
│   ├── vendor/                     # FNO / Transolver + GNOT 适配层 + 共享指标
│   └── experiments/
│       ├── common/                 # 训练、评估、表格与作业提交脚本
│       ├── E01_main/               # 外部基线：12 个配置 + 12 个已训练 run
│       ├── E02_ablation/           # MFGNO / MPNN A-B-C：12 个配置 + 12 个已训练 run
│       ├── E03_rollout/            # 长程自治 rollout
│       ├── E04_diagnostics_cost/   # 空间诊断、剖面、成本、数据质检
│       ├── figures/                # 数据集图与参数覆盖图
│       ├── logs/                   # 36 份训练日志（训练时长依据）
│       ├── protocol/               # 冻结协议（划分、指标、时间、损失、参数）
│       └── tests/
├── results/
│   ├── metrics/                    # 汇总指标，含 E03 合并结果
│   ├── per_run/                    # 逐 run 的 eq9.json / pertraj.npz
│   ├── tables/                     # 8 张 LaTeX 表
│   └── figures/                    # 8 张图 + fig03/fig04 manifest
├── smoke/                          # 外部基线冒烟测试
├── environment/                    # 环境说明、版本清单与建环境脚本
├── solver_patch/                   # 公开 I2VIS 求解器的修改补丁
├── data/                           # 数据集发布信息 + I2VIS 数据生成脚本
├── CITATION.cff                    # 引用信息
├── LICENSE                         # MIT（代码）
├── PERMISSIONS.md                  # 许可记录（I2VIS patch）
├── THIRD_PARTY_NOTICES.md          # 第三方来源与许可证说明
└── THIRD_PARTY_LICENSES/           # 第三方许可证全文
```

上游 I2VIS 求解器与原始模拟数据**不随包提供**。`solver_patch/` 只包含修改、
算例配置和编译说明；上游代码从 OSF 获取（DOI `10.17605/OSF.IO/BNVTH`）。
数据生成驱动在本代码包内，solver patch 单独放置，以明确上游来源和许可边界。

## 2. 运行环境

- conda 环境 `fno`：Python 3.8、PyTorch 2.4.1+cu121、CUDA 12.1，
  `torch_geometric`、`torch_cluster`、`h5py`、`numpy`、`scipy`、`matplotlib`、
  `ruamel.yaml`、`dgl` 2.1.0（CPU 版，GNOT 使用）。
- 随包提交脚本使用天河默认值：GPU 分区 `gpu5`、CPU 分区 `cp2`、conda 环境
  `fno`。天河上用 `yhbatch` 提交（不是 `sbatch`）；其他 SLURM 集群可使用
  各自的批处理命令，脚本只依赖 `#SBATCH` 头。
- 换到其他集群时需要修改：
  1. `code/experiments/common/run_*.sh` 与 `smoke/smoke_test.sh` 中的
     `#SBATCH -p` / `#SBATCH -t`（SBATCH 指令不能读 shell 变量）；
  2. 设置 `GEOSIM_CONDA_ENV` 指定 conda 环境名（默认 `fno`）；
  3. 设置 `GEOSIM_BASHRC` 指定初始化 conda 的 shell 文件（默认 `~/.bashrc`）。
- 精确包版本、硬件探测结果与建环境脚本见 [`environment/`](environment/)。
- 所有命令从发布包根目录执行：

```bash
cd /path/to/geosim-plume
```

## 3. 需要配置的路径

### 3.1 数据根目录

```bash
export GEOSIM_DATA=/path/to/geosim-data
```

`GEOSIM_DATA` 下必须包含 `plume1d_corrected/` 与 `plume2/`。
`GEOSIM_ROOT` 可省略，`code/experiments/common/paths.py` 会按文件位置自动推导。

数据集单独发布，Hugging Face 链接、DOI、许可证、revision 和访问状态见
[`data/README.md`](data/README.md)。

### 3.2 YAML 中的路径

训练与 checkpoint 评估读取 YAML 里的 `dataset.saved_folder`，不使用
`GEOSIM_DATA`。随包 YAML 中是绝对路径，需统一改成本机数据路径：

```bash
OLD=/fs1/home/USER/lhl/neuraloeprator-main/data
NEW=/path/to/geosim-data
grep -rl "$OLD" code/experiments/E0*/configs code/experiments/E0*/runs smoke/configs \
  | xargs sed -i "s#$OLD#$NEW#g"
```

stage 配置（`.../configs/*.yaml`）和 run 目录里的配置副本
（`.../runs/<exp_id>/<exp_id>.yaml`）都要改：评估随包 checkpoint 时读的是
run 目录里的副本。

如果整个发布包换了位置并且要**重新训练**，还需把
`code/experiments/E0*/configs/*.yaml` 里的 `root_path` 改为新的包路径。

### 3.3 一个图脚本

`code/experiments/figures/make_fig01.py` 读取
`<包根>/data/plume1d_corrected/plume1d_test_ds_128x128_phys1e6_first21.hdf5`。
把该文件复制到包的 `data/` 目录，或修改脚本中的 `DATA_PATH`。

### 3.4 I2VIS solver patch

`solver_patch/` 包含 Plume1/Plume2 两个分支的 patch、各自的
`init.t3c`/`mode.t3c`/`Makefile` 和校验文件；不包含上游 I2VIS 源码。

```bash
# 从 https://doi.org/10.17605/OSF.IO/BNVTH 下载 codes.zip 并解压到
# /path/to/codes/codes
bash solver_patch/scripts/apply_patch.sh plume1 /path/to/codes/codes i2vis-plume1
# 或 plume2
```

编译要求和热力学数据库文件见 `solver_patch/README.md`。

### 3.5 GNOT 可选依赖

上游 GNOT 模型代码不随包。包内只有 GeoSim 适配层
`code/vendor/GNOT-master/plume1/`。如需训练或重新评估 GNOT，请把上游的
`models/mmgpt.py`、`models/mlp.py`、`utils.py` 及对应的 `__init__.py`
放到 `code/vendor/GNOT-master/`，具体目录见
[`code/vendor/GNOT-master/NOTICE.md`](code/vendor/GNOT-master/NOTICE.md)。
没有这些文件时，随包的 GNOT 指标和 checkpoint 仍可读取，但 GNOT 训练和
checkpoint 装载不可用。

## 4. 数据清单

| 用途 | `GEOSIM_DATA` 下的文件 |
|---|---|
| 短窗口训练与评估（21 帧） | `plume1d_corrected/plume1d_{train,test}_ds_128x128_phys1e6_first21.hdf5`<br>`plume2/plume2d_{train,test}_ds_128x128_phys1e6_first21_corrected.hdf5` |
| 长程 rollout（E03）、成本（E04） | `plume1d_corrected/plume1d_{train,test}_ds_128x128_phys1e6.hdf5`<br>`plume2/plume2d_{train,test}_ds_128x128_norm_corrected.hdf5` |
| 数据质检（V8，可选） | `plume1d_corrected/plume1d_train_401x309.hdf5`、`plume2/plume2d_train_401x309.hdf5`、`plume1d_corrected/remap_weights_401x309_to_128x128.npz` |

plume2 的 `_norm_corrected` 是修正后的全时程文件，其时间轴已换算为 Myr
（此处的 `_norm` 不代表 per-dataset 归一化）。

## 5. 实验配置总表

| 论文实验 | 阶段 | 配置 | run 数 |
|---|---|---|---|
| X1 短窗口外部基线 | `E01_main` | `2026-09-17-{fno,gnot,transolver}-E01-plume1d-seed{0,1,2}.yaml`（9 个）<br>`2026-09-22-{fno,gnot,transolver}-E01-plume2-corr-seed0.yaml`（3 个） | 12 |
| X2 短窗口 MFGNO / MPNN | `E02_ablation` | 复用下方 C（MFGNO）与 A（MPNN） | 复用 X3 |
| X3 设计归因 A/B/C | `E02_ablation` | `2026-09-17-{mpnn-E02-A,mfgno-E02-B-equal,mfgno-E02-C-equal}-plume1d-seed{0,1,2}.yaml`（9 个）<br>`2026-09-22-{mpnn-E02-A,mfgno-E02-B-equal,mfgno-E02-C-equal}-plume2-corr-seed0.yaml`（3 个） | 12 |
| V4/V5 长程 rollout / 失败判据 | `E03_rollout` | 逐 run 调用 `run_rollout.sh`；20 个 checkpoint | 合并结果已随包 |
| V6 速度剖面 | `E04_diagnostics_cost` | `run_diag.sh` + `velocity_profile.py` | plume1d |
| V7 成本 | `E04_diagnostics_cost` | `run_cost.sh` | 5 个模型 |
| V8 数据质检 | `E04_diagnostics_cost` | `data_qc.py` | 两个子集 |
| V9 训练元数据 | `common/collect_meta.py` | — | 8 个模型条目 |

每个 run 目录包含 `checkpoints/*-best.pt`、`eval/predictions.hdf5`
（E01 全部 run；E02 部分 run）、`eval/<exp_id>_eq9.json`，以及该 run 的配置副本。

## 6. 路线 A — 用随包结果复现表格与图（不训练）

这是最快的路径，只读取随包的 checkpoint、预测和指标文件。

### 6.1 表格

```bash
python -B code/experiments/common/make_tables.py            # 生成表 03–08
python -B code/experiments/common/make_tables.py --dry-run  # 只打印，不落盘
```

`results/tables/02_dataset.tex` 与 `09_solver_cost.tex` 是随包的静态表，
不由该脚本生成。

### 6.2 图

需要读数据的脚本统一走 `run_cpu.sh` 提交：

```bash
# 图 1 — 数据集总览（需要先处理 §3.3 的路径）
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/figures/make_fig01.py"

# 图 A01 — 参数覆盖（只读 protocol CSV）
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/figures/make_figA01.py"

# 图 4 / F04 — rollout 曲线（正文 + 附录）
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E03_rollout/src/plot_rollout_paper.py"

# 图 3 / F03b / F03c + 剖面 CSV（使用随包的 E01/E02 预测）
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E04_diagnostics_cost/src/plot_diagnostics.py" \
    --subset plume1d --install
```

| 产物 | 位置 |
|---|---|
| 图 1 | `code/experiments/figures/fig01_benchmark_overview.pdf`（含 manifest） |
| 图 A01 | `code/experiments/figures/figA01_parameter_coverage.pdf`（含 manifest） |
| 图 4 / F04 | `results/figures/fig04_rollout_main.pdf`、`fig04_rollout_full.pdf` |
| 图 3 / F03b / F03c | `results/figures/fig03_field_predictions.pdf`、`fig03b_profile.pdf`、`fig03c_failure_case.pdf`、`fig03_manifest.json` |
| 图 2 | 随包的 `results/figures/fig02_mfgno_overview.png` |

`plot_diagnostics.py` 同时会重写
`E04_diagnostics_cost/results/profile_errors.csv` 与 `profile_errors_summary.csv`；
随包副本位于 `results/metrics/E04_diagnostics_cost/`。

## 7. 路线 B — 用随包 checkpoint 重新评估（不训练）

### 7.1 E01 外部基线

```bash
yhbatch code/experiments/common/run_predict.sh --stage E01_main --subset plume1d
yhbatch code/experiments/common/run_predict.sh --stage E01_main --subset plume2

yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_predictions.py" \
    --stage E01_main --subset plume1d
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_predictions.py" \
    --stage E01_main --subset plume2
```

产物：`runs/<exp_id>/eval/predictions.hdf5`、`eval/<exp_id>_eq9.json`、
`E01_main/results/E01_main_eq9_external.csv`。

### 7.2 E02 GNN 模型（A/B/C）

```bash
yhbatch code/experiments/common/run_eval.sh --stage E02_ablation --subset plume1d --compare
yhbatch code/experiments/common/run_eval.sh --stage E02_ablation --subset plume2 --compare
```

产物：`E02_ablation/results/E02_ablation_eq9.csv`。

### 7.3 Persistence、逐轨迹指标、训练元数据

```bash
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_persistence.py" --subset plume1d
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_persistence.py" --subset plume2

yhbatch code/experiments/common/run_pertraj.sh --stage E02_ablation --subset plume1d

yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/collect_meta.py"
```

产物：`E01_main/results/persistence_<subset>_eq9.csv`、
`results/per_run/*_pertraj.npz`、`E04_diagnostics_cost/results/model_meta.json`。

### 7.4 E04 诊断与成本

```bash
yhbatch code/experiments/common/run_diag.sh plume1d
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E04_diagnostics_cost/src/velocity_profile.py" --subset plume1d

yhbatch code/experiments/common/run_cost.sh
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E04_diagnostics_cost/src/data_qc.py" --subset both
```

成本数字依赖硬件。论文使用的是随包的
`results/metrics/E04_diagnostics_cost/cost_*.json`（A800-SXM4-80GB，测量环境为
torch 2.4.1）；在本机重新测量可能得到不同数值。

### 7.5 E03 长程 rollout（可选）

图 4 使用的最终合并结果已随包提供：
`results/metrics/E03_rollout/formal_all5/`。如需重跑单个 checkpoint：

```bash
# 在 GPU 计算节点上运行；rollout 前必须先有该 run 的阈值。
python -B code/experiments/E03_rollout/src/thresholds.py --subset plume1d --range-only
python -B code/experiments/E03_rollout/src/thresholds.py --subset plume1d --run <run_dir>

yhbatch code/experiments/common/run_rollout.sh --subset plume1d --run <run_dir> \
    --mode auto --mode teacher --dump
```

随包的 `E03_rollout/results/tau_c.json` 只覆盖四个模型家族（不含 GNOT）的
seed-0 checkpoint；其他种子与 GNOT 需先单独计算阈值。完整的 campaign 汇总
目录（`cases.json` + 逐 case 目录）不随包提供，论文图请使用随包的合并结果。

## 8. 路线 C — 从头训练

1. 按 §3 配置路径。
2. 分批提交 E01（集群对并发提交数有限制）：

```bash
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed0
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed1
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed2
bash code/experiments/E01_main/run_all.sh --with-plume2 --only=plume2
```

3. 同样方式提交 E02：

```bash
bash code/experiments/E02_ablation/run_all.sh --only=plume1d-seed0
bash code/experiments/E02_ablation/run_all.sh --only=plume1d-seed1
bash code/experiments/E02_ablation/run_all.sh --only=plume1d-seed2
bash code/experiments/E02_ablation/run_all.sh --with-plume2 --only=plume2
```

每个作业写 `code/experiments/<stage>/runs/<exp_id>/`。训练为单 GPU，
墙钟上限 96 小时；保持 `root_path` 与 `exp_id` 不变以维持目录结构。

4. 训练完成后按 §7 重新评估，再按 §6 生成表格与图。

## 9. 验证

- `python -B code/experiments/common/make_tables.py --dry-run` 只打印六张生成表。
- 走完路线 A 后，`results/tables/` 与 `results/figures/` 中的产物应与随包版本一致。
- `yhbatch smoke/smoke_test.sh` 验证 FNO、Transolver 能从 `code/vendor` 导入并各训练
  2 epoch；只有 `code/vendor/GNOT-master/` 下放入上游 GNOT 代码时才检查 GNOT。
- 单元测试：`python -B code/experiments/tests/test_dt.py` 等（见 `protocol/`）。

## 10. 引用

使用本代码或数据集时，请引用 GeoSim 论文与数据集记录。机器可读的引用信息见
[`CITATION.cff`](CITATION.cff)。

## 11. 许可证

- **代码：** MIT License，Copyright (c) 2026 Hailong Liu，见 [`LICENSE`](LICENSE)。
- **数据集：** CC BY 4.0，数据集单独发布，见 [`data/README.md`](data/README.md)
  与 [`data/LICENSE`](data/LICENSE)。
- **第三方组件：** 见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 和
  [`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/)。
- **I2VIS patch/config：** 不属于本包 MIT 许可范围；patch 与配置已获得
  I2VIS 作者（T. Gerya）许可（2026-11-10 邮件确认），见
  [`solver_patch/NOTICE.md`](solver_patch/NOTICE.md) 与
  [`PERMISSIONS.md`](PERMISSIONS.md)。
- **GNOT：** 上游 GNOT 模型代码不随包；包内只有 GeoSim 适配层
  （`code/vendor/GNOT-master/plume1/`）。用户需自行获取上游 GNOT 代码并
  遵守其条款，见 [`code/vendor/GNOT-master/NOTICE.md`](code/vendor/GNOT-master/NOTICE.md)。
