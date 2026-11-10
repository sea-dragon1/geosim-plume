#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
# ⚠ 墙钟 96h（原 48h）。gpu5 分区的 TIMELIMIT 是 `infinite`，48h 是本脚本自己写的上限。
# 最贵的两个模型都贴着它：MPNN（A 组，6 个独立网络逐个前向）单 epoch 约 92 s、
# 2000 epochs ≈ 51 h，48h 会把它砍在约 88% 处；MFGNO 约 33 h、GNOT 4 h、FNO 3 h
# 会自然结束，提高上限对它们没有影响。见 KNOWN_ISSUES.md K3。
#SBATCH -t 96:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-train
# 发布包：日志写到**提交目录**（slurm-<jobid>.out/.err）。
# 原版把绝对路径写死到 experiments/logs/，换机器即失效；SBATCH 指令里不能用 shell 变量。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# 用法（两个参数都传**绝对路径**）：
#
#   yhbatch common/run_train.sh <trainer.py> <config.yaml>
#
# 参数顺序与 python 调用一致，方便直接对照。
#
# 写法与项目里已跑通的训练提交脚本保持一致：
#   - 不写 set -euo pipefail（-u 会被 source/bashrc/conda 引用的可选变量触发，
#     实测让作业在环境准备阶段就死掉，失败点与实验无关）
#   - 不做路径解析，REPO 直接硬编码，配置传绝对路径
# 早期版本加了这两样，连挂三次；简单比聪明可靠。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
# source "${GEOSIM_BASHRC:-$HOME/.bashrc}" 是必需的：conda activate 依赖其中的 conda init 块。
# （缺少这一句时 conda activate 会直接 CommandNotFoundError。）
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

echo "config : $2"
echo "trainer: $1"
echo "host   : $(hostname)"
echo "start  : $(date -Is)"

python -u "$1" "$2"
status=$?

echo "end    : $(date -Is)"
# 退出码必须显式传递。否则后面这条 echo 会把失败吞成 0，
# sacct 显示 COMPLETED —— 坏作业看起来和成功作业一模一样（已实测踩过：
# 3537156/3537157 因 trainer 路径不存在启动即退出，却报 COMPLETED）。
if [[ $status -ne 0 ]]; then
    echo "!! 训练失败：python 退出码 $status（完整堆栈见同名 .err）"
fi
exit $status