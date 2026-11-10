#!/bin/bash
#SBATCH -p cp2 -N 1 -c 4 --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J geosim-cpu
# 日志归位到 experiments/logs/，前缀 cpu- 与 train-/eval-/pred-/cost-/pertraj- 区分。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# 纯 CPU 作业：只读数据、不碰 GPU 的那些脚本。
#
# 为什么需要它：**登录节点对每个用户有 5 GiB 硬内存上限**
# （cgroup `user-<uid>.slice`）。而这一类脚本要整读 hdf5 —— 例如
# `score_persistence.py --subset plume2` 峰值需要几个 GB
# （`relative_l2_per_frame` 内部会把 (25,20,128,128,6) 转成 float64，
#   `fit_epsilon` 还会整读训练文件的每个场），实测在登录节点被
# OOM-kill（退出码 137），而 `free` 显示还有几十 GB —— 那是别人的配额，
# 不是你能用的。**别把这类脚本放在登录节点跑，也别据此以为机器没内存。**
#
# 用法（两个参数都传**绝对路径**，与 run_train.sh 同约定）：
#
#   yhbatch common/run_cpu.sh <script.py> [args...]
#
# 例：
#   yhbatch common/run_cpu.sh /abs/common/score_persistence.py --subset plume2
#
# 写法刻意与 run_train.sh 保持一致（不写 set -euo pipefail、不做路径解析、
# 显式传递退出码），理由见那边的注释。
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
# source "${GEOSIM_BASHRC:-$HOME/.bashrc}" 是必需的：conda activate 依赖其中的 conda init 块。
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

echo "script : $1"
echo "args   : ${@:2}"
echo "host   : $(hostname)"
echo "start  : $(date -Is)"
echo "memlim : $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)"

script="$1"; shift
python -u "$script" "$@"
status=$?

echo "end    : $(date -Is)"
# 退出码必须显式传递，否则后面这条 echo 会把失败吞成 0（见 run_train.sh 的注释）。
if [[ $status -ne 0 ]]; then
    echo "!! CPU 作业失败：python 退出码 $status（完整堆栈见同名 .err）"
fi
exit $status