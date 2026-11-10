#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-eval
# 日志同样归位到 experiments/logs/（EXPERIMENT_FRAMEWORK.md §7）。
# 前缀是 eval-，与训练的 train-<jobid> 区分开。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# 按论文 Eq.(9) 评估 checkpoint（协议口径），见 KNOWN_ISSUES.md 的 K2/F7。
#
# 用法（参数直接透传给 eval_checkpoint.py）：
#
#   yhbatch common/run_eval.sh                                   # 默认 E02 + plume1d
#   yhbatch common/run_eval.sh --stage E02_ablation --subset plume1d --compare
#   yhbatch common/run_eval.sh --runs <run_dir> [<run_dir> ...]
#
# 写法与 common/run_train.sh 保持一致：
#   - 不写 set -euo pipefail（-u 会被 bashrc/conda 引用的可选变量触发）
#   - 不做路径解析，REPO 直接硬编码
#   - 显式透传退出码，否则末尾的 echo 会把失败吞成 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

echo "args : $*"
echo "host : $(hostname)"
echo "start: $(date -Is)"

python -u $GEOSIM_ROOT/code/experiments/common/eval_checkpoint.py "$@"
status=$?

echo "end  : $(date -Is)"
if [[ $status -ne 0 ]]; then
    echo "!! 评估失败：退出码 $status（完整堆栈见同名 .err）"
fi
exit $status