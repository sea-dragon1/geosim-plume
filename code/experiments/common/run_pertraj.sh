#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-pertraj
# 日志归位到 experiments/logs/，前缀 pertraj- 与 train-/eval-/pred- 区分。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# 重跑 run 并保存**逐轨迹**指标（配对 bootstrap 用），见 common/eval_per_traj.py。
# 必须在 GPU 上跑：登录节点 CPU 单个 run 580s 都跑不完。
#
# 用法（参数直接透传给 eval_per_traj.py）：
#
#   yhbatch common/run_pertraj.sh --stage E02_ablation --subset plume1d
#   yhbatch common/run_pertraj.sh --runs <run_dir> [<run_dir> ...]
#
# 写法与 run_eval.sh / run_predict.sh 保持一致：
#   - 不写 set -euo pipefail（-u 会被 bashrc/conda 引用的可选变量触发）
#   - 不做路径解析，REPO 直接硬编码
#   - 显式透传退出码
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

echo "args : $*"
echo "host : $(hostname)"
echo "start: $(date -Is)"

python -u $GEOSIM_ROOT/code/experiments/common/eval_per_traj.py "$@"
status=$?

echo "end  : $(date -Is)"
if [[ $status -ne 0 ]]; then
    echo "!! 逐轨迹评估失败：退出码 $status（完整堆栈见同名 .err）"
fi
exit $status