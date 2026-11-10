#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-pred
# 日志归位到 experiments/logs/（EXPERIMENT_FRAMEWORK.md §7），前缀 pred- 与训练/评估区分。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# 统一推理层：FNO / GNOT / Transolver 在 12 条协议测试轨迹上推理。
# 见 common/predict_external.py 与 KNOWN_ISSUES.md 的 K4/K4c。
#
# 用法（参数透传给 predict_external.py）：
#
#   yhbatch common/run_predict.sh --stage E01_main --subset plume1d
#   yhbatch common/run_predict.sh --runs <run_dir> [<run_dir> ...]
#
# 写法与 run_train.sh / run_eval.sh 一致：
#   - 不写 set -euo pipefail（-u 会被 bashrc/conda 引用的可选变量触发）
#   - 不做路径解析，REPO 硬编码
#   - 显式透传退出码
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

echo "args : $*"
echo "host : $(hostname)"
echo "start: $(date -Is)"

python -u $GEOSIM_ROOT/code/experiments/common/predict_external.py "$@"
status=$?

echo "end  : $(date -Is)"
if [[ $status -ne 0 ]]; then
    echo "!! 推理失败：退出码 $status（完整堆栈见同名 .err）"
fi
exit $status