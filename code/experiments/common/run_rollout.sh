#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
# ⚠ 24h：比 run_eval.sh 的 4h 长得多 —— Transolver 是逐帧自回归的，12 个窗口
#    = 120 次前向/轨迹，12（plume2 是 25）条轨迹下来是主成本。实测单帧
#    0.3–1.2 s 的区间跨 4 倍，先用 E04 的 cost.py 量出真实值再回来收窄这个上限。
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-rollout
# 日志归位到 experiments/logs/（EXPERIMENT_FRAMEWORK.md §7）。
# 前缀 eval-rollout-，与训练的 train-<jobid> 区分开。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# E03 长程自治滚动的提交脚本，参数全量透传给 run_rollout.py。
#
# 用法：
#   yhbatch common/run_rollout.sh --subset plume1d --run <run_dir> \
#       --mode auto --mode teacher --dump
#
# ⚠ **一个作业只跑一个 run**：FNO / GNOT / Transolver 三者不能共存于同一进程
#   （各有一个 utils.py / train.py，见 predict_external.isolate_paths 的说明），
#   而 GNN 家族又依赖 vendor 的 REPO 路径。model_zoo 会拦住跨界装载并报错，
#   所以不要试图在一个 python 进程里循环多个 run。
#
# ⚠ 提交前确认 τ_c 已经算过（run_rollout.py 会硬性检查，缺了就报错退出）：
#   python -B E03_rollout/src/thresholds.py --subset <s> --range-only
#   python -B E03_rollout/src/thresholds.py --subset <s> --run <run_dir>
#
# 退出码显式透传 —— 不加 exit $status 的话末尾 echo 会把失败吞成 0，
# sacct 显示 COMPLETED（本项目已实测踩过，见 README 的坑列表）。

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH

source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

echo "args : $*"
echo "host : $(hostname)"
echo "start: $(date -Is)"

python -u $GEOSIM_ROOT/code/experiments/E03_rollout/src/run_rollout.py "$@"
status=$?

echo "end  : $(date -Is)"
if [[ $status -ne 0 ]]; then
    echo "!! 长程滚动失败：退出码 $status（完整堆栈见同名 .err）"
fi
exit $status