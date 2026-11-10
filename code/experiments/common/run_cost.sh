#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-cost
# 日志归位到 experiments/logs/，前缀 cost- 与 train-/eval-/pred-/pertraj- 区分。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# E04 效率实测：参数量 / 峰值显存 / 训练 step 时间 / 推理时延 / 图构建成本。
# 产出 E04_diagnostics_cost/results/cost_<model>_<subset>.json，喂给 tables/06_cost.tex。
#
# ⚠ **一进程一模型**：`model_zoo.isolate_paths` 会在切换模型时摘掉 REPO 路径，
# 而 GNN 家族（mfgno/mpnn）依赖它，且 FNO/GNOT/Transolver 各有一个 utils.py /
# train.py 会互相遮蔽。所以只能串行起 5 个进程，不能在一个进程里循环。
#
# 用法：
#   yhbatch common/run_cost.sh                  # 默认 plume1d
#   yhbatch common/run_cost.sh plume2           # 换子集
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

SUBSET="${1:-plume1d}"
COST=$GEOSIM_ROOT/code/experiments/E04_diagnostics_cost/src/cost.py

echo "subset: $SUBSET"
echo "host  : $(hostname)"
echo "start : $(date -Is)"

status=0
for m in mfgno mpnn fno transolver gnot; do
    echo ""
    echo "=============== $m ==============="
    python -u "$COST" --model "$m" --subset "$SUBSET"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!! $m 测量失败：退出码 $rc（继续下一个，最后统一报）"
        status=1
    fi
done

echo ""
echo "=============== 汇总 ==============="
python -u "$COST" --aggregate --subset "$SUBSET"

echo "end   : $(date -Is)"
if [[ $status -ne 0 ]]; then
    echo "!! 有模型未测成：见上面的失败行与同名 .err"
fi
exit $status