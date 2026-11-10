#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=80G
#SBATCH -J geosim-diag
# 日志归位到 experiments/logs/，前缀 diag- 与 train-/eval-/pred-/cost-/pertraj- 区分。
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 发布包：GEOSIM_ROOT 指向 geosim-plume/；未设置时回退到提交目录。
GEOSIM_ROOT="${GEOSIM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$GEOSIM_ROOT/code/vendor" ]]; then
    echo "!! 找不到 GEOSIM_ROOT/code/vendor。请设置 GEOSIM_ROOT，或从 geosim-plume/ 提交。" >&2
    exit 2
fi

#
# E04 空间诊断的**推理前置**：为五个模型产出单窗口协议预测。
#
# 产出 E04_diagnostics_cost/runs/<exp_id>/predictions.hdf5（pred + truth_*）。
# 它是下列产物共同的输入，缺了它们全都跑不动：
#     E04/src/plot_diagnostics.py       → F03 三张图 + fig03_manifest.json
#     E04/src/velocity_profile.py       → 剖面汇总（读的也是这个 hdf5）
#     E04/src/predict_test.py 自身      → single_window_eq9_*.csv（ε 触发计数）
#
# ⚠ **本脚本是 2026-09-25 补建的** —— `E04_diagnostics_cost/PLAN.md` 一直要求有它，
#   但仓库里从来没有，`E04/runs/` 也一直是空目录。后果是 `plot_diagnostics.py`
#   与 `velocity_profile.py` 都只能对**既有产物**工作、无法从零复现；
#   连 `plot_diagnostics.py` 里那处已登记的 km 换算口径问题也因此无法重算
#   （见该文件 `peak_shift_km` 段的注释）。
#
# ⚠ **一进程一模型**：`model_zoo.isolate_paths` 会在切换模型时摘掉 REPO 路径，
#   而 GNN 家族（mfgno/mpnn）依赖它，且 FNO/GNOT/Transolver 各有一个 utils.py /
#   train.py 会互相遮蔽。所以只能串行起 5 个进程，不能在一个进程里循环。
#
# 用法：
#   yhbatch common/run_diag.sh                  # 默认 plume1d
#   yhbatch common/run_diag.sh plume2           # 换子集
#
# 跑完之后接：
#   python -B E04_diagnostics_cost/src/plot_diagnostics.py --subset <s> --install
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$GEOSIM_ROOT/code/vendor:$PYTHONPATH
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

SUBSET="${1:-plume1d}"
SRC=$GEOSIM_ROOT/code/experiments/E04_diagnostics_cost/src

echo "subset: $SUBSET"
echo "host  : $(hostname)"
echo "start : $(date -Is)"

status=0
for m in mfgno mpnn fno transolver gnot; do
    echo ""
    echo "=============== $m ==============="
    python -u "$SRC/predict_test.py" --model "$m" --subset "$SUBSET"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!! $m 推理失败：退出码 $rc（继续下一个，最后统一报）"
        status=1
    fi
done

echo ""
echo "end   : $(date -Is)"
if [[ $status -ne 0 ]]; then
    echo "!! 有模型未成功：见上面的失败行与同名 .err"
    echo "   单窗口打分（Tables 06 的 E_vx/E_vy 与 ε 触发计数）会缺行，不要当作完整结果"
fi
exit $status