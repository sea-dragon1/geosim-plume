#!/bin/bash
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH --mem=80G
#SBATCH -J geosim-smoke
#SBATCH -o smoke/logs/smoke-%j.out
#SBATCH -e smoke/logs/smoke-%j.err
#
# 干净目录冒烟：只依赖 code/vendor，**不引用原仓库的代码树**。
#
# 验证三件事：
#   1. FNO / Transolver 能从 vendor 目录 import；GNOT 仅在上游代码已放入时检查；
#   2. 2-epoch 训练能跑通（数据、模型、优化器、保存全链路）；
#   3. 产物落在 smoke/runs/<exp_id>/。
#
# 数据仍指向现有 first21 文件（绝对路径）；最终发布版会改成 data/download.py 拉取。
#
# 用法：yhbatch smoke/smoke_test.sh
#
# ⚠ 不写 set -u / set -e：/etc/bashrc 会引用未定义的 BASHRCSOURCED，
#   开 -u 会让 `source "${GEOSIM_BASHRC:-$HOME/.bashrc}"` 直接退出（实测作业 3795822 就是这么失败的）。
#   与 common/run_train.sh 的处理一致。

# ⚠ 不能用 BASH_SOURCE：SLURM 会把批处理脚本复制到 /var/spool/slurm/d，
#   BASH_SOURCE 指向 spool 目录而不是仓库（实测作业 3795824 就是这么失败的）。
#   用 SLURM_SUBMIT_DIR（提交时的工作目录），并显式要求从 geosim-plume/ 提交。
PKG="${GEOSIM_PKG:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -d "$PKG/code/vendor" ]]; then
    echo "!! 找不到 code/vendor。请从 geosim-plume/ 目录提交：" >&2
    echo "     cd geosim-plume && yhbatch smoke/smoke_test.sh" >&2
    echo "   （或设置 GEOSIM_PKG 为 geosim-plume 的绝对路径）" >&2
    exit 2
fi
VENDOR="$PKG/code/vendor"

export PYTHONPATH="$VENDOR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${GEOSIM_BASHRC:-$HOME/.bashrc}"
conda activate "${GEOSIM_CONDA_ENV:-fno}"

cd "$PKG"

echo "PKG      : $PKG"
echo "VENDOR   : $VENDOR"
echo "PYTHONPATH: $PYTHONPATH"
echo "CWD      : $(pwd)"
echo "python   : $(which python)"
echo "host     : $(hostname)"
echo "start    : $(date -Is)"
echo

mkdir -p "$PKG/smoke/runs" "$PKG/smoke/logs"

FAIL=0

# ---------------------------------------------------------------- import 检查
# 每个模型单独一个进程 —— fno 与 GNOT 都有 utils.py，同进程会互相遮蔽。
check_import() {
    local name="$1" script="$2" path="$3"
    echo "--- import 检查: $name"
    python - "$script" "$path" <<'PY'
import importlib.util
import sys

script, path = sys.argv[1], sys.argv[2]
sys.path.insert(0, path)
spec = importlib.util.spec_from_file_location("__smoke_mod__", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("    OK:", script)
PY
    if [[ $? -ne 0 ]]; then
        echo "    !! import 失败: $name"
        return 1
    fi
    return 0
}

check_import "FNO"        "$VENDOR/fno/train_new.py"               "$VENDOR/fno"        || FAIL=1
check_import "Transolver" "$VENDOR/Transolver/train.py"            "$VENDOR/Transolver" || FAIL=1

HAS_GNOT=0
if [[ -f "$VENDOR/GNOT-master/models/mmgpt.py" && -f "$VENDOR/GNOT-master/utils.py" ]]; then
    check_import "GNOT" "$VENDOR/GNOT-master/plume1/train.py" "$VENDOR/GNOT-master/plume1" || FAIL=1
    HAS_GNOT=1
else
    echo "--- SKIP GNOT: 上游模型代码未随包（见 code/vendor/GNOT-master/NOTICE.md）"
fi

# ---------------------------------------------------------------- 2-epoch 训练
run_one() {
    local name="$1" trainer="$2" cfg="$3" exp_id="$4"
    echo
    echo "===== 训练: $name ====="
    python -u "$trainer" "$cfg"
    local status=$?
    if [[ $status -ne 0 ]]; then
        echo "!! $name 训练失败：退出码 $status"
        FAIL=1
        return 1
    fi
    if [[ ! -d "$PKG/smoke/runs/$exp_id" ]]; then
        echo "!! $name 没有生成 run 目录: smoke/runs/$exp_id"
        FAIL=1
        return 1
    fi
    echo "--- $name 产物:"
    find "$PKG/smoke/runs/$exp_id" -maxdepth 2 -type f -printf '    %P\n' | head -10
    return 0
}

run_one "FNO"        "$VENDOR/fno/train_new.py"            "$PKG/smoke/configs/SMOKE-vendor-fno.yaml"        "SMOKE-vendor-fno"
if [[ $HAS_GNOT -eq 1 ]]; then
    run_one "GNOT"   "$VENDOR/GNOT-master/plume1/train.py" "$PKG/smoke/configs/SMOKE-vendor-gnot.yaml"       "SMOKE-vendor-gnot"
fi
run_one "Transolver" "$VENDOR/Transolver/train.py"         "$PKG/smoke/configs/SMOKE-vendor-transolver.yaml" "SMOKE-vendor-transolver"

echo
echo "end: $(date -Is)"
if [[ $FAIL -ne 0 ]]; then
    echo "!! 冒烟失败：见上方日志"
    exit 1
fi
if [[ $HAS_GNOT -eq 1 ]]; then
    echo "冒烟通过：FNO / GNOT / Transolver 可从 vendor 目录跑通"
else
    echo "冒烟通过：FNO / Transolver 可从 vendor 目录跑通；GNOT 未随包，已跳过"
fi
exit 0
