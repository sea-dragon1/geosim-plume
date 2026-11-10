#!/bin/bash
#
# 提交 E01 的全部训练作业。
#
#   bash E01_main/run_all.sh                  # 只提交 plume1d
#   bash E01_main/run_all.sh --with-plume2    # 连 plume2 一起提交
#   bash E01_main/run_all.sh --with-plume2 --only=plume2   # 只提交 plume2
#
# ⚠ 默认**不**提交 plume2：不是因为它不可用（2026-09-21 起划分已冻结、
#   数据已就绪），而是因为提交数有上限（实测 6→9，会变）—— 两子集一起提会超限被拒。
#   排在 1D 之后提交，是配额问题不是有效性问题。
#
# ⚠ 本脚本只提交作业，不等待、不聚合结果。提交后：
#   squeue -u $USER          查看状态
#   yhcancel <jobid>         取消
#
# 训练入口从配置头部注释读取（配置随发布包冻结）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS="$(dirname "$HERE")"
RUNNER="$EXPERIMENTS/common/run_train.sh"

# 发布包：trainer 在 code/vendor，不再依赖 .git。
# 从本脚本位置推算 geosim-plume/（E01_main → experiments → code → geosim-plume）。
export GEOSIM_ROOT="${GEOSIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REPO="$GEOSIM_ROOT/code/vendor"
if [[ ! -d "$REPO" ]]; then
    echo "找不到 trainer 目录：$REPO" >&2
    exit 1
fi

INCLUDE_PLUME2=0
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --with-plume2) INCLUDE_PLUME2=1 ;;
        # --only=<片段>：只提交文件名含该片段的配置。
        # 单用户提交数有上限（实测 6→9，会变），一次性提交全部会被调度器拒绝，必须分批；
        # 分批也符合「先跑完一批、看结果再排下一批」的要求。
        --only=*)      ONLY="${arg#--only=}" ;;
    esac
done

if [[ ! -x "$RUNNER" && ! -f "$RUNNER" ]]; then
    echo "找不到提交脚本: $RUNNER" >&2
    exit 1
fi

# 在计划目录内提交，SLURM 日志（geosim-train-<jobid>.out/.err）就落在这里，
# 不会散落到仓库根或模块顶层（EXPERIMENT_FRAMEWORK.md §7）。
cd "$HERE"

submitted=0
skipped=0
rejected=0
for config in "$HERE"/configs/*.yaml; do
    name="$(basename "$config")"

    # SMOKE-*（冒烟）与 RESUME-*（分段续训）永不参与批量提交。
    # 前者只验证代码能跑通；后者必须等作业被墙钟杀掉后手动单独提交（见 K3）。
    if [[ "$name" == SMOKE-* || "$name" == RESUME-* ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "$name" == *plume2* && $INCLUDE_PLUME2 -eq 0 ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    if [[ -n "$ONLY" && "$name" != *"$ONLY"* ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    trainer="$(grep -oE '训练入口: \S+' "$config" | head -1 | awk '{print $2}')"
    if [[ -z "$trainer" ]]; then
        echo "跳过（配置头部缺少训练入口注释）: $name" >&2
        skipped=$((skipped + 1))
        continue
    fi

    # 配置头部写的是「仓库相对路径」；run_train.sh 不做路径解析，
    # 这里统一转成绝对路径再传（参数顺序与 python 调用一致：trainer 在前）。
    [[ "$trainer" = /* ]] || trainer="$REPO/$trainer"

    # 提交前自检：trainer 必须真实存在。否则作业在计算节点上启动即退出，
    # 只烧几秒钟 GPU，但会以 COMPLETED 混进结果 —— 拦在提交这一步代价最低。
    if [[ ! -f "$trainer" ]]; then
        echo "跳过（训练入口不存在）: $name -> $trainer" >&2
        skipped=$((skipped + 1))
        continue
    fi

    echo "提交 $name  ->  $trainer"
    # 单个提交被拒（提交数配额）不应中断整批 —— 记下来，等前一批跑完再补。
    if yhbatch "$RUNNER" "$trainer" "$config"; then
        submitted=$((submitted + 1))
    else
        echo "!! 提交被拒（提交数配额或调度策略）: $name" >&2
        rejected=$((rejected + 1))
    fi
done

echo
echo "已提交 $submitted 个作业，跳过 $skipped 个。"
if [[ $rejected -gt 0 ]]; then
    echo "!! 被拒 $rejected 个 —— 等前一批跑完再提交剩下的。"
fi
if [[ -n "$ONLY" ]]; then
    echo "（本次只提交文件名含 '$ONLY' 的配置）"
fi
if [[ $INCLUDE_PLUME2 -eq 0 ]]; then
    echo "（plume2 未提交：划分待冻结。复核后加 --with-plume2）"
fi
echo "查看状态: squeue -u \$USER"
