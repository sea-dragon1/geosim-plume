#!/bin/bash
#
# 提交 E02 的全部消融训练作业。
#
#   bash E02_ablation/run_all.sh                  # 只提交 plume1d
#   bash E02_ablation/run_all.sh --with-plume2    # 连 plume2 一起提交
#
# ⚠ 预算（每个子集）：
#     - A 组（MPNN×6）           3 个种子组 = 18 个单场模型
#     - B/C × 等权/速度加权 × 3 种子 = 12 个联合模型
#   合计 12 个联合 + 18 个单场。以历史 48h/次估算，一个子集约 1440 GPU·h。
#   提交前请确认队列配额。
#
# ⚠ 本脚本提交的全部作业都走 common/vendor/train_tubor.py（隔离副本），
#   因为只有它能挂上统一的逐场损失。上游的 l2re_loss 与 weighted_l2re_loss
#   基础形式不同，无法用于权重归因。
#
# ⚠ 建议先只提交 seed0 的 B-equal 与 C-equal 做冒烟验证（验证全流程能跑通），
#   但**不要**用冒烟结果筛选协议或种子 —— 它只用于验证代码。
#
# 训练入口从配置头部注释读取（配置随发布包冻结）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS="$(dirname "$HERE")"
RUNNER="$EXPERIMENTS/common/run_train.sh"

# 发布包：trainer 在 code/vendor，不再依赖 .git。
# 从本脚本位置推算 geosim-plume/（E02_ablation → experiments → code → geosim-plume）。
export GEOSIM_ROOT="${GEOSIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REPO="$GEOSIM_ROOT"
if [[ ! -d "$REPO" ]]; then
    echo "找不到 trainer 目录：$REPO" >&2
    exit 1
fi

INCLUDE_PLUME2=0
SMOKE_ONLY=0
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --with-plume2) INCLUDE_PLUME2=1 ;;
        --smoke)       SMOKE_ONLY=1 ;;
        # --only=<片段>：只提交文件名含该片段的配置。
        # 单用户提交数有上限（实测 6→9，会变），一次性提交全部会被调度器拒绝，
        # 必须分批；分批也符合「先跑完一批、看结果再排下一批」的要求。
        --only=*)      ONLY="${arg#--only=}" ;;
    esac
done

if [[ ! -f "$RUNNER" ]]; then
    echo "找不到提交脚本: $RUNNER" >&2
    exit 1
fi

cd "$HERE"

submitted=0
skipped=0
rejected=0
for config in "$HERE"/configs/*.yaml; do
    name="$(basename "$config")"

    # SMOKE-*（冒烟）与 RESUME-*（分段续训）永不参与批量提交。
    # 前者只验证代码能跑通；后者**必须在作业被墙钟杀掉之后**手动单独提交 ——
    # 提前提会从旧 checkpoint 重跑，且两个作业写同一个 run 目录（见 K3）。
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
if [[ $SMOKE_ONLY -eq 1 ]]; then
    echo "（冒烟模式：只提交了 B/C 的 equal 臂 seed0）"
fi
if [[ $INCLUDE_PLUME2 -eq 0 ]]; then
    echo "（plume2 未提交：提交数配额所限，排在 1D 之后；划分与数据均已就绪）"
fi
echo "查看状态: squeue -u \$USER"
