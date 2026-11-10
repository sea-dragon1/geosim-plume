"""逐场 L2RE 损失：等权与速度加权共用同一公式，只改权重。

与上游实现（未随发布包）的差异（详见 ``protocol/loss.md``）：

- ``l2re_loss``          : 对 ``dim=(1,2,3,4)`` **整体**求范数 —— 六场联合成一个标量
- ``weighted_l2re_loss`` : **逐场**求范数比后分组加权

两者基础形式不同，直接对比无法把差异归因于权重。本项目统一为逐场形式，
权重是唯一的受控变量。

本模块不 import 上游任何模块，可独立测试。
"""
from __future__ import annotations

import torch

# 通道顺序固定，与 protocol/manifest.yaml 一致
FIELD_ORDER = ("ro", "pr", "tk", "nu", "vx", "vy")

# 等权：变换后各通道系数一致
EQUAL_WEIGHTS = (1.0 / 6.0,) * 6

# 速度加权：历史 0.2 / 0.8 **分组均值**语义的逐场展开
#     0.2 * mean(ro, pr, tk, nu) + 0.8 * mean(vx, vy)
#  => 标量组每场 0.2/4 = 0.05，速度组每场 0.8/2 = 0.4
# 不能直接用 (0.2,0.2,0.2,0.2,0.8,0.8)：那会改变标量与速度的相对比例。
VELOCITY_WEIGHTED_WEIGHTS = (0.05, 0.05, 0.05, 0.05, 0.4, 0.4)

WEIGHT_PRESETS = {
    "equal": EQUAL_WEIGHTS,
    "velocity_weighted": VELOCITY_WEIGHTED_WEIGHTS,
}

DEFAULT_EPS = 1e-8


def get_weights(name):
    """按名称取权重预设。"""
    if name not in WEIGHT_PRESETS:
        raise KeyError(
            "unknown weight preset {!r}; available: {}".format(
                name, sorted(WEIGHT_PRESETS)
            )
        )
    return WEIGHT_PRESETS[name]


def per_field_l2re_loss(pred, target, weights, eps=DEFAULT_EPS):
    """逐场相对 L2 损失的加权和。

    ``L = Σ_c λ_c · mean_batch[ ‖pred_c − target_c‖₂ / (‖target_c‖₂ + ε) ]``

    Args:
        pred, target: ``[B, X, Y, T, C]`` 张量；``C`` 必须等于 6，
            通道顺序为 :data:`FIELD_ORDER`。
        weights: 长度 ``C`` 的权重序列，建议取自 :data:`WEIGHT_PRESETS`。
        eps: 分母稳定项。

    Returns:
        标量张量。
    """
    if pred.shape != target.shape:
        raise ValueError(
            "pred/target shape mismatch: {} vs {}".format(pred.shape, target.shape)
        )
    if pred.dim() != 5:
        raise ValueError("expected 5-D [B,X,Y,T,C], got {}-D".format(pred.dim()))

    n_channel = pred.shape[-1]
    if len(weights) != n_channel:
        raise ValueError(
            "weights length {} != channel count {}".format(len(weights), n_channel)
        )

    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    # 范数在 (X, Y, T) 上计算，保留 batch 维 —— 与上游 weighted_l2re_loss 同族
    numerator = torch.norm(pred - target, p=2, dim=(1, 2, 3))       # [B, C]
    denominator = torch.norm(target, p=2, dim=(1, 2, 3)) + eps      # [B, C]
    per_field = (numerator / denominator).mean(dim=0)               # [C]
    return (w * per_field).sum()


def weight_summary(weights):
    """返回权重向量的自检信息，供测试与配置记录使用。"""
    total = float(sum(weights))
    groups = {
        "scalar": [FIELD_ORDER[i] for i in range(4)],
        "velocity": [FIELD_ORDER[i] for i in range(4, 6)],
    }
    return {
        "weights": list(weights),
        "sum": total,
        "scalar_group_mass": float(sum(weights[:4])),
        "velocity_group_mass": float(sum(weights[4:])),
        "groups": groups,
    }
