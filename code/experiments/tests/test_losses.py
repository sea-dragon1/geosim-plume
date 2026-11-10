"""``common/losses.py`` 的单元测试。

重点核对权重语义：速度加权必须是历史 0.2/0.8 **分组均值**的逐场展开，
而不是把 0.2/0.8 直接当逐场系数。

运行： ``python -B tests/test_losses.py``（CPU，秒级，需要 torch）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import torch  # noqa: E402

import losses as L  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_tensors(seed=0, shape=(2, 4, 5, 3, 6)):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(*shape, generator=g)
    pred = target + 0.1 * torch.randn(*shape, generator=g)
    return pred, target


def test_weight_presets_sum_to_one():
    for name, w in L.WEIGHT_PRESETS.items():
        check("预设 {!r} 权重和为 1".format(name), abs(sum(w) - 1.0) < 1e-12,
              "sum={}".format(sum(w)))


def test_velocity_weight_is_group_mean_expansion():
    """速度加权 = 0.2·mean(4标量) + 0.8·mean(2速度) 的逐场展开。"""
    w = L.VELOCITY_WEIGHTED_WEIGHTS
    check("标量组每场 = 0.2/4", all(abs(x - 0.05) < 1e-12 for x in w[:4]), str(w[:4]))
    check("速度组每场 = 0.8/2", all(abs(x - 0.4) < 1e-12 for x in w[4:]), str(w[4:]))
    check("标量组总质量 = 0.2", abs(sum(w[:4]) - 0.2) < 1e-12, str(sum(w[:4])))
    check("速度组总质量 = 0.8", abs(sum(w[4:]) - 0.8) < 1e-12, str(sum(w[4:])))

    naive = (0.2, 0.2, 0.2, 0.2, 0.8, 0.8)
    check("与「直接当逐场系数」的朴素写法不同",
          not all(abs(a - b) < 1e-12 for a, b in zip(w, naive)))


def test_group_mean_equivalence():
    """逐场加权形式必须等价于历史的分组均值形式。"""
    pred, target = make_tensors(seed=1)
    per_field = []
    for c in range(6):
        num = torch.norm(pred[..., c] - target[..., c], p=2, dim=(1, 2, 3))
        den = torch.norm(target[..., c], p=2, dim=(1, 2, 3)) + 1e-8
        per_field.append((num / den).mean())
    per_field = torch.stack(per_field)

    group_mean = 0.2 * per_field[:4].mean() + 0.8 * per_field[4:].mean()
    ours = L.per_field_l2re_loss(pred, target, L.VELOCITY_WEIGHTED_WEIGHTS)
    check("加权损失 == 0.2·mean(标量) + 0.8·mean(速度)",
          torch.allclose(ours, group_mean, atol=1e-6),
          "ours={} group_mean={}".format(ours.item(), group_mean.item()))


def test_equal_weights_is_uniform_mean():
    pred, target = make_tensors(seed=2)
    per_field = []
    for c in range(6):
        num = torch.norm(pred[..., c] - target[..., c], p=2, dim=(1, 2, 3))
        den = torch.norm(target[..., c], p=2, dim=(1, 2, 3)) + 1e-8
        per_field.append((num / den).mean())
    manual = torch.stack(per_field).mean()
    ours = L.per_field_l2re_loss(pred, target, L.EQUAL_WEIGHTS)
    check("等权损失 == 六场简单平均", torch.allclose(ours, manual, atol=1e-6))


def test_perfect_prediction_zero_loss():
    _, target = make_tensors(seed=3)
    for name, w in L.WEIGHT_PRESETS.items():
        val = L.per_field_l2re_loss(target.clone(), target, w)
        check("完美预测损失为 0（{}）".format(name), val.item() < 1e-6, str(val.item()))


def test_velocity_weighted_penalizes_velocity_more():
    """同样的相对扰动下，速度加权的损失应高于等权（因为速度权重更大）。"""
    _, target = make_tensors(seed=4)
    pred = target.clone()
    # 只扰动速度场
    pred[..., 4:] = target[..., 4:] * 1.5
    eq = L.per_field_l2re_loss(pred, target, L.EQUAL_WEIGHTS).item()
    vw = L.per_field_l2re_loss(pred, target, L.VELOCITY_WEIGHTED_WEIGHTS).item()
    check("速度扰动时 加权 > 等权", vw > eq, "eq={} vw={}".format(eq, vw))

    # 反向：只扰动标量场
    pred2 = target.clone()
    pred2[..., :4] = target[..., :4] * 1.5
    eq2 = L.per_field_l2re_loss(pred2, target, L.EQUAL_WEIGHTS).item()
    vw2 = L.per_field_l2re_loss(pred2, target, L.VELOCITY_WEIGHTED_WEIGHTS).item()
    check("标量扰动时 加权 < 等权", vw2 < eq2, "eq={} vw={}".format(eq2, vw2))


def test_weight_length_mismatch_raises():
    pred, target = make_tensors(shape=(1, 2, 2, 2, 6))
    try:
        L.per_field_l2re_loss(pred, target, (1.0, 1.0))
        check("权重长度不匹配应报错", False)
    except ValueError:
        check("权重长度不匹配应报错", True)


def test_shape_mismatch_raises():
    pred, _ = make_tensors(shape=(1, 2, 2, 2, 6))
    _, target = make_tensors(shape=(1, 2, 2, 3, 6))
    try:
        L.per_field_l2re_loss(pred, target, L.EQUAL_WEIGHTS)
        check("pred/target 形状不匹配应报错", False)
    except ValueError:
        check("pred/target 形状不匹配应报错", True)


def test_get_weights_unknown_name_raises():
    try:
        L.get_weights("nonexistent")
        check("未知权重名应报错", False)
    except KeyError:
        check("未知权重名应报错", True)


def main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if FAILURES:
        print("失败 {} 项: {}".format(len(FAILURES), FAILURES))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
