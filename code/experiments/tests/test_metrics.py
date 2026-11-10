"""``common/eval_metrics.py`` 的单元测试。

用可解析验证的合成数据核对 Eq.(metric) 的每一处约定：
聚合次序、ε 生效条件、面积权重、persistence 锚点、bootstrap 可复现性。

运行： ``python -B tests/test_metrics.py``（CPU，秒级）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import numpy as np  # noqa: E402

import eval_metrics as M  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_truth(seed=0, shape=(3, 4, 5, 6, 6)):
    return np.random.RandomState(seed).randn(*shape)


def test_perfect_prediction_is_zero():
    truth = make_truth()
    eps = np.full(6, 1e-30)
    e = M.relative_l2_per_frame(truth.copy(), truth, eps)
    check("perfect prediction -> e == 0", np.allclose(e, 0.0),
          "max={}".format(np.abs(e).max()))


def test_known_scaled_ratio():
    """pred = (1-a) * truth  =>  e = a（逐点同比例缩放时范数比等于 a）。"""
    truth = make_truth(seed=1)
    a = 0.25
    pred = (1.0 - a) * truth
    eps = np.full(6, 1e-30)
    e = M.relative_l2_per_frame(pred, truth, eps)
    check("pred=(1-a)*truth -> e == a", np.allclose(e, a, atol=1e-12),
          "got mean={} expected={}".format(e.mean(), a))


def test_aggregation_order_is_frame_then_trajectory():
    """E[c] 必须是 mean_n(mean_q(e))，而不是对整条轨迹取一次范数。

    构造帧间误差比例差异大的例子：高能量帧误差大、低能量帧误差小。
    注意若 pred 是 truth 的**逐点同比例**缩放（如 0.5*truth），两种聚合方式
    在数学上必然相等，无法区分 —— 必须让不同帧的相对误差不同才有区分度。
    """
    truth = make_truth(seed=2, shape=(2, 3, 4, 5, 6))
    truth[:, 0] *= 50.0              # 第 0 帧能量远大于其余帧
    pred = truth.copy()
    pred[:, 0] = truth[:, 0] * 0.1   # 高能量帧：相对误差 0.9
    pred[:, 1] = truth[:, 1] * 0.9   # 低能量帧：相对误差 0.1
    pred[:, 2] = truth[:, 2] * 0.9

    eps = np.full(6, 1e-30)
    e = M.relative_l2_per_frame(pred, truth, eps)

    manual = np.zeros(6)
    for c in range(6):
        manual[c] = np.mean([e[n, :, c].mean() for n in range(e.shape[0])])
    check("E = mean_n(mean_q(e))", np.allclose(M.aggregate_E(e), manual))

    per_frame = M.aggregate_E(e)
    check("逐帧聚合 = (0.9+0.1+0.1)/3", np.allclose(per_frame, 1.1 / 3, atol=1e-9),
          "got {}".format(per_frame[:2]))

    # 反例：整条轨迹合并成一个比值，会被高能量帧主导 -> 接近 0.9
    diff = (pred - truth) ** 2
    whole = (np.sqrt(diff.sum(axis=(1, 2, 3, 4)))
             / np.sqrt((truth ** 2).sum(axis=(1, 2, 3, 4)))).mean()
    check("聚合次序确实有影响（反例不相等）",
          not np.allclose(per_frame.mean(), whole, atol=1e-3),
          "per_frame={:.4f} whole={:.4f}".format(per_frame.mean(), whole))


def test_epsilon_floor_applies_after_sqrt():
    """truth 全零时，分母应落到 ε 而不是 0。

    ⚠ 2026-09-25：期望值由 **6.0 改为 2.0**。这不是为了让代码通过而改测试 ——
    是**测试原先编码了未归一化的约定**，而那个约定与论文 Eq.9 不符：

      旧：``SSE = Σ diff²``（未归一化，权重和 = N）  → sqrt(9) = 3 → 3/0.5 = **6**
      新：``SSE = Σ a_i diff²``（a_i = 1/N，与 Eq.9 一致）→ sqrt(9/9) = 1 → 1/0.5 = **2**

    Eq.9 的分母是 ``sqrt(Σ a_i U²)``；均匀网格下 ``a_i = 1/N``，所以范数就是
    **均方根**。ε_c 也拟合在均方根尺度上（``fit_epsilon`` 用
    ``1e-6 × sqrt(mean(truth²))``）—— 两边同尺度，这个比较才有意义。
    旧约定拿未归一化的 ``ΣU²`` 去比 ε，判据会严 √N 倍（N=16384 时 128 倍）。

    本测试守护的**性质没变**：ε 在开方**之后**比较（不是对 ``U²`` 直接比）。
    当前数值上无影响 —— ε 从未触发过（22 个 ``*_eq9.json`` 全为 0）。
    """
    truth = np.zeros((1, 2, 3, 3, 6))
    pred = np.ones((1, 2, 3, 3, 6))
    eps = np.full(6, 0.5)
    e = M.relative_l2_per_frame(pred, truth, eps)
    # 每个空间点的 diff² = 1，共 9 点 → Σa_i diff² = 9/9 = 1；sqrt = 1
    # 分母 = max(sqrt(0), 0.5) = 0.5  =>  e = 1 / 0.5 = 2
    check("ε 在开方后生效", np.allclose(e, 2.0), "got {}".format(e[0, 0, 0]))


def test_epsilon_trigger_count():
    truth = np.zeros((1, 2, 3, 3, 6))
    pred = np.ones((1, 2, 3, 3, 6))
    eps = np.full(6, 0.5)
    sse, energy = M.weighted_squares(pred, truth)
    counts = M.epsilon_triggered(sse, energy, eps)
    check("ε 触发计数 = 全部帧", all(v == 2 for v in counts.values()),
          str(counts))


def test_uniform_area_weights_match_equal_weights():
    """常数面积权重等价于均匀网格（权重在分子分母中约去）。"""
    truth = make_truth(seed=3, shape=(2, 3, 4, 5, 6))
    pred = truth * 0.7
    eps = np.full(6, 1e-30)
    e_none = M.relative_l2_per_frame(pred, truth, eps, area_weights=None)
    w = np.full((4, 5), 1.0 / 20.0)
    e_w = M.relative_l2_per_frame(pred, truth, eps, area_weights=w)
    check("常数面积权重 == 均匀网格", np.allclose(e_none, e_w))


def test_area_weights_change_result_when_nonuniform():
    truth = make_truth(seed=4, shape=(1, 2, 4, 5, 6))
    pred = truth * 0.5
    eps = np.full(6, 1e-30)
    w = np.zeros((4, 5))
    w[0, 0] = 1.0  # 只保留一个格点
    e = M.relative_l2_per_frame(pred, truth, eps, area_weights=w)
    # 单点权重下仍然是同比例缩放，故比值不变 —— 但若换成 pred 只改这点的扰动就会变
    pred2 = truth * 0.5
    pred2[:, :, 0, 0, :] = truth[:, :, 0, 0, :]  # 把权重所在点改成精确
    e2 = M.relative_l2_per_frame(pred2, truth, eps, area_weights=w)
    check("面积权重确实参与计算", not np.allclose(e, e2))


def test_persistence_anchor_uses_only_anchor_frame():
    """persistence 只能复制锚点帧，绝不能读「上一帧真值」。"""
    truth = make_truth(seed=5, shape=(2, 5, 3, 3, 6))
    pred = M.persistence_prediction(truth, anchor_frame=0)
    check("persistence 每帧都等于锚点帧",
          np.allclose(pred, truth[:, 0:1].repeat(5, axis=1)))
    check("persistence 与 truth 非平凡（锚点不是最后一帧）",
          not np.allclose(pred[:, 1:], truth[:, 1:]))


def test_rmse_matches_manual():
    truth = make_truth(seed=6, shape=(2, 3, 4, 5, 6))
    pred = truth + 0.1
    eps = np.full(6, 1e-30)
    r = M.rmse_per_frame(pred, truth)
    manual = np.sqrt(((pred - truth) ** 2).sum(axis=(2, 3)) / 20.0)
    check("RMSE = sqrt(SSE/|Ω|)", np.allclose(r, manual))


def test_skill_undefined_when_baseline_zero():
    sse_model = np.array([[0.5, 1.0]])
    sse_base = np.array([[1.0, 0.0]])
    skill = M.skill_from_sse(sse_model, sse_base)
    check("skill 正常通道 = 1 - ratio", np.isclose(skill[0, 0], 0.5))
    check("baseline 为零 -> NaN（未定义）", np.isnan(skill[0, 1]))


def test_bootstrap_is_deterministic_and_shared():
    rng = M.make_bootstrap_resamples(12, 200, seed=42)
    rng2 = M.make_bootstrap_resamples(12, 200, seed=42)
    check("bootstrap 索引可复现", np.array_equal(rng, rng2))
    check("bootstrap 索引形状正确", rng.shape == (200, 12))
    check("bootstrap 索引覆盖全部轨迹",
          set(np.unique(rng).tolist()) == set(range(12)))


def test_bootstrap_ci_nan_on_nonfinite():
    vals = np.array([1.0, np.nan, 3.0])
    rng = M.make_bootstrap_resamples(3, 50, seed=0)
    m, lo, hi = M.bootstrap_ci(vals, rng)
    check("含 NaN 的轨迹值 -> CI 全 NaN", all(np.isnan(v) for v in (m, lo, hi)))


def test_shape_mismatch_raises():
    truth = make_truth(shape=(1, 2, 3, 3, 6))
    pred = make_truth(shape=(1, 2, 3, 3, 5))
    try:
        M.relative_l2_per_frame(pred, truth, eps=np.full(5, 1e-9))
        check("形状不匹配应报错", False)
    except ValueError:
        check("形状不匹配应报错", True)


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
