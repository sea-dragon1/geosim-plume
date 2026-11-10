"""评估指标：Eq.(metric) 的逐帧逐场相对 L2 误差，以及轨迹级 bootstrap。

对应论文 ``sections/05_experiments.tex`` 的式 ``eq:metric``，约定见
``protocol/metrics.md``。与训练损失是两套东西（见 ``protocol/loss.md``）。

本模块只用 numpy，可独立测试，且不 import 任何上游模块。
"""
from __future__ import annotations

import numpy as np

FIELD_ORDER = ("ro", "pr", "tk", "nu", "vx", "vy")

# persistence 基线在两种协议下的锚点帧
BASELINE_ANCHOR = {
    "window": "last_input_frame",   # 10->10 窗口预测：固定最后一个输入帧
    "rollout": "first_frame",       # 自治 rollout：固定首帧
}


def _as_float(a, name):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("{} contains non-finite values".format(name))
    return a


def _check_shapes(pred, truth, eps):
    if pred.shape != truth.shape:
        raise ValueError(
            "pred/truth shape mismatch: {} vs {}".format(pred.shape, truth.shape)
        )
    if pred.ndim != 5:
        raise ValueError("expected 5-D [n_traj, n_frame, X, Y, C], got {}-D".format(pred.ndim))
    eps = np.asarray(eps, dtype=np.float64)
    if eps.shape != (pred.shape[-1],):
        raise ValueError(
            "eps must have shape ({},), got {}".format(pred.shape[-1], eps.shape)
        )
    return eps


def weighted_squares(pred, truth, area_weights=None):
    """返回逐帧逐场的 (SSE, ENERGY)，形状 ``[n_traj, n_frame, C]``。

    ⚠ 返回的是 **Eq.9 的加权量** ``SSE = Σ a_i diff²``、``ENERGY = Σ a_i truth²``，
    权重已归一化（均匀网格 ``a_i = 1/N`` ⇒ 这里除以 N）。

    **为什么必须归一化**：``1/N`` 在**比值**里确实会约掉，但在 **ε 的比较**里不会。
    Eq.9 的分母是 ``sqrt(Σ a_i U²)``（= RMS），而 ε_c 也拟合在 RMS 尺度上
    （``fit_epsilon`` 用 ``sqrt(mean(truth²))`` 的 1e-6 倍）。若这里返回未归一化的
    ``Σ U²``，ε 的判据会**严 √N 倍** —— N = 16384 时是 128 倍，等于几乎不可能触发。

    原实现返回未归一化和，理由写的是「a_i 在分子分母中约去」—— 对**比值**成立，
    对 **ε** 不成立，是漏看了一半。
    """
    diff2 = (pred - truth) ** 2
    truth2 = truth ** 2
    if area_weights is None:
        n = float(pred.shape[2] * pred.shape[3])
        return diff2.sum(axis=(2, 3)) / n, truth2.sum(axis=(2, 3)) / n

    w = np.asarray(area_weights, dtype=np.float64)
    if w.shape != pred.shape[2:4]:
        raise ValueError(
            "area_weights shape {} != spatial shape {}".format(w.shape, pred.shape[2:4])
        )
    wsum = float(w.sum())
    if wsum <= 0:
        raise ValueError("area_weights 之和为 {}，无法归一化".format(wsum))
    wb = w[None, None, :, :, None]
    return (wb * diff2).sum(axis=(2, 3)) / wsum, (wb * truth2).sum(axis=(2, 3)) / wsum


def relative_l2_per_frame(pred, truth, eps, area_weights=None):
    """逐帧逐场相对 L2 误差 ``e[n,q,c]``，形状 ``[n_traj, n_frame, C]``。

    ``e = sqrt(Σ a_i diff²) / max(sqrt(Σ a_i truth²), ε_c)``

    ε 在开方**之后**比较，与论文式一致。
    """
    pred = _as_float(pred, "pred")
    truth = _as_float(truth, "truth")
    eps = _check_shapes(pred, truth, eps)
    sse, energy = weighted_squares(pred, truth, area_weights)
    den = np.maximum(np.sqrt(energy), eps[None, None, :])
    return np.sqrt(sse) / den


def epsilon_triggered(sse, energy, eps):
    """统计每个通道上 ``sqrt(energy) <= ε_c`` 的帧数（即 ε 生效的次数）。

    Returns:
        dict: 通道名 -> 触发帧数
    """
    eps = np.asarray(eps, dtype=np.float64)
    triggered = np.sqrt(energy) <= eps[None, None, :]
    counts = triggered.sum(axis=(0, 1))
    return {FIELD_ORDER[i]: int(counts[i]) for i in range(len(FIELD_ORDER))}


def aggregate_E(e):
    """``E[c] = mean_n(mean_q(e[n,q,c]))`` —— 先按帧平均，再按轨迹平均。

    聚合次序是硬性的；与"对整条轨迹取一个范数"不同。
    """
    return e.mean(axis=1).mean(axis=0)


def rmse_per_frame(pred, truth, area_weights=None):
    """逐帧逐场 RMSE，形状 ``[n_traj, n_frame, C]``。

    ``RMSE = sqrt(Σ a_i diff²)``。⚠ ``weighted_squares`` 现在返回的**已经是**
    按权重归一化过的量（见其 docstring），所以这里**不能再除一次** |Ω| ——
    原先写的是 ``sqrt(sse / n_points)``，那是配旧的非归一化 sse 用的；
    两个都改才对，只改一个会把 RMSE 打歪 √N ≈ 128 倍。
    """
    pred = _as_float(pred, "pred")
    truth = _as_float(truth, "truth")
    sse, _ = weighted_squares(pred, truth, area_weights)
    return np.sqrt(sse)


def skill_from_sse(sse_model, sse_baseline):
    """``skill = 1 - SSE_model / SSE_baseline``；分母为零记 NaN（未定义）。

    正值表示优于 persistence，负值表示更差。分母为零时记未定义，
    不得用极小常数生成任意大的数值。
    """
    sse_model = np.asarray(sse_model, dtype=np.float64)
    sse_baseline = np.asarray(sse_baseline, dtype=np.float64)
    ratio = np.full(sse_model.shape, np.nan, dtype=np.float64)
    np.divide(sse_model, sse_baseline, out=ratio, where=sse_baseline > 0)
    return 1.0 - ratio


def make_bootstrap_resamples(n_traj, repeats, seed):
    """生成固定的重采样索引 ``[repeats, n_traj]``。

    所有模型必须共用同一组索引，才能做配对比较。
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_traj, size=(repeats, n_traj))


def bootstrap_ci(values, resamples):
    """轨迹级 bootstrap 的 95% percentile CI。

    Args:
        values: ``[n_traj]`` 逐轨迹指标
        resamples: ``[repeats, n_traj]`` 索引，来自 :func:`make_bootstrap_resamples`

    Returns:
        ``(mean, lo, hi)``；``values`` 含非有限值时返回全 NaN。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be 1-D, got shape {}".format(values.shape))
    if not np.isfinite(values).all():
        return (float("nan"), float("nan"), float("nan"))
    boot = values[resamples].mean(axis=1)
    return (float(values.mean()),
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)))


def persistence_prediction(truth, anchor_frame):
    """构造 persistence 预测：把 ``anchor_frame`` 广播到全部目标帧。

    **不要**每步读前一个真值帧 —— 那是 teacher forcing，不是 persistence。
    """
    truth = np.asarray(truth)
    if truth.ndim != 5:
        raise ValueError("truth must be 5-D [n_traj, n_frame, X, Y, C]")
    if not (0 <= anchor_frame < truth.shape[1]):
        raise ValueError("anchor_frame {} out of range".format(anchor_frame))
    anchor = truth[:, anchor_frame:anchor_frame + 1]
    return np.broadcast_to(anchor, truth.shape).copy()
