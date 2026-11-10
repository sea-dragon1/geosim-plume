#!/usr/bin/env python3
# coding=utf-8
"""生成论文表格（当前版式：单表六列、百分比、有效精度）。

**数据来源**（全部是冻结协议口径的产物，不手工拼数字，也不手改表格）::

    E01_main/results/persistence_<subset>_eq9.csv   ← score_persistence.py
    E01_main/results/E01_main_eq9_external.csv      ← score_predictions.py
    E02_ablation/results/E02_ablation_eq9.csv       ← eval_checkpoint.py
    E02_ablation/runs/*/eval/*_pertraj.npz          ← eval_per_traj.py
    E04_diagnostics_cost/results/model_meta.json    ← collect_meta.py
    E04_diagnostics_cost/results/cost_*.json        ← cost.py

**产出**::

    tables/03_main_results.tex     tab:main1（plume1d，六列单表）
    tables/04_ablation.tex         tab:ablation
    tables/05_training.tex         tab:training
    tables/06_cost.tex             tab:cost
    tables/07_plume2_main.tex      tab:main2（plume2，附录）
    tables/08_ablation_stats.tex   tab:ablation_stats（B−C + 95% CI）

**报告纪律**（``protocol/metrics.md``）：六场逐场报告；种子均值 ± SD；
单种子不写 ± SD；配对比较先逐轨迹相减再重采样。

**版式规则**（2026-09-26 手改后固化，2026-10-31 写回脚本）：
- 数值 ×100，百分比口径；
- mean 显示 3 位有效数字（保留尾零，如 2.50、0.0110）；
- SD 与 mean 用相同小数位；若该精度下 SD 的相对舍入误差 >20%，SD 多保留一位
  （mean 不动）——这正是 ρ 等小 SD 列出现 ``0.0223±0.00007`` 的来源；
- 无 SD 的行（Persistence、单种子）用 3 位有效数字并去掉尾零；
- 每场加粗「学习模型中最好的」（Persistence 不参与）。

用法::

    python -B make_tables.py --dry-run          # 只打印，不落盘
    python -B make_tables.py                    # 写入 tables/
    python -B make_tables.py --out-dir /tmp/x   # 写到别处（比对用）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import numpy as np      # noqa: E402

import eval_metrics as em          # noqa: E402
import paths                       # noqa: E402

# 发布包：表格写到 results/tables/（不再是仓库的 tables/）
ICLR = paths.RESULTS

FIELDS = list(em.FIELD_ORDER)


# ============================================================ 数值格式
def _finite(x):
    return x is not None and np.isfinite(float(x))


def _sig3_plain(v, strip=True):
    """3 位有效数字（用于无 SD 的单元格）。

    ``strip=True`` 去掉尾零（plume1d 表的约定，如 0.035）；
    ``strip=False`` 保留尾零（plume2 表的约定，如 94.0、1.10）。
    两张表在 2026-09-26 的手改中形成了不同约定，这里按现状分别保留。
    """
    v = float(v)
    if not np.isfinite(v):
        return r"\pending"
    if v == 0.0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    s = ("%." + str(max(2 - e, 0)) + "f") % v
    if strip and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _sig3_keep(v):
    """3 位有效数字，保留尾零（用于 CI 的 plain 分支）。"""
    v = float(v)
    if not np.isfinite(v):
        return r"\pending"
    if v == 0.0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    return ("%." + str(max(2 - e, 0)) + "f") % v


def _sci(v, nd=2):
    """``1.16\\times10^{-5}``（不含 $）。"""
    v = float(v)
    if not np.isfinite(v):
        return r"\pending"
    if v == 0.0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    return ("%." + str(nd) + "f{\\times}10^{%d}") % (v / 10.0 ** e, e)


def _mean_sd_plain(m, s):
    """mean 3 位有效数字；SD 同小数位，误差 >20% 时多留一位。

    Returns:
        (mean_str, sd_str)；``s`` 为 None/NaN 时 sd_str 为 None。
    """
    m = float(m)
    if s is None or not _finite(s) or float(s) == 0.0:
        return _sig3_plain(m), None
    s = float(s)
    e = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
    dec_mean = max(2 - e, 0)
    dec = dec_mean
    while dec <= dec_mean + 5:
        rs = round(s, dec)
        if rs != 0.0 and abs(rs - s) / s <= 0.2:
            break
        dec += 1
    return (("%." + str(dec_mean) + "f") % m,
            ("%." + str(dec) + "f") % s)


def _cell(m, s, bold=False):
    ms, ss = _mean_sd_plain(m, s)
    body = ms if ss is None else ms + "{\\pm}" + ss
    return ("$\\mathbf{" + body + "}$") if bold else ("$" + body + "$")


def _cell_nosd(m, strip=True):
    return _sig3_plain(m, strip=strip)


# ============================================================ 读 CSV
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def field_values(row):
    return np.array([float(row["E_" + c]) for c in FIELDS]) * 100.0   # → 百分比


def mean_sd(mat):
    """``[n_seed, 6]`` → (mean, sd)；n=1 时 sd 为 NaN（不伪造）。"""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape[0] == 1:
        return mat[0], np.full(mat.shape[1], np.nan)
    return mat.mean(axis=0), mat.std(axis=0, ddof=1)


def _model_of_exp_id(exp_id):
    for m in ("fno", "gnot", "transolver"):
        if "-{}-".format(m) in exp_id:
            return m
    return None


def _seed_of_exp_id(exp_id):
    tail = str(exp_id).rsplit("-seed", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


def _ext_rows(subset, key):
    ext = read_csv(os.path.join(EXPERIMENTS, "E01_main", "results",
                                "E01_main_eq9_external.csv"))
    return [r for r in ext if r["subset"] == subset and _model_of_exp_id(r["exp_id"]) == key]


def _ab_rows(subset, variant):
    ab = read_csv(os.path.join(EXPERIMENTS, "E02_ablation", "results",
                               "E02_ablation_eq9.csv"))
    return sorted([r for r in ab if r["subset"] == subset and r["variant"] == variant],
                  key=lambda r: int(r["seed"]))


def _persistence(subset):
    rs = read_csv(os.path.join(EXPERIMENTS, "E01_main", "results",
                               "persistence_{}_eq9.csv".format(subset)))
    return field_values(rs[0]) if rs else None


# ============================================================ 组装行
def main_rows(subset):
    """返回 [(显示名, mean[6], sd[6]|None, is_learned)]，顺序固定。"""
    rows = {}
    per = _persistence(subset)
    if per is not None:
        rows["Persistence"] = (per, None, False)

    for key, disp in (("fno", "FNO"), ("gnot", "GNOT")):
        rs = _ext_rows(subset, key)
        if rs:
            m, s = mean_sd(np.stack([field_values(r) for r in rs]))
            rows[disp] = (m, None if len(rs) == 1 else s, True)

    rs = _ab_rows(subset, "A")
    if rs:
        m, s = mean_sd(np.stack([field_values(r) for r in rs]))
        rows["Independent MPNN"] = (m, None if len(rs) == 1 else s, True)

    rs = _ext_rows(subset, "transolver")
    if rs:
        m, s = mean_sd(np.stack([field_values(r) for r in rs]))
        rows["Transolver"] = (m, None if len(rs) == 1 else s, True)

    rs = _ab_rows(subset, "C")
    if rs:
        m, s = mean_sd(np.stack([field_values(r) for r in rs]))
        rows["MFGNO"] = (m, None if len(rs) == 1 else s, True)

    order = ["Persistence", "FNO", "GNOT", "Independent MPNN", "Transolver", "MFGNO"]
    return [(k,) + rows[k] for k in order if k in rows]


def ablation_rows():
    out = []
    for variant, disp in (("A", "A: Independent"),
                          ("B", "B: Joint, no diff."),
                          ("C", "C: Full \\model")):
        rs = _ab_rows("plume1d", variant)
        if not rs:
            continue
        m, s = mean_sd(np.stack([field_values(r) for r in rs]))
        out.append((disp, m, s))
    return out


def _best_mask(rows):
    """每个场取学习模型中 mean 最小者；返回 {row_index: set(field_idx)}。"""
    n = len(FIELDS)
    best = {}
    for i in range(n):
        cand = [(row[1][i], j) for j, row in enumerate(rows) if row[3]]
        if not cand:
            continue
        _, j = min(cand)
        best.setdefault(j, set()).add(i)
    return best


def render_rows(rows, bold=True, strip_zeros=True):
    mask = _best_mask(rows) if bold else {}
    lines = []
    for j, (name, m, s, learned) in enumerate(rows):
        name = r"\model" if name == "MFGNO" else name
        cells = []
        for i in range(len(FIELDS)):
            is_best = i in mask.get(j, set())
            if s is None:
                val = _cell_nosd(m[i], strip=strip_zeros)
                cells.append("$\\mathbf{" + val + "}$" if is_best else val)
            else:
                cells.append(_cell(m[i], s[i], bold=is_best))
        lines.append(name + "\n" + "\n".join("& " + c for c in cells) + " \\\\")
    return "\n\n".join(lines)


def render_ablation(rows):
    means = np.stack([r[1] for r in rows])
    lines = []
    for j, (name, m, s) in enumerate(rows):
        cells = [_cell(m[i], s[i], bold=(means[:, i].argmin() == j)) for i in range(len(FIELDS))]
        lines.append(name + "\n" + "\n".join("& " + c for c in cells) + " \\\\")
    return "\n\n".join(lines)


# ============================================================ B−C 统计
def _load_pertraj(exp_id):
    # 开发仓库：runs/<exp_id>/eval/；发布包：results/per_run/<exp_id>_pertraj.npz
    pats = glob.glob(os.path.join(EXPERIMENTS, "*", "runs", exp_id, "eval", "*_pertraj.npz"))
    if not pats:
        pats = glob.glob(os.path.join(ICLR, "per_run",
                                      "{}_pertraj.npz".format(exp_id)))
    if not pats:
        return None
    return np.asarray(np.load(pats[0])["E_traj"], dtype=np.float64)


def bc_stats():
    """B−C 配对差：逐轨迹相减 → 按种子平均 → 种子 mean±SD + 轨迹 bootstrap。"""
    B, C = _ab_rows("plume1d", "B"), _ab_rows("plume1d", "C")
    if not B or not C:
        return None
    traj_diffs, per_seed = [], []
    for rb in B:
        rc = [r for r in C if r["seed"] == rb["seed"]]
        if not rc:
            return None
        rc = rc[0]
        eb, ec = _load_pertraj(rb["exp_id"]), _load_pertraj(rc["exp_id"])
        if eb is None or ec is None or eb.shape != ec.shape:
            return None
        d = (eb - ec) * 100.0
        traj_diffs.append(d)
        per_seed.append(d.mean(axis=0))
    mat = np.stack(per_seed)
    m, s = mean_sd(mat)
    dbar = np.stack(traj_diffs).mean(axis=0)
    res = em.make_bootstrap_resamples(dbar.shape[0], 10000, 20260922)
    lo = np.zeros(len(FIELDS))
    hi = np.zeros(len(FIELDS))
    for i in range(len(FIELDS)):
        _, lo[i], hi[i] = em.bootstrap_ci(dbar[:, i], res)
    return m, s, lo, hi


def _ebc_cell(m, s, plus=False):
    e = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
    mm, ss = m / 10.0 ** e, s / 10.0 ** e
    fmt = "(+%.2f{\\pm}%.2f)" if plus else "(%.2f{\\pm}%.2f)"
    return "$" + (fmt % (mm, ss)) + "{\\times}10^{%d}$" % e


def _ci_cell(lo, hi):
    def one(v):
        if v != 0 and abs(v) < 0.01:
            return _sci(v)
        return _sig3_keep(v)
    return "$[" + one(lo) + ",\\," + one(hi) + "]$"


# ============================================================ 模板
T03 = r"""% 由 common/make_tables.py 生成 —— 请勿手改。
% 数据: persistence_plume1d_eq9.csv + E01_main_eq9_external.csv
%       + E02_ablation_eq9.csv（A 组作 Independent MPNN，C 组作 MFGNO）
% 2026-09-25: 主表改为单表六列、数值 ×100（百分比口径），Plume-2D 主表
% （原 tab:main2）拆出至 tables/07_plume2_main.tex（附录 app:plume2）。
% 2026-09-26: 按有效精度展示（mean 到 SD 的小数位，不加尾零），列右对齐；
% make_tables.py 重生成时需同步此拆分、百分比口径与有效精度格式。
\begin{table}[t]
\centering
\footnotesize
\caption{
Short-window field-wise relative $L^2$ error (\%) on
\pone\ under the common $10\!\to\!10$ prediction protocol.
Learned-model results are mean $\pm$ SD over three prespecified
training seeds; lower is better. Persistence is deterministic.
Best learned result in each field is shown in bold. FNO, GNOT, and
Transolver are external baselines; the independent-field MPNN is the
existing graph baseline; MFGNO is the model proposed here. External
baselines use their MSE objectives; graph models use the per-field relative
$L^2$ objective (Sec.~\ref{sec:setup}).
}
\label{tab:main1}
\setlength{\tabcolsep}{4.5pt}
\fitwidth{\begin{tabular}{lrrrrrr}
\toprule
Model
& $\rho$
& $P$
& $T$
& $\log_{10}\eta$
& $v_x$
& $v_y$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular}}
\end{table}
"""

T04 = r"""% 由 common/make_tables.py 生成 —— 请勿手改。
% 数据: E02_ablation_eq9.csv（3 seeds）
% 2026-09-25: 主文消融表精简为 A/B/C 六列单表、百分比口径；E_B-E_C 与
% 95% 逐轨迹 CI 行拆出至 tables/08_ablation_stats.tex（附录 app:ablation）。
% make_tables.py 重生成时需同步此拆分与百分比口径。
\begin{table}[t]
\centering
\small
\caption{
Controlled ablation on \pone. A is the existing independent-field MPNN
baseline; B is the MFGNO setting that removes the explicit neighboring-state
difference input; C is the full \model. A versus B/C compares the overall
design, while B versus C isolates the difference input. All variants use the
same per-field relative-$L^2$ objective with equal weights.
Values are relative $L^2$ error (\%), mean $\pm$ SD over three seeds.
}
\label{tab:ablation}
\setlength{\tabcolsep}{4.0pt}
\fitwidth{\begin{tabular}{lcccccc}
\toprule
Variant
& $\rho$
& $P$
& $T$
& $\log_{10}\eta$
& $v_x$
& $v_y$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular}}
\end{table}
"""

T07 = r"""% 由 common/make_tables.py 生成 —— 请勿手改。
% 数据: persistence_plume2_eq9.csv + E01_main_eq9_external.csv
%       + E02_ablation_eq9.csv（A 组作 Independent MPNN，C 组作 MFGNO）
% 2026-09-25：Plume-2D 主表从 tables/03_main_results.tex 拆出、移入附录
% （app:plume2）；数值 ×100（百分比口径）。
% make_tables.py 重生成时需同步此拆分与百分比口径。
\begin{table}[H]
\centering
\small
\caption{\ptwo\ short-window field-wise relative $L^2$ error (\%) under the
common $10\!\to\!10$ prediction protocol; lower is better. Rows are
single-seed runs and carry no seed standard deviation (FNO, GNOT,
Independent MPNN, Transolver, \model). Persistence holds the last observed
input frame (frame index 9) constant over the prediction window, and by
protocol carries no seed variance. Best learned result in each field is
shown in bold. External baselines use their MSE objectives; graph models
use the per-field relative $L^2$ objective (Sec.~\ref{sec:setup}).}
\label{tab:main2}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lcccccc}
\toprule
Model
& $\rho$
& $P$
& $T$
& $\log_{10}\eta$
& $v_x$
& $v_y$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular}
\end{table}
"""

T08 = r"""% 由 common/make_tables.py 生成 —— 请勿手改。
% 数据: E02_ablation_eq9.csv（3 seeds）+ eval_per_traj.py 的逐轨迹产物
% 2026-09-25: 从原 tables/04_ablation.tex 拆出，数值 ×100（百分比口径）。
% make_tables.py 重生成时需同步此拆分与百分比口径。
\begin{table}[H]
\centering
\small
\caption{Paired B--C differences and $95\%$ paired trajectory-bootstrap
intervals on \pone, in relative $L^2$ error (\%). $E_B-E_C$ reports the
mean $\pm$ SD over the three paired training seeds, while the $95\%$
intervals are trajectory-level paired bootstrap intervals ($10^{4}$
resamples, one index set shared across models) conditioning on the trained
checkpoints.}
\label{tab:ablation_stats}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lccc}
\toprule
 & $\rho$ & $P$ & $T$\\
\midrule
@@EBC1@@
@@CI1@@
\bottomrule
\end{tabular}

\vspace{4pt}

\begin{tabular}{lccc}
\toprule
 & $\log_{10}\eta$ & $v_x$ & $v_y$\\
\midrule
@@EBC2@@
@@CI2@@
\bottomrule
\end{tabular}
\end{table}
"""

T05 = r"""% 由 common/make_tables.py 生成 —— 请勿手改。
% 来源: E04_diagnostics_cost/results/model_meta.json（collect_meta.py 收集）
\begin{table}[h]
\centering\scriptsize
\caption{Configuration record. Width/depth and objective are summarized from the
per-run configuration files kept alongside each checkpoint; total parameters are
counted as $\sum\mathrm{numel}$ over \texttt{model\_state\_dict} of the selected
checkpoint (for the independent-field baseline this is the \emph{sum of all six
sub-networks}). Budget is training epochs and the batch size actually used.
Seed IDs list the training seeds used in the reported short-window
and ablation results; they differ between the two subsets, which are
stated separately and not averaged.}
\label{tab:training}
\begin{tabularx}{\linewidth}{@{}lXXXXX@{}}
\toprule
Model & Width / depth & Total params & Objective & Budget / batch & Seed IDs\\
\midrule
@@BODY@@
\bottomrule
\end{tabularx}
\end{table}
"""

T06 = r"""% 由 common/make_tables.py 生成 —— 请勿手改。
% 来源: model_meta.json + cost_<model>_<subset>.json（cost.py 实测）+ 主表 CSV
\begin{table}[h]
\centering\small
\caption{Measured accuracy--cost record. Parameters are counted from checkpoints;
training hours are the \emph{accumulated job wall time} from \texttt{sacct},
including segments killed by the original 48\,h submission limit and any
resumed runs (the released training script raises this limit to 96\,h to
avoid segmentation), so they bound cost, not pure compute. Single-call inference latency is batch
one, fp32, on the same GPU partition (GNOT's graph remains on the CPU); it is the measured latency of one model
invocation and should not be interpreted as end-to-end latency for an
identical ten-frame prediction task, because the number of invocations
required differs across model families (one per window for the joint graph
models and GNOT, six for the independent-field baseline, and ten for FNO and
Transolver). $E_{v_x}/E_{v_y}$ are the seed-mean errors from the common
$10\!\to\!10$ protocol reported in Tab.~\ref{tab:main1} in percent, so the
two tables can be read against each other. Peak memory is a batch-one probe: forward \emph{and backward} for the
graph models, forward only for the external baselines, whose training code was
not instrumented --- the five rows are comparable within those two groups and
not across them. The I2VIS data-generation cost is reported separately in
Table~\ref{tab:solvercost}.}
\label{tab:cost}
\fitwidth{\begin{tabular}{@{}lccccc@{}}
\toprule
Model & Params (M) & Peak GiB & Train hours & Single-call ms & $E_{v_x}/E_{v_y}$ (\%)\\
\midrule
@@BODY@@
\bottomrule
\end{tabular}}
\end{table}
"""


def _fill(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("@@{}@@".format(k), v)
    return out


# ============================================================ 05 / 06
OBJECTIVE_OVERRIDE = {
    "Independent MPNN": r"per-field relative $L^2$, equal weights",
    "MFGNO": r"per-field relative $L^2$, equal weights",
    "A control": r"per-field relative $L^2$, equal weights",
    "B control": r"per-field relative $L^2$, equal weights",
    "C control": r"per-field relative $L^2$, equal weights",
}


def _load_meta():
    p = os.path.join(EXPERIMENTS, "E04_diagnostics_cost", "results", "model_meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)


def _load_cost(model, subset):
    p = os.path.join(EXPERIMENTS, "E04_diagnostics_cost", "results",
                     "cost_{}_{}.json".format(model, subset))
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)


def _seeds_by_subset(model_disp):
    out = {}
    for s in ("plume1d", "plume2"):
        key = {"FNO": "fno", "GNOT": "gnot", "Transolver": "transolver"}.get(model_disp)
        if key:
            seeds = {_seed_of_exp_id(r["exp_id"]) for r in _ext_rows(s, key)}
        else:
            variant = {"Independent MPNN": "A", "MFGNO": "C",
                       "A control": "A", "B control": "B", "C control": "C"}.get(model_disp)
            seeds = {int(r["seed"]) for r in _ab_rows(s, variant)} if variant else set()
        out[s] = sorted(x for x in seeds if x is not None)
    return out


def _fmt_seeds(by_subset):
    def s(lst):
        return ", ".join(str(x) for x in lst) if lst else r"\pending"

    a, b = by_subset.get("plume1d", []), by_subset.get("plume2", [])
    if a == b:
        return s(a)
    return "{} (\\pone),\\newline {} (\\ptwo)".format(s(a), s(b))


def build_training():
    meta = _load_meta()
    order = ["FNO", "GNOT", "Independent MPNN", "Transolver", "MFGNO",
             "A control", "B control", "C control"]
    body = []
    for name in order:
        m = (meta or {}).get("models", {}).get(name)
        if not m:
            body.append("{} & \\pending & \\pending & \\pending & \\pending & \\pending\\\\"
                        .format(name))
            continue
        params = ("%.2f M" % (m["params"] / 1e6)) if m.get("params") else r"\pending"
        budget = (r"\pending" if not m.get("epochs")
                  else "%d ep, bs %s" % (m["epochs"], m.get("batch_size")))
        objective = OBJECTIVE_OVERRIDE.get(name) or m.get("objective") or r"\pending"
        body.append("{} & {} & {} & {} & {} & {}\\\\".format(
            name, m.get("width_depth") or r"\pending", params,
            objective, budget, _fmt_seeds(_seeds_by_subset(name))))
    return _fill(T05, BODY="\n".join(body))


def _E_of(subset, model_disp):
    """主表口径的逐场 E（百分比）：外部基线取种子均值，GNN 取 A/C。"""
    key = {"FNO": "fno", "GNOT": "gnot", "Transolver": "transolver"}.get(model_disp)
    if key:
        rows = _ext_rows(subset, key)
    else:
        variant = {"Independent MPNN": "A", "MFGNO": "C"}.get(model_disp)
        rows = _ab_rows(subset, variant) if variant else []
    if not rows:
        return None
    return np.stack([field_values(r) for r in rows]).mean(axis=0)


def build_cost(subset="plume1d"):
    meta = _load_meta()
    order = ["FNO", "GNOT", "Independent MPNN", "Transolver", "MFGNO"]
    costkey = {"FNO": "fno", "GNOT": "gnot", "Independent MPNN": "mpnn",
               "Transolver": "transolver", "MFGNO": "mfgno"}
    body = []
    for name in order:
        m = (meta or {}).get("models", {}).get(name) or {}
        c = _load_cost(costkey[name], subset)
        pm = ("%.2f" % (m["params"] / 1e6)) if m.get("params") else r"\pending"
        peak = ("%.2f" % c["peak_alloc_GiB"]) if (c and c.get("peak_alloc_GiB") is not None) else r"\pending"
        hrs = ("%.2f" % m["train_hours"]) if m.get("train_hours") else r"\pending"
        ms = ("%.1f" % c["infer_latency_ms"]) if (c and c.get("infer_latency_ms") is not None) else r"\pending"
        E = _E_of(subset, name)
        ev = ("{} / {}".format(_sig3_plain(E[4]), _sig3_plain(E[5]))) if E is not None else r"\pending"
        body.append("{} & {} & {} & {} & {} & {}\\\\".format(name, pm, peak, hrs, ms, ev))
    return _fill(T06, BODY="\n".join(body))


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="生成论文表格")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    t03 = _fill(T03, BODY=render_rows(main_rows("plume1d")))
    t04 = _fill(T04, BODY=render_ablation(ablation_rows()))
    t05 = build_training()
    t06 = build_cost()
    t07 = _fill(T07, BODY=render_rows(main_rows("plume2"), strip_zeros=False))
    st = bc_stats()
    if st is None:
        t08 = _fill(T08, EBC1="", CI1="", EBC2="", CI2="")
    else:
        m, s, lo, hi = st

        def _row(label, cells):
            return label + " & " + " & ".join(cells) + "\\\\"

        t08 = _fill(
            T08,
            EBC1=_row("$E_B-E_C$",
                      [_ebc_cell(m[i], s[i], plus=(i == 0)) for i in range(3)]),
            CI1=_row(r"95\% traj.~CI", [_ci_cell(lo[i], hi[i]) for i in range(3)]),
            EBC2=_row("$E_B-E_C$", [_ebc_cell(m[i], s[i]) for i in range(3, 6)]),
            CI2=_row(r"95\% traj.~CI", [_ci_cell(lo[i], hi[i]) for i in range(3, 6)]),
        )

    files = [("03_main_results.tex", t03), ("04_ablation.tex", t04),
             ("05_training.tex", t05), ("06_cost.tex", t06),
             ("07_plume2_main.tex", t07), ("08_ablation_stats.tex", t08)]

    if a.dry_run:
        for name, txt in files:
            print("=" * 72)
            print("%", name)
            print(txt)
        print("[dry-run] 不落盘")
        return 0

    out_dir = a.out_dir or os.path.join(ICLR, "tables")
    os.makedirs(out_dir, exist_ok=True)
    for name, txt in files:
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf8") as f:
            f.write(txt)
        print("写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
