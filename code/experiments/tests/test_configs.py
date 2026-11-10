"""随包冻结配置的完整性与自洽性检查。

覆盖三类风险：
  1. 键缺失或格式错误（会让上游训练脚本 KeyError 或 h5py 打不开）
  2. 消融矩阵错配（B/C 的 zero_var_diff 与 loss_mode 组合）
  3. 训练/测试模式残留（误把历史 checkpoint 或 if_training: False 带进来）

运行： ``python -B tests/test_configs.py``
"""
from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, ".."))

import yaml  # noqa: E402

FAILURES = []

# 每个训练入口必需的顶层键
REQUIRED_BY_TRAINER = {
    # E02 全部走隔离副本（需要统一损失）
    "code/experiments/common/vendor/train_tubor.py": [
        "root_path", "exp_id", "model_id", "pde_name", "temporal_domain",
        "resolution_t", "spatial_domain", "resolution", "num_outputs",
        "if_training", "continue_training", "model_path", "save_period",
        "seed", "time_window", "epochs", "loss_mode",
    ],
    "fno/train_new.py": [
        "root_path", "exp_id", "if_training", "continue_training", "model_path",
        "seed", "training_type", "t_train", "initial_step", "epochs",
    ],
    "GNOT-master/plume1/train.py": [
        "root_path", "exp_id", "if_training", "continue_training", "model_path",
        "seed", "time_window", "epochs",
    ],
    "Transolver/train.py": [
        "root_path", "exp_id", "if_training", "continue_training", "model_path",
        "seed", "initial_step", "t_train", "epochs",
    ],
}


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print("[{}] {}{}".format(status, name, ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def trainer_of(path):
    """从生成文件头部注释里读出训练入口。"""
    for line in open(path, encoding="utf8"):
        m = re.search(r"训练入口:\s*(\S+)", line)
        if m:
            return m.group(1)
        if not line.startswith("#") and line.strip():
            break
    return None


def is_manual(name):
    """手写配置不参与「生成配置」的结构检查 —— 它们绕开的正是那些检查项。

    - ``SMOKE-*``  ：冒烟，只验证代码能跑通，配置是从正式配置复制后改小 epochs
    - ``RESUME-*`` ：分段续训，``continue_training: True`` 且 ``model_path``
      指向历史 checkpoint —— 与「全部从头训练」这条检查天然冲突

    两者都必须排除，否则测试会对着**故意例外**的配置报假失败。
    """
    return name.startswith(("SMOKE-", "RESUME-"))


def load_all():
    return {p: yaml.safe_load(open(p, encoding="utf8"))
            for p in sorted(glob.glob(os.path.join(EXPERIMENTS, "E0*", "configs", "*.yaml")))
            if not is_manual(os.path.basename(p))}


def test_all_configs_parse():
    paths = sorted(glob.glob(os.path.join(EXPERIMENTS, "E0*", "configs", "*.yaml")))
    check("存在生成的配置", len(paths) > 0, str(len(paths)))
    bad = []
    for p in paths:
        try:
            yaml.safe_load(open(p, encoding="utf8"))
        except Exception as e:
            bad.append((os.path.basename(p), repr(e)))
    check("全部配置可被 YAML 解析（{} 个）".format(len(paths)), not bad, str(bad[:3]))


def test_required_keys_present():
    missing = []
    for path, cfg in load_all().items():
        tr = trainer_of(path)
        if tr is None or tr not in REQUIRED_BY_TRAINER:
            missing.append((os.path.basename(path), "unknown trainer {!r}".format(tr)))
            continue
        for key in REQUIRED_BY_TRAINER[tr]:
            if key not in cfg:
                missing.append((os.path.basename(path), "{} missing for {}".format(key, tr)))
    check("各训练入口的必需键齐全", not missing, str(missing[:5]))


def test_training_mode_flags():
    problems = []
    for path, cfg in load_all().items():
        name = os.path.basename(path)
        if cfg.get("if_training") is not True:
            problems.append((name, "if_training={!r}".format(cfg.get("if_training"))))
        if cfg.get("continue_training") is not False:
            problems.append((name, "continue_training={!r}".format(cfg.get("continue_training"))))
        if cfg.get("model_path") is not None:
            problems.append((name, "model_path={!r}（不应指向历史 checkpoint）".format(cfg.get("model_path"))))
    check("全部配置为「从头训练」模式", not problems, str(problems[:5]))


def test_dataset_paths_absolute_and_distinct():
    problems = []
    for path, cfg in load_all().items():
        name = os.path.basename(path)
        ds = cfg.get("dataset", {})
        folder = str(ds.get("saved_folder", ""))
        fname = str(ds.get("file_name", ""))
        if not folder.startswith("/"):
            problems.append((name, "saved_folder 非绝对路径: {}".format(folder)))
        if not fname.endswith(".hdf5"):
            problems.append((name, "file_name 异常: {}".format(fname)))
    check("数据路径均为绝对路径且文件名合法", not problems, str(problems[:5]))


def test_root_path_is_inside_experiments():
    problems = []
    for path, cfg in load_all().items():
        root = str(cfg.get("root_path", ""))
        if not root.startswith(EXPERIMENTS):
            problems.append((os.path.basename(path), root))
    check("所有 root_path 都落在 experiments/ 内", not problems, str(problems[:3]))


def load_runs():
    """加载随包的 run 配置副本（评估时实际读取的就是这些 YAML）。"""
    return {p: yaml.safe_load(open(p, encoding="utf8"))
            for p in sorted(glob.glob(os.path.join(EXPERIMENTS, "E0*", "runs", "*", "*.yaml")))}


def test_run_configs_paths_are_release_consistent():
    """run 配置不得再指向实际训练目录，root/model_path 必须落在发布包内。

    2026-11-08 修复 P0-4：``runs/*.yaml`` 原先保留
    ``.../Plume_data/iclr2027/experiments/...`` 旧路径；评估读的是 run 配置，
    换机器后会找不到数据或 checkpoint。
    """
    old_root = "/Plume_data/iclr2027/experiments/"
    problems = []
    for path, cfg in load_runs().items():
        name = os.path.basename(path)
        root = str(cfg.get("root_path", ""))
        ds = cfg.get("dataset", {})
        folder = str(ds.get("saved_folder", ""))
        model_path = cfg.get("model_path")

        if old_root in root:
            problems.append((name, "root_path 仍是旧训练目录: {}".format(root)))
        if not root.startswith(EXPERIMENTS):
            problems.append((name, "root_path 不在发布包内: {}".format(root)))
        if not folder.startswith("/"):
            problems.append((name, "saved_folder 非绝对路径: {}".format(folder)))
        if model_path and str(model_path).strip().lower() not in ("", "none", "null", "~"):
            model_path = str(model_path)
            if old_root in model_path:
                problems.append((name, "model_path 仍是旧训练目录: {}".format(model_path)))
            if not os.path.exists(model_path):
                problems.append((name, "model_path 不存在: {}".format(model_path)))
    check("run 配置路径均指向发布包/数据根", not problems, str(problems[:5]))


def test_ablation_matrix_complete():
    """E02 必须有完整的 A/B/C 结构对照（默认只生成等权臂）。

    速度加权臂的**代码能力保留**（``losses.py`` 的权重预设支持），但本轮不生成
    配置 —— 用户明确「这个损失是为了提升速度场精度加的，和论文主线无关，
    不加对比更准确」。
    """
    configs = {os.path.basename(p): cfg for p, cfg in load_all().items()}
    e02 = {k: v for k, v in configs.items() if "-E02-" in k}
    check("E02 配置存在", len(e02) > 0, str(len(e02)))

    # ⚠ 种子数**按子集区分**（决策记录见冻结配置）：
    #   plume1d —— 3 种子（B/C 已跑完、A 组在跑，结果已进台账，不可缩减）
    #   plume2  —— 本轮只有 seed0，与 E01 同策略（先把单种子闭环做完）
    seeds_by_subset = {"plume1d": (0, 1, 2), "plume2": (0,)}

    for subset in ("plume1d", "plume2"):
        for variant in ("B", "C"):
            for seed in seeds_by_subset[subset]:
                tag = "-E02-{}-equal-{}-seed{}".format(variant, SUBSET_TAG[subset], seed)
                match = [k for k in e02 if tag in k]
                check("E02 格子存在: {} equal {} seed{}".format(variant, subset, seed),
                      len(match) == 1, str(match))
                if len(match) == 1:
                    cfg = e02[match[0]]
                    expected_zero = (variant == "B")
                    got = cfg.get("model", {}).get("zero_var_diff")
                    check("  {} 的 zero_var_diff == {}".format(variant, expected_zero),
                          got is expected_zero, "got {!r}".format(got))
                    check("  loss_mode == equal", cfg.get("loss_mode") == "equal",
                          repr(cfg.get("loss_mode")))
                    check("  上游损失分支已关闭",
                          cfg.get("if_l2re_loss") == 0
                          and cfg.get("if_weighted_l2re_loss") == 0)

    # 加权臂本轮不生成（能力保留在代码里）
    weighted = [k for k in e02 if "velocity_weighted" in k]
    check("默认不生成加权臂配置", not weighted, str(weighted[:3]))


def test_gnn_models_all_use_vendor():
    """GNN 系（E01 与 E02）必须都走隔离副本 —— 否则两个阶段的 MFGNO 不是同一个模型。

    走副本的两个理由：统一逐场损失、本项目对时间处理的改动。
    """
    GNN = ("mfgno", "mpnn")
    bad = []
    for path in sorted(glob.glob(os.path.join(EXPERIMENTS, "E0*", "configs", "*.yaml"))):
        name = os.path.basename(path)
        model = name.split("-")[3] if len(name.split("-")) > 3 else ""
        tr = trainer_of(path) or ""
        if model in GNN and "vendor" not in tr:
            bad.append((name, tr))
    check("GNN 系全部走 vendor 副本", not bad, str(bad[:3]))

    # 非 GNN 系不应走 vendor
    bad2 = []
    for path in sorted(glob.glob(os.path.join(EXPERIMENTS, "E0*", "configs", "*.yaml"))):
        name = os.path.basename(path)
        model = name.split("-")[3] if len(name.split("-")) > 3 else ""
        tr = trainer_of(path) or ""
        if model in ("fno", "gnot", "transolver") and "vendor" in tr:
            bad2.append((name, tr))
    check("非 GNN 系不走 vendor", not bad2, str(bad2[:3]))


# 训练主口径数据集：两个子集都是**前 21 帧、Myr 标尺**。
#
# ⚠ 帧数相同（21）但**物理时长不同**：plume1d 覆盖 0→17.24 Myr、plume2 覆盖 0→12.99 Myr。
#   因此主表必须分别报告，不得把两子集的分混成一个平均值。
EXPECTED_TRAIN_DATA = {
    "plume1d": "plume1d_train_ds_128x128_phys1e6_first21.hdf5",
    # ⚠ 2026-09-25：plume2 的训练数据已换成**修正版**（保守重映射 + 正确坐标），
    #   非修正版的 `*_first21.hdf5` 不再是任何 run 的输入。此处原先仍写旧名，
    #   导致这条检查对着现行配置报假失败。
    "plume2": "plume2d_train_ds_128x128_phys1e6_first21_corrected.hdf5",
}

# 配置名里代表该子集的片段。plume2 换修正版数据后名字多了 `-corr`，
# 所以**不能再拿子集名直接拼 tag** —— 那样拼出来的是换代前的名字。
# 用子串匹配而不是完整文件名，是为了对文件名前缀的日期免疫
# （plume1d 是 2026-09-17、plume2 是 2026-09-22）。
SUBSET_TAG = {"plume1d": "plume1d", "plume2": "plume2-corr"}


def subset_of(name):
    for s in EXPECTED_TRAIN_DATA:
        if "-{}-".format(s) in name or "-{}.yaml".format(s) in name:
            return s
    return None


def test_training_data_is_first21():
    """两个子集的训练数据都必须是**前 21 帧** first21 版本。

    为什么是 21 帧：上游 train_loop:151 的
        assert nt - tw - tw*unrolled > tw
    与 create_data 的切片边界合起来要求数据至少 21 帧（详见 README）。

    同时守住两个易错点：不能误用抽稀的 ``_20step``（那是时间外推基准，不是训练口径），
    也不能退回 ``_norm.hdf5``（全时程 131 帧 + t 单位是年）。
    """
    bad, unknown = [], []
    for path, cfg in load_all().items():
        name = os.path.basename(path)
        subset = subset_of(name)
        if subset is None:
            unknown.append(name)
            continue
        fname = str(cfg.get("dataset", {}).get("file_name", ""))
        if not fname:
            continue
        if fname != EXPECTED_TRAIN_DATA[subset]:
            bad.append((name, fname))
    check("训练数据全部指向 first21（两子集）", not bad, str(bad[:3]))
    check("每个配置都能归到已知子集", not unknown, str(unknown[:3]))


def test_resolution_t_matches_data_frames():
    """resolution_t 必须等于所选数据的帧数（两子集都是 21）。

    两条约束同时作用：
      - train_loop:151  assert nt - tw - tw*unrolled > tw  → nt >= 21
      - create_data     dp[step : step+tw] 不得越界        → nt <= 帧数
    nt 不等于帧数时，要么 assert 失败，要么 labels 切片越界（stack 崩溃）。
    """
    bad, skipped = [], 0
    for path, cfg in load_all().items():
        name = os.path.basename(path)
        if "resolution_t" not in cfg:
            skipped += 1     # FNO / Transolver 用 t_train，没有这个键
            continue
        if cfg["resolution_t"] != 21:
            bad.append((name, "resolution_t={} 应为 21".format(cfg["resolution_t"])))
    check("所有配置的 resolution_t 与 first21 帧数一致", not bad, str(bad[:3]))
    print("     （{} 个配置无 resolution_t 键，已跳过）".format(skipped))


def test_e01_reuses_e02_for_gnn_arms():
    """E01 **不**为 MFGNO / MPNN 生成配置 —— 这两个臂直接复用 E02（2026-09-21 决定）。

    理由：E02 的 C 组就是 MFGNO、A 组就是 6 个独立 MPNN，数据/超参/损失/训练入口
    与 E01 逐字相同（已 diff 核对）。复用省 4 次训练约 84 GPU·h，更重要的是保证
    **主表的 MFGNO 一行与消融表的 C 一行是同一个 checkpoint**。

    本测试守护两点，任一处被改动都会失败：
      1. E01 的配置里不得重新出现 mfgno / mpnn（若真的需要独立跑，必须同步
         调整 E01/E02 的配置与复用关系）；
      2. 被复用的 E02 配置必须仍然存在，否则主表会**静默少两行**。
    """
    names = [os.path.basename(p) for p in load_all()]
    leaked = [n for n in names if "-E01-" in n and re.search(r"-(mfgno|mpnn)-E01-", n)]
    check("E01 不含 mfgno/mpnn 配置（应复用 E02）", not leaked, str(leaked[:3]))

    # 被复用的四条（两子集 × 两臂）必须都在，seed0 是主表用的那个。
    # ⚠ 用子串匹配、不拼完整文件名：原先写死了 `2026-09-17-` 前缀，
    #   而 plume2 的现行配置是 `2026-09-22-...-corr-...`，两处都对不上。
    need = ["-mfgno-E02-C-equal-{}-seed0".format(SUBSET_TAG[s]) for s in EXPECTED_TRAIN_DATA]
    need += ["-mpnn-E02-A-{}-seed0".format(SUBSET_TAG[s]) for s in EXPECTED_TRAIN_DATA]
    missing = [n for n in need if not any(n in name for name in names)]
    check("E01 要复用的 E02 配置齐全", not missing, str(missing))


def test_a_group_uses_unified_loss():
    """A 组必须也走统一损失，否则无法与 B/C 做同一目标的对照。"""
    configs = {os.path.basename(p): cfg for p, cfg in load_all().items()}
    a_cfgs = {k: v for k, v in configs.items() if "-E02-A-" in k}
    # 4 = plume1d 的 3 个种子 + plume2 的 1 个（同上，plume2 本轮只跑 seed0）
    check("A 组配置存在（1D×3 + 2D×1）", len(a_cfgs) == 4, str(len(a_cfgs)))
    for name, cfg in a_cfgs.items():
        check("A 组带 loss_mode: " + name, cfg.get("loss_mode") == "equal",
              repr(cfg.get("loss_mode")))
        check("A 组 model_id 为 MPNN: " + name, cfg.get("model_id") == "MPNN",
              repr(cfg.get("model_id")))


# phys1e6 标尺（单位 Myr）下合法的 temporal_domain 上限。
#
# 用白名单而非"数量级合理即可"，是为了让**换了数据却忘改 temporal_domain** 必然失败。
# 代价是每新增一个数据版本都要来此登记 —— 登记时请顺手核对上限与
# protocol/manifest.yaml 中该文件的 t 范围一致。
KNOWN_MYR_DOMAINS = {
    "(0, 8.7433)": "plume1d 10step（前 10 帧）",
    "(0, 16.7433)": "plume1d first21 的历史写法（保留兼容）",
    "(0, 17.2433)": "plume1d first21（前 21 帧，训练主口径）",
    "(0, 55.2433)": "plume1d 全时程",
    "(0, 12.9913)": "plume2 first21（前 21 帧，训练主口径）",
}


def test_phys1e6_uses_matching_temporal_domain():
    """用 phys1e6 数据时 temporal_domain 必须同步放大，否则 t_pos 归一化越界。

    fno 的配置没有 temporal_domain 键（它用 t_train / initial_step），
    因此只检查确实声明了该键的配置。

    ⚠ 两个子集的 Myr 上限**不同**（plume1d 17.24 vs plume2 12.99），不能通用一个值。
    """
    problems, skipped, wrong_scale = [], 0, []
    for path, cfg in load_all().items():
        if "temporal_domain" not in cfg:
            skipped += 1
            continue
        fname = str(cfg.get("dataset", {}).get("file_name", ""))
        if "phys1e6" not in fname:
            continue
        td = str(cfg["temporal_domain"])
        if td not in KNOWN_MYR_DOMAINS:
            problems.append((os.path.basename(path), td))
        # 兜底：即便误登记，年尺度（~1e7）也会被这条抓住
        try:
            hi = float(td.strip("()").split(",")[1])
            if hi > 1000:
                wrong_scale.append((os.path.basename(path), td))
        except (ValueError, IndexError):
            wrong_scale.append((os.path.basename(path), td))

    check("phys1e6 数据的 temporal_domain 匹配 Myr 尺度", not problems,
          str(problems[:3]) + "  已登记: " + str(sorted(KNOWN_MYR_DOMAINS)))
    check("phys1e6 的 temporal_domain 上限不是年尺度（<1000）", not wrong_scale,
          str(wrong_scale[:3]))
    print("     （{} 个配置无 temporal_domain 键，已跳过）".format(skipped))


def test_exp_ids_unique():
    ids = []
    for path in glob.glob(os.path.join(EXPERIMENTS, "E0*", "configs", "*.yaml")):
        ids.append(os.path.basename(path))
    check("exp_id（文件名）唯一", len(ids) == len(set(ids)))


def main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if FAILURES:
        print("失败 {} 项".format(len(FAILURES)))
        for f in FAILURES[:10]:
            print("  -", f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
