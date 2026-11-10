# GeoSIM-Plume — Reproduction Guide

This package reproduces the experiments, tables and figures of the GeoSIM benchmark
paper for two 2-D mantle-convection subsets:

- **GeoSim-Plume1** (`plume1d`, one varying parameter: plume temperature)
- **GeoSim-Plume2** (`plume2`, two varying parameters: plume temperature and lithospheric thickness)

## Associated paper

This code accompanies the manuscript:

> Hailong Liu, Dinghui Yang, Mengxue Liu, Rui Qi, Meng Zhao, Yihang Tang,
> Jin Wen, and Zhilong Fang (2026). *GeoSim: A Reproducible Benchmark
> Dataset and Multi-Field Graph Neural Operator for Mantle-Plume
> Dynamics.* Manuscript submitted to *Geoscientific Model Development*.

The dataset is archived separately at
<https://huggingface.co/datasets/irenier/GeoSim> (DOI
`10.57967/hf/11421`). Machine-readable citation metadata for this package
is in [`CITATION.cff`](CITATION.cff).

Chinese version: [README.zh-CN.md](README.zh-CN.md).

## 1. What is included

```
geosim-plume/
├── code/
│   ├── vendor/                     # FNO / Transolver + GNOT adapter + metrics
│   └── experiments/
│       ├── common/                 # trainers, evaluation, tables, submission scripts
│       ├── E01_main/               # external baselines: 12 configs + 12 trained runs
│       ├── E02_ablation/           # MFGNO / MPNN A-B-C: 12 configs + 12 trained runs
│       ├── E03_rollout/            # long-horizon rollout code
│       ├── E04_diagnostics_cost/   # diagnostics, profiles, cost, data QC
│       ├── figures/                # dataset and parameter-coverage figures
│       ├── logs/                   # 36 training logs (wall-clock evidence)
│       ├── protocol/               # frozen protocol (splits, metrics, time, loss, parameters)
│       └── tests/
├── results/
│   ├── metrics/                    # packaged metric summaries, incl. E03 merged results
│   ├── per_run/                    # per-run eq9.json / pertraj.npz
│   ├── tables/                     # 8 LaTeX tables
│   └── figures/                    # 8 figures + fig03/fig04 manifests
├── smoke/                          # vendor smoke test
├── environment/                    # version records + setup_env.sh
├── solver_patch/                   # modifications to the public I2VIS solver
├── data/                           # dataset pointer + I2VIS generation drivers
├── CITATION.cff                    # citation metadata
├── LICENSE                         # MIT (code)
├── PERMISSIONS.md                  # permission records (I2VIS patch)
├── THIRD_PARTY_NOTICES.md          # third-party provenance and licenses
└── THIRD_PARTY_LICENSES/           # full third-party license texts
```

The upstream I2VIS solver and raw simulations are **not** redistributed.
`solver_patch/` contains only the modifications, case configurations, and
build instructions; the upstream code is obtained from the OSF record
(DOI `10.17605/OSF.IO/BNVTH`). Data-regeneration drivers are maintained with the GeoSim data-generation
workflow; the solver patch is kept separate here so its upstream provenance
is explicit.

## 2. Environment

- conda environment `fno`: Python 3.8, PyTorch 2.4.1+cu121, CUDA 12.1,
  `torch_geometric`, `torch_cluster`, `h5py`, `numpy`, `scipy`, `matplotlib`,
  `ruamel.yaml`, `dgl` 2.1.0 (CPU build, used by GNOT).
- SLURM cluster: the shipped scripts use Tianhe defaults — GPU partition
  `gpu5`, CPU partition `cp2`, conda environment `fno`. On Tianhe, submit
  with `yhbatch` (not `sbatch`); other SLURM clusters can use their own
  batch command because the scripts only rely on `#SBATCH` headers.
- Cluster-specific settings on another machine:
  1. edit the `#SBATCH -p` / `#SBATCH -t` lines in
     `code/experiments/common/run_*.sh` and `smoke/smoke_test.sh`
     (SBATCH directives cannot read shell variables);
  2. set `GEOSIM_CONDA_ENV` to the conda environment name (default `fno`);
  3. set `GEOSIM_BASHRC` to the shell init file that defines `conda`
     (default `~/.bashrc`).
- Exact package versions, hardware probe and a recreation script:
  [`environment/`](environment/).
- Run all commands from the package root, e.g.

```bash
cd /path/to/geosim-plume
```

## 3. Paths to configure

### 3.1 Data root

```bash
export GEOSIM_DATA=/path/to/geosim-data
```

`GEOSIM_DATA` must contain `plume1d_corrected/` and `plume2/`.
`GEOSIM_ROOT` is optional; `code/experiments/common/paths.py` derives it from
the file location.

The dataset is published separately; see [`data/README.md`](data/README.md)
for the Hugging Face link, DOI, license, revision, and access status.

### 3.2 YAML paths

Training and checkpoint evaluation read `dataset.saved_folder` from YAML, not
`GEOSIM_DATA`. The shipped YAMLs contain absolute paths. Rewrite them once:

```bash
OLD=/fs1/home/USER/lhl/neuraloeprator-main/data
NEW=/path/to/geosim-data
grep -rl "$OLD" code/experiments/E0*/configs code/experiments/E0*/runs smoke/configs \
  | xargs sed -i "s#$OLD#$NEW#g"
```

Both the stage configs (`.../configs/*.yaml`) and the run copies
(`.../runs/<exp_id>/<exp_id>.yaml`) must be updated: evaluation of the shipped
checkpoints reads the run copy.

If the package itself is moved and you plan to **retrain**, also rewrite
`root_path` in `code/experiments/E0*/configs/*.yaml` to the new package path.

### 3.3 One figure script

`code/experiments/figures/make_fig01.py` reads
`<package>/data/plume1d_corrected/plume1d_test_ds_128x128_phys1e6_first21.hdf5`.
Either copy that file into the package `data/` directory, or edit `DATA_PATH`
in the script.

### 3.4 I2VIS solver patch

`solver_patch/` contains the two branch patches (`plume1`, `plume2`), their
`init.t3c`/`mode.t3c`/`Makefile`, and checksums. It does **not** contain the
upstream I2VIS source.

```bash
# Obtain the upstream codes.zip from https://doi.org/10.17605/OSF.IO/BNVTH
# and extract it to /path/to/codes/codes.
bash solver_patch/scripts/apply_patch.sh plume1 /path/to/codes/codes i2vis-plume1
# or: plume2
```

See `solver_patch/README.md` for build requirements and the required
thermodynamic database files.

### 3.5 Optional GNOT dependency

The upstream GNOT model code is not bundled. The package contains the GeoSim
adapter in `code/vendor/GNOT-master/plume1/`; to run or re-evaluate GNOT,
place the upstream `models/mmgpt.py`, `models/mlp.py`, `utils.py`, and their
package `__init__.py` files under `code/vendor/GNOT-master/` as described in
[`code/vendor/GNOT-master/NOTICE.md`](code/vendor/GNOT-master/NOTICE.md).
Without them, the shipped GNOT metrics and checkpoints remain readable, but
GNOT training and checkpoint loading are unavailable.

## 4. Data checklist

| Purpose | Files under `GEOSIM_DATA` |
|---|---|
| Short-window training and evaluation (21 frames) | `plume1d_corrected/plume1d_{train,test}_ds_128x128_phys1e6_first21.hdf5`<br>`plume2/plume2d_{train,test}_ds_128x128_phys1e6_first21_corrected.hdf5` |
| Long-horizon rollout (E03), cost (E04) | `plume1d_corrected/plume1d_{train,test}_ds_128x128_phys1e6.hdf5`<br>`plume2/plume2d_{train,test}_ds_128x128_norm_corrected.hdf5` |
| Data QC (V8, optional) | `plume1d_corrected/plume1d_train_401x309.hdf5`, `plume2/plume2d_train_401x309.hdf5`, `plume1d_corrected/remap_weights_401x309_to_128x128.npz` |

For `plume2`, `_norm_corrected` is the corrected full-time file whose time axis
is already in Myr (the suffix does not mean per-dataset normalization here).

## 5. Experiment inventory

| Paper item | Stage | Configs | Runs |
|---|---|---|---|
| X1 short-window external baselines | `E01_main` | `2026-09-17-{fno,gnot,transolver}-E01-plume1d-seed{0,1,2}.yaml` (9)<br>`2026-09-22-{fno,gnot,transolver}-E01-plume2-corr-seed0.yaml` (3) | 12 |
| X2 short-window MFGNO / MPNN | `E02_ablation` | A (MPNN) and C (MFGNO) runs below | reuses X3 |
| X3 design-attribution A/B/C | `E02_ablation` | `2026-09-17-{mpnn-E02-A,mfgno-E02-B-equal,mfgno-E02-C-equal}-plume1d-seed{0,1,2}.yaml` (9)<br>`2026-09-22-{mpnn-E02-A,mfgno-E02-B-equal,mfgno-E02-C-equal}-plume2-corr-seed0.yaml` (3) | 12 |
| V4/V5 long-horizon rollout / failure criterion | `E03_rollout` | per-run `run_rollout.sh`; 20 checkpoints | merged results shipped |
| V6 velocity profile | `E04_diagnostics_cost` | `run_diag.sh` + `velocity_profile.py` | plume1d |
| V7 cost | `E04_diagnostics_cost` | `run_cost.sh` | 5 models |
| V8 data QC | `E04_diagnostics_cost` | `data_qc.py` | both subsets |
| V9 training metadata | `common/collect_meta.py` | — | 8 model entries |

Each run directory contains `checkpoints/*-best.pt`, `eval/predictions.hdf5`
(E01 all runs; E02 only some), `eval/<exp_id>_eq9.json`, and the run YAML.

## 6. Path A — reproduce tables and figures from shipped results (no training)

This is the fastest route. It uses the shipped checkpoints/predictions and
metric files.

### 6.1 Tables

```bash
python -B code/experiments/common/make_tables.py            # writes tables 03–08
python -B code/experiments/common/make_tables.py --dry-run  # print only
```

`results/tables/02_dataset.tex` and `09_solver_cost.tex` are shipped static
tables, not generated by this script.

### 6.2 Figures

Data-reading scripts should be submitted through `run_cpu.sh`:

```bash
# Fig. 1 — dataset overview (needs the path fix in §3.3)
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/figures/make_fig01.py"

# Fig. A01 — parameter coverage (protocol CSV only)
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/figures/make_figA01.py"

# Fig. 4 / F04 — rollout curves (main + appendix)
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E03_rollout/src/plot_rollout_paper.py"

# Fig. 3 / F03b / F03c + profile CSVs (uses shipped E01/E02 predictions)
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E04_diagnostics_cost/src/plot_diagnostics.py" \
    --subset plume1d --install
```

| Output | Location |
|---|---|
| Fig. 1 | `code/experiments/figures/fig01_benchmark_overview.pdf` (+ manifest) |
| Fig. A01 | `code/experiments/figures/figA01_parameter_coverage.pdf` (+ manifest) |
| Fig. 4 / F04 | `results/figures/fig04_rollout_main.pdf`, `fig04_rollout_full.pdf` |
| Fig. 3 / F03b / F03c | `results/figures/fig03_field_predictions.pdf`, `fig03b_profile.pdf`, `fig03c_failure_case.pdf`, `fig03_manifest.json` |
| Fig. 2 | shipped `results/figures/fig02_mfgno_overview.png` |

`plot_diagnostics.py` also rewrites `E04_diagnostics_cost/results/profile_errors.csv`
and `profile_errors_summary.csv`; packaged copies are under
`results/metrics/E04_diagnostics_cost/`.

## 7. Path B — re-evaluate the shipped checkpoints (no training)

### 7.1 E01 external baselines

```bash
yhbatch code/experiments/common/run_predict.sh --stage E01_main --subset plume1d
yhbatch code/experiments/common/run_predict.sh --stage E01_main --subset plume2

yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_predictions.py" \
    --stage E01_main --subset plume1d
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_predictions.py" \
    --stage E01_main --subset plume2
```

Outputs: `runs/<exp_id>/eval/predictions.hdf5`, `eval/<exp_id>_eq9.json`,
`E01_main/results/E01_main_eq9_external.csv`.

### 7.2 E02 GNN models (A/B/C)

```bash
yhbatch code/experiments/common/run_eval.sh --stage E02_ablation --subset plume1d --compare
yhbatch code/experiments/common/run_eval.sh --stage E02_ablation --subset plume2 --compare
```

Output: `E02_ablation/results/E02_ablation_eq9.csv`.

### 7.3 Persistence, per-trajectory metrics, metadata

```bash
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_persistence.py" --subset plume1d
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/score_persistence.py" --subset plume2

yhbatch code/experiments/common/run_pertraj.sh --stage E02_ablation --subset plume1d

yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/common/collect_meta.py"
```

Outputs: `E01_main/results/persistence_<subset>_eq9.csv`,
`results/per_run/*_pertraj.npz`, `E04_diagnostics_cost/results/model_meta.json`.

### 7.4 E04 diagnostics and cost

```bash
yhbatch code/experiments/common/run_diag.sh plume1d
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E04_diagnostics_cost/src/velocity_profile.py" --subset plume1d

yhbatch code/experiments/common/run_cost.sh
yhbatch code/experiments/common/run_cpu.sh "$PWD/code/experiments/E04_diagnostics_cost/src/data_qc.py" --subset both
```

Cost numbers are hardware-dependent. The shipped
`results/metrics/E04_diagnostics_cost/cost_*.json` are the values used in the
paper (A800-SXM4-80GB, torch 2.4.1 for the measurement environment); re-measured
values may differ.

### 7.5 E03 rollout (optional)

The final merged rollout results used by Fig. 4 are shipped in
`results/metrics/E03_rollout/formal_all5/`. To re-run one checkpoint:

```bash
# Run on a GPU compute node; thresholds must exist before rollout.
python -B code/experiments/E03_rollout/src/thresholds.py --subset plume1d --range-only
python -B code/experiments/E03_rollout/src/thresholds.py --subset plume1d --run <run_dir>

yhbatch code/experiments/common/run_rollout.sh --subset plume1d --run <run_dir> \
    --mode auto --mode teacher --dump
```

Shipped `E03_rollout/results/tau_c.json` only covers the seed-0 checkpoint of
four model families; for other seeds and for GNOT, compute the per-run
thresholds first. Full campaign aggregation (`cases.json` + per-case
directories) is not shipped; use the packaged merged results for the paper
figures.

## 8. Path C — retrain from scratch

1. Configure paths (§3).
2. Submit E01 in batches (the cluster limits concurrent submissions):

```bash
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed0
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed1
bash code/experiments/E01_main/run_all.sh --only=plume1d-seed2
bash code/experiments/E01_main/run_all.sh --with-plume2 --only=plume2
```

3. Submit E02 the same way:

```bash
bash code/experiments/E02_ablation/run_all.sh --only=plume1d-seed0
bash code/experiments/E02_ablation/run_all.sh --only=plume1d-seed1
bash code/experiments/E02_ablation/run_all.sh --only=plume1d-seed2
bash code/experiments/E02_ablation/run_all.sh --with-plume2 --only=plume2
```

Each submission writes to `code/experiments/<stage>/runs/<exp_id>/`. Training
uses one GPU per job and a 96-hour wall-clock limit; keep the `root_path` and
`exp_id` fields unchanged to preserve the expected directory layout.

4. Re-run the evaluation steps in §7, then tables/figures in §6.

## 9. Verification

- `python -B code/experiments/common/make_tables.py --dry-run` prints all six
  generated tables without writing.
- After path A, `results/tables/` and `results/figures/` should contain the same
  artifacts as the shipped versions.
- `yhbatch smoke/smoke_test.sh` checks FNO and Transolver from `code/vendor`
  (2 epochs each); GNOT is checked only when its upstream files are present
  under `code/vendor/GNOT-master/`.
- Unit tests: `python -B code/experiments/tests/test_dt.py` etc. (see `protocol/`).

## 10. Citation

If you use this code or the dataset, cite the GeoSim paper and the dataset
record. Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## 11. License

- **Code:** MIT License, Copyright (c) 2026 Hailong Liu. See [`LICENSE`](LICENSE).
- **Dataset:** CC BY 4.0; the dataset is published separately, see
  [`data/README.md`](data/README.md) and [`data/LICENSE`](data/LICENSE).
- **Third-party components:** see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
  [`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/).
- **I2VIS patch/config:** not covered by the MIT license of this package; the
  patches and configurations are included with the permission of the I2VIS
  author (T. Gerya), confirmed by email on 10 November 2026. See
  [`solver_patch/NOTICE.md`](solver_patch/NOTICE.md) and
  [`PERMISSIONS.md`](PERMISSIONS.md).
- **GNOT:** the upstream GNOT model code is not bundled. The package contains
  only the GeoSim adapter (`code/vendor/GNOT-master/plume1/`); users must
  obtain the upstream GNOT code and comply with its terms. See
  [`code/vendor/GNOT-master/NOTICE.md`](code/vendor/GNOT-master/NOTICE.md).
