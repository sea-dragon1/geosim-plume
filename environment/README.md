# Environment

This directory records the environment used to validate the release: training,
evaluation, figure regeneration and the GPU probe. Raw data is not included.

## Validated environment

| Component | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu121 |
| CUDA runtime (torch) | 12.1 |
| cuDNN | 9.1.0 |
| PyG | torch-geometric 2.6.1 |
| PyG extensions | torch-cluster 1.6.3+pt24cu121, torch-scatter 2.1.2, torch-sparse 0.6.18 |
| DGL | 2.1.0 (CPU build) |
| numpy / scipy / h5py | 1.24.4 / 1.10.1 / 3.11.0 |
| pandas / matplotlib | 2.0.3 / 3.7.5 |
| imageio / einops / tqdm | 2.33.1 / 0.8.1 / 4.68.2 |
| ruamel.yaml / pyyaml | 0.18.15 / 6.0.2 |

Hardware probe (compute node, 2026-11-01):

| Item | Value |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB, 79.33 GiB, compute capability 8.0 |
| NVIDIA driver | 535.104.12 (driver CUDA 12.2) |
| SLURM partition | `gpu5` (Tianhe default; edit `#SBATCH -p` for other clusters), one GPU per job |
| E03 formal manifests | all 20 checkpoints measured on A800-SXM4-80GB |

## Files

| File | Content |
|---|---|
| `env.json` | Machine-readable hardware/runtime/package record (probe output) |
| `nvidia-smi.txt` | Raw `nvidia-smi` output from the probe node |
| `conda-list.txt` | Full `conda list -n fno` snapshot (213 entries) |
| `pip-freeze.txt` | Exact pip snapshot; conda-built local paths are informational |
| `requirements.txt` | Direct dependencies pinned to the validated versions |
| `setup_env.sh` | Script to recreate the `fno` environment |

## Recreating the environment

```bash
bash environment/setup_env.sh          # default env name: fno
conda activate "${GEOSIM_CONDA_ENV:-fno}"
```

The submit scripts also honor `GEOSIM_CONDA_ENV` (conda environment name,
default `fno`) and `GEOSIM_BASHRC` (shell init file, default `~/.bashrc`).
For another SLURM cluster, edit the `#SBATCH -p` / `#SBATCH -t` lines in
`code/experiments/common/run_*.sh` and `smoke/smoke_test.sh`.

The script installs PyTorch 2.4.1 from the cu121 wheel index and the PyG
extension wheels from `data.pyg.org`. A driver supporting CUDA 12.1
(driver >= 525 on Linux) is required. Internet access is needed for the
wheel indexes.

## Notes

- **GNOT uses the CPU build of DGL** even when the model runs on GPU. Its
  inference latency is therefore not directly comparable with the other
  models' GPU measurements.
- **Transolver** imports `timm` if available but falls back to
  `torch.nn.init.trunc_normal_`; `timm` is not required.
- The release code does not import `scikit-learn` or `deepxde`, although older
  project notes may list them.
- **Cost measurements (V7) are hardware-dependent.** The values used in the
  paper are the shipped `results/metrics/E04_diagnostics_cost/cost_*.json`
  (A800-SXM4-80GB); numbers measured on other GPUs will differ.
- Older notes referring to PyTorch 1.13.1 / CUDA 11.6 describe an earlier
  setup. This release is validated on the versions recorded here.
