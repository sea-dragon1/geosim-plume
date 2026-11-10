#!/bin/bash
# Recreate the `fno` environment used by this release.
#
# Validated versions (see env.json):
#   Python 3.8, PyTorch 2.4.1 (CUDA 12.1), PyG 2.6.1,
#   torch-cluster 1.6.3+pt24cu121, dgl 2.1.0 (CPU).
#
# The CUDA 12.1 wheel requires a GPU driver that supports CUDA 12.1
# (driver >= 525 on Linux).
set -euo pipefail

ENV_NAME="${1:-fno}"

conda create -y -n "$ENV_NAME" python=3.8
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1

# PyG extension wheels (built for torch 2.4.0 + cu121; ABI-compatible with 2.4.1)
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# DGL CPU build. GNOT uses dgl on the CPU even when the model runs on GPU.
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/repo.html

pip install numpy==1.24.4 scipy==1.10.1 h5py==3.11.0 pandas==2.0.3 \
    matplotlib==3.7.5 imageio==2.33.1 einops==0.8.1 tqdm==4.68.2 \
    "ruamel.yaml==0.18.15" pyyaml==6.0.2

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torch_geometric, dgl; print('pyg', torch_geometric.__version__, 'dgl', dgl.__version__)"
