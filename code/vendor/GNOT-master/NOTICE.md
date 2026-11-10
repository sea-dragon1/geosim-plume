# GNOT dependency and licence status

This directory contains only the GeoSim GNOT adapter:

- `plume1/train.py` — GeoSim training wrapper;
- `plume1/dataset.py` — GeoSim HDF5-to-DGL adapter.

It does **not** contain the upstream GNOT implementation.  The GNOT model,
`models/mmgpt.py`, `models/mlp.py`, and `utils.py`, are obtained separately
from the official repository:

<https://github.com/thu-ml/GNOT>

At the time of release preparation, that repository has no LICENSE file and
its GitHub license field is `null`.  Public availability is not a licence
grant.  We therefore do not redistribute the GNOT model code and make no
licence claim over it.  Users must obtain it from the upstream repository and
comply with the terms that apply there.  If the authors grant redistribution
permission, the files can be placed here and this notice updated.

## Making the GNOT baseline runnable

Copy the upstream GNOT model files into this directory, preserving the
expected import layout:

```text
GNOT-master/
├── __init__.py              # from upstream
├── utils.py                 # from upstream
├── models/
│   ├── __init__.py          # from upstream
│   ├── mlp.py               # from upstream
│   └── mmgpt.py             # from upstream
└── plume1/
    ├── dataset.py           # GeoSim adapter (already here)
    └── train.py             # GeoSim adapter (already here)
```

The adapter expects `from models.mmgpt import GNOT` and
`from utils import MultipleTensors, get_seed`.

Without the upstream files, the GNOT training and re-evaluation paths are
intentionally unavailable; the shipped GNOT metrics, tables, and checkpoints
remain in the package as results, but cannot be reloaded without the model
code.

The GeoSim adapter files are covered by the repository-level MIT license to
the extent they do not incorporate upstream GNOT code.
