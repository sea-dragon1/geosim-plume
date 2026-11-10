# Third-party notices

The GeoSim code is released under the MIT License in [`LICENSE`](LICENSE).
The GeoSim dataset is released separately under CC BY 4.0; see
[`data/README.md`](data/README.md).

This package includes or adapts the following third-party components.

## FNO model

`code/vendor/fno/model/fno.py` is adapted from the Fourier Neural Operator
implementation by Zongyi Li:

- Copyright (c) 2020 Zongyi Li
- MIT License; full text in [`THIRD_PARTY_LICENSES/MIT-FNO.txt`](THIRD_PARTY_LICENSES/MIT-FNO.txt)

## FNO training utilities and shared metric/CSV helpers

The FNO training entry point and the shared `metrics.py` / `pandas_data.py`
helpers follow the structure of the PDEBench examples:

- Copyright 2022 NEC Labs Europe GmbH, Stuttgart University, CSIRO, and
  PDEBench contributors
- MIT License, except where otherwise stated; full text in
  [`THIRD_PARTY_LICENSES/MIT-PDEBench.txt`](THIRD_PARTY_LICENSES/MIT-PDEBench.txt)

The NEC-licensed `utils.py` from the upstream example is **not** included in
this release.  `code/vendor/fno/utils.py` is a clean-room implementation of
the single-file 2D mantle reader written for GeoSim and is covered by the
GeoSim MIT license.

## Transolver

`code/vendor/Transolver/` is adapted from Transolver:

- Copyright (c) 2024 THUML @ Tsinghua University
- MIT License; full text in
  [`THIRD_PARTY_LICENSES/MIT-Transolver.txt`](THIRD_PARTY_LICENSES/MIT-Transolver.txt)
- Upstream: <https://github.com/thuml/Transolver>

## GNOT

The upstream GNOT model code is **not included** in this package.

- Upstream: <https://github.com/thu-ml/GNOT>
- The upstream repository does not declare a license (GitHub license field:
  none), so we do not redistribute it and make no license claim over it.
- `code/vendor/GNOT-master/plume1/train.py` and `dataset.py` are GeoSim
  adapters; they require the upstream GNOT model files to be placed in
  `code/vendor/GNOT-master/` as described in
  [`code/vendor/GNOT-master/NOTICE.md`](code/vendor/GNOT-master/NOTICE.md).
- Without those files, GNOT training and checkpoint re-evaluation are
  unavailable; the shipped GNOT metrics and checkpoints remain as results.
- Users are responsible for obtaining the upstream code and complying with
  its terms.

## I2VIS

The upstream I2VIS solver is not distributed with this package.  The
`solver_patch/` directory contains patches and case configurations against
the public OSF release:

- Gerya, T. (2021), I2VIS code, <https://doi.org/10.17605/OSF.IO/BNVTH>
- The OSF archive does not include a LICENSE file.
- The patches and case configurations are included with the permission of
  the I2VIS author (T. Gerya), confirmed by email on 10 November 2026.
  They are not covered by the GeoSim MIT license.  See
  `solver_patch/NOTICE.md` and [`PERMISSIONS.md`](PERMISSIONS.md).

## PDEBench

The upstream PDEBench code and data are distributed under the MIT license,
with the exception of files that carry a separate header.  No PDEBench data
files are included in this package.
