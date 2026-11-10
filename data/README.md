# GeoSim dataset

This directory does not contain the dataset itself. It contains the
dataset publication pointer and the I2VIS data-generation drivers; the
dataset is published separately and should be placed under `GEOSIM_DATA`
when running the code.

## Where the dataset is published

**Hugging Face Datasets:**

<https://huggingface.co/datasets/irenier/GeoSim>

**DOI:** `10.57967/hf/11421`

<https://doi.org/10.57967/hf/11421>

- Current revision: `755fbc770ca7c3c39a99b6a3960b75e3f85d8590`
  (last modified 2026-11-04)
- License: CC-BY-4.0
- Approximate size: 99.7 GB
- Contents: 237 two-dimensional mantle-plume trajectories (109 Plume1 +
  128 Plume2), six physical fields, native `401×309` and learning
  `128×128` representations, per-trajectory timestamps, and parameter
  metadata.
- Stored frames: Plume1 training pool 97 (maximum end time 55.24 Myr),
  Plume1 test set 131 (maximum end time 72.24 Myr). Each Plume2 trajectory
  contains 131 frames with trajectory-specific timestamps; 67.99 Myr is
  the maximum end time (67.92 Myr in the test file).

At the time of writing, the Hugging Face repository is access-controlled
(`gated: auto`). Request access through the Hugging Face page. The GMD
release version is intended to be login-free; check the repository page for
the current access status.

A Zenodo mirror/DOI is being prepared. If a Zenodo DOI is available, cite
that DOI in addition to the Hugging Face DOI; do not replace the
Hugging Face record, which is the primary data archive.

## How to use it

Set `GEOSIM_DATA` to the directory containing:

```text
plume1d_corrected/
plume2/
```

The expected file names are listed in the package-level `README.md` §4
(Data checklist). The training/evaluation code reads
`dataset.saved_folder` from the YAML configs, not `GEOSIM_DATA` directly;
update those paths as described in the package README.

The Hugging Face files use their own names
(`geosim_plume{1,2}_{train,test}_{128x128_norm,401x309_physical}.hdf5`).
They are content-equivalent to the code-facing `plume1d_*` / `plume2d_*`
files; rename or symlink them to the names in the package README. The
`*_first21` training files are the first 21 frames of the corresponding
`128x128` file.

## Data-generation drivers

The I2VIS batch, sampling, and PRN-to-HDF5 drivers are under
[`generation/`](generation/). They take the patched solver directory,
campaign directory, parameter table, Singularity image, and SLURM partition
as command-line arguments; no author-machine paths are embedded. See
[`generation/README.md`](generation/README.md) for the full workflow.

## Important data conventions

- The `128×128` files are max-normalized per subset; the native `401×309`
  files are in raw physical units.
- Coordinates are dimensionless `[0,1]` for the `128×128` files and meters
  for the native files.
- Time is in Myr for the downsampled files and years for the native files;
  the time axis is per-trajectory and non-uniform.
- The pressure channel is **total** pressure, not a dynamic pressure
  anomaly.
- The dataset is for surrogate-model benchmarking; it is not a replacement
  for the I2VIS solver.

## Citation

Cite the dataset DOI and the GeoSim paper. The exact paper citation will be
added here when the GMD version is published.
