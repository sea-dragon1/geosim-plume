# I2VIS data-generation drivers

These drivers recreate the Plume1 and Plume2 simulation campaigns from a
patched I2VIS tree. They are the release versions of the research batch
scripts; all machine-specific paths are command-line arguments.

## What is included

```text
data/generation/
├── common.py                 # shared path/config/case helpers
├── prepare_cases.py          # shared 01_in2fast_pal implementation
├── submit_batches.py         # shared 02_i2fast_pal implementation
├── run_i2j.py                # shared 03_i2j_pal implementation
├── prn2h5.py                 # parameterized PRN -> HDF5 converter
├── plume1/
│   ├── 00_lhs_sample.py      # 100-point LHS table (T_plume)
│   ├── 01_in2fast_pal.py
│   ├── 02_i2fast_pal.py
│   ├── 03_i2j_pal.py
│   └── 04_prn2h5.py
└── plume2/
    ├── 00_sobol_sample.py    # 128-point 10-D Sobol table (2 active)
    ├── 01_in2fast_pal.py
    ├── 02_i2fast_pal.py
    ├── 03_i2j_pal.py
    └── 04_prn2h5.py
```

## Prerequisites

1. Obtain the upstream I2VIS release from
   <https://doi.org/10.17605/OSF.IO/BNVTH>.
2. Apply the matching patch from `solver_patch/` (`plume1` or `plume2`) and
   compile with Intel `icx`/MKL.  See `solver_patch/README.md`.
3. The compiled solver directory passed as `--solver-dir` must contain the
   binaries (`i2fast`, `in2fast`, `i2jslab`), `run_in2fast`, `run_i2j`,
   `init.t3c`, `mode.t3c`, `Makefile`, and a `database/` directory.
4. A Singularity image containing the Intel/MKL runtime; the image path and
   bind specification are passed on the command line.
5. Python with `numpy`, `scipy`, and `h5py`; SLURM with `yhbatch`.
   Use `scipy==1.10.1` for bit-reproducible scrambled Sobol draws.

The drivers do not redistribute the upstream solver or the thermodynamic
database.

## Workflow

All examples use a campaign directory `/campaign` and must be run from this
`data/generation/` directory or with the corresponding script path.

### Plume1

```bash
# 1. Training-pool parameter table (100 LHS draws; T0-T96 enter the release)
python plume1/00_lhs_sample.py --outdir /campaign/lhs_train --n 100

# 2. Create cases, edit init.t3c, write batch scripts
python plume1/01_in2fast_pal.py \
  --solver-dir /path/to/patched-i2vis \
  --workdir /campaign/train \
  --params /campaign/lhs_train/lhs_params.csv \
  --indices 0-99 --prefix T \
  --singularity-image /path/to/ubuntu_intel.sif \
  --singularity-bind /path:/path --partition cp6 --batch-size 7

# 3. Submit seven-case i2fast batches
python plume1/02_i2fast_pal.py --workdir /campaign/train

# 4. Post-process each case with i2jslab
python plume1/03_i2j_pal.py --workdir /campaign/train --prefix T --indices 0-99

# 5. Convert PRNs to HDF5 (T0-T96: 97 frames, last decade 930)
python plume1/04_prn2h5.py --workdir /campaign/train --prefix T \
  --indices 0-96 --last-decade 930

# Test set: a separate 12-point LHS draw, same seed, prefix T_test
python plume1/00_lhs_sample.py --outdir /campaign/lhs_test --n 12
python plume1/01_in2fast_pal.py \
  --solver-dir /path/to/patched-i2vis --workdir /campaign/test \
  --params /campaign/lhs_test/lhs_params.csv \
  --indices 0-11 --prefix T_test \
  --singularity-image /path/to/ubuntu_intel.sif \
  --singularity-bind /path:/path --partition cp6 --batch-size 7
# ... 02/03 as above with --prefix T_test ...
python plume1/04_prn2h5.py --workdir /campaign/test --prefix T_test \
  --indices 0-11 --last-decade 1270
```

### Plume2

```bash
python plume2/00_sobol_sample.py --outdir /campaign/params --n 128
python plume2/01_in2fast_pal.py \
  --solver-dir /path/to/patched-i2vis --workdir /campaign/run \
  --params /campaign/params/physics_params.csv \
  --indices 0-127 --prefix run \
  --singularity-image /path/to/ubuntu_intel.sif \
  --singularity-bind /path:/path --partition cp6 --batch-size 7
python plume2/02_i2fast_pal.py --workdir /campaign/run
python plume2/03_i2j_pal.py --workdir /campaign/run --prefix run --indices 0-127
python plume2/04_prn2h5.py --workdir /campaign/run --prefix run \
  --indices 0-127 --last-decade 1270
```

`04_prn2h5.py` builds the HDF5 time axis from `plume0`–`plume3` plus
`plume10`–`plume<last-decade>`.  With `--last-decade 930` this is 97 frames
(the released Plume1 training pool); with `1270` it is 131 frames (Plume1
test and Plume2).

## Notes

- `01_in2fast_pal.py` copies the solver tree into each case directory and
  edits `init.t3c` in place.  Use `--no-copy` only when the case files
  already exist.
- `--submit-in2fast` optionally submits `sing_run_in2fast.sh` while
  preparing cases; `02_i2fast_pal.py` submits the seven-case batches.
- The batch scripts use `OMP_NUM_THREADS=4` and `yhbatch`; change
  `--partition` and `--submit-cmd` for another SLURM site.
- `03_i2j_pal.py` and `02_i2fast_pal.py` submit jobs; use `--dry-run` to
  inspect the commands first.
- The original research scripts used `/fs2` paths and a site-specific
  Singularity image.  Those paths are intentionally absent here.
