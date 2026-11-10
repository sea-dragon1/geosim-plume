# I2VIS plume patches

This package contains **only the modifications** made to the public I2VIS
code for the GeoSim mantle-plume simulations. It is **not** a complete
I2VIS distribution. The unmodified upstream files must be obtained from
the original OSF release.

Two source branches are provided because the two GeoSim subsets were
generated with slightly different modified copies:

| Subset | Patch | Configuration | Build |
|---|---|---|---|
| Plume1 (LHS, 109 cases) | `patch/i2vis-plume1.patch` | `config/plume1/` | `-O3 -xCORE-AVX512` |
| Plume2 (Sobol, 128 cases) | `patch/i2vis-plume2.patch` | `config/plume2/` | `-O2 -qopenmp` |

## Upstream code

- Gerya, T. (2021), *I2VIS code*, OSF:
  <https://doi.org/10.17605/OSF.IO/BNVTH>
- OSF project title: *Slab segmentation*
- Base archive: `codes.zip` from the OSF record, extracted as
  `codes/codes/`

Both patches were generated against the upstream `codes/codes/` tree.
The expected hashes of the 14 required base files are in
`UPSTREAM.sha256`; do not apply a patch to a different I2VIS revision
without checking those hashes.

## What is included

```
solver_patch/
├── patch/
│   ├── i2vis-plume1.patch
│   └── i2vis-plume2.patch
├── config/
│   ├── plume1/{init.t3c,mode.t3c,Makefile}
│   └── plume2/{init.t3c,mode.t3c,Makefile}
├── scripts/apply_patch.sh
├── workflow/README.md
├── database/README.md
├── UPSTREAM.sha256
├── PATCHED.plume1.sha256
├── PATCHED.plume2.sha256
├── UPSTREAM.md
├── NOTICE.md
└── MANIFEST.sha256
```

### Modified C files

Plume1 changes eight files:

`gausfast.c`, `headfast.c`, `headjslab.c`, `in2fast.c`,
`loadjslab.c`, `loadslab.c`, `markslab.c`, `movefast.c`

Plume2 changes the same eight files plus `moveslab.c` (extra diagnostic
prints).

The unmodified upstream files, required by the build, are:

`gausslab.c`, `heatfast.c`, `heatslab.c`, `i2fast.c`, `i2jslab.c`

They are intentionally not copied here and are verified by
`UPSTREAM.sha256`.

## Quick start

```bash
# 1. Obtain the upstream code from OSF and extract codes.zip.
#    You need the directory containing gausfast.c, i2fast.c, etc.
UPSTREAM=/path/to/codes/codes

# 2. Apply one branch (recommended).
bash scripts/apply_patch.sh plume1 "$UPSTREAM" i2vis-plume1
# or:
bash scripts/apply_patch.sh plume2 "$UPSTREAM" i2vis-plume2

# 3. Provide the thermodynamic database files (see database/README.md).
cd i2vis-plume1   # or i2vis-plume2
mkdir -p database
# copy pdry_*, pwet_*, ... into database/

# 4. Build.
make

# 5. Run one case.
./in2fast        # writes plume0.prn
./i2fast         # writes plume01.prn ... plume1300.prn
./i2jslab       # post-processing / conversion step
```

Do not run `apply_patch.sh` into an existing directory; it refuses to
overwrite. Manual application is also possible:

```bash
cd /path/to/codes/codes
sha256sum -c /path/to/geosim-plume/solver_patch/UPSTREAM.sha256
patch -p1 < /path/to/geosim-plume/solver_patch/patch/i2vis-plume1.patch
cp /path/to/geosim-plume/solver_patch/config/plume1/{Makefile,init.t3c,mode.t3c} .
```

## Build notes

The two supplied Makefiles differ:

```make
# Plume1
CFLAGS  = -O3 -xCORE-AVX512 -mcmodel=medium
LDFLAGS = -qmkl -lpthread -qopt-zmm-usage=high

# Plume2
CFLAGS  = -O2 -qopenmp -mcmodel=medium
LDFLAGS = -qmkl -lpthread -lm -Wl,--no-relax
```

Both use Intel `icx`, Intel MKL, OpenMP, and `-mcmodel=medium`.
The C changes are independent of the optimization flags, but the source
copy and patch must match the intended branch.

## Run configuration

`config/plume1/` and `config/plume2/` contain the corresponding
`init.t3c`, `mode.t3c`, and `Makefile`:

- domain: 1150 km × 660 km
- grid: 401 × 309
- 5 × 5 Lagrangian markers per cell
- 1,300 output records and 14,500 solver cycles
- output records reuse file names; 137 unique `.prn` files are retained

The configuration files contain no absolute paths. The thermodynamic
database is expected under `database/` relative to the run directory.

## Data-generation drivers

The Python scripts that create case directories, rewrite `init.t3c`,
submit `in2fast`/`i2fast` batches, and convert PRN to HDF5 are maintained
with the GeoSim data-generation workflow, not this solver patch. They are
intentionally excluded here so that this package contains only the I2VIS
modifications. `workflow/README.md` records this boundary.

## Citation

If you use these patches, cite the upstream I2VIS code and the GeoSim
paper:

- Gerya, T. (2021). *I2VIS code*. OSF.
  <https://doi.org/10.17605/OSF.IO/BNVTH>
- The GeoSim benchmark paper (see the accompanying manuscript for the
  final citation).

## Important boundary

This package does **not** redistribute the upstream I2VIS release. It
contains patches and case configurations derived from modified I2VIS
copies, included with the permission of the I2VIS author. Upstream code
must be obtained from OSF and used under its terms. See `NOTICE.md`.
