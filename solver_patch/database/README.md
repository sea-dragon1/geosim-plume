# Thermodynamic database files

The modified `loadslab.c` and `loadjslab.c` open the thermodynamic tables
with a `database/` prefix. These files are **not** part of this patch and
are not redistributed here.

Expected names (relative to the run directory):

```text
pdry_ro  pdry_hh  pdry_vp  pdry_vs
pwet_ro  pwet_hh  pwet_vp  pwet_vs  pwet_wa
pmlt_ro  pmlt_hh  pmlt_vp  pmlt_vs  pmlt_wa
gwet_ro  gwet_hh  gwet_vp  gwet_vs  gwet_wa
gmlt_ro  gmlt_hh  gmlt_vp  gmlt_vs  gmlt_wa
swet_ro  swet_hh  swet_vp  swet_vs  swet_wa
smlt_ro  smlt_hh  smlt_vp  smlt_vs  smlt_wa
bwet_ro  bwet_hh  bwet_vp  bwet_vs  bwet_wa
bmlt_ro  bmlt_hh  bmlt_vp  bmlt_vs  bmlt_wa
m895_ro  m895_hh  m895_vp  m895_vs
morn_ro  morn_hh  morn_vp  morn_vs
```

`stop.yn` is created by the code at runtime.

Obtain these tables from the I2VIS thermodynamic database distribution
used for the original runs, or from the corresponding material-property
source. They are third-party data and are not covered by this patch.
