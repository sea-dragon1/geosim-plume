# Data-generation drivers

This directory is intentionally empty of runnable driver scripts.

The Python scripts that create case directories, rewrite `init.t3c`,
submit `in2fast`/`i2fast` batches, and convert PRN to HDF5 are maintained
with the **GeoSim data-generation workflow**, not the I2VIS solver patch. Keeping them
out of this package makes the boundary clear:

- this package changes the solver;
- the GeoSim package drives the benchmark workflow around the patched
  solver.

The drivers used for this study assume the patched executables
`in2fast`, `i2fast`, and `i2jslab` are present in each case directory,
and use the configuration files in `../config/`.
