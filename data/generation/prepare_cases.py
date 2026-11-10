#!/usr/bin/env python3
# coding=utf-8
"""Create I2VIS case directories, edit init.t3c, and write batch scripts.

This is the release counterpart of the research drivers
``01_in2fast_pal.py`` (Plume1/Plume2).  It does not contain author-machine
paths; all roots are command-line arguments.

Example
-------
python 01_in2fast_pal.py \
  --solver-dir /path/to/patched-i2vis \
  --workdir /path/to/campaign \
  --params lhs_params.csv \
  --indices 0-99 --prefix T \
  --singularity-image /path/to/ubuntu_intel.sif \
  --singularity-bind /path:/path \
  --partition cp6 --batch-size 7
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    copy_solver_tree, edit_init, parse_indices, read_params,
    submit_script, write_batch_scripts, write_i2j_slurm, write_in2fast_slurm,
)


def main(default_subset=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=("plume1", "plume2"), default=default_subset)
    ap.add_argument("--solver-dir", required=True,
                    help="directory containing the compiled patched I2VIS tree")
    ap.add_argument("--workdir", required=True,
                    help="campaign directory; case folders are created here")
    ap.add_argument("--params", required=True,
                    help="parameter CSV from 00_lhs_sample.py / 00_sobol_sample.py")
    ap.add_argument("--indices", default=None,
                    help="case indices, e.g. 0-99 or 0,1,20-30 (default: all rows)")
    ap.add_argument("--prefix", default=None, help="case prefix (default: T/run)")
    ap.add_argument("--singularity-image", required=True)
    ap.add_argument("--singularity-bind", required=True,
                    help="bind specification passed to singularity, e.g. /fs2:/fs2")
    ap.add_argument("--partition", default="cp6")
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=7)
    ap.add_argument("--submit-in2fast", action="store_true",
                    help="submit sing_run_in2fast.sh after preparing each case")
    ap.add_argument("--no-copy", action="store_true",
                    help="do not copy the solver tree (case files must already exist)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit-cmd", default="yhbatch")
    a = ap.parse_args()

    if a.subset is None:
        ap.error("--subset is required")
    prefix = a.prefix or ("T" if a.subset == "plume1" else "run")
    params = read_params(a.params, a.subset)
    indices = parse_indices(a.indices, n=max(params) + 1)
    missing = [i for i in indices if i not in params]
    if missing:
        raise SystemExit("indices absent from parameter table: {}".format(missing[:10]))

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cases = []
    for i in indices:
        case = workdir / "{}{}".format(prefix, i)
        if case.exists() and not a.overwrite:
            print("skip existing", case)
            cases.append(case)
            continue
        cases.append(case)
        if not a.no_copy:
            if a.dry_run:
                print("DRY-RUN copy solver ->", case)
            else:
                copy_solver_tree(a.solver_dir, case)
        if a.dry_run:
            print("DRY-RUN edit", case / "init.t3c", params[i])
            continue
        edit_init(case, a.subset, params[i])
        write_in2fast_slurm(case, a.singularity_image, a.singularity_bind, a.partition)
        write_i2j_slurm(case, a.singularity_image, a.singularity_bind, a.partition)
        if a.submit_in2fast:
            submit_script(case / "sing_run_in2fast.sh", submit_cmd=a.submit_cmd,
                          dry_run=a.dry_run, cwd=case)

    # One batch per consecutive group; the last case index is the batch id.
    for start in range(0, len(cases), a.batch_size):
        group = cases[start:start + a.batch_size]
        if a.dry_run:
            print("DRY-RUN batch", group[0].name, "...", group[-1].name)
            continue
        submit = write_batch_scripts(
            workdir, group, a.singularity_image, a.singularity_bind,
            a.partition, a.omp_threads, group[-1].name.replace(prefix, "") or group[-1].name,
        )
        print("wrote", submit)
    print("prepared {} cases in {}".format(len(cases), workdir))


if __name__ == "__main__":
    main()
