# coding=utf-8
"""Shared helpers for the I2VIS data-generation drivers.

The drivers are deliberately path-agnostic: the solver tree, campaign
directory, parameter table, container image and SLURM partition are all
supplied on the command line.  They do not contain author-machine paths.
"""
from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def parse_indices(spec, n=None):
    """Parse ``0-9,12,20-`` into a sorted list of integer indices."""
    if spec is None:
        if n is None:
            raise ValueError("either indices or n is required")
        return list(range(n))
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 0
            end = int(b) if b else (n - 1 if n is not None else None)
            if end is None:
                raise ValueError("open-ended index range needs a case count")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return sorted(out)


def read_params(path, subset):
    """Read a parameter CSV keyed by ``Sample_ID``.

    plume1 columns: ``T_plume`` (K)
    plume2 columns: ``T_anomaly`` (K), ``Mantle_H`` (km)
    """
    rows = {}
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            idx = int(row["Sample_ID"])
            if subset == "plume1":
                rows[idx] = {"T_plume": float(row["T_plume"])}
            elif subset == "plume2":
                rows[idx] = {
                    "T_anomaly": float(row["T_anomaly"]),
                    "Mantle_H": float(row["Mantle_H"]),
                }
            else:
                raise ValueError(subset)
    return rows


def _replace_fields(line, values):
    parts = line.strip().split()
    for col, value in values.items():
        parts[col] = str(value)
    return "    ".join(parts) + "\n"


def edit_init(case_dir, subset, params):
    """Apply the published plume-temperature / lithosphere changes."""
    path = Path(case_dir) / "init.t3c"
    lines = path.read_text(encoding="utf8").splitlines(keepends=True)

    t = round(float(params["T_plume"] if subset == "plume1" else params["T_anomaly"]), 2)
    pipe = (240, {9: t, 10: t + 100, 11: t, 12: t + 100})
    bottom = (242, {9: t + 100, 10: t + 100, 11: t + 100, 12: t + 100})
    for line_no, values in (pipe, bottom):
        lines[line_no - 1] = _replace_fields(lines[line_no - 1], values)

    if subset == "plume2":
        h_km = float(params["Mantle_H"])
        bottom_depth = 60000.0 + (h_km - 60.0) * 1000.0
        top_depth = 60000.0
        n_layers = 8
        layer = (bottom_depth - top_depth) / n_layers
        boundaries = [top_depth + i * layer for i in range(n_layers + 1)]
        # Original I2VIS configuration: eight two-line layers starting at 187.
        for i in range(n_layers):
            line_idx = 187 + i * 2
            content = lines[line_idx].strip().split()
            content[3] = "m{:.1f}".format(boundaries[i])
            content[5] = "m{:.1f}".format(boundaries[i + 1])
            content[7] = "m{:.0f}".format(boundaries[i])
            content[9] = "m{:.1f}".format(boundaries[i + 1])
            lines[line_idx] = "  " + "      ".join(content) + "\n"
        parts = lines[236].strip().split()
        h = "m{:.1f}".format(bottom_depth)
        parts[4] = h
        parts[8] = h
        lines[236] = "  " + "      ".join(parts) + "\n"

    path.write_text("".join(lines), encoding="utf8")


def copy_solver_tree(solver_dir, case_dir,
                     skip_dirs=("database", "batch_run", "datah5", "results")):
    """Copy top-level solver files and link the thermodynamic database."""
    solver_dir = Path(solver_dir)
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(solver_dir.iterdir()):
        if item.name in skip_dirs:
            continue
        dst = case_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    db = solver_dir / "database"
    if db.exists():
        link = case_dir / "database"
        if link.is_symlink() or link.exists():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(db.resolve())


def write_in2fast_slurm(case_dir, image, bind, partition):
    text = """#!/bin/bash
#SBATCH -N 1
#SBATCH -p {partition}

export LC_ALL=C
module load singularity/3.11.0

singularity run --writable --bind {bind} {image} bash run_in2fast
""".format(partition=partition, bind=bind, image=image)
    (Path(case_dir) / "sing_run_in2fast.sh").write_text(text, encoding="utf8")


def write_i2j_slurm(case_dir, image, bind, partition):
    text = """#!/bin/bash
#SBATCH -N 1
#SBATCH -p {partition}

module load singularity/3.11.0

singularity run --writable --bind {bind} {image} ./i2jslab
""".format(partition=partition, bind=bind, image=image)
    (Path(case_dir) / "sing_run_i2j.sh").write_text(text, encoding="utf8")


def write_batch_scripts(workdir, cases, image, bind, partition, omp_threads, batch_id):
    """Write batch_run<id>.sh and batch_submit<id>.sh for a group of cases."""
    workdir = Path(workdir)
    batch_dir = workdir / "batch_run"
    batch_dir.mkdir(parents=True, exist_ok=True)
    run = batch_dir / "batch_run{}.sh".format(batch_id)
    lines = ["#!/bin/bash\n", "module list\n"]
    for case in cases:
        lines.append("cd {}\n".format(shlex.quote(str(case))))
        lines.append("export OMP_NUM_THREADS={}\n".format(int(omp_threads)))
        lines.append("nohup ./i2fast 2>&1 > {}.log &\n".format(case.name))
        lines.append("\n")
    lines.append("wait\n")
    run.write_text("".join(lines), encoding="utf8")
    submit = batch_dir / "batch_submit{}.sh".format(batch_id)
    submit.write_text(
        """#!/bin/bash
#SBATCH -N 1
#SBATCH -p {partition}

module load singularity/3.11.0
singularity run --writable --bind {bind} {image} bash batch_run{bid}.sh
""".format(partition=partition, bind=bind, image=image, bid=batch_id),
        encoding="utf8",
    )
    return submit


def submit_script(script, submit_cmd="yhbatch", dry_run=False, cwd=None):
    cmd = [submit_cmd, str(script)]
    if dry_run:
        print("DRY-RUN:", " ".join(shlex.quote(x) for x in cmd))
        return
    subprocess.run(cmd, check=True, cwd=cwd)
