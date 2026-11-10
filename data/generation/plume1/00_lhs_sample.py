#!/usr/bin/env python3
# coding=utf-8
"""Generate the Plume1 Latin-hypercube temperature table.

The released Plume1 design is ``LatinHypercube(d=1, seed=42)`` mapped
affinely to 1600--1800 K, rounded to two decimals.  The training pool was
drawn with 100 points; the published 97-case training pool excludes T97 and
the two unfinished cases T98/T99.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=".", help="directory for lhs_params.csv")
    ap.add_argument("--n", type=int, default=100,
                    help="number of LHS draws (default 100; release uses 0-96)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lo", type=float, default=1600.0)
    ap.add_argument("--hi", type=float, default=1800.0)
    a = ap.parse_args()

    sampler = qmc.LatinHypercube(d=1, seed=a.seed)
    samples = sampler.random(n=a.n)
    values = np.round(a.lo + (a.hi - a.lo) * samples.flatten(), 2)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "lhs_params.csv"
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["Sample_ID", "T_plume"])
        for i, t in enumerate(values):
            w.writerow([i, "{:.2f}".format(t)])
    print("wrote", path, "n =", a.n)


if __name__ == "__main__":
    main()
