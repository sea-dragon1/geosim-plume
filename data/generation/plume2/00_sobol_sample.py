#!/usr/bin/env python3
# coding=utf-8
"""Generate the Plume2 ten-dimensional Sobol table.

Only the first two coordinates are active; the remaining eight dimensions
are reserved placeholders.  Values are mapped to the published intervals:
T_anomaly 2000--2150 K and Mantle_H 160--220 km.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = qmc.Sobol(d=10, scramble=True, seed=a.seed)
    samples = sampler.random(n=a.n)
    with open(out / "physics_params.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["Sample_ID", "T_anomaly", "Mantle_H"] +
                   ["Reserved_{}".format(i) for i in range(3, 11)])
        for i, row in enumerate(samples):
            t = float(np.round(2000.0 + row[0] * 150.0, 2))
            h = float(np.round(160.0 + row[1] * 60.0, 2))
            w.writerow([i, "{:.2f}".format(t), "{:.2f}".format(h)] +
                       ["{:.6f}".format(x) for x in row[2:]])
    print("wrote", out / "physics_params.csv", "n =", a.n)


if __name__ == "__main__":
    main()
