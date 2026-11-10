#!/usr/bin/env python3
# coding=utf-8
"""Submit the batch_run jobs created by ``01_in2fast_pal.py``."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--batches", default=None,
                    help="comma-separated batch ids, e.g. 6,13,20 (default: all)")
    ap.add_argument("--submit-cmd", default="yhbatch")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    batch_dir = Path(a.workdir) / "batch_run"
    files = sorted(batch_dir.glob("batch_submit*.sh"))
    if a.batches:
        wanted = {"batch_submit{}.sh".format(x.strip()) for x in a.batches.split(",")}
        files = [p for p in files if p.name in wanted]
    if not files:
        raise SystemExit("no batch_submit*.sh under {}".format(batch_dir))
    for p in files:
        cmd = [a.submit_cmd, p.name]
        if a.dry_run:
            print("DRY-RUN:", " ".join(cmd))
            continue
        print("SUBMIT", p.name)
        subprocess.run(cmd, check=True, cwd=batch_dir)
    print("processed", len(files), "batches", "(dry-run)" if a.dry_run else "")


if __name__ == "__main__":
    main()
