#!/usr/bin/env python3
# coding=utf-8
"""Submit ``sing_run_i2j.sh`` for each prepared case."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_indices, submit_script  # noqa: E402


def main(default_subset=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=("plume1", "plume2"), default=default_subset)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--indices", default="0-")
    ap.add_argument("--submit-cmd", default="yhbatch")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    prefix = a.prefix or ("T" if a.subset == "plume1" else "run")
    workdir = Path(a.workdir)
    # ``0-`` means all case directories present.
    n = len([p for p in workdir.glob("{}*".format(prefix)) if p.is_dir()])
    indices = parse_indices(a.indices, n=n)
    for i in indices:
        case = workdir / "{}{}".format(prefix, i)
        script = case / "sing_run_i2j.sh"
        if not script.exists():
            print("skip missing", script)
            continue
        print("SUBMIT", script)
        submit_script(script, submit_cmd=a.submit_cmd, dry_run=a.dry_run, cwd=case)
    print("submitted i2j jobs:", len(indices))


if __name__ == "__main__":
    main()
