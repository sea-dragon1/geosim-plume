#!/usr/bin/env python3
# coding=utf-8
"""Plume1 i2j wrapper."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_i2j

if __name__ == "__main__":
    run_i2j.main(default_subset="plume1")
