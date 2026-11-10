#!/usr/bin/env python3
# coding=utf-8
"""Plume2 PRN→HDF5 wrapper (prefix run, 131 frames)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prn2h5

if __name__ == "__main__":
    prn2h5.main()
