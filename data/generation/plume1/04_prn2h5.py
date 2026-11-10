#!/usr/bin/env python3
# coding=utf-8
"""Plume1 PRN→HDF5 wrapper.

Example:
  python 04_prn2h5.py --workdir /campaign --prefix T --indices 0-96 \
      --last-decade 930
  python 04_prn2h5.py --workdir /campaign --prefix T_test --indices 0-11 \
      --last-decade 1270
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prn2h5

if __name__ == "__main__":
    prn2h5.main()
