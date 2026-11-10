#!/usr/bin/env python3
# coding=utf-8
"""Plume2 wrapper for the shared case-preparation driver."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_cases

if __name__ == "__main__":
    prepare_cases.main(default_subset="plume2")
