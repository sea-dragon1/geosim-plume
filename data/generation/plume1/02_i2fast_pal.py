#!/usr/bin/env python3
# coding=utf-8
"""Plume1 batch-submit wrapper."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submit_batches

if __name__ == "__main__":
    submit_batches.main()
