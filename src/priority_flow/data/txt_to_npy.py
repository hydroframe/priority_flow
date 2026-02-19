#!/usr/bin/env python3
"""
Convert DEM, watershed_mask, and river_mask from .txt to .npy in this directory.
Run from this directory or from the project root; paths are relative to this script.
The .txt files were created by calling write.table() in R from the .RData files in the
PriorityFlow/Rpkg/data/ directory.
"""

import numpy as np
from pathlib import Path

# Directory where this script lives (and where the txt/npy files are)
SCRIPT_DIR = Path(__file__).resolve().parent

FILES = [
    ("DEM.txt", "DEM.npy"),
    ("watershed_mask.txt", "watershed_mask.npy"),
    ("river_mask.txt", "river_mask.npy"),
]


def main():
    for txt_name, npy_name in FILES:
        txt_path = SCRIPT_DIR / txt_name
        npy_path = SCRIPT_DIR / npy_name
        if not txt_path.exists():
            print(f"Skipping {txt_name}: file not found")
            continue
        arr = np.loadtxt(txt_path)
        np.save(npy_path, arr)
        print(f"Saved {arr.shape} -> {npy_name}")


if __name__ == "__main__":
    main()
