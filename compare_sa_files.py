#!/usr/bin/env python3
"""
Compare two ParFlow .sa (ASCII) files point-by-point.

Usage:
  python compare_sa_files.py <file1.sa> <file2.sa> [--tolerance 1e-9]
  python compare_sa_files.py SlopeX.sa SlopeX_ref.sa

.sa format: first line "nx ny 1", then nx*ny lines with one value each (column-major order).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_parflow_sa(filepath: str) -> np.ndarray:
    """Read a ParFlow .sa file; return 2D array of shape (nx, ny)."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(path) as f:
        line1 = f.readline().strip().split()
        if len(line1) < 2:
            raise ValueError(f"Invalid header in {filepath}: expected 'nx ny [nz]'")
        nx, ny = int(line1[0]), int(line1[1])
        values = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(float(line))
    n = nx * ny
    if len(values) != n:
        raise ValueError(
            f"File {filepath}: expected {n} values (nx*ny={nx}*{ny}), got {len(values)}"
        )
    # Column-major order (same as write_parflow_ascii: j varies slow, i fast)
    arr = np.array(values, dtype=np.float64).reshape((nx, ny), order="F")
    return arr


def compare_sa_files(
    path1: str,
    path2: str,
    tolerance: float = 1e-9,
    relative_tolerance: float = 0.0,
) -> bool:
    """
    Compare two .sa files point-by-point.
    Returns True if they match within tolerance, False otherwise.
    Prints a short summary.
    """
    a = read_parflow_sa(path1)
    b = read_parflow_sa(path2)

    if a.shape != b.shape:
        print(f"Shape mismatch: {path1} {a.shape} vs {path2} {b.shape}")
        return False

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    abs_ok = diff <= tolerance

    if relative_tolerance > 0:
        denom = np.maximum(np.abs(a), np.abs(b))
        np.place(denom, denom == 0, 1.0)
        rel_diff = diff / denom
        rel_ok = rel_diff <= relative_tolerance
        ok = abs_ok & rel_ok
    else:
        ok = abs_ok

    n_total = a.size
    n_diff = int(np.sum(~ok))
    max_abs_diff = float(np.max(diff))
    if n_diff > 0:
        max_diff_at = np.unravel_index(np.argmax(diff), a.shape)
        print(f"Files: {path1}  vs  {path2}")
        print(f"Shape: {a.shape}")
        print(f"Points differing (|a-b| > {tolerance}): {n_diff} / {n_total}")
        print(f"Max absolute difference: {max_abs_diff}")
        print(f"Max diff at index (i,j): {max_diff_at}  values: {a[max_diff_at]:.6g} vs {b[max_diff_at]:.6g}")
        return False
    else:
        print(f"Files match (max |diff| = {max_abs_diff:.3e} <= {tolerance})")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two ParFlow .sa files point-by-point."
    )
    parser.add_argument("file1", help="First .sa file")
    parser.add_argument("file2", help="Second .sa file")
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-9,
        help="Absolute tolerance for equality (default: 1e-9)",
    )
    parser.add_argument(
        "--relative",
        "-r",
        type=float,
        default=0.0,
        help="Relative tolerance (optional; 0 = not used)",
    )
    args = parser.parse_args()

    try:
        ok = compare_sa_files(
            args.file1,
            args.file2,
            tolerance=args.tolerance,
            relative_tolerance=args.relative,
        )
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
