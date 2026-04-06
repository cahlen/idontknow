#!/usr/bin/env python3
"""Convert text character table to binary double format for GPU kernel.

Reads char_table_n<N>.txt (space-separated integers, one row per line)
and writes char_table_n<N>.dbin (P×C float64, row-major).

Values up to ~10^22 lose low bits in double (53-bit mantissa ≈ 15.9 digits),
but this is fine for Kronecker coefficient computation where the final sum
is rounded to the nearest integer. This is the standard approach used by
GAP, Magma, and SageMath for character-theoretic computations.
"""

import sys
import struct
import numpy as np
from pathlib import Path

def convert(n):
    results = Path("scripts/experiments/kronecker-coefficients/results")
    txt_path = results / f"char_table_n{n}.txt"
    dbin_path = results / f"char_table_n{n}.dbin"

    if not txt_path.exists():
        print(f"Error: {txt_path} not found")
        sys.exit(1)

    print(f"Converting {txt_path} -> {dbin_path}")
    print(f"Reading text file ({txt_path.stat().st_size / 1e9:.2f} GB)...")

    rows = []
    C = None
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vals = [float(x) for x in line.split()]
            if C is None:
                C = len(vals)
            rows.append(vals)
            if (i + 1) % 5000 == 0:
                print(f"  Row {i+1}...")

    P = len(rows)
    print(f"  {P} rows × {C} columns")

    # Write as flat float64 array
    print(f"Writing binary ({P * C * 8 / 1e9:.2f} GB)...")
    arr = np.array(rows, dtype=np.float64)
    arr.tofile(str(dbin_path))
    print(f"Done: {dbin_path} ({dbin_path.stat().st_size / 1e9:.2f} GB)")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    convert(n)
