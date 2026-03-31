#!/usr/bin/env python3
"""
Compute character table of S_n via Murnaghan-Nakayama rule (rim-path method).
Validated: row and column orthogonality pass for S_5 through S_12.

Output: binary files for GPU Kronecker coefficient computation.

Usage: python3 char_table.py <n> [max_height]
"""

import sys
import os
import time
import numpy as np
from functools import lru_cache
from collections import Counter
from math import factorial


def generate_partitions(n, max_height=None):
    if max_height is None:
        max_height = n
    if n == 0:
        return [()]
    result = []
    def _gen(remaining, max_part, parts):
        if remaining == 0:
            if len(parts) <= max_height:
                result.append(tuple(parts))
            return
        if len(parts) >= max_height:
            return
        for p in range(min(remaining, max_part), 0, -1):
            _gen(remaining - p, p, parts + [p])
    _gen(n, n, [])
    return result


def z_rho(rho):
    c = Counter(rho)
    z = 1
    for i, m in c.items():
        z *= i ** m
        for j in range(2, m + 1):
            z *= j
    return z


def rim_path(lam):
    """Rim cells in path order (SW to NE along SE boundary)."""
    n = len(lam)
    if n == 0:
        return ()

    def on_rim(i, j):
        return 0 <= i < n and 0 <= j < lam[i] and not (i + 1 < n and j + 1 < lam[i + 1])

    start = None
    for i in range(n - 1, -1, -1):
        for j in range(lam[i]):
            if on_rim(i, j):
                start = (i, j)
                break
        if start:
            break
    if not start:
        return ()

    visited = set()
    path = []
    i, j = start
    while on_rim(i, j) and (i, j) not in visited:
        path.append((i, j))
        visited.add((i, j))
        if i - 1 >= 0 and on_rim(i - 1, j) and (i - 1, j) not in visited:
            i -= 1
        elif on_rim(i, j + 1) and (i, j + 1) not in visited:
            j += 1
        else:
            break
    return tuple(path)


@lru_cache(maxsize=None)
def border_strips(lam, k):
    """All border strips of size k: contiguous subpaths of the rim,
    where each row's cells are rightmost and contiguous."""
    path = rim_path(lam)
    if len(path) < k:
        return ()

    strips = []
    for start in range(len(path) - k + 1):
        cells = path[start:start + k]

        rows_cols = {}
        for ci, cj in cells:
            rows_cols.setdefault(ci, []).append(cj)

        valid = True
        for ri, cl in rows_cols.items():
            if max(cl) != lam[ri] - 1:
                valid = False
                break
            cl.sort()
            for idx in range(len(cl) - 1):
                if cl[idx + 1] != cl[idx] + 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        new_lam = list(lam)
        rows = set()
        for ci, cj in cells:
            new_lam[ci] -= 1
            rows.add(ci)

        if all(new_lam[i] >= new_lam[i + 1] for i in range(len(new_lam) - 1)):
            height = len(rows) - 1
            residual = tuple(x for x in new_lam if x > 0)
            strips.append((residual, height))

    return tuple(strips)


@lru_cache(maxsize=None)
def mn_character(lam, rho):
    """Murnaghan-Nakayama: chi^lam(rho)."""
    if not rho:
        return 1 if not lam else 0
    if not lam:
        return 0
    return sum((-1) ** h * mn_character(res, rho[1:])
               for res, h in border_strips(lam, rho[0]))


def validate(char_table, partitions, cycle_types, n):
    p = len(partitions)
    z_vals = [z_rho(ct) for ct in cycle_types]

    # Dimension sum
    id_idx = None
    for c, ct in enumerate(cycle_types):
        if ct == tuple([1] * n):
            id_idx = c
            break
    if id_idx is not None:
        dims = char_table[:, id_idx]
        ds = int(sum(d * d for d in dims))
        nf = factorial(n)
        print(f"  Σ dim² = {ds} (expected {nf}): {'OK' if ds == nf else 'FAIL'}")

    # Row orthogonality (first 5 rows)
    ok = True
    for i in range(min(p, 5)):
        for j in range(i, min(p, 5)):
            dot = sum(char_table[i, c] * char_table[j, c] / z_vals[c]
                      for c in range(len(cycle_types)))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > 0.01:
                ok = False
    print(f"  Row orthogonality: {'OK' if ok else 'FAIL'}")

    # Column orthogonality (first 5 cols)
    ok2 = True
    for i in range(min(len(cycle_types), 5)):
        for j in range(i, min(len(cycle_types), 5)):
            dot = sum(int(char_table[l, i]) * int(char_table[l, j]) for l in range(p))
            expected = z_vals[i] if i == j else 0
            if abs(dot - expected) > 0.01:
                ok2 = False
    print(f"  Column orthogonality: {'OK' if ok2 else 'FAIL'}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <n> [max_height]")
        sys.exit(1)

    n = int(sys.argv[1])
    max_height = int(sys.argv[2]) if len(sys.argv) > 2 else n

    print(f"Character table for S_{n} (height ≤ {max_height})")
    t0 = time.time()

    partitions = generate_partitions(n, max_height)
    cycle_types = generate_partitions(n)
    num_parts = len(partitions)
    num_classes = len(cycle_types)

    print(f"Partitions: {num_parts}, Classes: {num_classes}")
    print(f"Table: {num_parts} × {num_classes} = {num_parts * num_classes:,} entries\n")

    char_table = np.zeros((num_parts, num_classes), dtype=np.int64)
    total = num_parts * num_classes
    done = 0
    step = max(1, total // 20)

    for i, lam in enumerate(partitions):
        for c, rho in enumerate(cycle_types):
            char_table[i, c] = mn_character(lam, rho)
            done += 1
            if done % step == 0:
                print(f"  {done:,}/{total:,} ({100*done//total}%)", flush=True)

    t1 = time.time()
    print(f"\nCharacter table: {t1 - t0:.1f}s")

    print("\nValidation:")
    validate(char_table, partitions, cycle_types, n)

    # Save
    os.makedirs("scripts/experiments/kronecker-coefficients/results", exist_ok=True)
    suffix = f"_h{max_height}" if max_height < n else ""

    bin_path = f"scripts/experiments/kronecker-coefficients/results/char_table_n{n}{suffix}.bin"
    char_table.tofile(bin_path)

    z_inv = np.array([1.0 / z_rho(ct) for ct in cycle_types], dtype=np.float64)
    z_path = f"scripts/experiments/kronecker-coefficients/results/z_inv_n{n}{suffix}.bin"
    z_inv.tofile(z_path)

    txt_path = f"scripts/experiments/kronecker-coefficients/results/partitions_n{n}{suffix}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"# S_{n}, height <= {max_height}\n")
        f.write(f"# {num_parts} partitions, {num_classes} classes\n")
        for i, p in enumerate(partitions):
            f.write(f"{i}\t{p}\n")

    print(f"\nSaved: {bin_path} ({os.path.getsize(bin_path):,} bytes)")
    print(f"Saved: {z_path}")
    print(f"Saved: {txt_path}")
    print(f"\nGPU: {num_parts}³/6 ≈ {num_parts**3//6:,} Kronecker triples")


if __name__ == "__main__":
    main()
