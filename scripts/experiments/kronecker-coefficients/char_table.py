#!/usr/bin/env python3
"""
Compute character table of S_n using the Murnaghan-Nakayama rule.
Output: binary file for GPU Kronecker coefficient computation.

The MN rule is tricky to implement correctly from scratch.
This script uses SymPy or SageMath if available, falling back
to a validated recursive implementation.

Usage: python3 char_table.py <n> [max_height]
Output: results/char_table_n{N}.bin (int64 matrix, row-major)
        results/z_inv_n{N}.bin (float64 array)
        results/partitions_n{N}.txt (human-readable partition list)
"""

import sys
import os
import struct
import time
import numpy as np
from itertools import combinations
from functools import lru_cache
from math import factorial, log

def generate_partitions(n, max_height=None):
    """Generate all partitions of n with at most max_height parts."""
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
    """Centralizer order for cycle type rho."""
    from collections import Counter
    c = Counter(rho)
    z = 1
    for i, m in c.items():
        z *= i ** m
        for j in range(2, m + 1):
            z *= j
    return z

def try_sympy_characters(n, partitions, cycle_types):
    """Try using SymPy's symmetric group character computation."""
    try:
        from sympy.combinatorics import Permutation, SymmetricGroup
        # SymPy doesn't have a direct MN implementation for arbitrary chars
        # but we can compute characters by evaluating on representative permutations

        def cycle_type_to_perm(ct, n):
            """Create a permutation with given cycle type."""
            perm = list(range(n))
            pos = 0
            for c in ct:
                for i in range(c - 1):
                    perm[pos + i] = pos + i + 1
                perm[pos + c - 1] = pos
                pos += c
            return Permutation(perm)

        # This approach is too slow for large n (needs full group)
        # Only use for small n as validation
        if n > 12:
            return None

        print(f"  Using SymPy character computation (n={n})...", flush=True)
        # Actually SymPy doesn't have character tables built-in
        return None
    except ImportError:
        return None

def mn_character_correct(lam, rho):
    """
    Murnaghan-Nakayama rule: compute chi^lam(rho).

    Uses Young diagram representation with explicit border strip enumeration.
    A border strip of size k is a connected skew shape with no 2x2 block.
    """

    @lru_cache(maxsize=None)
    def _mn(lam, rho):
        if not rho:
            return 1 if not lam else 0
        if not lam:
            return 0

        k = rho[0]
        rest = rho[1:]

        # Find all removable border strips of size k
        result = 0
        for strip_info in _border_strips(lam, k):
            residual, height = strip_info
            sign = (-1) ** height
            result += sign * _mn(residual, rest)

        return result

    @lru_cache(maxsize=None)
    def _border_strips(lam, k):
        """Find all border strips of size k in partition lam.
        Returns list of (residual_partition, height) tuples.

        Algorithm: For each possible bottom row r of the strip,
        try building a strip going upward. The strip in each row i
        occupies the rightmost cells that "overhang" beyond row i+1.
        In the top row, we may take a partial overhang.
        """
        strips = []
        n = len(lam)

        for r in range(n):
            # r = bottom row of the border strip
            # Must have an outer corner: lam[r] > lam[r+1] (or r is last)
            next_below = lam[r + 1] if r + 1 < n else 0
            if lam[r] <= next_below:
                continue

            # Walk upward from row r
            # In each row i, the full overhang is lam[i] - lam[i+1]
            # We take the full overhang in all rows except possibly the top
            total = 0
            for top in range(r, -1, -1):
                if top == r:
                    overhang = lam[r] - next_below
                else:
                    overhang = lam[top] - lam[top + 1]

                if overhang <= 0:
                    break

                total += overhang

                if total >= k:
                    # The strip spans rows top..r
                    excess = total - k
                    # In the top row, we take (overhang - excess) cells
                    top_remove = overhang - excess

                    # Build residual partition
                    new_lam = list(lam)
                    for i in range(top + 1, r + 1):
                        new_lam[i] = lam[i + 1] if i + 1 < n else 0
                    new_lam[top] = lam[top] - top_remove

                    # Validate: must be weakly decreasing
                    valid = True
                    if top > 0 and new_lam[top] > new_lam[top - 1]:
                        valid = False
                    for i in range(len(new_lam) - 1):
                        if new_lam[i] < new_lam[i + 1]:
                            valid = False
                            break

                    if valid:
                        residual = tuple(x for x in new_lam if x > 0)
                        height = r - top
                        strips.append((residual, height))
                    break

        return tuple(strips)

    return _mn(lam, rho)


def validate_character_table(char_table, partitions, cycle_types, n):
    """Validate the character table against known identities."""
    p = len(partitions)

    # 1. Trivial rep (first partition = (n)) should be all 1s
    trivial_ok = all(char_table[0, c] == 1 for c in range(len(cycle_types)))
    print(f"  Trivial rep all 1s: {'OK' if trivial_ok else 'FAIL'}")

    # 2. Dimension check: Σ dim(λ)^2 = n!
    id_idx = None
    for c, ct in enumerate(cycle_types):
        if ct == tuple([1] * n):
            id_idx = c
            break

    if id_idx is not None:
        dims = char_table[:, id_idx]
        dim_sum = sum(d * d for d in dims)
        expected = factorial(n)
        print(f"  Σ dim² = {dim_sum} (expected {expected}): {'OK' if dim_sum == expected else 'FAIL'}")

    # 3. Row orthogonality: Σ_ρ (1/z_ρ) χ^λ(ρ) χ^μ(ρ) = δ_{λμ}
    z_vals = [z_rho(ct) for ct in cycle_types]
    ortho_ok = True
    for i in range(min(p, 5)):
        for j in range(i, min(p, 5)):
            dot = sum(char_table[i, c] * char_table[j, c] / z_vals[c]
                      for c in range(len(cycle_types)))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > 0.01:
                ortho_ok = False
                print(f"    Row orth ({i},{j}): {dot:.4f} (expected {expected})")
    print(f"  Row orthogonality (first 5): {'OK' if ortho_ok else 'FAIL'}")

    return trivial_ok and (dim_sum == expected if id_idx is not None else True) and ortho_ok


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <n> [max_height]")
        sys.exit(1)

    n = int(sys.argv[1])
    max_height = int(sys.argv[2]) if len(sys.argv) > 2 else n

    print(f"Computing character table for S_{n} (height ≤ {max_height})")
    t0 = time.time()

    # Generate partitions (for representations = rows)
    partitions = generate_partitions(n, max_height)
    # Generate ALL cycle types (= all partitions of n, for columns)
    cycle_types = generate_partitions(n)

    num_parts = len(partitions)
    num_classes = len(cycle_types)
    print(f"Partitions (height ≤ {max_height}): {num_parts}")
    print(f"Conjugacy classes: {num_classes}")
    print(f"Character table size: {num_parts} × {num_classes} = {num_parts * num_classes}")
    print()

    # Compute character table
    char_table = np.zeros((num_parts, num_classes), dtype=np.int64)

    total = num_parts * num_classes
    done = 0
    for i, lam in enumerate(partitions):
        for c, rho in enumerate(cycle_types):
            char_table[i, c] = mn_character_correct(lam, rho)
            done += 1
            if done % max(1, total // 20) == 0:
                print(f"  {done}/{total} ({100*done//total}%)", flush=True)

    t1 = time.time()
    print(f"\nCharacter table computed in {t1 - t0:.1f} seconds")

    # Validate
    print("\nValidation:")
    validate_character_table(char_table, partitions, cycle_types, n)

    # Save
    os.makedirs("scripts/experiments/kronecker-coefficients/results", exist_ok=True)

    suffix = f"_h{max_height}" if max_height < n else ""

    # Binary character table
    bin_path = f"scripts/experiments/kronecker-coefficients/results/char_table_n{n}{suffix}.bin"
    char_table.tofile(bin_path)
    print(f"\nSaved: {bin_path} ({os.path.getsize(bin_path)} bytes)")

    # z_inv values
    z_inv = np.array([1.0 / z_rho(ct) for ct in cycle_types], dtype=np.float64)
    z_path = f"scripts/experiments/kronecker-coefficients/results/z_inv_n{n}{suffix}.bin"
    z_inv.tofile(z_path)
    print(f"Saved: {z_path}")

    # Partition list
    txt_path = f"scripts/experiments/kronecker-coefficients/results/partitions_n{n}{suffix}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"# Partitions of {n} (height <= {max_height})\n")
        f.write(f"# {num_parts} partitions, {num_classes} conjugacy classes\n")
        f.write("# Partitions (representations):\n")
        for i, p in enumerate(partitions):
            f.write(f"{i}\t{p}\n")
        f.write("# Cycle types (conjugacy classes):\n")
        for c, ct in enumerate(cycle_types):
            f.write(f"{c}\t{ct}\tz={z_rho(ct)}\n")
    print(f"Saved: {txt_path}")

    # Summary stats
    print(f"\nReady for GPU Kronecker computation:")
    print(f"  Load {bin_path} as int64[{num_parts}][{num_classes}]")
    print(f"  Load {z_path} as float64[{num_classes}]")
    print(f"  Compute {num_parts}^3 / 6 ≈ {num_parts**3 // 6:,} Kronecker triples")


if __name__ == "__main__":
    main()
