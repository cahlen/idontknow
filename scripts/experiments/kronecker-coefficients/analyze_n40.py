#!/usr/bin/env python3
"""
Analyze S_40 character table and compute targeted Kronecker coefficients.

The full Kronecker computation (8.68T triples) requires int128 GPU arithmetic
and is a multi-day job. This script extracts scientifically interesting results
from the precomputed character table using exact Python arithmetic:

1. Character table statistics (value distribution, sparsity, dimensions)
2. Kronecker coefficients for all hook x hook x hook triples (40^3/6 ~ 10K)
3. Kronecker coefficients for near-rectangular partitions (GCT-relevant)
4. Random-sample estimate of overall nonzero fraction
5. Trend analysis across n = 5, 20, 30, 40

Usage: python3 analyze_n40.py
"""

import os
import sys
import time
import json
import random
import numpy as np
from fractions import Fraction
from collections import Counter
from math import factorial

RESULTS_DIR = "scripts/experiments/kronecker-coefficients/results"
OUTPUT_DIR = "scripts/experiments/kronecker-coefficients/results"


def load_partitions(n):
    """Load partition list and identify special partitions."""
    path = os.path.join(RESULTS_DIR, f"partitions_n{n}.txt")
    partitions = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) == 2:
                idx = int(parts[0])
                # Parse tuple string like "(40,)" or "(20, 20)"
                tup_str = parts[1].strip()
                tup = tuple(int(x.strip()) for x in tup_str.strip("()").split(",") if x.strip())
                partitions.append(tup)
    return partitions


def z_rho(rho):
    """Centralizer order z_ρ = Π_i i^{m_i} * m_i!"""
    c = Counter(rho)
    z = 1
    for i, m in c.items():
        z *= i ** m
        for j in range(2, m + 1):
            z *= j
    return z


def identify_hooks(partitions, n):
    """Hook partitions: (n-k, 1^k) for k=0,...,n-1."""
    hooks = {}
    for idx, p in enumerate(partitions):
        if len(p) == 1 and p[0] == n:
            hooks[0] = idx  # trivial
        elif len(p) >= 2 and p[0] == n - len(p) + 1 and all(p[i] == 1 for i in range(1, len(p))):
            k = len(p) - 1
            hooks[k] = idx
    return hooks


def identify_near_rectangles(partitions, n):
    """Near-rectangular partitions: (a^b) ± small perturbation."""
    results = {}
    for idx, p in enumerate(partitions):
        if len(set(p)) == 1:  # exact rectangle
            a, b = p[0], len(p)
            if a * b == n:
                results[f"({a}^{b})"] = idx
        # One-row perturbation: (a+1, a^{b-2}, a-1) etc.
        if len(p) >= 2:
            vals = set(p)
            if len(vals) == 2:
                v1, v2 = sorted(vals)
                if v2 - v1 == 1:
                    c1 = p.count(v1)
                    c2 = p.count(v2)
                    # Check if close to rectangle
                    avg = n / len(p)
                    if abs(avg - v1) <= 1.5 and abs(avg - v2) <= 1.5:
                        label = f"({v2}^{c2},{v1}^{c1})"
                        results[label] = idx
    return results


def load_char_rows(n, row_indices):
    """Load specific rows from the character table (memory efficient).
    Returns dict: row_index -> list of Python ints."""
    path = os.path.join(RESULTS_DIR, f"char_table_n{n}.txt")
    row_set = set(row_indices)
    rows = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in row_set:
                rows[i] = [int(x) for x in line.strip().split()]
                if len(rows) == len(row_set):
                    break
            if i % 5000 == 0 and i > 0:
                print(f"  Scanning row {i}/{37338}...", flush=True)
    return rows


def load_z_inv(n):
    """Load z_inv values (FP64 from binary file)."""
    path = os.path.join(RESULTS_DIR, f"z_inv_n{n}.bin")
    return np.fromfile(path, dtype=np.float64)


def kronecker_exact(chi_i, chi_j, chi_k, cycle_types):
    """Compute g(λ_i, λ_j, λ_k) using exact rational arithmetic.
    g = Σ_ρ (1/z_ρ) χ^λ_i(ρ) χ^μ_j(ρ) χ^ν_k(ρ)
    """
    total = Fraction(0)
    for c in range(len(cycle_types)):
        product = chi_i[c] * chi_j[c] * chi_k[c]
        if product != 0:
            total += Fraction(product, cycle_types[c])
    return int(total)


def char_table_stats_streaming(n):
    """Compute character table statistics by streaming through the file."""
    path = os.path.join(RESULTS_DIR, f"char_table_n{n}.txt")

    total_entries = 0
    zero_count = 0
    positive_count = 0
    negative_count = 0
    max_abs = 0
    abs_value_bins = Counter()  # log10 bins
    dimensions = []  # χ^λ(identity) = dimension

    # Identity class is the last class (cycle type (1^n))
    # Need to find it from partitions
    partitions = load_partitions(n)
    cycle_types = load_partitions(n)  # all partitions = all cycle types
    id_idx = None
    for c, ct in enumerate(cycle_types):
        if ct == tuple([1] * n):
            id_idx = c
            break

    print(f"  Identity class index: {id_idx}")
    t0 = time.time()

    with open(path) as f:
        for row_num, line in enumerate(f):
            vals = line.strip().split()
            for ci, v_str in enumerate(vals):
                v = int(v_str)
                total_entries += 1
                if v == 0:
                    zero_count += 1
                elif v > 0:
                    positive_count += 1
                else:
                    negative_count += 1

                av = abs(v)
                if av > max_abs:
                    max_abs = av

                if av > 0:
                    import math
                    bin_idx = int(math.log10(av))
                    abs_value_bins[bin_idx] += 1

            # Extract dimension
            if id_idx is not None and id_idx < len(vals):
                dimensions.append(int(vals[id_idx]))

            if row_num % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  Row {row_num}/37338 ({100*row_num//37338}%) [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    nonzero = total_entries - zero_count

    stats = {
        "n": n,
        "num_partitions": len(partitions),
        "total_entries": total_entries,
        "zero_count": zero_count,
        "nonzero_count": nonzero,
        "nonzero_fraction": nonzero / total_entries,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "max_abs_value": max_abs,
        "max_abs_value_str": str(max_abs),
        "max_abs_log10": float(f"{np.log10(float(max_abs)):.2f}") if max_abs > 0 else 0,
        "value_distribution_log10": {str(k): v for k, v in sorted(abs_value_bins.items())},
        "scan_time_seconds": elapsed,
    }

    if dimensions:
        dims_abs = [abs(d) for d in dimensions]
        stats["max_dimension"] = max(dims_abs)
        stats["max_dimension_str"] = str(max(dims_abs))
        stats["min_nonzero_dimension"] = min(d for d in dims_abs if d > 0) if any(d > 0 for d in dims_abs) else 0
        stats["dim_sum_squares"] = str(sum(d * d for d in dimensions))
        stats["n_factorial"] = str(factorial(n))
        stats["dim_sum_check"] = str(sum(d * d for d in dimensions)) == str(factorial(n))
        # Top 10 largest dimensions
        sorted_dims = sorted(enumerate(dimensions), key=lambda x: abs(x[1]), reverse=True)[:10]
        stats["top_10_dimensions"] = [(partitions[i], abs(d)) for i, d in sorted_dims]

    return stats


def compute_hook_kronecker(n, partitions, cycle_types_z):
    """Compute all Kronecker coefficients g(hook_i, hook_j, hook_k)."""
    hooks = identify_hooks(partitions, n)
    print(f"  Found {len(hooks)} hook partitions")

    if not hooks:
        return {}

    hook_indices = sorted(hooks.values())
    print(f"  Loading {len(hook_indices)} character table rows...")
    char_rows = load_char_rows(n, hook_indices)

    results = {}
    total = 0
    nonzero = 0
    max_g = 0

    hook_items = sorted(hooks.items())  # (k, idx) pairs
    num_hooks = len(hook_items)
    total_triples = num_hooks * (num_hooks + 1) * (num_hooks + 2) // 6

    print(f"  Computing {total_triples} hook triples...")
    t0 = time.time()

    for ai in range(num_hooks):
        k_a, idx_a = hook_items[ai]
        chi_a = char_rows[idx_a]
        for bi in range(ai, num_hooks):
            k_b, idx_b = hook_items[bi]
            chi_b = char_rows[idx_b]
            for ci in range(bi, num_hooks):
                k_c, idx_c = hook_items[ci]
                chi_c = char_rows[idx_c]

                g = kronecker_exact(chi_a, chi_b, chi_c, cycle_types_z)
                total += 1
                if g != 0:
                    nonzero += 1
                    if abs(g) > max_g:
                        max_g = abs(g)
                    results[(k_a, k_b, k_c)] = g

        if ai % 5 == 0:
            elapsed = time.time() - t0
            print(f"    Hook row {ai}/{num_hooks} ({100*ai//num_hooks}%) [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  Hook Kronecker: {nonzero}/{total} nonzero ({100*nonzero/total:.1f}%), max={max_g}, {elapsed:.1f}s")

    return {
        "total_triples": total,
        "nonzero": nonzero,
        "nonzero_fraction": nonzero / total if total > 0 else 0,
        "max_coefficient": max_g,
        "time_seconds": elapsed,
        "top_20": sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)[:20],
    }


def compute_near_rect_kronecker(n, partitions, cycle_types_z):
    """Compute Kronecker coefficients for near-rectangular partitions."""
    near_rects = identify_near_rectangles(partitions, n)
    print(f"  Found {len(near_rects)} near-rectangular partitions:")
    for label, idx in sorted(near_rects.items(), key=lambda x: x[1]):
        print(f"    {label} -> index {idx}")

    if len(near_rects) < 2:
        return {}

    rect_indices = sorted(near_rects.values())
    print(f"  Loading {len(rect_indices)} character table rows...")
    char_rows = load_char_rows(n, rect_indices)

    results = {}
    total = 0
    nonzero = 0
    max_g = 0

    rect_items = sorted(near_rects.items(), key=lambda x: x[1])
    num_rects = len(rect_items)
    total_triples = num_rects * (num_rects + 1) * (num_rects + 2) // 6

    print(f"  Computing {total_triples} near-rectangular triples...")
    t0 = time.time()

    for ai in range(num_rects):
        label_a, idx_a = rect_items[ai]
        chi_a = char_rows[idx_a]
        for bi in range(ai, num_rects):
            label_b, idx_b = rect_items[bi]
            chi_b = char_rows[idx_b]
            for ci in range(bi, num_rects):
                label_c, idx_c = rect_items[ci]
                chi_c = char_rows[idx_c]

                g = kronecker_exact(chi_a, chi_b, chi_c, cycle_types_z)
                total += 1
                if g != 0:
                    nonzero += 1
                    if abs(g) > max_g:
                        max_g = abs(g)
                    results[(label_a, label_b, label_c)] = g

    elapsed = time.time() - t0
    print(f"  Near-rect Kronecker: {nonzero}/{total} nonzero ({100*nonzero/total:.1f}%), max={max_g}, {elapsed:.1f}s")

    return {
        "total_triples": total,
        "nonzero": nonzero,
        "nonzero_fraction": nonzero / total if total > 0 else 0,
        "max_coefficient": max_g,
        "time_seconds": elapsed,
        "all_values": {str(k): v for k, v in sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)},
    }


def sample_kronecker(n, partitions, cycle_types_z, num_samples=1000):
    """Estimate overall nonzero fraction by random sampling."""
    P = len(partitions)
    print(f"  Sampling {num_samples} random triples from {P} partitions...")

    # Select random triples (i <= j <= k)
    random.seed(42)
    sample_indices = set()
    triples = []
    while len(triples) < num_samples:
        i, j, k = sorted(random.sample(range(P), 3))
        key = (i, j, k)
        if key not in sample_indices:
            sample_indices.add(key)
            triples.append(key)

    # Collect all unique row indices needed
    all_rows = set()
    for i, j, k in triples:
        all_rows.update([i, j, k])

    print(f"  Loading {len(all_rows)} unique character table rows...")
    char_rows = load_char_rows(n, all_rows)

    nonzero = 0
    max_g = 0
    t0 = time.time()

    for ti, (i, j, k) in enumerate(triples):
        chi_i = char_rows[i]
        chi_j = char_rows[j]
        chi_k = char_rows[k]
        g = kronecker_exact(chi_i, chi_j, chi_k, cycle_types_z)
        if g != 0:
            nonzero += 1
            if abs(g) > max_g:
                max_g = abs(g)

        if ti % 100 == 0:
            elapsed = time.time() - t0
            print(f"    Sample {ti}/{num_samples} ({100*ti//num_samples}%) [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    frac = nonzero / num_samples
    # Wilson score interval (95%)
    import math
    z = 1.96
    denom = 1 + z**2 / num_samples
    center = (frac + z**2 / (2 * num_samples)) / denom
    margin = z * math.sqrt((frac * (1 - frac) + z**2 / (4 * num_samples)) / num_samples) / denom

    print(f"  Sample: {nonzero}/{num_samples} nonzero = {100*frac:.1f}% (95% CI: [{100*(center-margin):.1f}%, {100*(center+margin):.1f}%])")
    print(f"  Max sampled |g| = {max_g}, time = {elapsed:.1f}s")

    return {
        "num_samples": num_samples,
        "nonzero": nonzero,
        "nonzero_fraction": frac,
        "ci_low": center - margin,
        "ci_high": center + margin,
        "max_sampled_coefficient": max_g,
        "time_seconds": elapsed,
    }


def main():
    n = 40
    print(f"{'='*60}")
    print(f"Kronecker S_{n} Analysis")
    print(f"{'='*60}\n")

    # Load partitions and compute z_ρ for each cycle type
    print("Loading partitions...")
    partitions = load_partitions(n)
    cycle_types = load_partitions(n)  # cycle types = all partitions of n
    P = len(partitions)
    print(f"  {P} partitions of {n}")

    # Precompute z_ρ for exact arithmetic
    print("Computing centralizer orders...")
    cycle_types_z = [z_rho(ct) for ct in cycle_types]

    all_results = {
        "n": n,
        "num_partitions": P,
        "unique_triples": P * (P + 1) * (P + 2) // 6,
        "char_table_time_seconds": 34283.4,  # from log
        "max_abs_chi": "58965081685061803130880",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Phase 1: Character table statistics
    print(f"\n--- Phase 1: Character Table Statistics ---")
    char_stats = char_table_stats_streaming(n)
    all_results["char_table_stats"] = char_stats
    print(f"  Nonzero entries: {char_stats['nonzero_count']:,} / {char_stats['total_entries']:,} ({100*char_stats['nonzero_fraction']:.1f}%)")
    print(f"  Max |χ|: {char_stats['max_abs_value_str']}")
    if 'dim_sum_check' in char_stats:
        print(f"  Σ dim² = {n}!: {char_stats['dim_sum_check']}")

    # Phase 2: Hook Kronecker coefficients
    print(f"\n--- Phase 2: Hook Kronecker Coefficients ---")
    hook_results = compute_hook_kronecker(n, partitions, cycle_types_z)
    all_results["hook_kronecker"] = hook_results

    # Phase 3: Near-rectangular Kronecker coefficients
    print(f"\n--- Phase 3: Near-Rectangular Kronecker Coefficients ---")
    rect_results = compute_near_rect_kronecker(n, partitions, cycle_types_z)
    all_results["near_rectangular_kronecker"] = rect_results

    # Phase 4: Random sample
    print(f"\n--- Phase 4: Random Sample (nonzero fraction estimate) ---")
    sample_results = sample_kronecker(n, partitions, cycle_types_z, num_samples=1000)
    all_results["random_sample"] = sample_results

    # Phase 5: Trend summary
    print(f"\n--- Phase 5: Cross-n Trend Summary ---")
    trends = {
        "5":  {"partitions": 7,     "nonzero_frac_full": None, "max_g": None},
        "20": {"partitions": 627,   "nonzero_frac_full": 0.795, "max_g": 6408361},
        "30": {"partitions": 5604,  "nonzero_frac_full": 0.899, "max_g": 24233221539853},
        "40": {
            "partitions": P,
            "nonzero_frac_sample": sample_results["nonzero_fraction"],
            "nonzero_frac_ci": [sample_results["ci_low"], sample_results["ci_high"]],
            "max_g_hook": hook_results.get("max_coefficient", 0),
            "max_g_sampled": sample_results["max_sampled_coefficient"],
        },
    }
    all_results["trends"] = trends
    print(f"  n=20: 79.5% nonzero, max g = 6,408,361")
    print(f"  n=30: 89.9% nonzero, max g = 24,233,221,539,853")
    print(f"  n=40: ~{100*sample_results['nonzero_fraction']:.1f}% nonzero (sampled), max g (hooks) = {hook_results.get('max_coefficient', 'N/A')}")

    # Save results
    out_path = os.path.join(OUTPUT_DIR, f"analysis_n{n}.json")
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        if isinstance(obj, tuple):
            return [serialize(v) for v in obj]
        return make_serializable(obj)

    with open(out_path, 'w') as f:
        json.dump(serialize(all_results), f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"S_{n} Analysis Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
