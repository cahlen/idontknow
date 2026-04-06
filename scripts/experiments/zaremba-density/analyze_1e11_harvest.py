#!/usr/bin/env python3
"""
Analysis of Zaremba density results at 10^11.

Harvests all completed density logs, computes:
1. {1,k} pair hierarchy and power-law fit
2. {1,k}/{2,k} amplification ratios (inverse-square law test)
3. {1,2,k} exception set convergence across decades
4. {1,3,5} closed-set convergence
5. No-digit-1 pair densities

Outputs: analysis_1e11.json with all computed values.
"""

import json
import os
import re
import sys
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

def parse_log(path):
    content = path.read_text()
    if "RESULTS" not in content:
        return None
    m_digits = re.search(r'Digit set: \{(.+?)\}', content)
    m_range = re.search(r'Range: d = 1 to (\d+)', content)
    m_covered = re.search(r'Covered: (\d+) / (\d+)', content)
    m_density = re.search(r'Density: ([\d.]+)%', content)
    m_uncovered = re.search(r'Uncovered: (\d+)', content)
    m_time = re.search(r'Time: ([\d.]+)s', content)
    m_uncov_list = re.search(r'Uncovered d: ([\d ]+)', content)
    if not m_digits:
        return None
    return {
        'digits': m_digits.group(1),
        'max_d': int(m_range.group(1)) if m_range else 0,
        'covered': int(m_covered.group(1)) if m_covered else 0,
        'total': int(m_covered.group(2)) if m_covered else 0,
        'density_pct': float(m_density.group(1)) if m_density else 0,
        'uncovered': int(m_uncovered.group(1)) if m_uncovered else 0,
        'time_s': float(m_time.group(1)) if m_time else 0,
        'uncovered_list': m_uncov_list.group(1).strip() if m_uncov_list else None,
    }

def main():
    # Parse all logs
    all_results = {}
    for f in sorted(RESULTS_DIR.glob("gpu_A*.log")):
        entry = parse_log(f)
        if entry:
            key = (entry['digits'], entry['max_d'])
            all_results[key] = entry

    analysis = {"date": "2026-04-06", "source": "zaremba_density_gpu on 8xB200"}

    # 1. {1,k} pair hierarchy at 1e11
    print("=== {1,k} Pair Hierarchy at 10^11 ===")
    pair_1k = []
    for k in range(2, 11):
        key = (f"1,{k}", 100000000000)
        if key in all_results:
            e = all_results[key]
            pair_1k.append({"k": k, "density_pct": e['density_pct'], "covered": e['covered']})
            print(f"  k={k}: {e['density_pct']:.10f}%")

    # Power-law fit: density ~ C * k^alpha
    # log(density) = log(C) + alpha * log(k)
    if len(pair_1k) >= 3:
        xs = [math.log(p['k']) for p in pair_1k]
        ys = [math.log(p['density_pct']) for p in pair_1k]
        n = len(xs)
        sx = sum(xs); sy = sum(ys); sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
        alpha = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        log_c = (sy - alpha * sx) / n
        C = math.exp(log_c)
        # R^2
        y_mean = sy / n
        ss_tot = sum((y - y_mean)**2 for y in ys)
        ss_res = sum((y - (log_c + alpha * x))**2 for x, y in zip(xs, ys))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"\n  Power-law fit: density ~ {C:.4f} * k^{alpha:.3f}  (R²={r_sq:.6f})")
        analysis["pair_1k_power_law"] = {"C": C, "alpha": alpha, "R_squared": r_sq}

    analysis["pair_1k_hierarchy_1e11"] = pair_1k

    # 2. {1,k}/{2,k} amplification ratios at 1e10 and 1e11
    print("\n=== Inverse-Square Amplification ===")
    print("k | ratio@1e10 | ratio@1e11 | growth | predicted(1633/k²)")
    amplification = []
    for k in range(3, 11):
        r10_1k = all_results.get((f"1,{k}", 10000000000))
        r10_2k = all_results.get((f"2,{k}", 10000000000))
        r11_1k = all_results.get((f"1,{k}", 100000000000))
        r11_2k = all_results.get((f"2,{k}", 100000000000))

        ratio_10 = r10_1k['density_pct'] / r10_2k['density_pct'] if r10_1k and r10_2k and r10_2k['density_pct'] > 0 else None
        ratio_11 = r11_1k['density_pct'] / r11_2k['density_pct'] if r11_1k and r11_2k and r11_2k['density_pct'] > 0 else None
        predicted = 1633.0 / (k * k)

        if ratio_10 and ratio_11:
            growth = ratio_11 / ratio_10
            print(f"  {k} | {ratio_10:.1f} | {ratio_11:.1f} | {growth:.2f}x | {predicted:.1f}")
            amplification.append({"k": k, "ratio_1e10": ratio_10, "ratio_1e11": ratio_11, "growth": growth, "predicted": predicted})
        elif ratio_11:
            print(f"  {k} | - | {ratio_11:.1f} | - | {predicted:.1f}")
            amplification.append({"k": k, "ratio_1e11": ratio_11, "predicted": predicted})

    analysis["amplification_ratios"] = amplification

    # 3. {1,2,k} exception set convergence
    print("\n=== {1,2,k} Exception Sets Across Decades ===")
    print("Set       | 1e9    | 1e10   | 1e11   | Status")
    exception_sets = []
    for k in range(3, 11):
        digits = f"1,2,{k}" if k >= 3 else f"1,2"
        counts = {}
        for decade in [9, 10, 11]:
            key = (digits, 10**decade)
            if key in all_results:
                counts[f"1e{decade}"] = all_results[key]['uncovered']

        if counts:
            c9 = counts.get('1e9', '-')
            c10 = counts.get('1e10', '-')
            c11 = counts.get('1e11', '-')

            # Determine status
            if isinstance(c10, int) and isinstance(c11, int) and c10 == c11:
                status = "CLOSED (confirmed)"
            elif isinstance(c10, int) and isinstance(c11, int) and c11 > c10:
                status = f"OPEN (grew by {c11-c10})"
            else:
                status = "partial data"

            print(f"  {{1,2,{k}}} | {str(c9):>6} | {str(c10):>6} | {str(c11):>6} | {status}")
            exception_sets.append({"digits": digits, **counts, "status": status})

    analysis["exception_sets"] = exception_sets

    # 4. {1,3,5} convergence
    print("\n=== {1,3,5} Convergence ===")
    counts_135 = {}
    for decade in [9, 10, 11]:
        key = ("1,3,5", 10**decade)
        if key in all_results:
            counts_135[f"1e{decade}"] = all_results[key]['uncovered']

    if len(counts_135) >= 3:
        c9, c10, c11 = counts_135['1e9'], counts_135['1e10'], counts_135['1e11']
        delta_1 = c10 - c9
        delta_2 = c11 - c10
        decel = delta_1 / delta_2 if delta_2 > 0 else float('inf')
        print(f"  1e9: {c9}  →  1e10: {c10} (Δ={delta_1})  →  1e11: {c11} (Δ={delta_2})")
        print(f"  Deceleration: {decel:.1f}x per decade")
        print(f"  Extrapolated limit: ~{c11 + int(delta_2/decel)} (if deceleration continues)")
        analysis["convergence_135"] = {
            "counts": counts_135, "delta_9_10": delta_1, "delta_10_11": delta_2,
            "deceleration": decel,
            "extrapolated_limit": c11 + int(delta_2 / decel) if delta_2 > 0 else c11
        }

    # 5. No-digit-1 pairs at 1e11
    print("\n=== No-Digit-1 Pairs at 10^11 ===")
    no1_pairs = []
    for j in range(2, 6):
        for k in range(j+1, 11):
            key = (f"{j},{k}", 100000000000)
            if key in all_results:
                e = all_results[key]
                print(f"  {{{j},{k}}}: {e['density_pct']:.10f}%")
                no1_pairs.append({"digits": f"{j},{k}", "density_pct": e['density_pct']})

    analysis["no_digit_1_pairs_1e11"] = no1_pairs

    # 6. {1,2,6} closed set confirmation
    print("\n=== {1,2,6} Closed Set Confirmation ===")
    for decade in [9, 10, 11]:
        key = ("1,2,6", 10**decade)
        if key in all_results:
            e = all_results[key]
            print(f"  1e{decade}: {e['uncovered']} uncovered")
            if e.get('uncovered_list'):
                print(f"    exceptions: {e['uncovered_list'][:200]}")

    # Save analysis
    out_path = RESULTS_DIR / "analysis_1e11.json"
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
