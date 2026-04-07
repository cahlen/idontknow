#!/usr/bin/env python3
"""
Multi-constant PSLQ scanner for unmatched converged CFs from ramanujan_v2.

Takes the unmatched CSV (CFs that converged but didn't match any single constant
at double precision) and runs PSLQ against vectors of multiple constants
simultaneously to find relations like CF = (3*pi + 2*e) / 7.

Usage:
    python3 pslq_scan.py results/v2_unmatched_a2_b4_r6_6.csv
    python3 pslq_scan.py results/v2_unmatched_a2_b4_r6_6.csv --digits 150
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from mpmath import (
    mp, mpf, pi, e, log, sqrt, zeta, catalan, euler, agm,
    pslq, fabs, nstr, glaisher, khinchin
)

DEFAULT_DIGITS = 120
CF_DEPTH = 1000

def parse_coeffs(s):
    s = s.strip().strip('"').strip('(').strip(')')
    return [int(x) for x in s.split(',')]

def eval_cf_hp(a_coeffs, b_coeffs, depth=CF_DEPTH):
    deg_a = len(a_coeffs) - 1
    deg_b = len(b_coeffs) - 1
    def A(n):
        return sum(mpf(a_coeffs[k]) * mpf(n)**k for k in range(deg_a+1))
    def B(n):
        return sum(mpf(b_coeffs[k]) * mpf(n)**k for k in range(deg_b+1))
    val = A(depth)
    for n in range(depth-1, 0, -1):
        if fabs(val) < mpf(10)**(-mp.dps+10): return None
        val = A(n) + B(n+1) / val
    if fabs(val) < mpf(10)**(-mp.dps+10): return None
    return A(0) + B(1) / val

def build_constants():
    K = {}
    K["pi"] = pi; K["e"] = e; K["ln2"] = log(2)
    K["gamma"] = euler; K["Catalan"] = catalan
    K["zeta3"] = zeta(3); K["zeta5"] = zeta(5)
    K["sqrt2"] = sqrt(2); K["phi"] = (1+sqrt(5))/2
    K["pi2"] = pi**2; K["sqrtpi"] = sqrt(pi)
    K["Glaisher"] = glaisher
    return K

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Unmatched CSV from ramanujan_v2")
    parser.add_argument("--digits", type=int, default=DEFAULT_DIGITS)
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    mp.dps = args.digits + 30

    constants = build_constants()
    const_names = list(constants.keys())
    const_vals = [constants[k] for k in const_names]

    print(f"Multi-constant PSLQ scanner")
    print(f"Precision: {args.digits} digits")
    print(f"Constants: {', '.join(const_names)}")
    print(f"{'='*60}")

    with open(args.file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) > args.max_rows:
        print(f"Sampling {args.max_rows} of {len(rows)} rows")
        import random; random.seed(42)
        rows = random.sample(rows, args.max_rows)

    print(f"Processing {len(rows)} unmatched CFs")
    t0 = time.time()

    discoveries = []
    checked = 0
    errors = 0

    for i, row in enumerate(rows):
        a_coeffs = parse_coeffs(row["a_coeffs"])
        b_coeffs = parse_coeffs(row["b_coeffs"])

        try:
            cf = eval_cf_hp(a_coeffs, b_coeffs)
        except Exception:
            errors += 1; continue
        if cf is None:
            errors += 1; continue

        if fabs(cf) < mpf(10)**(-10) or fabs(cf) > mpf(10)**10:
            continue

        checked += 1

        # Single-constant PSLQ (linear and quadratic)
        for j, (cname, cval) in enumerate(zip(const_names, const_vals)):
            # Linear: a + b*K + c*V = 0
            rel = pslq([1, cval, cf], maxcoeff=200)
            if rel is not None:
                a, b, c = rel
                if c != 0 and (a != 0 or b != 0):
                    expected = -(mpf(a) + mpf(b)*cval) / mpf(c)
                    rd = fabs(cf - expected) / (fabs(cf) + mpf(10)**(-50))
                    digits = -int(mp.log10(rd + mpf(10)**(-300)))
                    if digits >= args.digits // 2:
                        disc = {
                            "a_coeffs": a_coeffs, "b_coeffs": b_coeffs,
                            "type": "linear", "constant": cname,
                            "relation": f"{a} + {b}*{cname} + {c}*V = 0",
                            "simplified": f"V = ({-a} + {-b}*{cname}) / {c}",
                            "digits": digits, "cf_value": nstr(cf, 25),
                        }
                        discoveries.append(disc)
                        print(f"  FOUND: a={a_coeffs} b={b_coeffs} -> {disc['simplified']} [{digits}d]")
                        break  # one match per CF is enough

        # Multi-constant PSLQ: [1, K1, K2, V] for pairs
        if checked <= 5000:  # expensive, limit
            for j in range(len(const_vals)):
                for k in range(j+1, len(const_vals)):
                    rel = pslq([1, const_vals[j], const_vals[k], cf], maxcoeff=100)
                    if rel is not None:
                        a, b, c, d = rel
                        if d != 0 and (b != 0 or c != 0):
                            expected = -(mpf(a) + mpf(b)*const_vals[j] + mpf(c)*const_vals[k]) / mpf(d)
                            rd = fabs(cf - expected) / (fabs(cf) + mpf(10)**(-50))
                            digits = -int(mp.log10(rd + mpf(10)**(-300)))
                            if digits >= args.digits // 2:
                                disc = {
                                    "a_coeffs": a_coeffs, "b_coeffs": b_coeffs,
                                    "type": "multi-constant",
                                    "constants": [const_names[j], const_names[k]],
                                    "relation": f"{a}+{b}*{const_names[j]}+{c}*{const_names[k]}+{d}*V=0",
                                    "digits": digits, "cf_value": nstr(cf, 25),
                                }
                                discoveries.append(disc)
                                print(f"  MULTI: a={a_coeffs} b={b_coeffs} -> {disc['relation']} [{digits}d]")

        if (i+1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] {elapsed:.0f}s, {checked} checked, {len(discoveries)} found")

    elapsed = time.time() - t0

    # Save results
    outpath = args.file.replace("unmatched", "pslq_discoveries").replace(".csv", ".json")
    result = {
        "input_file": args.file,
        "total_rows": len(rows),
        "checked": checked,
        "errors": errors,
        "discoveries": len(discoveries),
        "discovery_details": discoveries,
        "elapsed_seconds": elapsed,
    }
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"PSLQ SCAN COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Checked: {checked}, Errors: {errors}")
    print(f"  Discoveries: {len(discoveries)}")
    for d in discoveries:
        tag = "NEW!" if d["type"] == "multi-constant" else ""
        print(f"    {d.get('simplified', d['relation'])} [{d['digits']}d] {tag}")
    print(f"\n  Results: {outpath}")

if __name__ == "__main__":
    main()
