#!/usr/bin/env python3
"""
Test whether deeper CF evaluation reveals hidden matches.
The GPU evaluates at depth 200 — some CFs may converge slowly.
"""

import csv
import os
import time
import json
from mpmath import mp, mpf, pi, e, log, sqrt, zeta, catalan, euler, agm, pslq, fabs, nstr

mp.dps = 170  # 150 digits + guard

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def parse_coeffs(s):
    s = s.strip().strip('"').strip('(').strip(')')
    return [int(x) for x in s.split(',')]

def eval_cf(p_coeffs, q_coeffs, depth):
    deg = len(p_coeffs) - 1
    def P(n): return sum(mpf(p_coeffs[k]) * mpf(n)**k for k in range(deg+1))
    def Q(n): return sum(mpf(q_coeffs[k]) * mpf(n)**k for k in range(deg+1))
    val = P(depth)
    for n in range(depth-1, 0, -1):
        if fabs(val) < mpf(10)**(-150): return None
        val = P(n) + Q(n+1)/val
    if fabs(val) < mpf(10)**(-150): return None
    return P(0) + Q(1)/val

def build_constants():
    K = {}
    K["pi"] = pi; K["e"] = e; K["ln(2)"] = log(2)
    K["gamma"] = euler; K["Catalan"] = catalan; K["zeta(3)"] = zeta(3)
    K["zeta(5)"] = zeta(5); K["zeta(7)"] = zeta(7)
    K["Gauss"] = 1/agm(1,sqrt(2))
    K["Lemniscate"] = 2*pi/agm(1,sqrt(2))
    K["sqrt(2)"] = sqrt(2); K["phi"] = (1+sqrt(5))/2
    K["pi^2"] = pi**2; K["pi^2/6"] = pi**2/6
    K["e*pi"] = e*pi; K["pi/e"] = pi/e
    K["ln(10)"] = log(10); K["sqrt(pi)"] = sqrt(pi)
    K["sqrt(3)"] = sqrt(3); K["sqrt(5)"] = sqrt(5)
    return K

def main():
    constants = build_constants()

    # Load hits from deg4_range7 and deg5_range5
    rows = []
    for fname in ["hits_deg4_range7.csv", "hits_deg5_range5.csv"]:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path): continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source"] = fname
                rows.append(row)

    print(f"Loaded {len(rows)} hits from deg4_range7 + deg5_range5")
    t0 = time.time()

    slow_convergers = []
    new_matches = []
    total_checked = 0

    for i, row in enumerate(rows):
        p = parse_coeffs(row["P_coeffs"])
        q = parse_coeffs(row["Q_coeffs"])

        v200 = eval_cf(p, q, 200)
        v5000 = eval_cf(p, q, 5000)

        if v200 is None or v5000 is None:
            continue
        total_checked += 1

        reldiff = fabs(v200 - v5000) / (fabs(v5000) + mpf(10)**(-100))

        if reldiff > mpf(10)**(-12):
            slow_convergers.append({
                "P": row["P_coeffs"], "Q": row["Q_coeffs"],
                "v200": float(v200), "v5000": nstr(v5000, 30),
                "reldiff_200_5000": float(reldiff),
                "gpu_constant": row.get("constant", ""),
                "source": row["_source"],
            })

            # Check if v5000 is converged (compare with v1000)
            v1000 = eval_cf(p, q, 1000)
            if v1000 is None: continue
            conv_diff = fabs(v1000 - v5000) / (fabs(v5000) + mpf(10)**(-100))
            if conv_diff > mpf(10)**(-50):
                continue  # still not converged even at 5000

            # PSLQ scan at depth-5000 value
            for cname, cval in constants.items():
                rel = pslq([1, cval, v5000], maxcoeff=200)
                if rel is not None:
                    a, b, c = rel
                    if c != 0:
                        expected = -(mpf(a) + mpf(b)*cval) / mpf(c)
                        rd = fabs(v5000 - expected) / (fabs(v5000) + mpf(10)**(-100))
                        digits = -int(mp.log10(rd + mpf(10)**(-200)))
                        if digits >= 70:
                            match = {
                                "P": row["P_coeffs"], "Q": row["Q_coeffs"],
                                "constant": cname,
                                "relation": f"{a} + {b}*{cname} + {c}*V = 0",
                                "simplified": f"V = ({-a} + {-b}*{cname}) / {c}",
                                "digits": digits,
                                "cf_value": nstr(v5000, 30),
                                "source": row["_source"],
                            }
                            new_matches.append(match)
                            print(f"  MATCH: P={row['P_coeffs']} Q={row['Q_coeffs']} -> {cname} ({digits}d)")

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] {elapsed:.0f}s, {len(slow_convergers)} slow, {len(new_matches)} matches")

    elapsed = time.time() - t0

    # Save results
    result = {
        "total_hits": len(rows),
        "total_checked": total_checked,
        "slow_convergers": len(slow_convergers),
        "new_matches": len(new_matches),
        "matches": new_matches,
        "slow_converger_details": slow_convergers[:50],  # first 50
        "elapsed_seconds": elapsed,
    }
    outpath = os.path.join(RESULTS_DIR, "test_deep_cf_results.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"DEEP CF TEST COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Total checked: {total_checked}")
    print(f"  Slow convergers (depth 200 vs 5000 differ >1e-12): {len(slow_convergers)}")
    print(f"  New PSLQ matches at depth 5000: {len(new_matches)}")
    if new_matches:
        for m in new_matches:
            print(f"    {m['simplified']} [{m['constant']}, {m['digits']}d]")
            print(f"      P={m['P']}, Q={m['Q']}")
    print(f"\n  Results saved to: {outpath}")

if __name__ == "__main__":
    main()
