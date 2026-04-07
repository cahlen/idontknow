#!/usr/bin/env python3
"""
Test whether an expanded constant library catches matches the GPU kernel missed.
Generates random CFs and scans against MZVs, Glaisher, Khinchin, Soldner, etc.
"""

import json
import os
import random
import time
from mpmath import (
    mp, mpf, pi, e, log, sqrt, zeta, catalan, euler, agm,
    pslq, fabs, nstr, glaisher, khinchin, polylog
)

mp.dps = 180  # 150 + guard

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
random.seed(2026)

def eval_cf(p_coeffs, q_coeffs, depth):
    deg = len(p_coeffs) - 1
    def P(n): return sum(mpf(p_coeffs[k]) * mpf(n)**k for k in range(deg+1))
    def Q(n): return sum(mpf(q_coeffs[k]) * mpf(n)**k for k in range(deg+1))
    val = P(depth)
    for n in range(depth-1, 0, -1):
        if fabs(val) < mpf(10)**(-160): return None
        val = P(n) + Q(n+1)/val
    if fabs(val) < mpf(10)**(-160): return None
    return P(0) + Q(1)/val

def build_expanded_constants():
    K = {}
    # Standard
    K["pi"] = pi; K["e"] = e; K["ln(2)"] = log(2)
    K["gamma"] = euler; K["Catalan"] = catalan; K["zeta(3)"] = zeta(3)
    K["sqrt(2)"] = sqrt(2); K["phi"] = (1+sqrt(5))/2
    # MZVs
    K["zeta(5)"] = zeta(5); K["zeta(7)"] = zeta(7); K["zeta(9)"] = zeta(9)
    # Products
    K["pi*gamma"] = pi*euler; K["pi*ln(2)"] = pi*log(2)
    K["e*gamma"] = e*euler; K["gamma^2"] = euler**2
    K["ln(2)^2"] = log(2)**2; K["pi*zeta(3)"] = pi*zeta(3)
    # Special
    K["Glaisher"] = glaisher; K["Khinchin"] = khinchin
    K["Soldner"] = mpf("1.4513692348833810502839684858920274863157395220741")
    # Extras
    K["Li2(1/2)"] = polylog(2, mpf("0.5"))
    K["ln(3)"] = log(3); K["ln(5)"] = log(5)
    K["sqrt(3)"] = sqrt(3); K["sqrt(5)"] = sqrt(5)
    K["pi^2"] = pi**2; K["pi^2/6"] = pi**2/6
    K["sqrt(pi)"] = sqrt(pi); K["e*pi"] = e*pi
    K["pi/e"] = pi/e; K["ln(10)"] = log(10)
    return K

def gen_random_cf(deg, coeff_range):
    p = [random.randint(-coeff_range, coeff_range) for _ in range(deg+1)]
    q = [random.randint(-coeff_range, coeff_range) for _ in range(deg+1)]
    # Ensure Q not all zero
    if all(c == 0 for c in q):
        q[0] = 1
    return p, q

def main():
    constants = build_expanded_constants()
    print(f"Expanded constant library: {len(constants)} constants")
    print(f"Constants: {', '.join(constants.keys())}")

    # Generate random CFs
    cfs = []
    for _ in range(500): cfs.append(("deg2", *gen_random_cf(2, 5)))
    for _ in range(500): cfs.append(("deg3", *gen_random_cf(3, 3)))
    for _ in range(500): cfs.append(("deg4", *gen_random_cf(4, 2)))
    for _ in range(500): cfs.append(("deg5", *gen_random_cf(5, 2)))

    print(f"Generated {len(cfs)} random CFs")
    t0 = time.time()

    converged = 0
    matches = []

    for i, (label, p, q) in enumerate(cfs):
        v1000 = eval_cf(p, q, 1000)
        v800 = eval_cf(p, q, 800)

        if v1000 is None or v800 is None:
            continue
        if fabs(v1000) < mpf(10)**(-10) or fabs(v1000) > mpf(10)**10:
            continue

        conv = fabs(v1000 - v800) / (fabs(v1000) + mpf(10)**(-100))
        if conv > mpf(10)**(-50):
            continue

        converged += 1

        # PSLQ against each constant
        for cname, cval in constants.items():
            # Linear
            rel = pslq([1, cval, v1000], maxcoeff=500)
            if rel is not None:
                a, b, c = rel
                if c != 0:
                    expected = -(mpf(a) + mpf(b)*cval) / mpf(c)
                    rd = fabs(v1000 - expected) / (fabs(v1000) + mpf(10)**(-100))
                    digits = -int(mp.log10(rd + mpf(10)**(-300)))
                    if digits >= 70:
                        m = {
                            "type": "linear", "label": label,
                            "P": p, "Q": q, "constant": cname,
                            "relation": f"V = ({-a} + {-b}*{cname}) / {c}",
                            "digits": digits, "cf_value": nstr(v1000, 30),
                        }
                        matches.append(m)
                        print(f"  LINEAR MATCH: {label} P={p} Q={q} -> {cname} ({digits}d)")

            # Quadratic
            rel = pslq([1, cval, cval**2, v1000], maxcoeff=500)
            if rel is not None:
                a, b, c, d = rel
                if d != 0:
                    expected = -(mpf(a) + mpf(b)*cval + mpf(c)*cval**2) / mpf(d)
                    rd = fabs(v1000 - expected) / (fabs(v1000) + mpf(10)**(-100))
                    digits = -int(mp.log10(rd + mpf(10)**(-300)))
                    if digits >= 70:
                        m = {
                            "type": "quadratic", "label": label,
                            "P": p, "Q": q, "constant": cname,
                            "relation": f"{a}+{b}*{cname}+{c}*{cname}^2+{d}*V=0",
                            "digits": digits, "cf_value": nstr(v1000, 30),
                        }
                        matches.append(m)
                        print(f"  QUAD MATCH: {label} P={p} Q={q} -> {cname} ({digits}d)")

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(cfs)}] {elapsed:.0f}s, {converged} converged, {len(matches)} matches")

    elapsed = time.time() - t0

    result = {
        "total_cfs": len(cfs),
        "converged": converged,
        "matches": len(matches),
        "match_details": matches,
        "elapsed_seconds": elapsed,
    }
    outpath = os.path.join(RESULTS_DIR, "test_expanded_constants_results.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"EXPANDED CONSTANTS TEST — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Converged CFs: {converged}/{len(cfs)}")
    print(f"  Matches: {len(matches)}")
    for m in matches:
        is_alg = m["constant"] in ("sqrt(2)","sqrt(3)","sqrt(5)","phi")
        tag = "algebraic" if is_alg else "TRANSCENDENTAL"
        print(f"    [{tag}] {m['relation']} ({m['digits']}d)")
    print(f"\n  Results: {outpath}")

if __name__ == "__main__":
    main()
