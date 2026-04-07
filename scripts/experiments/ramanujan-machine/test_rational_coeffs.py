#!/usr/bin/env python3
"""
Test whether rational (non-integer) polynomial coefficients unlock new CF formulas.
Tests half-integers, third-integers, and hypergeometric-type families.
"""

import json
import os
import random
import time
from itertools import product as cartprod
from mpmath import (
    mp, mpf, pi, e, log, sqrt, zeta, catalan, euler, agm,
    pslq, fabs, nstr, glaisher
)

mp.dps = 180

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
random.seed(42)

def eval_cf(p_coeffs, q_coeffs, depth):
    """Evaluate CF with mpf coefficients."""
    deg = len(p_coeffs) - 1
    def P(n):
        nn = mpf(n)
        return sum(p_coeffs[k] * nn**k for k in range(deg+1))
    def Q(n):
        nn = mpf(n)
        return sum(q_coeffs[k] * nn**k for k in range(deg+1))
    val = P(depth)
    for n in range(depth-1, 0, -1):
        if fabs(val) < mpf(10)**(-160): return None
        val = P(n) + Q(n+1)/val
    if fabs(val) < mpf(10)**(-160): return None
    return P(0) + Q(1)/val

def build_constants():
    K = {}
    K["pi"] = pi; K["e"] = e; K["ln(2)"] = log(2)
    K["gamma"] = euler; K["zeta(3)"] = zeta(3); K["Catalan"] = catalan
    K["sqrt(2)"] = sqrt(2); K["phi"] = (1+sqrt(5))/2
    K["pi^2"] = pi**2; K["pi^2/6"] = pi**2/6
    K["sqrt(pi)"] = sqrt(pi); K["Glaisher"] = glaisher
    return K

def check_cf(p_coeffs, q_coeffs, constants, label, matches):
    """Evaluate CF and run PSLQ. Appends to matches list."""
    v1 = eval_cf(p_coeffs, q_coeffs, 1000)
    v2 = eval_cf(p_coeffs, q_coeffs, 800)
    if v1 is None or v2 is None: return False
    if fabs(v1) < mpf(10)**(-10) or fabs(v1) > mpf(10)**10: return False
    conv = fabs(v1 - v2) / (fabs(v1) + mpf(10)**(-100))
    if conv > mpf(10)**(-50): return False

    for cname, cval in constants.items():
        # Linear PSLQ
        rel = pslq([1, cval, v1], maxcoeff=500)
        if rel is not None:
            a, b, c = rel
            if c != 0:
                expected = -(mpf(a) + mpf(b)*cval) / mpf(c)
                rd = fabs(v1 - expected) / (fabs(v1) + mpf(10)**(-100))
                digits = -int(mp.log10(rd + mpf(10)**(-300)))
                if digits >= 70:
                    p_str = [str(x) for x in p_coeffs]
                    q_str = [str(x) for x in q_coeffs]
                    m = {
                        "type": "linear", "label": label,
                        "P": p_str, "Q": q_str, "constant": cname,
                        "relation": f"V = ({-a} + {-b}*{cname}) / {c}",
                        "digits": digits, "cf_value": nstr(v1, 30),
                    }
                    matches.append(m)
                    print(f"  MATCH [{label}]: P={p_str} Q={q_str} -> {cname} ({digits}d)")
                    return True

        # Quadratic PSLQ
        rel = pslq([1, cval, cval**2, v1], maxcoeff=500)
        if rel is not None:
            a, b, c, d = rel
            if d != 0:
                expected = -(mpf(a) + mpf(b)*cval + mpf(c)*cval**2) / mpf(d)
                rd = fabs(v1 - expected) / (fabs(v1) + mpf(10)**(-100))
                digits = -int(mp.log10(rd + mpf(10)**(-300)))
                if digits >= 70:
                    p_str = [str(x) for x in p_coeffs]
                    q_str = [str(x) for x in q_coeffs]
                    m = {
                        "type": "quadratic", "label": label,
                        "P": p_str, "Q": q_str, "constant": cname,
                        "relation": f"{a}+{b}*{cname}+{c}*{cname}^2+{d}*V=0",
                        "digits": digits, "cf_value": nstr(v1, 30),
                    }
                    matches.append(m)
                    print(f"  QUAD [{label}]: P={p_str} Q={q_str} -> {cname} ({digits}d)")
                    return True
    return False

def main():
    constants = build_constants()
    print(f"Constants: {len(constants)} — {', '.join(constants.keys())}")
    t0 = time.time()

    matches = []
    converged = 0
    total = 0

    # --- Family 1: Half-integer degree-1 CFs ---
    print("\n=== Family 1: Half-integer degree-1 CFs ===")
    half_ints = [mpf(k)/2 for k in range(-6, 7)]
    fam1_samples = 5000
    fam1_tested = 0
    for _ in range(fam1_samples):
        p = [random.choice(half_ints), random.choice(half_ints)]
        q = [random.choice(half_ints), random.choice(half_ints)]
        if all(c == 0 for c in q): continue
        total += 1
        fam1_tested += 1
        if check_cf(p, q, constants, "half-int-deg1", matches):
            converged += 1
        elif fam1_tested % 1000 == 0:
            print(f"  [{fam1_tested}/{fam1_samples}] {time.time()-t0:.0f}s")

    # --- Family 2: Third-integer degree-2 CFs ---
    print("\n=== Family 2: Third-integer degree-2 CFs ===")
    third_ints = [mpf(k)/3 for k in range(-6, 7)]
    fam2_samples = 5000
    fam2_tested = 0
    for _ in range(fam2_samples):
        p = [random.choice(third_ints) for _ in range(3)]
        q = [random.choice(third_ints) for _ in range(3)]
        if all(c == 0 for c in q): continue
        total += 1
        fam2_tested += 1
        if check_cf(p, q, constants, "third-int-deg2", matches):
            converged += 1
        elif fam2_tested % 1000 == 0:
            print(f"  [{fam2_tested}/{fam2_samples}] {time.time()-t0:.0f}s")

    # --- Family 3: Hypergeometric-type CFs ---
    # P(n) = n, Q(n) = -(n+a)(n+b) for rational a,b
    print("\n=== Family 3: Hypergeometric Q(n) = -(n+a)(n+b) ===")
    sixth_ints = [mpf(k)/6 for k in range(1, 13)]
    fam3_tested = 0
    for a in sixth_ints:
        for b in sixth_ints:
            if a > b: continue  # symmetry
            # P(n) = n, Q(n) = -(n+a)(n+b) = -(n^2 + (a+b)n + ab)
            p_coeffs = [mpf(0), mpf(1)]  # P(n) = n
            q_coeffs = [-a*b, -(a+b), mpf(-1)]  # Q(n) = -(ab + (a+b)n + n^2)
            total += 1
            fam3_tested += 1
            check_cf(p_coeffs, q_coeffs, constants, f"hyper(a={a},b={b})", matches)

    # Also try P(n) = 2n+1 (Gauss-type)
    print("\n=== Family 3b: Gauss-type P(n)=2n+1, Q(n)=-(n+a)(n+b) ===")
    for a in sixth_ints:
        for b in sixth_ints:
            if a > b: continue
            p_coeffs = [mpf(1), mpf(2)]
            q_coeffs = [-a*b, -(a+b), mpf(-1)]
            total += 1
            fam3_tested += 1
            check_cf(p_coeffs, q_coeffs, constants, f"gauss(a={a},b={b})", matches)

    # P(n) = 2n+1, Q(n) = n^2 (classic for pi)
    print("\n=== Family 3c: P(n)=an+b, Q(n)=cn^2 variants ===")
    for a_coeff in range(1, 5):
        for b_coeff in range(-3, 4):
            for c_coeff in [-4, -3, -2, -1, 1, 2, 3, 4]:
                p_coeffs = [mpf(b_coeff), mpf(a_coeff)]
                q_coeffs = [mpf(0), mpf(0), mpf(c_coeff)]
                total += 1
                check_cf(p_coeffs, q_coeffs, constants, f"P={a_coeff}n+{b_coeff},Q={c_coeff}n^2", matches)

    elapsed = time.time() - t0

    # Separate transcendental from algebraic
    alg_consts = {"sqrt(2)", "sqrt(3)", "sqrt(5)", "phi"}
    trans_matches = [m for m in matches if m["constant"] not in alg_consts]
    alg_matches = [m for m in matches if m["constant"] in alg_consts]

    result = {
        "total_tested": total,
        "matches": len(matches),
        "transcendental_matches": len(trans_matches),
        "algebraic_matches": len(alg_matches),
        "match_details": matches,
        "elapsed_seconds": elapsed,
    }
    outpath = os.path.join(RESULTS_DIR, "test_rational_coeffs_results.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"RATIONAL COEFFICIENTS TEST — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Total tested: {total}")
    print(f"  Transcendental matches: {len(trans_matches)}")
    print(f"  Algebraic matches: {len(alg_matches)}")
    for m in trans_matches:
        print(f"    {m['label']}: {m['relation']} [{m['constant']}, {m['digits']}d]")
    print(f"\n  Results: {outpath}")

if __name__ == "__main__":
    main()
