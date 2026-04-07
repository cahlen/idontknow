#!/usr/bin/env python3
"""
High-precision PSLQ verification of Ramanujan Machine GPU hits.

The GPU kernel screens at double precision (~15 digits), producing many false
positives for transcendental constants. This script re-evaluates each CF at
100+ digits using mpmath and runs PSLQ to confirm or reject claimed relations.

Usage:
    python3 verify_hits.py                          # verify all hit CSVs
    python3 verify_hits.py hits_deg4_range7.csv     # verify one file
    python3 verify_hits.py --algebraic-only         # only check algebraic hits
    python3 verify_hits.py --digits 200             # use 200-digit precision

Output:
    results/verified_hits.csv         — confirmed formulas
    results/verified_summary.log      — run log with statistics
"""

import csv
import glob
import os
import sys
import time
from collections import defaultdict
from mpmath import (
    mp, mpf, pi, e, log, sqrt, zeta, catalan, euler as euler_gamma,
    pslq, nstr, fabs, isinf, isnan, power, agm
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Precision for verification (digits)
DEFAULT_DIGITS = 100
CF_DEPTH = 1000  # deeper than GPU's 200-500 for convergence

# ── Known constants at arbitrary precision ────────────────────────

def build_constants():
    """Build the same constant library as the GPU kernel, at current mp.dps precision."""
    K = {}

    # Base constants (match d_constants[] in CUDA)
    K["pi"] = pi
    K["e"] = e
    K["ln(2)"] = log(2)
    K["gamma"] = euler_gamma
    K["Catalan"] = catalan
    K["zeta(3)"] = zeta(3)
    K["Gauss"] = 1 / agm(1, sqrt(2))
    K["Lemniscate"] = 2 * pi / agm(1, sqrt(2))  # 2*omega = 2*pi/agm(1,sqrt(2))
    K["sqrt(2)"] = sqrt(2)
    K["phi"] = (1 + sqrt(5)) / 2

    # Compound constants (match d_compounds[] in CUDA)
    K["1/pi"] = 1 / pi
    K["1/e"] = 1 / e
    K["1/ln(2)"] = 1 / log(2)
    K["4/pi"] = 4 / pi
    K["pi/4"] = pi / 4
    K["pi/2"] = pi / 2
    K["pi/3"] = pi / 3
    K["pi/6"] = pi / 6
    K["pi^2"] = pi ** 2
    K["pi^2/6"] = pi ** 2 / 6
    K["pi^2/4"] = pi ** 2 / 4
    K["pi^2/12"] = pi ** 2 / 12
    # K["ln(2)"] already defined
    K["2*ln(2)"] = 2 * log(2)
    K["ln(10)"] = log(10)
    K["e*pi"] = e * pi
    K["e/pi"] = e / pi
    K["pi/e"] = pi / e
    K["pi*ln(2)"] = pi * log(2)
    K["sqrt(pi)"] = sqrt(pi)
    K["1/sqrt(pi)"] = 1 / sqrt(pi)
    K["2/sqrt(pi)"] = 2 / sqrt(pi)
    K["sqrt(e)"] = sqrt(e)
    K["1/sqrt(e)"] = 1 / sqrt(e)
    K["sqrt(2pi)"] = sqrt(2 * pi)
    K["1/sqrt(2pi)"] = 1 / sqrt(2 * pi)
    K["1/(e*pi)"] = 1 / (e * pi)
    K["sqrt(3)"] = sqrt(3)
    K["sqrt(5)"] = sqrt(5)

    return K


# ── CF evaluation at arbitrary precision ──────────────────────────

def parse_coeffs(s):
    """Parse '(1,-2,3)' into [1, -2, 3]."""
    s = s.strip().strip('"').strip('(').strip(')')
    return [int(x) for x in s.split(',')]


def eval_cf_hp(p_coeffs, q_coeffs, depth=CF_DEPTH):
    """Evaluate polynomial CF at current mpmath precision using backward recurrence."""
    deg = len(p_coeffs) - 1

    def P(n):
        return sum(mpf(p_coeffs[k]) * mpf(n) ** k for k in range(deg + 1))

    def Q(n):
        return sum(mpf(q_coeffs[k]) * mpf(n) ** k for k in range(deg + 1))

    # Backward recurrence from n=depth
    val = P(depth)
    for n in range(depth - 1, 0, -1):
        if fabs(val) < mpf(10) ** (-(mp.dps - 10)):
            return None  # divergence
        val = P(n) + Q(n + 1) / val

    # Final: CF = P(0) + Q(1) / val
    if fabs(val) < mpf(10) ** (-(mp.dps - 10)):
        return None
    return P(0) + Q(1) / val


def check_convergence_hp(p_coeffs, q_coeffs, depth=CF_DEPTH):
    """Evaluate at two depths to verify convergence."""
    v1 = eval_cf_hp(p_coeffs, q_coeffs, depth)
    v2 = eval_cf_hp(p_coeffs, q_coeffs, depth - 100)

    if v1 is None or v2 is None:
        return None

    if fabs(v1) < mpf(10) ** (-20) or fabs(v1) > mpf(10) ** 20:
        return None

    reldiff = fabs(v1 - v2) / fabs(v1)
    # Require convergence to at least half our working precision
    if reldiff > mpf(10) ** (-(mp.dps // 2)):
        return None

    return v1


# ── PSLQ matching ─────────────────────────────────────────────────

ALGEBRAIC_CONSTS = {"sqrt(2)", "sqrt(3)", "sqrt(5)", "phi"}


def verify_linear_relation(cf_val, const_name, c0, c1, c2, constants):
    """Check if CF = (c0 + c2*K) / c1 holds at high precision."""
    if const_name not in constants:
        return 0

    K = constants[const_name]
    expected = (mpf(c0) + mpf(c2) * K) / mpf(c1)

    if fabs(expected) < mpf(10) ** (-20):
        return 0

    reldiff = fabs(cf_val - expected) / fabs(expected)
    if reldiff == 0:
        return mp.dps  # exact

    digits = -int(mp.log10(reldiff))
    return max(0, digits)


def verify_power_relation(cf_val, const_name, p, q, constants):
    """Check if CF = K^(p/q) holds at high precision."""
    if const_name not in constants:
        return 0

    K = constants[const_name]
    try:
        expected = power(K, mpf(p) / mpf(q))
    except Exception:
        return 0

    if fabs(expected) < mpf(10) ** (-20):
        return 0

    reldiff = fabs(cf_val - expected) / fabs(expected)
    if reldiff == 0:
        return mp.dps

    digits = -int(mp.log10(reldiff))
    return max(0, digits)


def pslq_scan(cf_val, constants, max_coeff=100):
    """Run PSLQ against all constants to find integer relations.

    Tries to find integers (a, b, c) such that a + b*K + c*cf_val = 0
    for each constant K. Also tries quadratic: a + b*K + c*K^2 + d*cf_val = 0.
    """
    matches = []

    for cname, cval in constants.items():
        # Linear: a + b*K + c*V = 0  →  V = -(a + b*K)/c
        rel = pslq([1, cval, cf_val], maxcoeff=max_coeff)
        if rel is not None:
            a, b, c = rel
            if c != 0 and abs(a) <= max_coeff and abs(b) <= max_coeff:
                # Verify: reconstruct
                expected = -(mpf(a) + mpf(b) * cval) / mpf(c)
                reldiff = fabs(cf_val - expected) / (fabs(cf_val) + mpf(10) ** (-50))
                digits = -int(mp.log10(reldiff + mpf(10) ** (-200)))
                if digits >= mp.dps // 2:
                    matches.append({
                        "type": "linear",
                        "constant": cname,
                        "relation": f"{a} + {b}*{cname} + {c}*V = 0",
                        "simplified": f"V = ({-a} + {-b}*{cname}) / {c}",
                        "digits": digits,
                        "coeffs": (a, b, c),
                    })

        # Quadratic: a + b*K + c*K^2 + d*V = 0
        rel = pslq([1, cval, cval ** 2, cf_val], maxcoeff=max_coeff)
        if rel is not None:
            a, b, c, d = rel
            if d != 0 and max(abs(a), abs(b), abs(c)) <= max_coeff:
                expected = -(mpf(a) + mpf(b) * cval + mpf(c) * cval ** 2) / mpf(d)
                reldiff = fabs(cf_val - expected) / (fabs(cf_val) + mpf(10) ** (-50))
                digits = -int(mp.log10(reldiff + mpf(10) ** (-200)))
                if digits >= mp.dps // 2:
                    matches.append({
                        "type": "quadratic",
                        "constant": cname,
                        "relation": f"{a} + {b}*{cname} + {c}*{cname}^2 + {d}*V = 0",
                        "simplified": f"V = ({-a} + {-b}*{cname} + {-c}*{cname}^2) / {d}",
                        "digits": digits,
                        "coeffs": (a, b, c, d),
                    })

    return matches


# ── Main verification loop ────────────────────────────────────────

def verify_file(filepath, constants, digits, algebraic_only=False, run_pslq_scan=False):
    """Verify all hits in a CSV file. Returns (confirmed, rejected, errors)."""
    confirmed = []
    rejected = []
    errors = []

    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fname = os.path.basename(filepath)
    total = len(rows)
    if total == 0:
        return confirmed, rejected, errors

    # Skip files with only algebraic hits if we want transcendentals
    transcendental_count = sum(
        1 for r in rows
        if r.get("constant", "") not in ALGEBRAIC_CONSTS
        and r.get("constant", "") != ""
    )

    print(f"\n  {fname}: {total} hits ({transcendental_count} transcendental)")

    for i, row in enumerate(rows):
        const_name = row.get("constant", "")
        if not const_name:
            continue

        is_algebraic = const_name in ALGEBRAIC_CONSTS
        if algebraic_only and not is_algebraic:
            continue
        if not algebraic_only and is_algebraic:
            continue  # skip algebraic when doing transcendental pass

        p_coeffs = parse_coeffs(row["P_coeffs"])
        q_coeffs = parse_coeffs(row["Q_coeffs"])
        c0 = int(row["c0"])
        c1 = int(row["c1"])
        c2 = int(row["c2"])
        gpu_value = float(row["value"])

        try:
            cf_val = check_convergence_hp(p_coeffs, q_coeffs)
            if cf_val is None:
                errors.append({"row": row, "reason": "diverged_or_not_converged"})
                continue

            is_power = (c2 == -999)

            if is_power:
                matching_digits = verify_power_relation(cf_val, const_name, c0, c1, constants)
            else:
                matching_digits = verify_linear_relation(cf_val, const_name, c0, c1, c2, constants)

            threshold = digits // 2  # require at least half our precision

            if matching_digits >= threshold:
                result = {
                    "P_coeffs": row["P_coeffs"],
                    "Q_coeffs": row["Q_coeffs"],
                    "constant": const_name,
                    "c0": c0, "c1": c1, "c2": c2,
                    "matching_digits": matching_digits,
                    "cf_value_hp": nstr(cf_val, 30),
                    "source": fname,
                }
                confirmed.append(result)
            else:
                rej = {
                    "P_coeffs": row["P_coeffs"],
                    "Q_coeffs": row["Q_coeffs"],
                    "constant": const_name,
                    "claimed": f"({c0}+{c2}*{const_name})/{c1}" if not is_power else f"{const_name}^({c0}/{c1})",
                    "matching_digits": matching_digits,
                    "source": fname,
                }

                # For rejected transcendentals, optionally try PSLQ scan
                if run_pslq_scan and not is_algebraic and matching_digits < 5:
                    pslq_matches = pslq_scan(cf_val, constants)
                    if pslq_matches:
                        best = max(pslq_matches, key=lambda m: m["digits"])
                        rej["pslq_discovery"] = best["simplified"]
                        rej["pslq_digits"] = best["digits"]
                        rej["pslq_constant"] = best["constant"]
                        if best["digits"] >= threshold:
                            # PSLQ found a DIFFERENT relation that holds!
                            result = {
                                "P_coeffs": row["P_coeffs"],
                                "Q_coeffs": row["Q_coeffs"],
                                "constant": best["constant"],
                                "relation": best["simplified"],
                                "relation_type": best["type"],
                                "matching_digits": best["digits"],
                                "cf_value_hp": nstr(cf_val, 30),
                                "source": fname,
                                "discovery": "PSLQ",
                                "c0": 0, "c1": 0, "c2": 0,
                            }
                            confirmed.append(result)
                            continue

                rejected.append(rej)

        except Exception as ex:
            errors.append({"row": row, "reason": str(ex)})

    return confirmed, rejected, errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify Ramanujan Machine hits at high precision")
    parser.add_argument("files", nargs="*", help="Specific CSV files to verify (default: all)")
    parser.add_argument("--digits", type=int, default=DEFAULT_DIGITS, help="Working precision in digits")
    parser.add_argument("--algebraic-only", action="store_true", help="Only verify algebraic hits")
    parser.add_argument("--pslq-scan", action="store_true", help="Run PSLQ scan on rejected hits (slow)")
    parser.add_argument("--all", action="store_true", help="Verify both algebraic and transcendental")
    args = parser.parse_args()

    mp.dps = args.digits + 20  # extra guard digits

    print(f"Ramanujan Machine — High-Precision PSLQ Verification")
    print(f"Working precision: {args.digits} digits (mpmath dps={mp.dps})")
    print(f"CF evaluation depth: {CF_DEPTH}")
    print(f"PSLQ scan: {'ON' if args.pslq_scan else 'OFF'}")
    print(f"{'=' * 60}")

    constants = build_constants()
    print(f"Loaded {len(constants)} constants")

    # Find files to process
    if args.files:
        files = []
        for f in args.files:
            if os.path.isabs(f):
                files.append(f)
            elif os.path.exists(f):
                files.append(f)
            else:
                path = os.path.join(RESULTS_DIR, f)
                if os.path.exists(path):
                    files.append(path)
                else:
                    print(f"WARNING: {f} not found")
    else:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hits_deg*.csv")))

    if not files:
        print("No hit files found!")
        return

    print(f"Processing {len(files)} files")

    t0 = time.time()
    all_confirmed = []
    all_rejected = []
    all_errors = []

    passes = []
    if args.all or not args.algebraic_only:
        passes.append(("transcendental", False))
    if args.all or args.algebraic_only:
        passes.append(("algebraic", True))

    if not passes:
        passes = [("transcendental", False)]

    for pass_name, alg_only in passes:
        print(f"\n{'=' * 60}")
        print(f"Pass: {pass_name}")
        print(f"{'=' * 60}")

        for filepath in files:
            confirmed, rejected, errors = verify_file(
                filepath, constants, args.digits,
                algebraic_only=alg_only,
                run_pslq_scan=args.pslq_scan,
            )
            all_confirmed.extend(confirmed)
            all_rejected.extend(rejected)
            all_errors.extend(errors)

            nc, nr = len(confirmed), len(rejected)
            if nc > 0:
                print(f"    ✓ {nc} CONFIRMED")
            if nr > 0:
                print(f"    ✗ {nr} rejected (false positives)")

    elapsed = time.time() - t0

    # Write verified hits
    verified_path = os.path.join(RESULTS_DIR, "verified_hits.csv")
    if all_confirmed:
        with open(verified_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "P_coeffs", "Q_coeffs", "constant", "c0", "c1", "c2",
                "matching_digits", "cf_value_hp", "source",
                "relation", "relation_type", "discovery",
            ])
            writer.writeheader()
            for hit in all_confirmed:
                writer.writerow({k: hit.get(k, "") for k in writer.fieldnames})

    # Write summary log
    log_path = os.path.join(RESULTS_DIR, "verified_summary.log")
    with open(log_path, "w") as f:
        f.write(f"Ramanujan Machine — PSLQ Verification Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Precision: {args.digits} digits\n")
        f.write(f"CF depth: {CF_DEPTH}\n")
        f.write(f"Files processed: {len(files)}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")

        f.write(f"CONFIRMED: {len(all_confirmed)}\n")
        f.write(f"REJECTED (false positive): {len(all_rejected)}\n")
        f.write(f"ERRORS (diverged/etc): {len(all_errors)}\n\n")

        if all_confirmed:
            f.write(f"{'─' * 60}\n")
            f.write(f"Confirmed formulas:\n")
            f.write(f"{'─' * 60}\n")
            by_const = defaultdict(list)
            for h in all_confirmed:
                by_const[h["constant"]].append(h)
            for cname in sorted(by_const):
                hits = by_const[cname]
                f.write(f"\n  {cname}: {len(hits)} formulas\n")
                for h in hits[:10]:
                    if h.get("discovery") == "PSLQ":
                        f.write(f"    P={h['P_coeffs']}, Q={h['Q_coeffs']}\n")
                        f.write(f"      {h['relation']} ({h['matching_digits']}d) [PSLQ discovery]\n")
                    elif h.get("c2") == -999 or h.get("c2") == "-999":
                        f.write(f"    P={h['P_coeffs']}, Q={h['Q_coeffs']}\n")
                        f.write(f"      = {cname}^({h['c0']}/{h['c1']}) ({h['matching_digits']}d)\n")
                    else:
                        f.write(f"    P={h['P_coeffs']}, Q={h['Q_coeffs']}\n")
                        f.write(f"      = ({h['c0']} + {h['c2']}*{cname}) / {h['c1']} ({h['matching_digits']}d)\n")
                if len(hits) > 10:
                    f.write(f"    ... and {len(hits) - 10} more\n")

        if all_rejected:
            f.write(f"\n{'─' * 60}\n")
            f.write(f"False positive breakdown:\n")
            f.write(f"{'─' * 60}\n")
            by_const = defaultdict(int)
            for r in all_rejected:
                by_const[r["constant"]] += 1
            for cname in sorted(by_const, key=lambda k: -by_const[k]):
                f.write(f"  {cname}: {by_const[cname]} false positives\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  CONFIRMED:      {len(all_confirmed)}")
    print(f"  FALSE POSITIVE: {len(all_rejected)}")
    print(f"  ERRORS:         {len(all_errors)}")

    if all_confirmed:
        print(f"\n  Verified hits written to: {verified_path}")
        by_const = defaultdict(int)
        for h in all_confirmed:
            by_const[h["constant"]] += 1
        for cname in sorted(by_const, key=lambda k: -by_const[k]):
            print(f"    {cname}: {by_const[cname]}")

    print(f"  Full log: {log_path}")


if __name__ == "__main__":
    main()
