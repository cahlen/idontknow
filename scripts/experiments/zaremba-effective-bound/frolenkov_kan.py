"""
Frolenkov-Kan Effective Bound for Zaremba's Conjecture

Uses the elementary sieve approach from:
  Frolenkov-Kan, "A strengthening of the Bourgain-Kontorovich theorem" (2014)

Instead of the circle method (which has non-effective minor arc constants),
F-K bound the exceptional set directly using a combinatorial sieve:

  #{d <= T : d NOT Zaremba} <= sum_m |E_m(T)|

where |E_m(T)| is bounded using the spectral gap sigma_m of the
congruence transfer operator.

The key formula:
  |E_m(T)| <= C(m) * T * (1 - sigma_m)^{floor(log T / log m)}

where C(m) depends on the orbit structure mod m.

If the total sum < 1 for all T > Q_0, then every d > Q_0 is Zaremba,
and we verify d <= Q_0 computationally.

Usage:
    python frolenkov_kan.py --spectral-data logs/transfer-operator/phase2_m2000.log
"""

import argparse
import math
import re
import json
import sys


def parse_spectral_data(filepath):
    """Parse spectral gap data from transfer operator log."""
    data = []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m and float(m.group(4)) > 0.99:
                mod = int(m.group(1))
                orbits = int(m.group(3))
                lam_non = float(m.group(5))
                gap = float(m.group(6))
                data.append({
                    'm': mod,
                    'orbits': orbits,
                    'lam_non': lam_non,
                    'gap': gap,
                })
    return data


def is_squarefree(n):
    for p in range(2, int(n**0.5) + 1):
        if n % (p * p) == 0:
            return False
    return True


def frolenkov_kan_bound(spectral_data, T):
    """
    Compute the F-K upper bound on #{d <= T : d not Zaremba}.

    For each modulus m with spectral gap sigma_m:
      |E_m(T)| <= orbits(m) * (1 - sigma_m)^{floor(log(T) / log(m))}

    The total exceptional count is bounded by sum_m |E_m(T)|.

    Note: This is a simplified version. The full F-K bound involves
    Mobius function inclusion-exclusion and more careful orbit counting.
    We use the upper bound that each orbit contributes at most 1 to
    the exceptional count per residue class.
    """
    total = 0.0
    log_T = math.log(T)

    contributions = []
    for entry in spectral_data:
        m = entry['m']
        sigma = entry['gap']
        orbits = entry['orbits']

        if m <= 1:
            continue

        log_m = math.log(m)
        exponent = int(log_T / log_m)

        if exponent <= 0:
            continue

        # The contraction factor per "step" in the sieve
        contraction = (1.0 - sigma)

        # Contribution from this modulus
        # C(m) accounts for the number of non-trivial orbits
        # and the proportion of residue classes that could be missed
        C_m = (orbits - 1) / orbits  # fraction of non-trivial residues

        contribution = C_m * T * (contraction ** exponent)

        total += contribution
        contributions.append((m, contribution, exponent, contraction))

    return total, contributions


def find_Q0(spectral_data, target_exceptions=0.5):
    """
    Find Q_0 such that for all T > Q_0, the F-K bound gives
    fewer than target_exceptions exceptions.

    Binary search on T.
    """
    # Start with T = 10^6, go up
    T_lo = 1e6
    T_hi = 1e30  # start high

    # First check if we can even bound it
    bound_at_hi, _ = frolenkov_kan_bound(spectral_data, T_hi)
    if bound_at_hi > target_exceptions:
        return None, bound_at_hi  # Can't bound — need more spectral data

    # Binary search
    for _ in range(200):
        T_mid = math.sqrt(T_lo * T_hi)  # geometric midpoint
        bound, _ = frolenkov_kan_bound(spectral_data, T_mid)

        if bound > target_exceptions:
            T_lo = T_mid
        else:
            T_hi = T_mid

        if T_hi / T_lo < 1.01:
            break

    return T_hi, frolenkov_kan_bound(spectral_data, T_hi)[0]


def main():
    parser = argparse.ArgumentParser(description="Frolenkov-Kan effective bound")
    parser.add_argument("--spectral-data", required=True,
                        help="Path to spectral gap log file")
    parser.add_argument("--target-T", type=float, default=None,
                        help="Evaluate bound at specific T")
    parser.add_argument("--find-Q0", action="store_true",
                        help="Find Q_0 where exceptions < 0.5")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data = parse_spectral_data(args.spectral_data)
    print(f"Loaded spectral data: {len(data)} moduli")
    print(f"Min gap: {min(d['gap'] for d in data):.6f} at m={min(data, key=lambda d: d['gap'])['m']}")
    print(f"Max gap: {max(d['gap'] for d in data):.6f}")
    print(f"Mean gap: {sum(d['gap'] for d in data)/len(data):.6f}")
    print()

    if args.target_T:
        T = args.target_T
        bound, contribs = frolenkov_kan_bound(data, T)
        print(f"F-K bound at T = {T:.2e}:")
        print(f"  Upper bound on exceptions: {bound:.6f}")
        print(f"  (Need < 1 for Zaremba to hold for all d > T)")
        print()

        if args.verbose:
            # Show top contributors
            contribs.sort(key=lambda x: -x[1])
            print("Top 20 contributing moduli:")
            for m, contrib, exp, contr in contribs[:20]:
                print(f"  m={m:5d}: contribution={contrib:.6e} "
                      f"(contraction={contr:.4f})^{exp} = {contr**exp:.6e}")

    if args.find_Q0:
        print("Searching for Q_0...")
        print()

        # Check at various T values first
        for log_T in range(6, 40):
            T = 10.0 ** log_T
            bound, _ = frolenkov_kan_bound(data, T)
            status = "< 1 ✓" if bound < 1 else "> 1 ✗"
            print(f"  T = 10^{log_T:2d}: bound = {bound:15.6f}  {status}")

            if bound < 1e-10:
                print(f"  (bound negligible, stopping)")
                break

        print()
        Q0, bound = find_Q0(data)
        if Q0 is not None:
            print(f"*** Q_0 = {Q0:.6e} ***")
            print(f"*** (bound at Q_0 = {bound:.6f}) ***")
            print()
            print(f"If this is correct, Zaremba's Conjecture holds for all d > {Q0:.2e}")
            print(f"Combined with brute-force verification to 10^9,")
            if Q0 < 1e9:
                print(f"THIS WOULD PROVE THE CONJECTURE (Q_0 < 10^9).")
            else:
                print(f"we would need to extend verification to {Q0:.2e}.")
        else:
            print(f"Could not find Q_0 — bound at 10^30 is still {bound:.6f}")
            print("Need more spectral data or a tighter bound formula.")


if __name__ == "__main__":
    main()
