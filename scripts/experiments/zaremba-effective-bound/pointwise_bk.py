"""
Pointwise Bourgain-Kontorovich Circle Method for Zaremba's Conjecture

For a SPECIFIC integer d, compute:
  Main(d) vs Error(d)
to determine if the B-K circle method proves d is a Zaremba denominator.

Framework:
  R_N(d) = Main(d) + Error_major(d) + Error_minor(d)

  Main(d)        = C_delta * N^{2*delta-1} * S(d) * Psi_hat(d/N)
  Error_major(d) = C_delta * N^{2*delta-1} * sum_{q<=Q} phi(q) * (1-sigma_q)^K
  Error_minor(d) = [requires separate bound]

We can CHECK: Main(d) > Error_major(d) using spectral gap data.
The minor arc problem remains unless Q = N (which requires sigma_min > delta).

Key insight: We separate the computation into what we CAN bound (major arcs)
and what we CANNOT yet bound (minor arcs), and compute exact numerical values
for the computable parts.

Usage:
    python pointwise_bk.py --spectral-data ../../logs/transfer-operator/phase2_m2000.log
    python pointwise_bk.py --spectral-data ../../logs/transfer-operator/phase2_m2000.log --target-d 1000003
    python pointwise_bk.py --spectral-data ../../logs/transfer-operator/phase2_m2000.log --scan 100 1000000
"""

import argparse
import math
import re
import sys
from collections import defaultdict


# Constants
DELTA = 0.836829443681208
TWO_DELTA_MINUS_1 = 2 * DELTA - 1  # 0.673658887362416
PHI_GOLD = (1 + math.sqrt(5)) / 2  # 1.6180339887...
LOG_PHI = math.log(PHI_GOLD)        # 0.48121182506...


def parse_spectral_data(filepath):
    """Parse spectral gap data from transfer operator log.

    Returns dict: squarefree m -> (lambda_non, gap)
    """
    data = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(
                r'\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                line
            )
            if m and float(m.group(4)) > 0.99:
                mod = int(m.group(1))
                lam_non = float(m.group(5))
                gap = float(m.group(6))
                data[mod] = (lam_non, gap)
    return data


def sieve_primes(limit):
    """Sieve of Eratosthenes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]


def factorize(n):
    """Return list of (prime, exponent) pairs."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                exp += 1
                n //= d
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def euler_phi(n):
    """Euler's totient function."""
    result = n
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            while temp % d == 0:
                temp //= d
            result -= result // d
        d += 1
    if temp > 1:
        result -= result // temp
    return result


def singular_series(d):
    """
    Compute the singular series S(d) for Zaremba with alphabet {1,...,5}.

    Since Gamma_{1,...,5} acts transitively on P^1(F_p) for ALL primes p,
    the local factor at prime p dividing d is:

        S_p = p^2 / (p^2 - 1)

    The full singular series is:
        S(d) = product_{p | d, p prime} p^2 / (p^2 - 1)

    This is >= 1 for all d >= 2, and equals 1 for d = 1.
    """
    factors = factorize(d)
    S = 1.0
    for p, _ in factors:
        S *= p**2 / (p**2 - 1)
    return S


def get_spectral_gap_for_q(q, spectral_data, prime_gaps):
    """
    Get spectral gap for modulus q.

    For squarefree q in our data, use directly.
    For composite q = product of coprime parts, use:
        sigma_q >= min(sigma_{p_i}) over prime factors p_i

    This follows from the tensor product decomposition of L_{delta,q}.

    Returns (gap, source) where source describes how we got it.
    """
    # Direct lookup for squarefree moduli
    if q in spectral_data:
        return spectral_data[q][1], f"direct(m={q})"

    # Factor q and use minimum over prime factor gaps
    factors = factorize(q)
    primes = [p for p, _ in factors]

    gaps = []
    for p in primes:
        if p in prime_gaps:
            gaps.append(prime_gaps[p])
        else:
            # Prime not in our data -- we don't have a bound
            return None, f"missing_prime({p})"

    if gaps:
        return min(gaps), f"factored(primes={primes})"
    return None, "no_data"


def compute_prime_gaps(spectral_data):
    """Extract spectral gaps for individual primes from the data."""
    prime_gaps = {}
    for m, (lam_non, gap) in spectral_data.items():
        factors = factorize(m)
        if len(factors) == 1 and factors[0][1] == 1:
            # m is prime
            prime_gaps[m] = gap
    return prime_gaps


def cf_length_scale(N):
    """
    Approximate CF length for denominators of size N.

    A CF [0; a1, ..., ak] with a_i in {1,...,5} has denominator
    growing roughly as phi^k (for average digit ~ 1) to 5^k (all 5s).

    Typical growth rate with random digits in {1,...,5} is about
    exp(Lyapunov exponent) per step. The Lyapunov exponent for
    the Patterson-Sullivan measure is log(N)/k, so k ~ log(N)/lyap.

    For A = {1,...,5}, the Lyapunov exponent lambda ~ 1.18 (computed
    from the transfer operator).

    Conservative estimate: use log(N)/log(phi) which gives the LONGEST
    possible CF (all 1s), giving the WEAKEST spectral contraction.
    """
    return int(math.log(max(N, 2)) / LOG_PHI)


def pointwise_check(d, spectral_data, prime_gaps, theta=None, verbose=False):
    """
    Pointwise B-K check for a specific integer d.

    Computes:
    1. S(d) -- singular series
    2. K(d) -- CF length scale
    3. Error_major(d) / Main(d) ratio for the chosen Q = d^theta

    The check passes if Error_major < Main, i.e., ratio < 1.

    NOTE: This does NOT bound the minor arc. The minor arc contribution
    exists for theta < 1 and requires separate analysis (sum-product
    estimates or direct exponential sum computation).

    Parameters:
        d: target integer
        spectral_data: dict of squarefree m -> (lambda_non, gap)
        prime_gaps: dict of prime p -> gap
        theta: major arc parameter Q = d^theta (default: optimize)
    """
    N = d  # scale
    K = cf_length_scale(N)
    S_d = singular_series(d)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Pointwise B-K analysis for d = {d}")
        print(f"{'='*60}")
        print(f"  delta           = {DELTA:.15f}")
        print(f"  2*delta - 1     = {TWO_DELTA_MINUS_1:.15f}")
        print(f"  N ~ d           = {d}")
        print(f"  K (CF length)   = {K}")
        print(f"  S(d)            = {S_d:.10f}")
        factors = factorize(d)
        if factors:
            print(f"  factorization   = {' * '.join(f'{p}^{e}' if e > 1 else str(p) for p,e in factors)}")
            for p, e in factors:
                print(f"    S_{p} = {p}^2/({p}^2-1) = {p**2/(p**2-1):.10f}")

    # Try multiple theta values to find optimal
    if theta is None:
        best_theta = None
        best_ratio = float('inf')

        for theta_try in [i * 0.01 for i in range(1, 100)]:
            Q = max(2, int(d**theta_try))
            if Q > d:
                break

            ratio = compute_major_error_ratio(d, Q, K, S_d, spectral_data, prime_gaps)
            if ratio is not None and ratio < best_ratio:
                best_ratio = ratio
                best_theta = theta_try

        theta = best_theta if best_theta is not None else 0.3

    Q = max(2, int(d**theta))

    if verbose:
        print(f"  theta           = {theta:.4f}")
        print(f"  Q = d^theta     = {Q}")

    # Compute major arc error / main term ratio
    ratio, error_sum, missing_count, top_contribs = compute_major_error_ratio_detailed(
        d, Q, K, S_d, spectral_data, prime_gaps
    )

    if verbose:
        print(f"\n  Major arc error analysis:")
        print(f"    sum_{{q<=Q}} phi(q) * (1-sigma_q)^K = {error_sum:.6e}")
        print(f"    S(d)                                = {S_d:.10f}")
        print(f"    Ratio Error/Main                    = {ratio:.6e}" if ratio is not None else "    Ratio: UNDEFINED (missing gaps)")
        print(f"    Moduli with missing gaps             = {missing_count}")

        if top_contribs:
            print(f"\n    Top 10 contributing moduli to major arc error:")
            for q, contrib, gap, source in top_contribs[:10]:
                print(f"      q={q:6d}: phi(q)*(1-gap)^K = {contrib:.6e}  "
                      f"(gap={gap:.6f}, {source})")

        # Minor arc analysis
        print(f"\n  Minor arc analysis:")
        print(f"    Q/N = d^{{theta-1}} = d^{{{theta-1:.4f}}} = {d**(theta-1):.6e}")
        print(f"    Minor arc region: |alpha - a/q| > 1/(q*Q) for all q <= Q")
        print(f"    Minor arc measure: ~ 1 - {min(1.0, 12/math.pi**2 * theta):.4f}")
        if theta < 1:
            print(f"    *** MINOR ARC EXISTS -- cannot bound without sum-product ***")
            print(f"    To eliminate minor arc: need Q = d, i.e., theta = 1")
            print(f"    But theta=1 requires sigma_min > {1/(2.39):.4f} = 1/2.39")
            min_gap = min(prime_gaps.values()) if prime_gaps else 0
            print(f"    Your sigma_min = {min_gap:.6f}")
            print(f"    Required for theta=1: 2.39 * sigma_min = {2.39*min_gap:.4f} > 2")
            print(f"    Status: {'SATISFIED' if 2.39*min_gap > 2 else 'NOT SATISFIED'}")

        # What theta=1 would give
        if d <= 1999:
            Q_full = d
            ratio_full, _, _, _ = compute_major_error_ratio_detailed(
                d, Q_full, K, S_d, spectral_data, prime_gaps
            )
            print(f"\n  Full major arc check (theta=1, Q=d={d}):")
            if ratio_full is not None:
                print(f"    Error/Main ratio = {ratio_full:.6e}")
                print(f"    {'PASSES' if ratio_full < 1 else 'FAILS'}")
            else:
                print(f"    Cannot compute -- missing spectral gaps")

    return {
        'd': d,
        'S_d': S_d,
        'K': K,
        'theta': theta,
        'Q': Q,
        'error_main_ratio': ratio,
        'error_sum': error_sum,
        'missing_gaps': missing_count,
        'major_arc_passes': ratio is not None and ratio < 1,
        'minor_arc_exists': theta < 1,
    }


def compute_major_error_ratio(d, Q, K, S_d, spectral_data, prime_gaps):
    """Compute Error_major / Main = [sum phi(q)*(1-sigma_q)^K] / S(d)."""
    error_sum = 0.0
    missing = 0

    for q in range(2, Q + 1):
        gap, source = get_spectral_gap_for_q(q, spectral_data, prime_gaps)
        if gap is None:
            missing += 1
            continue

        contraction = (1.0 - gap) ** K
        error_sum += euler_phi(q) * contraction

    if missing > 0:
        return None  # Can't compute -- missing data

    return error_sum / S_d


def compute_major_error_ratio_detailed(d, Q, K, S_d, spectral_data, prime_gaps):
    """Like compute_major_error_ratio but returns detailed breakdown."""
    error_sum = 0.0
    missing = 0
    contribs = []

    for q in range(2, Q + 1):
        gap, source = get_spectral_gap_for_q(q, spectral_data, prime_gaps)
        if gap is None:
            missing += 1
            continue

        contraction = (1.0 - gap) ** K
        contrib = euler_phi(q) * contraction
        error_sum += contrib
        contribs.append((q, contrib, gap, source))

    contribs.sort(key=lambda x: -x[1])

    ratio = error_sum / S_d if missing == 0 else None
    return ratio, error_sum, missing, contribs


def scan_range(start, end, spectral_data, prime_gaps, theta=0.3):
    """Scan a range of d values and report results."""
    print(f"\nScanning d in [{start}, {end}] with theta={theta}")
    print(f"{'d':>10s}  {'S(d)':>10s}  {'K':>4s}  {'Q':>8s}  {'Error/Main':>12s}  {'Major OK?':>9s}  {'Note'}")
    print("-" * 80)

    passes = 0
    fails = 0
    skips = 0

    for d in range(start, end + 1):
        result = pointwise_check(d, spectral_data, prime_gaps, theta=theta)

        ratio = result['error_main_ratio']
        if ratio is None:
            status = "SKIP"
            note = f"missing {result['missing_gaps']} gaps"
            skips += 1
        elif ratio < 1:
            status = "PASS"
            note = ""
            passes += 1
        else:
            status = "FAIL"
            note = f"ratio={ratio:.4f}"
            fails += 1

        if status != "PASS" or d % 1000 == 0:
            print(f"{d:10d}  {result['S_d']:10.6f}  {result['K']:4d}  "
                  f"{result['Q']:8d}  "
                  f"{ratio:12.6e}  " if ratio is not None else f"{'N/A':>12s}  ",
                  end="")
            print(f"{status:>9s}  {note}")

    print(f"\nSummary: {passes} pass, {fails} fail, {skips} skip out of {end-start+1}")


def critical_theta_analysis(spectral_data, prime_gaps):
    """
    Analyze what theta values are feasible given our spectral gaps.

    For the major arc error to be smaller than the main term, we need:
        sum_{q<=Q} phi(q) * (1-sigma_q)^K < S(d)

    The dominant contribution comes from q with the smallest gap.
    The rough requirement is:
        Q^2 * (1 - sigma_min)^K < 1
    i.e.:
        2*theta*log(d) + K*log(1-sigma_min) < 0
        2*theta*log(d) < K * sigma_min  (approx, for small gaps)
        2*theta < (log(d)/log(phi)) * sigma_min / log(d)  -- WRONG, let me redo

    K ~ log(d)/log(phi), so:
        2*theta*log(d) < (log(d)/log(phi)) * sigma_min
        2*theta < sigma_min / log(phi)
        theta < sigma_min / (2 * log(phi))
        theta < sigma_min / 0.9624
    """
    if not prime_gaps:
        print("No prime gap data available")
        return

    sigma_min = min(prime_gaps.values())
    sigma_min_p = min(prime_gaps, key=prime_gaps.get)

    theta_max = sigma_min / (2 * LOG_PHI)

    print(f"\nCritical theta analysis:")
    print(f"  sigma_min = {sigma_min:.6f} (at p = {sigma_min_p})")
    print(f"  log(phi)  = {LOG_PHI:.6f}")
    print(f"  Rough theta bound: theta < sigma_min / (2*log(phi)) = {theta_max:.6f}")
    print(f"  For theta = 1 (no minor arc): need sigma_min > {2*LOG_PHI:.4f} = 2*log(phi)")
    print(f"  Your sigma_min = {sigma_min:.6f} {'>' if sigma_min > 2*LOG_PHI else '<'} {2*LOG_PHI:.4f}")
    print()

    # The precise bound for specific d values
    for log_d in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        d = 10**log_d
        K = cf_length_scale(d)
        # Need: Q^2 * (1-sigma_min)^K < 1
        # Q = d^theta
        # 2*theta*log(d) + K*log(1-sigma_min) < 0
        # theta < -K*log(1-sigma_min) / (2*log(d))
        if sigma_min < 1:
            theta_bound = -K * math.log(1 - sigma_min) / (2 * math.log(d))
        else:
            theta_bound = float('inf')

        # What fraction of [0,1] is covered by major arcs of level Q?
        # Approximately min(1, 6/pi^2 * Q / d) for the Farey dissection
        Q_at_bound = d**min(theta_bound, 1.0)
        coverage = min(1.0, 6 / math.pi**2 * Q_at_bound / d)

        print(f"  d ~ 10^{log_d:2d}: K={K:5d}, theta_max = {theta_bound:.4f}, "
              f"Q_max ~ d^{{{theta_bound:.3f}}} = {Q_at_bound:.2e}, "
              f"arc coverage ~ {coverage:.4f}")

    print()
    print("INTERPRETATION:")
    print(f"  With sigma_min = {sigma_min:.6f}, you can take theta up to ~{theta_max:.3f}")
    print(f"  This means major arcs cover Q ~ d^{{{theta_max:.3f}}} out of d")
    print(f"  The minor arc region [d^{{{theta_max:.3f}}}, d] CANNOT be bounded")
    print(f"  from spectral gaps alone.")
    print()
    print("OPTIONS TO CLOSE THE GAP:")
    print("  1. Sum-product estimate: bound exp sums on minor arc")
    print("     (Bourgain's theorem gives this non-effectively)")
    print("  2. Direct computation: for SPECIFIC d, evaluate the")
    print("     exponential sum F_N(alpha) on a grid over the minor arc")
    print("     This is feasible on GPU for moderate d.")
    print("  3. Use the Frolenkov-Kan sieve instead (avoids circle method)")
    print("     See frolenkov_kan.py")
    print("  4. Increase spectral gaps: if sigma_min > 0.963, theta=1 works")
    print("     (impossible with current semigroup -- gaps are real)")


def main():
    parser = argparse.ArgumentParser(
        description="Pointwise Bourgain-Kontorovich analysis for Zaremba"
    )
    parser.add_argument("--spectral-data", required=True,
                        help="Path to spectral gap log file")
    parser.add_argument("--target-d", type=int, default=None,
                        help="Analyze specific d")
    parser.add_argument("--scan", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Scan range of d values")
    parser.add_argument("--theta", type=float, default=None,
                        help="Major arc parameter (default: optimize)")
    parser.add_argument("--analysis", action="store_true",
                        help="Run critical theta analysis")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load spectral data
    spectral_data = parse_spectral_data(args.spectral_data)
    prime_gaps = compute_prime_gaps(spectral_data)

    print(f"Loaded spectral data: {len(spectral_data)} squarefree moduli")
    print(f"Prime gaps available: {len(prime_gaps)} primes")
    if prime_gaps:
        min_p = min(prime_gaps, key=prime_gaps.get)
        print(f"Min gap: {prime_gaps[min_p]:.6f} at p = {min_p}")
        print(f"Max gap: {max(prime_gaps.values()):.6f}")

    if args.analysis:
        critical_theta_analysis(spectral_data, prime_gaps)

    if args.target_d:
        pointwise_check(
            args.target_d, spectral_data, prime_gaps,
            theta=args.theta, verbose=True
        )

    if args.scan:
        scan_range(
            args.scan[0], args.scan[1],
            spectral_data, prime_gaps,
            theta=args.theta or 0.3
        )

    if not args.target_d and not args.scan and not args.analysis:
        # Default: run analysis and check a few interesting d values
        critical_theta_analysis(spectral_data, prime_gaps)

        print("\n" + "="*60)
        print("Sample pointwise checks")
        print("="*60)

        for d in [7, 17, 34, 100, 997, 1009, 1999]:
            pointwise_check(d, spectral_data, prime_gaps, verbose=True)


if __name__ == "__main__":
    main()
