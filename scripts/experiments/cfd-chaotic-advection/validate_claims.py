#!/usr/bin/env python3
"""
Validate Lyapunov claims for the Chirikov standard map experiment.

Cross-checks:
  1. Convergence vs iteration count (finite-time -> asymptotic estimate)
  2. Agreement with ln(K/2) large-K formula
  3. Greene K_crit literature value (no refinement claimed)
  4. Symplectic pairing: for 2D area-preserving maps, λ1 + λ2 = 0 (trace of Jacobian)

Usage:
  python3 scripts/experiments/cfd-chaotic-advection/validate_claims.py
  python3 scripts/experiments/cfd-chaotic-advection/validate_claims.py --quick
"""

import argparse
import csv
import math
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"
DEEP_CSV = RESULTS / "lyapunov_k2048_ic8192_iter50000.csv"

K_CRIT_LIT = 0.971635406
TWO_PI = 2.0 * math.pi


def mod2pi(x: float) -> float:
    x = math.fmod(x, TWO_PI)
    if x < 0:
        x += TWO_PI
    return x


def benettin_largest(K: float, theta0: float, p0: float, n_iters: int, seed: int) -> float:
    """CPU reference: single tangent vector, renormalize every step."""
    rng = random.Random(seed)
    theta = mod2pi(theta0)
    p = mod2pi(p0)
    v0, v1 = 1.0, 0.0
    total = 0.0
    for _ in range(n_iters):
        c = math.cos(theta)
        j00 = 1.0 + K * c
        j01 = 1.0
        j10 = K * c
        j11 = 1.0
        w0 = j00 * v0 + j01 * v1
        w1 = j10 * v0 + j11 * v1
        norm = math.hypot(w0, w1)
        if norm <= 0.0 or math.isnan(norm) or math.isinf(norm):
            return float("nan")
        total += math.log(norm)
        v0, v1 = w0 / norm, w1 / norm
        p_new = mod2pi(p + K * math.sin(theta))
        theta = mod2pi(theta + p_new)
        p = p_new
    return total / n_iters


def benettin_spectrum(K: float, theta0: float, p0: float, n_iters: int, seed: int):
    """CPU reference: two tangent vectors, Gram-Schmidt each step -> λ1, λ2."""
    theta = mod2pi(theta0)
    p = mod2pi(p0)
    # orthonormal tangent basis
    v10, v11 = 1.0, 0.0
    v20, v21 = 0.0, 1.0
    s1 = s2 = 0.0

    def apply_j(v0, v1, c):
        j00 = 1.0 + K * c
        j01 = 1.0
        j10 = K * c
        j11 = 1.0
        return j00 * v0 + j01 * v1, j10 * v0 + j11 * v1

    for step in range(n_iters):
        c = math.cos(theta)
        w10, w11 = apply_j(v10, v11, c)
        n1 = math.hypot(w10, w11)
        if n1 <= 0:
            return float("nan"), float("nan")
        s1 += math.log(n1)
        v10, v11 = w10 / n1, w11 / n1

        w20, w21 = apply_j(v20, v21, c)
        # Gram-Schmidt against v1
        dot = w20 * v10 + w21 * v11
        w20 -= dot * v10
        w21 -= dot * v11
        n2 = math.hypot(w20, w21)
        if n2 <= 0:
            return float("nan"), float("nan")
        s2 += math.log(n2)
        v20, v21 = w20 / n2, w21 / n2

        p_new = mod2pi(p + K * math.sin(theta))
        theta = mod2pi(theta + p_new)
        p = p_new

    return s1 / n_iters, s2 / n_iters


def load_csv(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def nearest_row(rows, K_target):
    return min(rows, key=lambda r: abs(float(r["K"]) - K_target))


def convergence_study(K: float, n_ic: int, iter_list, seed_base: int = 42):
    print(f"\n=== Convergence at K={K} ({n_ic} random ICs, CPU reference) ===")
    print(f"{'iters':>8} {'mean Λ':>12} {'std':>10} {'|Δ vs 100k|':>12}")
    ref = None
    results = {}
    for n_iters in iter_list:
        vals = [
            benettin_largest(K, random.Random(seed_base + i).random() * TWO_PI,
                             random.Random(seed_base + i + 100000).random() * TWO_PI,
                             n_iters, seed_base + i)
            for i in range(n_ic)
        ]
        vals = [v for v in vals if not math.isnan(v)]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        results[n_iters] = mean
        if n_iters == max(iter_list):
            ref = mean
    for n_iters in iter_list:
        delta = abs(results[n_iters] - ref) if ref else 0
        print(f"{n_iters:8d} {results[n_iters]:12.6f} {0:10.6f} {delta:12.2e}")


def symplectic_pairing_check(K: float, n_ic: int = 200, n_iters: int = 20000):
    print(f"\n=== Symplectic pairing at K={K} (2D area-preserving) ===")
    print("For symplectic maps in 2D: λ1 + λ2 should ≈ 0 (Liouville).")
    sums = []
    for i in range(n_ic):
        rng = random.Random(9000 + i)
        l1, l2 = benettin_spectrum(K, rng.random() * TWO_PI, rng.random() * TWO_PI, n_iters, i)
        if not math.isnan(l1):
            sums.append(l1 + l2)
    if not sums:
        print("  FAILED: no valid samples")
        return
    mean_sum = sum(sums) / len(sums)
    max_abs = max(abs(s) for s in sums)
    print(f"  mean(λ1+λ2) = {mean_sum:.2e}  max|λ1+λ2| = {max_abs:.2e}  ({n_ic} ICs)")
    ok = max_abs < 1e-10
    print(f"  Pairing claim: {'VALIDATED' if ok else 'NEEDS REVIEW (finite-time noise)'}")


def compare_gpu_csv(csv_path: Path):
    if not csv_path.exists():
        print(f"\nSkipping GPU CSV check — {csv_path} not found")
        return
    rows = load_csv(csv_path)
    print(f"\n=== GPU deep sweep cross-check ({csv_path.name}) ===")
    for K_target, label in [(0.0, "K=0"), (K_CRIT_LIT, "K_crit"), (5.0, "K=5")]:
        row = nearest_row(rows, K_target)
        K = float(row["K"])
        mean = float(row["mean_lyapunov"])
        mx = float(row["max_lyapunov"])
        mn = float(row["min_lyapunov"])
        print(f"  {label}: K={K:.4f} mean={mean:.6f} min={mn:.2e} max={mx:.6f}")
        if label == "K=5":
            theory = math.log(K / 2.0) if K > 0 else 0
            print(f"    ln(K/2) = {theory:.6f}  rel err = {100*abs(mean-theory)/theory:.2f}%")
        if label == "K_crit":
            print(f"    Literature range ~0.03–0.06 (Greene 1979; Lichtenberg & Lieberman 1992)")


def scale_claim_audit():
    print("\n=== Scale / 'largest' claim audit ===")
    print("  TERMINOLOGY: 'largest Lyapunov exponent' = maximal LCE (standard math term),")
    print("               NOT a claim that our computed VALUES are world records.")
    print("  TRAJECTORY COUNT: 16.8M (2048×8192) is large for a single-GPU parameter sweep,")
    print("               but NOT the largest standard-map study in literature:")
    print("               - Chirikov & Shepelyansky (1984+): ensemble / diffusion studies")
    print("               - dynamical-systems.org StdMap: 10^6+ iter per orbit, Lyapunov-colored sections")
    print("               - GPU Lyapunov packages (Chaoticus 2025, Julia ChaosTools): comparable parallelism")
    print("  NUMERICAL VALUES: we REPRODUCE published ranges; we do NOT claim new maxima.")
    print("  max_lyapunov column: per-K maximum over 8192 ICs (ensemble spread), not global record.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n_ic = 256 if args.quick else 1024
    iter_list = [5000, 20000, 50000, 100000] if args.quick else [5000, 10000, 20000, 50000, 100000]

    print("Chirikov standard map — claim validation")
    scale_claim_audit()
    convergence_study(5.0, n_ic, iter_list)
    convergence_study(K_CRIT_LIT, min(n_ic, 512), iter_list)
    symplectic_pairing_check(5.0, n_ic=min(n_ic, 200), n_iters=50000 if not args.quick else 20000)
    compare_gpu_csv(DEEP_CSV)

    print("\n=== Summary ===")
    print("  Safe claims: maximal LCE via Benettin; values consistent with literature; open pipeline.")
    print("  Do NOT claim: largest-ever computation; refined K_crit; saturated Lyapunov at small K without convergence test.")


if __name__ == "__main__":
    main()
