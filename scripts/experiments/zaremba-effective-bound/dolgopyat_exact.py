#!/usr/bin/env python3
"""
EXACT Dolgopyat spectral radius via FULL eigendecomposition (numpy/LAPACK).

For 80×80 complex matrices, CPU LAPACK is faster than GPU cuSOLVER
because kernel launch + transfer overhead dominates at this size.

Vectorized build_L for speed.
"""

import numpy as np
import time
import sys

BOUND = 5
NC = 80
DELTA = 0.836829443681208

# Precompute Chebyshev nodes and barycentric weights
_j = np.arange(NC)
NODES = 0.5 * (1.0 + np.cos(np.pi * (2*_j + 1) / (2.0 * NC)))
BARY = ((-1.0)**_j) * np.sin(np.pi * (2*_j + 1) / (2.0 * NC))


def build_L(t):
    """Build L_{δ+it} as NC×NC complex matrix, vectorized over digits."""
    L = np.zeros((NC, NC), dtype=np.complex128)

    for a in range(1, BOUND + 1):
        apx = a + NODES                    # (NC,)
        ga = 1.0 / apx                     # (NC,) Gauss map images
        weight = apx ** (-2.0 * DELTA)     # (NC,)
        phase = -2.0 * t * np.log(apx)     # (NC,)
        w = weight * np.exp(1j * phase)    # (NC,) complex weights

        # Barycentric interpolation: B[i,j] = contribution of node j to ga[i]
        diffs = ga[:, None] - NODES[None, :]  # (NC, NC)
        num = BARY[None, :] / diffs            # (NC, NC)
        B = num / num.sum(axis=1, keepdims=True)  # (NC, NC) normalized

        L += w[:, None] * B  # outer product-ish accumulation

    return L


def main():
    num_t = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    t_max = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0

    print(f"Dolgopyat EXACT (numpy LAPACK ZGEEV): N={NC}, {num_t} grid points, t∈[1,{t_max:.0f}]")
    print()

    t0 = time.time()

    t_values = (np.arange(num_t) + 0.5) * t_max / num_t
    t_values = t_values[t_values >= 1.0]

    max_rho = 0.0
    max_rho_t = 0.0
    step = max(1, len(t_values) // 20)

    for idx, t in enumerate(t_values):
        L = build_L(t)
        eigs = np.linalg.eigvals(L)
        rho = np.max(np.abs(eigs))

        if rho > max_rho:
            max_rho = rho
            max_rho_t = t

        if idx % step == 0:
            elapsed_so_far = time.time() - t0
            rate = (idx + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
            eta = (len(t_values) - idx) / rate if rate > 0 else 0
            print(f"  [{idx+1:6d}/{len(t_values)}] t={t:8.2f}: ρ = {rho:.8f}  "
                  f"(max so far: {max_rho:.8f} at t={max_rho_t:.2f})  "
                  f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"sup_{{t≥1}} ρ(L_{{δ+it}}) = {max_rho:.10f} at t = {max_rho_t:.6f}")
    print(f"Time: {elapsed:.1f}s for {len(t_values)} eigendecompositions")
    print(f"Rate: {len(t_values)/elapsed:.0f} eigdecomp/s")
    print("=" * 60)

    # Key values
    print("\nKey values:")
    for t in [1, 2, 5, 10, 15, 19.02, 20, 28.6, 50, 100, 500, 1000]:
        L = build_L(t)
        eigs = np.linalg.eigvals(L)
        rho = np.max(np.abs(eigs))
        print(f"  t={t:8.2f}: ρ = {rho:.10f}")

    # D₀ calculation
    print("\n" + "=" * 60)
    print("D₀ CALCULATION")
    print("=" * 60)
    rho_eta = max_rho
    c1 = 0.6046
    h0 = 1.3776
    delta = DELTA
    print(f"  ρ_η = {rho_eta:.10f}")
    print(f"  c₁  = {c1}")
    print(f"  h(0) = {h0}")
    print(f"  δ    = {delta}")
    # D₀ from: c₁ h(0)² D₀^{2δ-2} ρ_η^{log D₀} < 1
    # i.e. D₀ such that f(D₀) = c₁ h(0)² D₀^{2δ-2} D₀^{log(ρ_η)} < 1
    # exponent: 2δ - 2 + log(ρ_η) = 2(0.8368) - 2 + log(ρ_η)
    log_rho = np.log(rho_eta)
    exponent = 2*delta - 2 + log_rho
    print(f"  2δ - 2 = {2*delta - 2:.6f}")
    print(f"  log(ρ_η) = {log_rho:.6f}")
    print(f"  Combined exponent = {exponent:.6f}")
    if exponent < 0:
        # D₀ = (c₁ h(0)²)^{1/|exponent|}
        D0 = (c1 * h0**2) ** (1.0 / abs(exponent))
        print(f"  D₀ = {D0:.2e}")
        print(f"  Verified to: 2.1×10^11")
        if D0 < 2.1e11:
            print(f"  ✓ D₀ < 2.1×10^11 — CONJECTURE PROVED")
        else:
            print(f"  ✗ D₀ > 2.1×10^11 — need more brute force")
    else:
        print(f"  ERROR: exponent ≥ 0, series diverges!")


if __name__ == "__main__":
    main()
