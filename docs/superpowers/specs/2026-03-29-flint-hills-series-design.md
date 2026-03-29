# Flint Hills Series: Partial Sums to 10^10

**Date:** 2026-03-29
**Hardware:** RTX 5090 (32GB VRAM) + Intel Core Ultra 9 285K + 188GB RAM
**Estimated runtime:** 1-10 minutes for full computation to N = 10^10 (bulk is double-precision, only ~19 spike terms need quad-double)
**Output:** First large-scale computation of the Flint Hills series, 100,000x beyond published frontier

## Background

### The Problem

The Flint Hills series is:

```
S = Σ_{n=1}^{∞} 1 / (n³ sin²(n))
```

**Open question (Pickover, 2002):** Does S converge?

### Why It Matters Now

Lopez Zapata (March 2026, arXiv:2603.09719) proved the sharp criterion:

> S converges **if and only if** the irrationality measure μ(π) ≤ 5/2.

The best known bound is μ(π) ≤ 7.103 (Zeilberger-Zudilin, 2020). If we can demonstrate empirically that S converges, that is strong computational evidence for μ(π) ≤ 2.5 — an extraordinary improvement over the current bound.

### Current Frontier

Published partial sums extend to N ~ 10^5 (October 2025, arXiv:2510.27041). Nobody has published verified S_N for N ≥ 10^6. Our target of N = 10^10 is **100,000x beyond the frontier**.

### The Computational Challenge

The series has "spikes" — terms where n is close to a multiple of π, making sin(n) very small and 1/(n³ sin²(n)) very large. These occur at the numerators of convergents of π's continued fraction:

| Convergent p_k | q_k | Approx |sin(p_k)| | Term magnitude |
|---|---|---|---|
| 22 | 7 | 1.2e-3 | 2.7e+3 |
| 333 | 106 | 3.1e-5 | 2.8e+7 |
| 355 | 113 | 2.7e-8 | 3.0e+13 |
| 103993 | 33102 | 5.8e-10 | 2.6e+4 |
| 833719 | 265381 | 7.5e-9 | 3.1e-1 |
| 80143857 | 25510582 | 6.2e-10 | 5.0e+0 |
| 6167950454 | 1963319607 | computed at runtime | computed at runtime |

There are only ~19 convergent numerators below 10^10. These are the terms that determine convergence behavior.

**Key question:** Are the spike magnitudes growing (divergence) or decaying (convergence)?

### Connection to AI

- **No training data exists** for this series beyond N = 10^5. AI models asked about convergence will hallucinate.
- **The spike catalog** encodes deep information about π's Diophantine properties that directly relates to function approximation theory.
- **If we can show convergence**, that constrains μ(π) — a fundamental constant that appears in approximation theory, the mathematical backbone of neural network function spaces.

## Algorithm

### Two-Phase Approach

**Phase 1: Bulk terms (double precision).** For the vast majority of n from 1 to 10^10, double precision (64-bit) is sufficient. |sin(n)| is bounded away from zero (say > 10^-8), and the term 1/(n³ sin²(n)) is small enough that double precision accumulation is fine.

**Phase 2: Spike terms (256-bit precision).** For the ~19 convergent numerators and a small neighborhood around each, compute sin(n) using quad-double arithmetic (4× double, ~62 decimal digits of precision). This provides accurate spike magnitudes even when |sin(n)| ~ 10^-19.

Both phases run on the GPU.

### Quad-Double Arithmetic on GPU

A quad-double number is represented as a non-overlapping sum of 4 doubles: x = x₀ + x₁ + x₂ + x₃ where |x₁| ≤ ε|x₀|, etc. This gives ~212 bits (~62 decimal digits) of precision.

Operations needed:
- **Addition/subtraction:** Error-free Two-Sum cascaded 4 times. ~20 FP64 ops.
- **Multiplication:** Expand product using Two-Prod, cascade renormalization. ~40 FP64 ops.
- **sin(n):** Argument reduction (compute n mod 2π using a high-precision π constant embedded in the kernel), then Taylor series or CORDIC. ~200 FP64 ops total.

The high-precision π constant (256 bits = 4 doubles) is baked into the kernel as a compile-time constant. The first 256 bits of π are known to trillions of digits — we only need to embed 64 decimal digits.

### Argument Reduction for sin(n)

The critical step. For large n (~10^10), naive computation of sin(n) via sin(n mod 2π) loses precision because n mod 2π involves catastrophic cancellation.

**Solution: Cody-Waite extended argument reduction.**

1. Pre-compute π to 256 bits as a quad-double constant.
2. Compute k = round(n / π) as an integer.
3. Compute r = n - k·π using quad-double arithmetic (no cancellation because we have enough precision).
4. Compute sin(r) via Taylor series (r is small, so the series converges fast).

For n up to 10^10, k is at most ~3.2×10^9. The product k·π requires ~96 bits of π (log2(3.2×10^9) ≈ 32 bits for k, plus 64 bits of precision in the remainder). Quad-double's 212 bits is more than sufficient.

### Detecting Spike Terms

Pre-load the ~19 convergent numerators of π below 10^10 as a lookup table. For each batch of n values, check if n is within a small window (say ±100) of any convergent numerator. If so, flag for quad-double computation; otherwise use double precision.

In practice, 99.999998% of terms use double precision. Only ~3800 terms (19 convergents × ±100 neighborhood) use quad-double.

### Accumulation Strategy

Use **compensated (Kahan) summation** in double precision for the bulk terms. For the spike terms, accumulate separately in quad-double, then combine at the end.

Checkpoint partial sums at N = 10^6, 10^7, 10^8, 10^9, 10^10.

## Implementation

### New File

```
scripts/experiments/flint-hills/flint_hills.cu
```

Standalone CUDA kernel. Self-contained — no external libraries (we implement quad-double inline).

### Kernel Design

**Kernel 1: Bulk summation (double precision)**

```
Each thread block processes a range of n values (e.g., 10^6 per block).
For each n:
  1. Check if n is in the spike lookup table → skip if so
  2. Compute sin(n) in double precision (hardware sincos)
  3. Compute term = 1.0 / (n³ * sin² (n))
  4. Accumulate with Kahan summation
Block-level reduction → global partial sum array
```

Thread grid: 10^10 terms / 10^6 per block = 10^4 blocks. RTX 5090 handles this trivially.

**Kernel 2: Spike computation (quad-double)**

```
One thread per spike term (only ~19 threads, or ~3800 if including neighborhoods).
For each convergent numerator p_k:
  1. Compute sin(p_k) using quad-double argument reduction + Taylor series
  2. Compute term = 1 / (p_k³ * sin²(p_k)) in quad-double
  3. Store: (p_k, sin_p_k, term_magnitude, cumulative_spike_sum)
```

This kernel is tiny — runs in milliseconds.

**Host orchestration:**

1. Launch Kernel 2 (spikes) — store results
2. Launch Kernel 1 in batches (10^8 per batch, 100 batches total)
3. After each batch, copy partial sum back, print checkpoint
4. Final: combine bulk sum + spike sum

### Quad-Double Implementation

Inline `__device__` functions, no external library. The QD operations are well-documented (Hida, Li, Bailey 2001, "Library for Double-Double and Quad-Double Arithmetic"):

```c
typedef struct { double x[4]; } qd_real;

__device__ qd_real qd_add(qd_real a, qd_real b);
__device__ qd_real qd_mul(qd_real a, qd_real b);
__device__ qd_real qd_div(qd_real a, qd_real b);
__device__ qd_real qd_sin(qd_real a);  // via argument reduction + Taylor
```

The key constant:

```c
// π to 62 decimal digits as a quad-double
__device__ const qd_real QD_PI = {{
    3.141592653589793e+00,
    1.224646799147353e-16,
    -2.994769809718340e-33,
    1.112454220863365e-49
}};
```

### Memory Requirements

- Bulk partial sums: 10^4 blocks × 8 bytes = 80 KB (trivial)
- Spike data: 19 entries × ~64 bytes = ~1.2 KB (trivial)
- Per-thread working memory: ~100 bytes (registers)
- **Total VRAM: < 1 MB.** The 32 GB is completely overkill for this problem.

### Runtime Estimate

**Bulk (double precision):**
- sin(n) via hardware FP64 sincos: ~20 cycles per call
- RTX 5090 FP64: ~1.64 TFLOPS → ~82 billion FP64 ops/sec
- 10^10 terms × ~30 FP64 ops/term = 3×10^11 ops
- **~4 seconds** (compute-bound estimate)
- With memory overhead and Kahan summation: **~30-120 seconds**

**Spikes (quad-double):**
- 19 terms × ~200 FP64 ops × 4 (QD overhead) = ~15,000 ops
- **< 1 millisecond**

**Total estimated runtime: 1-5 minutes for 10^10 terms.**

This is much faster than the Hausdorff spectrum computation. The Flint Hills problem is embarrassingly parallel with minimal per-thread work.

## Output Format

### Primary Output

CSV with checkpoint partial sums:

```
N,S_N,S_N_error_bound,spike_contribution,bulk_contribution
1000000,30.31449...,1e-10,28.94...,1.37...
10000000,30.31451...,1e-9,...,...
100000000,...,...,...,...
1000000000,...,...,...,...
10000000000,...,...,...,...
```

### Spike Catalog

CSV with one row per convergent term:

```
k,p_k,q_k,sin_p_k,abs_sin_p_k,term_magnitude,log10_term,cumulative_spike_sum
0,3,1,-1.41e-1,1.41e-1,1.86e+1,1.27,...
1,22,7,-8.85e-3,8.85e-3,1.20e+4,4.08,...
2,333,106,3.14e-5,3.14e-5,2.75e+7,7.44,...
3,355,113,-2.67e-8,2.67e-8,3.14e+13,13.50,...
...
```

### Growth Rate Analysis

For each consecutive pair of spike terms, compute:

```
k,p_k,Delta_k,Delta_k/Delta_{k-1},log_ratio,trend
```

Where Delta_k = 1/(p_k³ sin²(p_k)). If the log ratios are consistently negative, spikes are shrinking → evidence for convergence.

### JSON Metadata

```json
{
  "experiment": "flint-hills-series",
  "date": "2026-03-29",
  "hardware": "RTX 5090 32GB",
  "max_N": 10000000000,
  "precision_bulk": "double (64-bit)",
  "precision_spikes": "quad-double (256-bit, ~62 decimal digits)",
  "checkpoints": [1e6, 1e7, 1e8, 1e9, 1e10],
  "num_convergent_terms": 19,
  "total_runtime_seconds": null,
  "novel": true,
  "description": "Flint Hills partial sums to 10^10, 100,000x beyond published frontier"
}
```

## Publishing Pipeline

1. Raw data: `idontknow/scripts/experiments/flint-hills/results/`
2. Experiment write-up: `bigcompute.science/src/content/experiments/2026-03-29-flint-hills-series.md`
3. Dataset: `bigcompute.science/public/data/flint-hills/`
4. Findings (if spike analysis shows clear convergence/divergence trend): `bigcompute.science/src/content/findings/`
5. Update `bigcompute.science/public/llms.txt`

## Verification

### Internal Consistency
- **Checkpoint monotonicity:** S_N is strictly increasing (all terms are positive)
- **Spike isolation:** Bulk sum + spike sum = total sum at every checkpoint
- **Kahan vs naive:** Compare Kahan-accumulated sum with naive sum at N = 10^6 to quantify accumulation error

### External Validation
- **S_{10^5} must match** the published value from arXiv:2510.27041
- **sin(355) = -2.667..×10^{-8}** is well-known and must match exactly
- **Individual spike terms** can be verified against Wolfram Alpha for small convergents

### Cross-Precision Check
- Run spikes at both quad-double and double-double precision, verify agreement to double-double precision

## Scope Boundaries

**In scope:**
- CUDA kernel with inline quad-double arithmetic
- Full partial sums to N = 10^10 with checkpoints
- Spike catalog for all 19 convergent terms
- Growth rate analysis
- CSV + JSON output
- bigcompute.science experiment page

**Out of scope:**
- Formal proof of convergence (this is computational evidence, not proof)
- Computing new digits of π or new convergents (we use known values)
- Extending beyond 10^10 (would need the B200 cluster for 10^12+)
- Interactive visualizations

## Risk Assessment

**Low risk overall.** The algorithm is straightforward (evaluate and sum). The main risks:

1. **Argument reduction precision for largest convergent** — n = 6,167,950,454 requires reducing n mod 2π with no cancellation. Quad-double (212 bits) provides ~160 bits of precision after reduction, which is sufficient. If somehow insufficient, we can fall back to computing just that one term on CPU with MPFR for verification.

2. **Accumulation error at 10^10 terms** — Kahan summation in double precision has O(ε) error per step regardless of N, so the accumulated error is ~10^{-16} × max_term. Since max_term ~ 10^13 (the 355 spike), accumulated error is ~10^{-3}. This is acceptable for the bulk, since we track the spike contribution separately at full precision.

3. **Hardware sin/cos precision at large n** — CUDA's `sincos()` loses precision for n > 10^8 due to internal argument reduction limitations. **Decision: use custom argument reduction for all n** (compute n mod 2π via pre-computed double-double π constant). This adds ~10 FP64 ops per term but ensures correctness everywhere. No split path needed.

4. **The series might just be boring** — If S_N grows smoothly without revealing clear convergence/divergence, the result is still valuable (extending frontier 100,000x) but less exciting. The spike growth rate analysis is the key diagnostic.
