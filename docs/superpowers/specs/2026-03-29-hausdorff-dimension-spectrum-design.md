# Hausdorff Dimension Spectrum of Continued Fraction Cantor Sets

**Date:** 2026-03-29
**Hardware:** RTX 5090 (32GB VRAM) + Intel Core Ultra 9 285K + 188GB RAM
**Estimated runtime:** 5-15 minutes for full spectrum (n=20), conservatively up to 1 hour with overhead
**Output:** First-ever complete dimension spectrum for all 2^n - 1 subsets of {1,...,n}

## Background

### The Mathematical Object

For a non-empty set A of positive integers, define:

```
E_A = { α ∈ (0,1) : every partial quotient of α's continued fraction is in A }
```

E_A is a Cantor-like fractal subset of [0,1]. Its Hausdorff dimension dim_H(E_A) measures the "size" of the set of real numbers whose continued fraction digits are restricted to A.

Classical examples:
- E_{1,2,3,...} = (0,1) \ Q, so dim_H = 1
- E_{1} = {[0; 1, 1, 1, ...]} = {1/φ} (single point), so dim_H = 0 (degenerate: only one CF with all digits = 1)
- E_{1,2} ≈ 0.5312 (Jenkinson-Pollicott, 100+ digits)
- E_{1,...,5} ≈ 0.8368 (computed in our Zaremba transfer operator experiment)

### Why This Matters

**For mathematics:** The function A → dim_H(E_A) encodes deep information about Diophantine approximation. Numbers in E_A are "badly approximable" at a rate controlled by A. The complete spectrum — how dimension varies over all subsets — has never been mapped. Individual values are scattered across papers; the combinatorial landscape is unexplored.

**For AI:** Current models have zero training data on dim_H(E_A) beyond a handful of published values. A complete dataset of 1,048,575 dimension values (all subsets of {1,...,20}) with 15-digit precision would be:
- The first structured dataset connecting subset combinatorics to fractal geometry
- Training signal for AI reasoning about operators, spectra, and approximation theory
- A reference that prevents hallucination when models are asked about these objects

The transfer operator L_s used here is mathematically identical to the kernel operators that appear in Gaussian processes and neural tangent kernels — teaching AI about how its spectrum depends on the digit set directly informs understanding of function approximation.

### Connection to Existing Work

Our Zaremba transfer operator experiment (`transfer_operator.cu`) already computes dim_H(E_{1,...,5}) as Phase 1. That code uses Chebyshev collocation + power iteration + bisection — exactly the algorithm needed here. The new experiment generalizes from one fixed digit set to all possible digit sets.

No overlap with the B200 cluster work: the cluster experiments focus on congruence spectral gaps (Phase 2) and large-modulus analysis. This experiment is purely Phase 1 style — small matrices, CPU or single-GPU, run locally on the 5090.

## Algorithm

### Transfer Operator Method

For digit set A and parameter s > 0, the transfer operator is:

```
(L_s f)(x) = Σ_{a ∈ A} (a + x)^{-2s} · f(1/(a+x))
```

dim_H(E_A) = the unique s = δ where the leading eigenvalue λ_0(s) = 1.

**Discretization:** Represent f on N Chebyshev nodes in [0,1]. L_s becomes an N×N matrix. The leading eigenvalue is found by power iteration (300 iterations, guaranteed convergence for this operator).

**Bisection:** λ_0(s) is strictly decreasing in s. Start with s_lo = 0.01, s_hi = 1.0. Bisect 55 times to get δ to ~15 decimal digits (2^{-55} ≈ 3×10^{-17}).

### Per-Subset Cost

For a single subset A with max element ≤ 20:
- Matrix size: N×N where N = 40 (sufficient for 15 digits)
- Matrix construction: |A| matrix builds, each O(N²) = O(1600)
- Power iteration: 300 iterations of O(N²) matrix-vector multiply
- Bisection: 55 bisection steps, each requiring one matrix build + power iteration
- **Total per subset:** ~55 × (1600·|A| + 300·1600) ≈ 55 × 500K = ~27.5M flops
- **Wall time per subset:** ~0.01-0.1ms on RTX 5090

### Total Cost

- Subsets of {1,...,20}: 2^20 - 1 = 1,048,575
- At 0.05ms average per subset: ~52 seconds
- With overhead (memory transfers, output): estimate 5-15 minutes for n=20
- For n=10 (1,023 subsets): seconds
- For n=15 (32,767 subsets): ~2 minutes

## Implementation

### New File

```
scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu
```

Standalone CUDA kernel. Does NOT modify the existing Zaremba transfer operator code — that stays focused on its mission.

### Kernel Design

**GPU batching strategy:** Process multiple subsets in parallel. Each subset needs an N×N matrix (40×40 = 12,800 bytes at double precision). At 32GB VRAM, we could hold millions of matrices simultaneously, but the bottleneck is compute not memory. Instead:

1. **Outer loop:** Iterate over subsets in batches of ~1024
2. **Per batch:** Each CUDA thread block handles one subset's full bisection
3. **Inner loop (per thread block):** Build matrix, power iterate, bisect — all in shared memory or registers

The N=40 matrix fits entirely in shared memory (12.8KB, well under the 5090's 128KB/SM limit). This means zero global memory traffic during the inner loop.

**Subset encoding:** Each subset A ⊆ {1,...,n} is encoded as a bitmask (uint32_t). Subset iteration is just incrementing from 1 to 2^n - 1.

### Code Structure

```c
// Per-subset dimension computation (adapted from transfer_operator.cu Phase 1)
__device__ double compute_dimension(uint32_t subset_mask, int max_digit, int N);

// Batched computation across all subsets
__global__ void batch_hausdorff(uint32_t start_mask, uint32_t count,
                                 int max_digit, int N, double *results);

// Host: orchestrate batches, collect results, write output
int main(int argc, char **argv);
```

### Key Differences from Zaremba Transfer Operator

| Aspect | Zaremba (existing) | Hausdorff Spectrum (new) |
|---|---|---|
| Digit set | Fixed: {1,...,5} | Variable: any A ⊆ {1,...,n} |
| Matrix size | N×N, N up to 200 | N×N, N = 40 (fixed) |
| GPU usage | cuBLAS dgemm for large implicit Kronecker products | Shared-memory matrix ops, no cuBLAS needed |
| Parallelism | Multi-GPU across moduli m | Single-GPU across subsets |
| Output | Single δ + spectral gaps table | 2^n - 1 dimension values |
| Phase 2 | Congruence gaps (the hard part) | None |

### Precision

N = 40 Chebyshev nodes with 55 bisection steps yields ~15 significant digits for dim_H(E_A). This matches or exceeds the precision of most published values.

For the record-setting computation of dim_H(E_{1,2}) (Jenkinson-Pollicott's 100+ digits), we would need multiprecision arithmetic and larger N. That's a separate follow-up, not part of this experiment.

## Output Format

### Primary Dataset

CSV file with columns:

```
subset_mask,subset_digits,cardinality,max_digit,dimension,precision_estimate
1,{1},1,1,0.000000000000000,1e-15
2,{2},1,2,0.000000000000000,1e-15
3,{1,2},2,2,0.531280506277205,1e-15
...
1048575,{1,2,...,20},20,20,0.999999999999...,1e-15
```

### Summary Statistics

Alongside the CSV, compute and report:
- Distribution of dim_H(E_A) by |A| (cardinality)
- Monotonicity verification: A ⊂ B implies dim_H(E_A) < dim_H(E_B)
- Dimension vs cardinality scatter data
- Extremal subsets: for each cardinality k, which k-element subset has highest/lowest dimension?
- Growth rate: how does dim_H(E_{1,...,n}) approach 1 as n increases?

### Agent-Consumable Metadata

JSON companion file with:
```json
{
  "experiment": "hausdorff-dimension-spectrum",
  "date": "2026-03-29",
  "hardware": "RTX 5090 32GB",
  "max_digit": 20,
  "num_subsets": 1048575,
  "chebyshev_order": 40,
  "bisection_steps": 55,
  "precision_digits": 15,
  "total_runtime_seconds": null,
  "novel": true,
  "description": "First complete Hausdorff dimension spectrum for all subsets of {1,...,20}"
}
```

## Publishing Pipeline

1. Raw data: `idontknow/scripts/experiments/hausdorff-spectrum/results/`
2. Experiment write-up: `bigcompute.science/src/content/experiments/2026-03-29-hausdorff-dimension-spectrum.md`
3. Dataset: `bigcompute.science/public/data/hausdorff-spectrum/`
4. Update `bigcompute.science/public/llms.txt` with new experiment entry

## Verification

### Internal Consistency
- **Monotonicity:** For every pair A ⊂ B, verify dim_H(E_A) < dim_H(E_B). Any violation indicates a bug.
- **Boundary cases:** dim_H(E_{1,...,n}) approaches 1 as n → ∞ (asymptotic — at n=20 it will still be meaningfully below 1). All singletons E_{a} are single points with dim_H = 0 (the unique CF [0; a, a, a, ...] converges to one quadratic irrational). Need |A| ≥ 2 for positive dimension.
- **Known values:** dim_H(E_{1,...,5}) must match our Zaremba result (0.836829443681208).

### External Validation
- dim_H(E_{1,2}) ≈ 0.5312805062772051 (Jenkinson-Pollicott)
- dim_H(E_{1,2,3}) published values in Hensley's work
- Any other published individual values we can find

## Scope Boundaries

**In scope:**
- CUDA kernel for batched dimension computation
- Full spectrum for n = 20 (1,048,575 subsets)
- CSV + JSON output
- Summary statistics
- bigcompute.science experiment page

**Out of scope:**
- Multiprecision computation (100+ digit precision for individual sets)
- Phase 2 style congruence gap analysis per subset
- Modifying the existing Zaremba transfer operator code
- Interactive visualizations (future follow-up)

## Risk Assessment

**Low risk overall.** The algorithm is proven, the code pattern exists, the computation fits trivially in 32GB VRAM (each matrix is 12.8KB). The main risks:

1. **Numerical instability for small subsets with large digits** — e.g., singletons E_{a} have very small dimensions. Power iteration may converge slowly. Mitigation: increase iteration count for |A| = 1 cases. Handle E_{1} as a special case (dim = 0, degenerate single point — the transfer operator is 1×1 trivially).
2. **Runtime estimate is wrong** — could take longer if shared memory approach has bank conflicts. Mitigation: start with n=10, measure, extrapolate.
3. **N=40 insufficient for some subsets** — unlikely given that N=40 gives 15 digits for E_{1,...,5}, which is a "harder" case than most. Can validate by running a few subsets at N=60 and comparing.
