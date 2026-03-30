# Ramsey R(5,5) GPU Search — Design Notes

## The Bug

The incremental K₅ counter (`ramsey_incremental.cu`) has a verified bug:
354 "solutions" for n=44 were ALL false positives (verified fitness 1300-4200).
Even n=43 (known to have solutions) produced zero verified solutions.
The incremental fitness drifted to negative values over many steps.

**Root cause under investigation.** The counting logic appears correct on paper
(only K₅ containing both endpoints of the flipped edge are affected), but
the implementation produces systematic negative drift.

## Current Correct Baseline

`ramsey_fullcount.cu` — full recount on every step. Correct but slow:
- 18M flips/sec across 8 B200 GPUs
- n=43: best fitness 155 after 100K steps/walker (didn't reach 0)
- Need ~500K+ steps to find n=43 solutions

## Matrix-Based Optimization Plan

### Observation
For edge (u,v), K₅ through (u,v) = #triangles in subgraph induced by
common neighbors. Flipping (u,v) only changes K₅ containing both u and v.

### Approach: Maintain Triangle Count Matrix T

Define T[i][j] = number of common same-color neighbors of i and j
(= (A²)[i][j] where A is the adjacency matrix)

When we flip edge (u,v):
1. T changes only for pairs that include u or v
2. For each vertex w: T[u,w] and T[w,u] change by ±1 depending on
   whether v was/is a common neighbor of u and w
3. Total update: O(n) changes to T

### K₅ from T
K₅ through edge (u,v) = triangles in common-neighbor subgraph
                       = Σ_{a<b<c ∈ common} [a-b][b-c][a-c]

This is NOT directly available from T alone — need the actual adjacency
within the common-neighbor set. T[a][b] gives the count but not for the
restricted subgraph.

### Better: Batched cuBLAS Approach

For n=44: A is 44×44. Use cuBLAS SGEMM (float) for matrix operations.

Maintain per walker:
- A (44×44, float): red adjacency matrix
- A² = A×A: common neighbor counts

Full K₅ count via trace formula:
  #K₅ = (1/120) [tr(A⁵) - 10·m·tr(A³) + 20·Σ(A³_diag) + ...]

where corrections involve products of simpler subgraph counts.

Per-step with cuBLAS batched GEMM:
- A² = batched_GEMM(A, A) — for all walkers simultaneously
- A³ = batched_GEMM(A, A²)
- A⁵ = batched_GEMM(A², A³)
- Extract tr(A⁵) and corrections

After edge flip: A changes by rank-2, so A² update is 2 GEMV calls.

### Estimated Performance
- 10K walkers per GPU, 44×44 matrices
- Batched GEMM: 10K × 44³ = 852M ops, ~0.01ms on B200
- 3 GEMMs per step: ~0.03ms for ALL walkers
- 500K steps: 15 seconds total
- Compare: current full-recount takes ~220 seconds for 5K walkers × 100K steps

### Complication: Trace Formula for K₅

The trace formula for complete subgraph counting is complex.
tr(A^k) counts closed WALKS of length k, not k-cliques.

For K₅: #K₅ = (1/120) Σ_{σ∈S₅} (-1)^{5-c(σ)} Π tr(A^{cycle lengths of σ})

More practically:
#K₅ = (1/120)(p₅ − 10p₃ + 20p₂p₁ − 15p₁² + ...)
where pₖ = tr(Aᵏ) and corrections involve degree sequences.

The exact formula requires careful derivation. See:
- Alon, Yuster, Zwick (1997) "Finding and counting given length cycles"
- Floderus et al. (2015) "Detecting monotone monochromatic K₅"

### Alternative: Direct Bitmask with Correct Incremental

The simplest correct approach may be to fix the incremental counter.
The current bitmask approach is already fast (O(n² × popcount) per edge).
If the incremental delta is correct, each step is O(n²) ≈ 1936 ops
instead of O(n⁵/120) ≈ 1.3M ops — a 670× speedup.

**Priority: debug the incremental counter on CPU first, then port to GPU.**
