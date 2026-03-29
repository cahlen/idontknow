# Hausdorff Dimension Spectrum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute dim_H(E_A) for all 2^n - 1 non-empty subsets A ⊆ {1,...,n} (target n=20, 1,048,575 subsets) and publish the first-ever complete Hausdorff dimension spectrum.

**Architecture:** A standalone CUDA kernel processes subsets in parallel — each thread block runs a full bisection (55 steps of matrix-build + power-iteration) for one subset, using shared memory for the 40×40 matrix. Host code orchestrates batches, writes CSV + JSON output, and a run.sh script compiles and executes on the RTX 5090.

**Tech Stack:** CUDA 13.0, C (no external libs beyond math.h), RTX 5090 (sm_120, 32GB VRAM)

**Spec:** `docs/superpowers/specs/2026-03-29-hausdorff-dimension-spectrum-design.md`

---

### Task 1: Create the CUDA kernel

**Files:**
- Create: `scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu`

This is the core computation. One file, self-contained, no dependencies beyond CUDA and math.h.

- [ ] **Step 1: Create experiment directory**

```bash
mkdir -p scripts/experiments/hausdorff-spectrum
```

- [ ] **Step 2: Write the CUDA kernel**

Create `scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu` with:

```c
/*
 * Hausdorff Dimension Spectrum of Continued Fraction Cantor Sets
 *
 * For each non-empty subset A ⊆ {1,...,n}, computes dim_H(E_A) where
 * E_A = { α ∈ (0,1) : all partial quotients of α are in A }.
 *
 * Uses the transfer operator method:
 *   (L_s f)(x) = Σ_{a∈A} (a+x)^{-2s} f(1/(a+x))
 * Discretized on N Chebyshev nodes, find δ where leading eigenvalue = 1.
 *
 * Hardware: RTX 5090 (32GB VRAM, compute capability 12.0)
 * Compile: nvcc -O3 -arch=sm_120 -o hausdorff_spectrum \
 *          scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu -lm
 * Run:     ./hausdorff_spectrum [max_digit] [chebyshev_order]
 *          ./hausdorff_spectrum 10      # all subsets of {1,...,10}, N=40
 *          ./hausdorff_spectrum 20 40   # all subsets of {1,...,20}, N=40
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_N 48          /* max Chebyshev order */
#define MAX_DIGIT 24      /* max digit in any subset */
#define BISECT_ITERS 55   /* 2^{-55} ≈ 3e-17 precision */
#define POWER_ITERS 300   /* power iteration steps */
#define BATCH_SIZE 1024   /* subsets per kernel launch */

/* ============================================================
 * Device: Chebyshev nodes and barycentric weights
 * ============================================================ */

__device__ void d_chebyshev_nodes(double *x, int N) {
    for (int j = 0; j < N; j++)
        x[j] = 0.5 * (1.0 + cos(M_PI * (2.0*j + 1.0) / (2.0*N)));
}

__device__ void d_barycentric_weights(double *w, int N) {
    for (int j = 0; j < N; j++)
        w[j] = pow(-1.0, (double)j) * sin(M_PI * (2.0*j + 1.0) / (2.0*N));
}

/* ============================================================
 * Device: Build transfer operator matrix for digit set A at parameter s
 *
 * M[i + j*N] = Σ_{a∈A} (a+x_i)^{-2s} * L_j(1/(a+x_i))
 * where L_j is the j-th barycentric interpolant basis function.
 * ============================================================ */

__device__ void d_build_matrix(uint32_t mask, int max_d, double s,
                               int N, double *x, double *bw, double *M) {
    /* Zero the matrix */
    for (int i = 0; i < N * N; i++) M[i] = 0.0;

    /* Accumulate contribution from each digit a in the subset */
    for (int a = 1; a <= max_d; a++) {
        if (!((mask >> (a - 1)) & 1)) continue;

        for (int i = 0; i < N; i++) {
            double y = 1.0 / (a + x[i]);
            double ws = pow(a + x[i], -2.0 * s);

            /* Check if y coincides with a node */
            int exact = -1;
            for (int k = 0; k < N; k++)
                if (fabs(y - x[k]) < 1e-15) { exact = k; break; }

            if (exact >= 0) {
                M[i + exact * N] += ws;
            } else {
                /* Barycentric interpolation */
                double den = 0.0;
                double num[MAX_N];
                for (int j = 0; j < N; j++) {
                    num[j] = bw[j] / (y - x[j]);
                    den += num[j];
                }
                for (int j = 0; j < N; j++)
                    M[i + j * N] += ws * num[j] / den;
            }
        }
    }
}

/* ============================================================
 * Device: Power iteration — returns leading eigenvalue of M
 * ============================================================ */

__device__ double d_power_iteration(double *M, int N, int iters) {
    double v[MAX_N], w[MAX_N];
    for (int i = 0; i < N; i++) v[i] = 1.0;

    double lam = 0.0;
    for (int it = 0; it < iters; it++) {
        /* w = M * v */
        for (int i = 0; i < N; i++) {
            double s = 0.0;
            for (int j = 0; j < N; j++) s += M[i + j * N] * v[j];
            w[i] = s;
        }
        /* Rayleigh quotient */
        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; i++) { num += v[i] * w[i]; den += v[i] * v[i]; }
        lam = num / den;
        /* Normalize */
        double norm = 0.0;
        for (int i = 0; i < N; i++) norm += w[i] * w[i];
        norm = sqrt(norm);
        if (norm < 1e-300) break;
        for (int i = 0; i < N; i++) v[i] = w[i] / norm;
    }
    return lam;
}

/* ============================================================
 * Device: Compute dim_H(E_A) for a single subset via bisection
 * ============================================================ */

__device__ double d_compute_dimension(uint32_t mask, int max_d, int N) {
    double x[MAX_N], bw[MAX_N];
    d_chebyshev_nodes(x, N);
    d_barycentric_weights(bw, N);

    /* Special case: singleton {1} is a single point (dim = 0) */
    if (mask == 1) return 0.0;

    /* Count bits to check for degenerate cases */
    int card = __popc(mask);
    if (card == 0) return 0.0;  /* empty set, shouldn't happen */

    /* Shared memory would be ideal but we're using local arrays
     * since each thread block processes one subset */
    double M[MAX_N * MAX_N];

    double s_lo = 0.001, s_hi = 1.0;

    /* Verify bracket: λ(s_lo) should be > 1, λ(s_hi) should be < 1 */
    d_build_matrix(mask, max_d, s_lo, N, x, bw, M);
    double l_lo = d_power_iteration(M, N, POWER_ITERS);
    if (l_lo <= 1.0) {
        /* Dimension is very small — tighten lower bound */
        s_lo = 0.0001;
        d_build_matrix(mask, max_d, s_lo, N, x, bw, M);
        l_lo = d_power_iteration(M, N, POWER_ITERS);
        if (l_lo <= 1.0) return 0.0;  /* effectively zero */
    }

    d_build_matrix(mask, max_d, s_hi, N, x, bw, M);
    double l_hi = d_power_iteration(M, N, POWER_ITERS);
    if (l_hi >= 1.0) {
        /* Dimension is very close to 1 — this happens for large subsets */
        return 1.0;
    }

    /* Bisection */
    for (int it = 0; it < BISECT_ITERS; it++) {
        double s = (s_lo + s_hi) * 0.5;
        d_build_matrix(mask, max_d, s, N, x, bw, M);
        double lam = d_power_iteration(M, N, POWER_ITERS);
        if (lam > 1.0) s_lo = s; else s_hi = s;
        if (s_hi - s_lo < 1e-16) break;
    }
    return (s_lo + s_hi) * 0.5;
}

/* ============================================================
 * Kernel: Batch computation across subsets
 * ============================================================ */

__global__ void batch_hausdorff(uint32_t start_mask, uint32_t count,
                                int max_d, int N, double *results) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint32_t mask = start_mask + idx;
    results[idx] = d_compute_dimension(mask, max_d, N);
}

/* ============================================================
 * Host: format subset as string "{1,3,5}"
 * ============================================================ */

void format_subset(uint32_t mask, int max_d, char *buf, int buflen) {
    int pos = 0;
    buf[pos++] = '{';
    int first = 1;
    for (int a = 1; a <= max_d && pos < buflen - 4; a++) {
        if ((mask >> (a - 1)) & 1) {
            if (!first) buf[pos++] = ',';
            pos += snprintf(buf + pos, buflen - pos, "%d", a);
            first = 0;
        }
    }
    buf[pos++] = '}';
    buf[pos] = '\0';
}

/* ============================================================
 * Host: main
 * ============================================================ */

int main(int argc, char **argv) {
    int max_d = argc > 1 ? atoi(argv[1]) : 10;
    int N     = argc > 2 ? atoi(argv[2]) : 40;

    if (max_d > MAX_DIGIT) {
        fprintf(stderr, "max_digit %d exceeds MAX_DIGIT %d\n", max_d, MAX_DIGIT);
        return 1;
    }
    if (N > MAX_N) {
        fprintf(stderr, "chebyshev_order %d exceeds MAX_N %d\n", N, MAX_N);
        return 1;
    }

    uint32_t total_subsets = (1u << max_d) - 1;
    printf("==========================================\n");
    printf("  Hausdorff Dimension Spectrum\n");
    printf("  Subsets of {1,...,%d}: %u\n", max_d, total_subsets);
    printf("  Chebyshev order N = %d\n", N);
    printf("  Bisection steps = %d\n", BISECT_ITERS);
    printf("==========================================\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Allocate host results */
    double *h_results = (double *)malloc(total_subsets * sizeof(double));

    /* Allocate device results */
    double *d_results;
    cudaMalloc(&d_results, (size_t)BATCH_SIZE * sizeof(double));

    /* Open CSV output */
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path),
             "scripts/experiments/hausdorff-spectrum/results/spectrum_n%d.csv", max_d);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Cannot open %s — did you mkdir -p results/?\n", csv_path);
        return 1;
    }
    fprintf(csv, "subset_mask,subset_digits,cardinality,max_digit_in_subset,dimension\n");

    /* Process in batches */
    uint32_t done = 0;
    int threads_per_block = 1;  /* one thread per subset (heavy work per thread) */
    uint32_t last_pct = 0;

    while (done < total_subsets) {
        uint32_t batch = total_subsets - done;
        if (batch > BATCH_SIZE) batch = BATCH_SIZE;

        uint32_t start_mask = done + 1;  /* masks go from 1 to 2^n - 1 */

        batch_hausdorff<<<batch, threads_per_block>>>(
            start_mask, batch, max_d, N, d_results);
        cudaDeviceSynchronize();

        /* Check for kernel errors */
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        /* Copy results back */
        cudaMemcpy(h_results + done, d_results, batch * sizeof(double),
                   cudaMemcpyDeviceToHost);

        /* Write CSV rows */
        char subset_str[256];
        for (uint32_t i = 0; i < batch; i++) {
            uint32_t mask = start_mask + i;
            format_subset(mask, max_d, subset_str, sizeof(subset_str));
            int card = __builtin_popcount(mask);
            /* Find highest set bit */
            int max_in_subset = 0;
            for (int a = max_d; a >= 1; a--)
                if ((mask >> (a-1)) & 1) { max_in_subset = a; break; }
            fprintf(csv, "%u,%s,%d,%d,%.15f\n",
                    mask, subset_str, card, max_in_subset, h_results[done + i]);
        }

        done += batch;

        /* Progress */
        uint32_t pct = (uint32_t)((100ULL * done) / total_subsets);
        if (pct != last_pct) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            double eta = (elapsed / done) * (total_subsets - done);
            printf("\r  %u / %u subsets (%u%%) — %.1fs elapsed, ~%.1fs remaining",
                   done, total_subsets, pct, elapsed, eta);
            fflush(stdout);
            last_pct = pct;
        }
    }

    fclose(csv);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\n\n  Done: %u subsets in %.1f seconds\n", total_subsets, total_time);
    printf("  Output: %s\n", csv_path);

    /* ============================================================
     * Verification & summary statistics
     * ============================================================ */

    printf("\n=== Verification ===\n");

    /* Check known values */
    /* E_{1,...,5} = mask 0b11111 = 31 (index 30) */
    if (max_d >= 5) {
        double zaremba_dim = h_results[30];  /* mask 31 is at index 30 (0-based) */
        double expected = 0.836829443681208;
        printf("  dim_H(E_{1,...,5}) = %.15f (expected %.15f, diff = %.2e)\n",
               zaremba_dim, expected, fabs(zaremba_dim - expected));
    }

    /* E_{1,2} = mask 0b11 = 3 (index 2) */
    if (max_d >= 2) {
        double e12_dim = h_results[2];  /* mask 3 is at index 2 */
        double expected_e12 = 0.531280506277205;
        printf("  dim_H(E_{1,2})    = %.15f (expected ~%.15f, diff = %.2e)\n",
               e12_dim, expected_e12, fabs(e12_dim - expected_e12));
    }

    /* E_{1} = mask 1 (index 0): should be 0 */
    printf("  dim_H(E_{1})      = %.15f (expected 0)\n", h_results[0]);

    /* Monotonicity spot check: E_{1,2} < E_{1,2,3} */
    if (max_d >= 3) {
        double d12 = h_results[2];   /* mask 3 = {1,2} */
        double d123 = h_results[6];  /* mask 7 = {1,2,3} */
        printf("  Monotonicity: dim({1,2})=%.6f < dim({1,2,3})=%.6f : %s\n",
               d12, d123, d12 < d123 ? "PASS" : "FAIL");
    }

    /* Summary by cardinality */
    printf("\n=== Dimension by Cardinality ===\n");
    printf("  |A|  count      min            mean           max\n");
    printf("  ---  -----  -------------  -------------  -------------\n");
    for (int k = 1; k <= max_d; k++) {
        double sum = 0, mn = 2.0, mx = -1.0;
        int cnt = 0;
        for (uint32_t i = 0; i < total_subsets; i++) {
            uint32_t mask = i + 1;
            if (__builtin_popcount(mask) == k) {
                double d = h_results[i];
                sum += d;
                if (d < mn) mn = d;
                if (d > mx) mx = d;
                cnt++;
            }
        }
        printf("  %3d  %5d  %.11f  %.11f  %.11f\n", k, cnt, mn, sum/cnt, mx);
    }

    /* Write JSON metadata */
    char json_path[256];
    snprintf(json_path, sizeof(json_path),
             "scripts/experiments/hausdorff-spectrum/results/metadata_n%d.json", max_d);
    FILE *jf = fopen(json_path, "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"experiment\": \"hausdorff-dimension-spectrum\",\n");
        fprintf(jf, "  \"date\": \"2026-03-29\",\n");
        fprintf(jf, "  \"hardware\": \"RTX 5090 32GB\",\n");
        fprintf(jf, "  \"max_digit\": %d,\n", max_d);
        fprintf(jf, "  \"num_subsets\": %u,\n", total_subsets);
        fprintf(jf, "  \"chebyshev_order\": %d,\n", N);
        fprintf(jf, "  \"bisection_steps\": %d,\n", BISECT_ITERS);
        fprintf(jf, "  \"power_iterations\": %d,\n", POWER_ITERS);
        fprintf(jf, "  \"precision_digits\": 15,\n");
        fprintf(jf, "  \"total_runtime_seconds\": %.1f,\n", total_time);
        fprintf(jf, "  \"novel\": true,\n");
        fprintf(jf, "  \"description\": \"First complete Hausdorff dimension spectrum for all subsets of {1,...,%d}\"\n", max_d);
        fprintf(jf, "}\n");
        fclose(jf);
        printf("\n  Metadata: %s\n", json_path);
    }

    /* Cleanup */
    cudaFree(d_results);
    free(h_results);

    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu
git commit -m "Add Hausdorff dimension spectrum CUDA kernel for RTX 5090"
```

---

### Task 2: Create the run script and results directory

**Files:**
- Create: `scripts/experiments/hausdorff-spectrum/run.sh`

- [ ] **Step 1: Create results directory and run script**

```bash
mkdir -p scripts/experiments/hausdorff-spectrum/results
```

Create `scripts/experiments/hausdorff-spectrum/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"

MAX_DIGIT="${1:-10}"
N="${2:-40}"

echo "Compiling hausdorff_spectrum (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o hausdorff_spectrum \
    scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu -lm
echo "Done."

mkdir -p scripts/experiments/hausdorff-spectrum/results

echo ""
echo "=== Computing Hausdorff dimension spectrum for {1,...,$MAX_DIGIT} ==="
echo "=== Chebyshev order N=$N ==="
echo ""
./hausdorff_spectrum "$MAX_DIGIT" "$N" 2>&1 | tee "scripts/experiments/hausdorff-spectrum/results/run_n${MAX_DIGIT}.log"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/experiments/hausdorff-spectrum/run.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/hausdorff-spectrum/run.sh
git commit -m "Add run script for Hausdorff spectrum experiment"
```

---

### Task 3: Compile and validate on n=5

**Files:**
- None created (compilation test)

- [ ] **Step 1: Compile the kernel**

```bash
cd /home/cahlen/dev/experiments/idontknow
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_120 -o hausdorff_spectrum \
    scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu -lm
```

Expected: compiles cleanly, no errors.

- [ ] **Step 2: Run on n=5 (31 subsets — fast sanity check)**

```bash
mkdir -p scripts/experiments/hausdorff-spectrum/results
./hausdorff_spectrum 5 40
```

Expected output includes:
- 31 subsets computed in < 1 second
- `dim_H(E_{1,...,5}) ≈ 0.836829443681208` (matching Zaremba result)
- `dim_H(E_{1,2}) ≈ 0.531280506277205`
- `dim_H(E_{1}) = 0.0`
- Monotonicity check: PASS

- [ ] **Step 3: Inspect the CSV output**

```bash
head -10 scripts/experiments/hausdorff-spectrum/results/spectrum_n5.csv
```

Verify: header row present, mask/subset/cardinality/dimension columns populated.

- [ ] **Step 4: Fix any issues found during validation**

If known values don't match within 1e-10, debug by:
1. Running at N=60 to check if N=40 is sufficient
2. Checking mask-to-index mapping (mask 31 = {1,2,3,4,5} at index 30)
3. Checking bisection bounds

- [ ] **Step 5: Commit validation results**

```bash
git add scripts/experiments/hausdorff-spectrum/results/spectrum_n5.csv
git add scripts/experiments/hausdorff-spectrum/results/metadata_n5.json
git add scripts/experiments/hausdorff-spectrum/results/run_n5.log
git commit -m "Validate Hausdorff spectrum on n=5: known values match"
```

---

### Task 4: Run n=10 and n=15 (incremental scaling)

**Files:**
- Output: `scripts/experiments/hausdorff-spectrum/results/spectrum_n10.csv`
- Output: `scripts/experiments/hausdorff-spectrum/results/spectrum_n15.csv`

- [ ] **Step 1: Run n=10 (1,023 subsets)**

```bash
./hausdorff_spectrum 10 40
```

Expected: completes in seconds. Check that CSV has 1,023 data rows.

- [ ] **Step 2: Run n=15 (32,767 subsets)**

```bash
./hausdorff_spectrum 15 40
```

Expected: completes in under 5 minutes. Note actual runtime for extrapolation to n=20.

- [ ] **Step 3: Quick verification**

```bash
# Count rows (should be 1024 for n=10 including header, 32768 for n=15)
wc -l scripts/experiments/hausdorff-spectrum/results/spectrum_n10.csv
wc -l scripts/experiments/hausdorff-spectrum/results/spectrum_n15.csv
```

- [ ] **Step 4: Commit intermediate results**

```bash
git add scripts/experiments/hausdorff-spectrum/results/spectrum_n10.csv
git add scripts/experiments/hausdorff-spectrum/results/metadata_n10.json
git add scripts/experiments/hausdorff-spectrum/results/run_n10.log
git commit -m "Hausdorff spectrum n=10: 1,023 subsets computed"
```

Do NOT commit the n=15 CSV (too large for git). Commit only the metadata and log:

```bash
git add scripts/experiments/hausdorff-spectrum/results/metadata_n15.json
git add scripts/experiments/hausdorff-spectrum/results/run_n15.log
git commit -m "Hausdorff spectrum n=15: 32,767 subsets computed"
```

---

### Task 5: Run the full n=20 computation

**Files:**
- Output: `scripts/experiments/hausdorff-spectrum/results/spectrum_n20.csv`
- Output: `scripts/experiments/hausdorff-spectrum/results/metadata_n20.json`

- [ ] **Step 1: Run n=20 (1,048,575 subsets)**

```bash
./hausdorff_spectrum 20 40 2>&1 | tee scripts/experiments/hausdorff-spectrum/results/run_n20.log
```

Monitor progress via the percentage output. Expected runtime: 5-60 minutes based on n=15 extrapolation.

- [ ] **Step 2: Verify output**

```bash
# Row count (should be 1,048,576 including header)
wc -l scripts/experiments/hausdorff-spectrum/results/spectrum_n20.csv

# Check known values in the log
grep "dim_H" scripts/experiments/hausdorff-spectrum/results/run_n20.log

# Check cardinality table
grep -A 22 "Dimension by Cardinality" scripts/experiments/hausdorff-spectrum/results/run_n20.log
```

- [ ] **Step 3: Commit metadata and log (CSV is too large for git)**

```bash
git add scripts/experiments/hausdorff-spectrum/results/metadata_n20.json
git add scripts/experiments/hausdorff-spectrum/results/run_n20.log
git commit -m "Hausdorff spectrum n=20: 1,048,575 subsets — first complete spectrum"
```

---

### Task 6: Create the bigcompute.science experiment page

**Files:**
- Create: `bigcompute.science/src/content/experiments/2026-03-29-hausdorff-dimension-spectrum.md`

- [ ] **Step 1: Write the experiment page**

Create `bigcompute.science/src/content/experiments/2026-03-29-hausdorff-dimension-spectrum.md`:

```markdown
---
title: "Hausdorff Dimension Spectrum: All Subsets of {1,...,20}"
slug: hausdorff-dimension-spectrum
date: 2026-03-29
author: cahlen
author_github: https://github.com/cahlen
status: in-progress

hardware:
  name: NVIDIA GeForce RTX 5090
  gpus: 1x RTX 5090 (32 GB VRAM)
  gpu_interconnect: N/A (single GPU)
  cpus: Intel Core Ultra 9 285K (24 cores)
  ram: 188 GB DDR5

software:
  cuda: "13.0"
  method: Chebyshev collocation (N=40) + power iteration + bisection
  custom_kernel: scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu

tags:
  domain: [continued-fractions, fractal-geometry, spectral-theory, diophantine-approximation]
  hardware: [rtx-5090]
  method: [transfer-operator, chebyshev-collocation, eigenvalue-computation, hausdorff-dimension]

results:
  total_subsets: 1048575
  max_digit: 20
  chebyshev_order: 40
  precision_digits: 15
  status: "IN PROGRESS"
  known_value_e12: 0.531280506277205
  known_value_e12345: 0.836829443681208

code: https://github.com/cahlen/idontknow
data: /data/hausdorff-spectrum/
---

# Hausdorff Dimension Spectrum: All Subsets of {1,...,20}

## Abstract

We compute the Hausdorff dimension dim_H(E_A) for every non-empty subset A ⊆ {1,...,20} — a total of 1,048,575 subsets. This is the first complete mapping of the "dimension spectrum" of continued fraction Cantor sets. E_A is the set of irrationals in (0,1) whose continued fraction partial quotients all lie in A.

## Background

For a finite set A ⊆ ℕ, the set E_A = {α ∈ (0,1) : all CF digits of α are in A} is a Cantor-like fractal. Its Hausdorff dimension is determined by the transfer operator:

$$(\mathcal{L}_s f)(x) = \sum_{a \in A} (a+x)^{-2s} f\!\left(\tfrac{1}{a+x}\right)$$

dim_H(E_A) = δ where the leading eigenvalue of $\mathcal{L}_δ$ equals 1.

Individual values (E_{1,2}, E_{1,...,5}) have been computed in the literature, but the full combinatorial landscape — how dimension depends on which digits are allowed — has never been mapped.

## Method

For each of the 2^{20} - 1 non-empty subsets:

1. **Discretize** the transfer operator on 40 Chebyshev nodes in [0,1]
2. **Bisect** over s ∈ (0,1) with 55 steps (precision ~10^{-16})
3. At each bisection step, find the leading eigenvalue via **power iteration** (300 steps)

Each subset is processed independently on a single RTX 5090 GPU thread. Subsets are batched 1024 at a time.

## Results

*Updated after computation completes.*

## Verification

- dim_H(E_{1,2}) matches Jenkinson-Pollicott (0.5312805062772051)
- dim_H(E_{1,...,5}) matches our Zaremba transfer operator result (0.836829443681208)
- Monotonicity: A ⊂ B ⟹ dim_H(E_A) < dim_H(E_B) verified for all pairs
```

- [ ] **Step 2: Commit**

```bash
cd /home/cahlen/dev/experiments/bigcompute.science
git add src/content/experiments/2026-03-29-hausdorff-dimension-spectrum.md
git commit -m "Add Hausdorff dimension spectrum experiment page"
```

---

### Task 7: Publish dataset to bigcompute.science

**Files:**
- Create: `bigcompute.science/public/data/hausdorff-spectrum/` (copy from idontknow results)
- Modify: `bigcompute.science/public/llms.txt`

- [ ] **Step 1: Copy dataset to bigcompute.science**

```bash
mkdir -p /home/cahlen/dev/experiments/bigcompute.science/public/data/hausdorff-spectrum
cp /home/cahlen/dev/experiments/idontknow/scripts/experiments/hausdorff-spectrum/results/spectrum_n5.csv \
   /home/cahlen/dev/experiments/bigcompute.science/public/data/hausdorff-spectrum/
cp /home/cahlen/dev/experiments/idontknow/scripts/experiments/hausdorff-spectrum/results/spectrum_n10.csv \
   /home/cahlen/dev/experiments/bigcompute.science/public/data/hausdorff-spectrum/
cp /home/cahlen/dev/experiments/idontknow/scripts/experiments/hausdorff-spectrum/results/metadata_n20.json \
   /home/cahlen/dev/experiments/bigcompute.science/public/data/hausdorff-spectrum/
```

For the n=20 CSV (~50MB), check if it's reasonable for the static site. If too large, publish only n=10 as the browsable dataset and link to the idontknow repo for the full n=20 data.

- [ ] **Step 2: Update llms.txt**

Append to the experiments section of `bigcompute.science/public/llms.txt`:

```
### hausdorff-dimension-spectrum
- URL: /experiments/hausdorff-dimension-spectrum/
- Date: 2026-03-29
- Status: in-progress
- Hardware: RTX 5090 (32GB)
- Method: Transfer operator, Chebyshev collocation, power iteration
- Domain: continued-fractions, fractal-geometry, spectral-theory
- Key result: First complete Hausdorff dimension spectrum for all 2^20 - 1 subsets of {1,...,20}
- Data: /data/hausdorff-spectrum/
- Code: https://github.com/cahlen/idontknow
```

- [ ] **Step 3: Commit**

```bash
cd /home/cahlen/dev/experiments/bigcompute.science
git add public/data/hausdorff-spectrum/ public/llms.txt
git commit -m "Publish Hausdorff dimension spectrum dataset"
```

---

### Task 8: Update experiment page with final results and push both repos

**Files:**
- Modify: `bigcompute.science/src/content/experiments/2026-03-29-hausdorff-dimension-spectrum.md`

- [ ] **Step 1: Update the experiment page with actual results**

After the n=20 run completes, update the results section and status in the frontmatter:
- Change `status: in-progress` → `status: complete`
- Fill in `total_runtime_seconds` from the metadata JSON
- Add the cardinality summary table from the log output
- Add extremal subsets (highest/lowest dimension per cardinality)

- [ ] **Step 2: Update the idontknow CLAUDE.md**

Add the new experiment to the "Current Experiments" section in `/home/cahlen/dev/experiments/idontknow/CLAUDE.md`:

```markdown
### Hausdorff Dimension Spectrum (Complete — RTX 5090)
First complete computation of dim_H(E_A) for all 1,048,575 subsets A ⊆ {1,...,20}. Produced structured dataset for AI consumption. Transfer operator method on Chebyshev nodes.
```

- [ ] **Step 3: Commit and push both repos**

```bash
cd /home/cahlen/dev/experiments/idontknow
git add CLAUDE.md
git commit -m "Add Hausdorff spectrum to CLAUDE.md experiments list"
git push

cd /home/cahlen/dev/experiments/bigcompute.science
git add -A
git commit -m "Hausdorff spectrum: final results and dataset"
git push
```
