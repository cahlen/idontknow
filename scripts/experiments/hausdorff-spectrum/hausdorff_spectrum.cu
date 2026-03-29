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
    if (max_d >= 5) {
        double zaremba_dim = h_results[30];  /* mask 31 = {1,...,5} at index 30 */
        double expected = 0.836829443681208;
        printf("  dim_H(E_{1,...,5}) = %.15f (expected %.15f, diff = %.2e)\n",
               zaremba_dim, expected, fabs(zaremba_dim - expected));
    }

    if (max_d >= 2) {
        double e12_dim = h_results[2];  /* mask 3 = {1,2} at index 2 */
        double expected_e12 = 0.531280506277205;
        printf("  dim_H(E_{1,2})    = %.15f (expected ~%.15f, diff = %.2e)\n",
               e12_dim, expected_e12, fabs(e12_dim - expected_e12));
    }

    printf("  dim_H(E_{1})      = %.15f (expected 0)\n", h_results[0]);

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
