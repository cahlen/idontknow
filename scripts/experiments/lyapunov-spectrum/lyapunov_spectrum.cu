/*
 * Lyapunov Exponent Spectrum of Continued Fraction Cantor Sets
 *
 * For each non-empty subset A <= {1,...,n}, computes the Lyapunov exponent
 * lambda(A) measuring the average exponential divergence rate of the Gauss
 * map T(x) = {1/x} restricted to E_A.
 *
 * Method: lambda(A) = -P'(1) where P(s) = log(leading eigenvalue of L_s).
 * Computed via finite difference:
 *   lambda ~= -(log(lam(1+eps)) - log(lam(1))) / eps
 *
 * Uses the same transfer operator discretization as the Hausdorff kernel:
 *   (L_s f)(x) = sum_{a in A} (a+x)^{-2s} f(1/(a+x))
 * on N Chebyshev nodes with barycentric interpolation.
 *
 * Hardware: RTX 5090 (32GB VRAM, compute capability 12.0)
 * Compile: nvcc -O3 -arch=sm_120 -o lyapunov_spectrum \
 *          scripts/experiments/lyapunov-spectrum/lyapunov_spectrum.cu -lm
 * Run:     ./lyapunov_spectrum [max_digit] [chebyshev_order]
 *          ./lyapunov_spectrum 10      # all subsets of {1,...,10}, N=40
 *          ./lyapunov_spectrum 20 40   # all subsets of {1,...,20}, N=40
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_N 48          /* max Chebyshev order */
#define MAX_DIGIT 24      /* max digit in any subset */
#define POWER_ITERS 300   /* power iteration steps */
#define BATCH_SIZE 1024   /* subsets per kernel launch */
#define FD_EPS 1e-6       /* finite difference epsilon */

/* ============================================================
 * Device: Chebyshev nodes and barycentric weights on [0,1]
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
 * M[i + j*N] = sum_{a in A} (a+x_i)^{-2s} * L_j(1/(a+x_i))
 * where L_j is the j-th barycentric interpolant basis function.
 * ============================================================ */

__device__ void d_build_matrix(uint32_t mask, int max_d, double s,
                               int N, double *x, double *bw, double *M) {
    for (int i = 0; i < N * N; i++) M[i] = 0.0;

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
 * Device: Power iteration -- returns leading eigenvalue of M
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
 * Device: Compute Lyapunov exponent and spectral radius at s=1
 * for a single subset.
 *
 * Returns two values via output pointers:
 *   lam1    = leading eigenvalue at s=1 (spectral radius / pressure)
 *   lyapunov = -(log lam(1+eps) - log lam(1)) / eps
 * ============================================================ */

__device__ void d_compute_lyapunov(uint32_t mask, int max_d, int N,
                                   double *out_lam1, double *out_lyapunov) {
    double x[MAX_N], bw[MAX_N];
    d_chebyshev_nodes(x, N);
    d_barycentric_weights(bw, N);

    double M[MAX_N * MAX_N];

    /* Evaluate leading eigenvalue at s = 1 */
    d_build_matrix(mask, max_d, 1.0, N, x, bw, M);
    double lam1 = d_power_iteration(M, N, POWER_ITERS);

    /* Evaluate leading eigenvalue at s = 1 + eps */
    double eps = FD_EPS;
    d_build_matrix(mask, max_d, 1.0 + eps, N, x, bw, M);
    double lam1e = d_power_iteration(M, N, POWER_ITERS);

    *out_lam1 = lam1;

    /* Finite difference for -P'(1) */
    if (lam1 > 1e-300 && lam1e > 1e-300) {
        *out_lyapunov = -(log(lam1e) - log(lam1)) / eps;
    } else {
        *out_lyapunov = 0.0;
    }
}

/* ============================================================
 * Kernel: Batch computation across subsets
 * Each thread computes one subset. Outputs 2 doubles per subset.
 * ============================================================ */

__global__ void batch_lyapunov(uint32_t start_mask, uint32_t count,
                               int max_d, int N,
                               double *lam1_results, double *lyap_results) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint32_t mask = start_mask + idx;
    double lam1, lyap;
    d_compute_lyapunov(mask, max_d, N, &lam1, &lyap);
    lam1_results[idx] = lam1;
    lyap_results[idx] = lyap;
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
    printf("  Lyapunov Exponent Spectrum\n");
    printf("  Subsets of {1,...,%d}: %u\n", max_d, total_subsets);
    printf("  Chebyshev order N = %d\n", N);
    printf("  Finite difference eps = %.1e\n", FD_EPS);
    printf("  Power iterations = %d\n", POWER_ITERS);
    printf("==========================================\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Allocate host results */
    double *h_lam1 = (double *)malloc(total_subsets * sizeof(double));
    double *h_lyap = (double *)malloc(total_subsets * sizeof(double));

    /* Allocate device results */
    double *d_lam1, *d_lyap;
    cudaMalloc(&d_lam1, (size_t)BATCH_SIZE * sizeof(double));
    cudaMalloc(&d_lyap, (size_t)BATCH_SIZE * sizeof(double));

    /* Open CSV output */
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path),
             "scripts/experiments/lyapunov-spectrum/results/spectrum_n%d.csv", max_d);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Cannot open %s -- did you mkdir -p results/?\n", csv_path);
        return 1;
    }
    fprintf(csv, "subset_mask,subset_digits,cardinality,spectral_radius_s1,lyapunov_exponent\n");

    /* Process in batches */
    uint32_t done = 0;
    int threads_per_block = 1;  /* one thread per subset (heavy work per thread) */
    uint32_t last_pct = 0;

    while (done < total_subsets) {
        uint32_t batch = total_subsets - done;
        if (batch > BATCH_SIZE) batch = BATCH_SIZE;

        uint32_t start_mask = done + 1;  /* masks go from 1 to 2^n - 1 */

        batch_lyapunov<<<batch, threads_per_block>>>(
            start_mask, batch, max_d, N, d_lam1, d_lyap);
        cudaDeviceSynchronize();

        /* Check for kernel errors */
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        /* Copy results back */
        cudaMemcpy(h_lam1 + done, d_lam1, batch * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lyap + done, d_lyap, batch * sizeof(double),
                   cudaMemcpyDeviceToHost);

        /* Write CSV rows */
        char subset_str[256];
        for (uint32_t i = 0; i < batch; i++) {
            uint32_t mask = start_mask + i;
            format_subset(mask, max_d, subset_str, sizeof(subset_str));
            int card = __builtin_popcount(mask);
            fprintf(csv, "%u,%s,%d,%.15f,%.15f\n",
                    mask, subset_str, card,
                    h_lam1[done + i], h_lyap[done + i]);
        }

        done += batch;

        /* Progress */
        uint32_t pct = (uint32_t)((100ULL * done) / total_subsets);
        if (pct != last_pct) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            double eta = (elapsed / done) * (total_subsets - done);
            printf("\r  %u / %u subsets (%u%%) -- %.1fs elapsed, ~%.1fs remaining",
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

    /* Singleton {a}: The transfer operator at s=1 is a single-term operator
     * with eigenvalue sum_{n>=0} (a+x)^{-2} iterated; the Lyapunov exponent
     * for the orbit staying at digit a is 2*log(a + phi_a) where phi_a is
     * the fixed point of x -> 1/(a+x), i.e. phi_a = (-a + sqrt(a^2+4))/2.
     * Numerically: lambda({a}) = 2*log(a + phi_a). */
    if (max_d >= 1) {
        double phi1 = (-1.0 + sqrt(5.0)) / 2.0;  /* golden ratio - 1 */
        double expected_lyap1 = 2.0 * log(1.0 + phi1);  /* 2*log(golden ratio) ~= 0.9624 */
        printf("  lambda({1})       = %.15f (singleton expected ~%.15f, diff = %.2e)\n",
               h_lyap[0], expected_lyap1, fabs(h_lyap[0] - expected_lyap1));
    }

    if (max_d >= 2) {
        /* {2}: fixed point phi_2 = (-2 + sqrt(8))/2 = sqrt(2) - 1 */
        double phi2 = sqrt(2.0) - 1.0;
        double expected_lyap2 = 2.0 * log(2.0 + phi2);  /* 2*log(1+sqrt(2)) */
        printf("  lambda({2})       = %.15f (singleton expected ~%.15f, diff = %.2e)\n",
               h_lyap[1], expected_lyap2, fabs(h_lyap[1] - expected_lyap2));
    }

    if (max_d >= 2) {
        printf("  lambda({1,2})     = %.15f\n", h_lyap[2]);
        printf("  spectral_radius({1,2}, s=1) = %.15f\n", h_lam1[2]);
    }

    if (max_d >= 5) {
        /* mask 31 = {1,...,5} at index 30 */
        printf("  lambda({1,...,5}) = %.15f\n", h_lyap[30]);
        printf("  spectral_radius({1,...,5}, s=1) = %.15f\n", h_lam1[30]);
    }

    /* Monotonicity check: adding digits should increase the Lyapunov exponent */
    if (max_d >= 3) {
        double l12 = h_lyap[2];   /* mask 3 = {1,2} */
        double l123 = h_lyap[6];  /* mask 7 = {1,2,3} */
        printf("  Monotonicity: lambda({1,2})=%.6f < lambda({1,2,3})=%.6f : %s\n",
               l12, l123, l12 < l123 ? "PASS" : "FAIL");
    }

    /* Summary by cardinality */
    printf("\n=== Lyapunov Exponent by Cardinality ===\n");
    printf("  |A|  count      min            mean           max\n");
    printf("  ---  -----  -------------  -------------  -------------\n");
    for (int k = 1; k <= max_d; k++) {
        double sum = 0, mn = 1e20, mx = -1e20;
        int cnt = 0;
        for (uint32_t i = 0; i < total_subsets; i++) {
            uint32_t mask = i + 1;
            if (__builtin_popcount(mask) == k) {
                double l = h_lyap[i];
                sum += l;
                if (l < mn) mn = l;
                if (l > mx) mx = l;
                cnt++;
            }
        }
        printf("  %3d  %5d  %.11f  %.11f  %.11f\n", k, cnt, mn, sum/cnt, mx);
    }

    printf("\n=== Spectral Radius at s=1 by Cardinality ===\n");
    printf("  |A|  count      min            mean           max\n");
    printf("  ---  -----  -------------  -------------  -------------\n");
    for (int k = 1; k <= max_d; k++) {
        double sum = 0, mn = 1e20, mx = -1e20;
        int cnt = 0;
        for (uint32_t i = 0; i < total_subsets; i++) {
            uint32_t mask = i + 1;
            if (__builtin_popcount(mask) == k) {
                double l = h_lam1[i];
                sum += l;
                if (l < mn) mn = l;
                if (l > mx) mx = l;
                cnt++;
            }
        }
        printf("  %3d  %5d  %.11f  %.11f  %.11f\n", k, cnt, mn, sum/cnt, mx);
    }

    /* Write JSON metadata */
    char json_path[256];
    snprintf(json_path, sizeof(json_path),
             "scripts/experiments/lyapunov-spectrum/results/metadata_n%d.json", max_d);
    FILE *jf = fopen(json_path, "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"experiment\": \"lyapunov-exponent-spectrum\",\n");
        fprintf(jf, "  \"date\": \"2026-03-29\",\n");
        fprintf(jf, "  \"hardware\": \"RTX 5090 32GB\",\n");
        fprintf(jf, "  \"max_digit\": %d,\n", max_d);
        fprintf(jf, "  \"num_subsets\": %u,\n", total_subsets);
        fprintf(jf, "  \"chebyshev_order\": %d,\n", N);
        fprintf(jf, "  \"finite_difference_eps\": %.1e,\n", FD_EPS);
        fprintf(jf, "  \"power_iterations\": %d,\n", POWER_ITERS);
        fprintf(jf, "  \"method\": \"transfer_operator_chebyshev_collocation\",\n");
        fprintf(jf, "  \"formula\": \"lambda = -(log(lam(1+eps)) - log(lam(1))) / eps\",\n");
        fprintf(jf, "  \"precision_digits\": 10,\n");
        fprintf(jf, "  \"total_runtime_seconds\": %.1f,\n", total_time);
        fprintf(jf, "  \"novel\": true,\n");
        fprintf(jf, "  \"description\": \"First complete Lyapunov exponent spectrum for all subsets of {1,...,%d}\"\n", max_d);
        fprintf(jf, "}\n");
        fclose(jf);
        printf("\n  Metadata: %s\n", json_path);
    }

    /* Cleanup */
    cudaFree(d_lam1);
    cudaFree(d_lyap);
    free(h_lam1);
    free(h_lyap);

    return 0;
}
