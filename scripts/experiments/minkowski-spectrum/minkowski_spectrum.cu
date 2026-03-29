/*
 * Multifractal Singularity Spectrum of the Minkowski Question Mark Function
 *
 * Computes f(α) — the Hausdorff dimension of the set of points where
 * the Minkowski ?(x) function has local Hölder exponent α.
 *
 * Method: thermodynamic formalism via the Gauss map transfer operator
 *   (L_q f)(x) = Σ_{a=1}^{A_max} (a+x)^{-2q} f(1/(a+x))
 *
 * For each q on a grid:
 *   1. Build L_q on Chebyshev nodes, find leading eigenvalue λ(q)
 *   2. Pressure: P(q) = log(λ(q))
 *   3. Free energy: τ(q) = P(q) / log(2)
 *   4. Legendre transform: α(q) = -τ'(q), f(α) = q·α + τ(q)
 *
 * Hardware: RTX 5090 (32GB VRAM, compute capability 12.0)
 * Compile: nvcc -O3 -arch=sm_120 -o minkowski_spectrum \
 *          scripts/experiments/minkowski-spectrum/minkowski_spectrum.cu -lm
 * Run:     ./minkowski_spectrum [A_max] [chebyshev_order]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_N 48          /* max Chebyshev order */
#define MAX_AMAX 100      /* max number of CF branches */
#define POWER_ITERS 300   /* power iteration steps */

/* q grid parameters */
#define Q_MIN  -20.0
#define Q_MAX   20.0
#define Q_STEP  0.004
#define Q_COUNT 10001     /* (Q_MAX - Q_MIN) / Q_STEP + 1 */

/* ============================================================
 * Device: Chebyshev nodes on [0,1] and barycentric weights
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
 * Device: Build transfer operator matrix at parameter q
 *
 * M[i + j*N] = Σ_{a=1}^{A_max} (a+x_i)^{-2q} * L_j(1/(a+x_i))
 * where L_j is the j-th barycentric interpolant basis function.
 * ============================================================ */

__device__ void d_build_matrix(int A_max, double q,
                               int N, double *x, double *bw, double *M) {
    for (int i = 0; i < N * N; i++) M[i] = 0.0;

    for (int a = 1; a <= A_max; a++) {
        for (int i = 0; i < N; i++) {
            double y = 1.0 / (a + x[i]);
            double ws = pow(a + x[i], -2.0 * q);

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
 * Device: Compute λ(q) for a single q value
 * ============================================================ */

__device__ double d_compute_lambda(double q, int A_max, int N) {
    double x[MAX_N], bw[MAX_N];
    d_chebyshev_nodes(x, N);
    d_barycentric_weights(bw, N);

    double M[MAX_N * MAX_N];
    d_build_matrix(A_max, q, N, x, bw, M);
    return d_power_iteration(M, N, POWER_ITERS);
}

/* ============================================================
 * Kernel: each thread computes λ(q) for one q value
 * ============================================================ */

__global__ void compute_eigenvalues(int num_q, double q_min, double q_step,
                                    int A_max, int N, double *lambda_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_q) return;

    double q = q_min + idx * q_step;
    lambda_out[idx] = d_compute_lambda(q, A_max, N);
}

/* ============================================================
 * Host: main
 * ============================================================ */

int main(int argc, char **argv) {
    int A_max = argc > 1 ? atoi(argv[1]) : 50;
    int N     = argc > 2 ? atoi(argv[2]) : 40;

    if (A_max > MAX_AMAX) {
        fprintf(stderr, "A_max %d exceeds MAX_AMAX %d\n", A_max, MAX_AMAX);
        return 1;
    }
    if (N > MAX_N) {
        fprintf(stderr, "chebyshev_order %d exceeds MAX_N %d\n", N, MAX_N);
        return 1;
    }

    int num_q = Q_COUNT;
    double q_min = Q_MIN;
    double q_step = Q_STEP;

    printf("==========================================\n");
    printf("  Minkowski ? Singularity Spectrum\n");
    printf("  A_max = %d, Chebyshev N = %d\n", A_max, N);
    printf("  q range: [%.1f, %.1f], step %.4f (%d values)\n",
           q_min, Q_MAX, q_step, num_q);
    printf("  Power iterations = %d\n", POWER_ITERS);
    printf("==========================================\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Allocate device memory */
    double *d_lambda;
    cudaMalloc(&d_lambda, num_q * sizeof(double));

    /* Launch kernel — one thread per q value */
    int threads_per_block = 32;
    int num_blocks = (num_q + threads_per_block - 1) / threads_per_block;

    printf("  Launching %d blocks x %d threads...\n", num_blocks, threads_per_block);
    fflush(stdout);

    compute_eigenvalues<<<num_blocks, threads_per_block>>>(
        num_q, q_min, q_step, A_max, N, d_lambda);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Copy results back */
    double *h_lambda = (double *)malloc(num_q * sizeof(double));
    cudaMemcpy(h_lambda, d_lambda, num_q * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_lambda);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("  GPU computation: %.1f seconds\n\n", gpu_time);

    /* ============================================================
     * Host: compute P(q), τ(q), and Legendre transform
     * ============================================================ */

    double *h_q     = (double *)malloc(num_q * sizeof(double));
    double *h_P     = (double *)malloc(num_q * sizeof(double));
    double *h_tau   = (double *)malloc(num_q * sizeof(double));
    double *h_alpha = (double *)malloc(num_q * sizeof(double));
    double *h_f     = (double *)malloc(num_q * sizeof(double));

    double log2_val = log(2.0);

    for (int i = 0; i < num_q; i++) {
        h_q[i] = q_min + i * q_step;
        h_P[i] = log(h_lambda[i]);
        h_tau[i] = h_P[i] / log2_val;
    }

    /* Finite differences for τ'(q) → α(q), then f(α) = q·α + τ */
    /* Use central differences for interior, one-sided at boundaries */
    for (int i = 0; i < num_q; i++) {
        double dtau;
        if (i == 0) {
            dtau = (h_tau[1] - h_tau[0]) / q_step;
        } else if (i == num_q - 1) {
            dtau = (h_tau[num_q-1] - h_tau[num_q-2]) / q_step;
        } else {
            dtau = (h_tau[i+1] - h_tau[i-1]) / (2.0 * q_step);
        }
        h_alpha[i] = -dtau;
        h_f[i] = h_q[i] * h_alpha[i] + h_tau[i];
    }

    /* ============================================================
     * Write CSV output
     * ============================================================ */

    const char *csv_path = "scripts/experiments/minkowski-spectrum/results/spectrum.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Cannot open %s — did you mkdir -p results/?\n", csv_path);
        return 1;
    }
    fprintf(csv, "q,lambda_q,P_q,tau_q,alpha_q,f_alpha\n");
    for (int i = 0; i < num_q; i++) {
        fprintf(csv, "%.4f,%.15e,%.15e,%.15e,%.15e,%.15e\n",
                h_q[i], h_lambda[i], h_P[i], h_tau[i], h_alpha[i], h_f[i]);
    }
    fclose(csv);
    printf("  Output: %s\n", csv_path);

    /* ============================================================
     * Summary statistics
     * ============================================================ */

    /* Find max f(α) and corresponding α */
    double f_max = -1e30;
    double alpha_at_fmax = 0.0;
    int idx_fmax = 0;
    for (int i = 0; i < num_q; i++) {
        if (h_f[i] > f_max) {
            f_max = h_f[i];
            alpha_at_fmax = h_alpha[i];
            idx_fmax = i;
        }
    }

    /* Find α_min and α_max (support of spectrum) */
    /* α(q) is decreasing: α_min = α(q_max), α_max = α(q_min) */
    double alpha_min = h_alpha[num_q - 1];
    double alpha_max = h_alpha[0];

    /* Sanity: ensure α_min < α_max */
    if (alpha_min > alpha_max) {
        double tmp = alpha_min;
        alpha_min = alpha_max;
        alpha_max = tmp;
    }

    printf("\n=== Singularity Spectrum Summary ===\n");
    printf("  max f(α)   = %.15f\n", f_max);
    printf("  at α       = %.15f\n", alpha_at_fmax);
    printf("  at q       = %.4f\n", h_q[idx_fmax]);
    printf("  α_min      = %.15f  (q = %.1f)\n", alpha_min, Q_MAX);
    printf("  α_max      = %.15f  (q = %.1f)\n", alpha_max, Q_MIN);
    printf("  support    = [%.6f, %.6f]\n", alpha_min, alpha_max);

    /* Verification: at q=0, λ should equal the number of branches A_max
     * because L_0 f(x) = Σ f(1/(a+x)), spectral radius = # branches for f=1 */
    int idx_q0 = (int)((0.0 - q_min) / q_step + 0.5);
    printf("\n=== Verification ===\n");
    printf("  λ(q=0) = %.15f (expected: sum of branch contributions)\n", h_lambda[idx_q0]);
    printf("  P(q=0) = %.15f = log(λ(0))\n", h_P[idx_q0]);
    printf("  τ(q=0) = %.15f = P(0)/log(2)\n", h_tau[idx_q0]);

    /* At q=1, λ(1) should be 1 (since Σ 1/(a+x)^2 is the Gauss density normalizer) */
    int idx_q1 = (int)((1.0 - q_min) / q_step + 0.5);
    printf("  λ(q=1) = %.15f (expected ~1 for Gauss measure)\n", h_lambda[idx_q1]);
    printf("  P(q=1) = %.15e\n", h_P[idx_q1]);

    printf("\n  GPU time: %.1f seconds\n", gpu_time);

    /* ============================================================
     * Write JSON metadata
     * ============================================================ */

    const char *json_path = "scripts/experiments/minkowski-spectrum/results/metadata.json";
    FILE *jf = fopen(json_path, "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"experiment\": \"minkowski-question-mark-singularity-spectrum\",\n");
        fprintf(jf, "  \"date\": \"2026-03-29\",\n");
        fprintf(jf, "  \"hardware\": \"RTX 5090 32GB\",\n");
        fprintf(jf, "  \"A_max\": %d,\n", A_max);
        fprintf(jf, "  \"chebyshev_order\": %d,\n", N);
        fprintf(jf, "  \"q_min\": %.1f,\n", q_min);
        fprintf(jf, "  \"q_max\": %.1f,\n", Q_MAX);
        fprintf(jf, "  \"q_step\": %.4f,\n", q_step);
        fprintf(jf, "  \"num_q_values\": %d,\n", num_q);
        fprintf(jf, "  \"power_iterations\": %d,\n", POWER_ITERS);
        fprintf(jf, "  \"gpu_time_seconds\": %.1f,\n", gpu_time);
        fprintf(jf, "  \"f_alpha_max\": %.15f,\n", f_max);
        fprintf(jf, "  \"alpha_at_f_max\": %.15f,\n", alpha_at_fmax);
        fprintf(jf, "  \"alpha_min\": %.15f,\n", alpha_min);
        fprintf(jf, "  \"alpha_max\": %.15f,\n", alpha_max);
        fprintf(jf, "  \"description\": \"Multifractal singularity spectrum f(alpha) of the Minkowski question mark function via thermodynamic formalism\"\n");
        fprintf(jf, "}\n");
        fclose(jf);
        printf("  Metadata: %s\n", json_path);
    }

    /* Cleanup */
    free(h_lambda);
    free(h_q);
    free(h_P);
    free(h_tau);
    free(h_alpha);
    free(h_f);

    return 0;
}
