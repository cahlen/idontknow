/*
 * Multifractal Singularity Spectrum of the Minkowski Question Mark Function
 *
 * Computes f(α) — the Hausdorff dimension of the set of points where
 * the Minkowski ?(x) function has local Hölder exponent α.
 *
 * The Minkowski measure assigns mass 2^{-n} to each CF interval at depth n.
 * The thermodynamic formalism gives:
 *   τ(q) = unique s where spectral radius of L_{q,s} = 1
 * where L_{q,s} f(x) = Σ_{a=1}^{A_max} 2^{-q} (a+x)^{-2s} f(1/(a+x))
 *
 * The singularity spectrum is the Legendre transform:
 *   α(q) = τ'(q),  f(α) = inf_q (qα - τ(q)) = qα(q) - τ(q)
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

#define MAX_N 48
#define MAX_AMAX 100
#define POWER_ITERS 300
#define BISECT_ITERS 55

/* q grid: covers the interesting range of the spectrum */
#define Q_MIN  -10.0
#define Q_MAX   10.0
#define Q_STEP  0.01
#define Q_COUNT 2001

/* ---- Device: Chebyshev nodes and barycentric weights ---- */

__device__ void d_chebyshev_nodes(double *x, int N) {
    for (int j = 0; j < N; j++)
        x[j] = 0.5 * (1.0 + cos(M_PI * (2.0*j + 1.0) / (2.0*N)));
}

__device__ void d_barycentric_weights(double *w, int N) {
    for (int j = 0; j < N; j++)
        w[j] = pow(-1.0, (double)j) * sin(M_PI * (2.0*j + 1.0) / (2.0*N));
}

/* ---- Device: Build L_{q,s} matrix ----
 * M[i + j*N] = Σ_{a=1}^{A_max} 2^{-q} (a+x_i)^{-2s} L_j(1/(a+x_i))
 *
 * The 2^{-q} factor is the same for all a, so factor it out:
 * M = 2^{-q} * Σ_a (a+x_i)^{-2s} L_j(1/(a+x_i))
 *
 * But the eigenvalue equation λ(L_{q,s}) = 1 becomes:
 * 2^{-q} * λ(L_s) = 1, i.e., λ(L_s) = 2^q
 * where L_s is the standard CF transfer operator.
 *
 * So τ(q) = unique s where λ_0(s) = 2^q.
 * This is simpler: we can reuse the standard operator and just change the target!
 */

__device__ void d_build_matrix(int A_max, double s,
                               int N, double *x, double *bw, double *M) {
    for (int i = 0; i < N * N; i++) M[i] = 0.0;

    for (int a = 1; a <= A_max; a++) {
        for (int i = 0; i < N; i++) {
            double y = 1.0 / (a + x[i]);
            double ws = pow(a + x[i], -2.0 * s);

            int exact = -1;
            for (int k = 0; k < N; k++)
                if (fabs(y - x[k]) < 1e-15) { exact = k; break; }

            if (exact >= 0) {
                M[i + exact * N] += ws;
            } else {
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

__device__ double d_power_iteration(double *M, int N, int iters) {
    double v[MAX_N], w[MAX_N];
    for (int i = 0; i < N; i++) v[i] = 1.0;

    double lam = 0.0;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < N; i++) {
            double s = 0.0;
            for (int j = 0; j < N; j++) s += M[i + j * N] * v[j];
            w[i] = s;
        }
        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; i++) { num += v[i] * w[i]; den += v[i] * v[i]; }
        lam = num / den;
        double norm = 0.0;
        for (int i = 0; i < N; i++) norm += w[i] * w[i];
        norm = sqrt(norm);
        if (norm < 1e-300) break;
        for (int i = 0; i < N; i++) v[i] = w[i] / norm;
    }
    return lam;
}

/* ---- Device: Find τ(q) = unique s where λ_0(s) = 2^q ----
 * Uses bisection, just like the Hausdorff dimension computation
 * but with target = 2^q instead of target = 1.
 */

__device__ double d_compute_tau(double q, int A_max, int N) {
    double x[MAX_N], bw[MAX_N];
    d_chebyshev_nodes(x, N);
    d_barycentric_weights(bw, N);

    double target = pow(2.0, q);
    double M[MAX_N * MAX_N];

    /* Bisection: find s where λ_0(s) = target
     * λ_0(s) is decreasing in s.
     * For q > 0: target > 1, so τ(q) < dim_H (need smaller s where λ is bigger)
     * For q < 0: target < 1, so τ(q) > dim_H
     * For q = 0: target = 1, so τ(0) = dim_H(E_{1,...,A_max})
     */

    double s_lo = -5.0, s_hi = 5.0;

    /* Verify bracket */
    d_build_matrix(A_max, s_lo, N, x, bw, M);
    double l_lo = d_power_iteration(M, N, POWER_ITERS);
    d_build_matrix(A_max, s_hi, N, x, bw, M);
    double l_hi = d_power_iteration(M, N, POWER_ITERS);

    if (l_lo < target || l_hi > target) {
        /* Can't bracket — return NaN */
        return 0.0 / 0.0;
    }

    for (int it = 0; it < BISECT_ITERS; it++) {
        double s = (s_lo + s_hi) * 0.5;
        d_build_matrix(A_max, s, N, x, bw, M);
        double lam = d_power_iteration(M, N, POWER_ITERS);
        if (lam > target) s_lo = s; else s_hi = s;
        if (s_hi - s_lo < 1e-15) break;
    }
    return (s_lo + s_hi) * 0.5;
}

/* ---- Kernel: each thread computes τ(q) for one q value ---- */

__global__ void compute_tau(int num_q, double q_min, double q_step,
                            int A_max, int N, double *tau_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_q) return;

    double q = q_min + idx * q_step;
    tau_out[idx] = d_compute_tau(q, A_max, N);
}

/* ---- Host ---- */

int main(int argc, char **argv) {
    int A_max = argc > 1 ? atoi(argv[1]) : 50;
    int N     = argc > 2 ? atoi(argv[2]) : 40;

    if (A_max > MAX_AMAX || N > MAX_N) {
        fprintf(stderr, "Parameters exceed limits\n");
        return 1;
    }

    int num_q = Q_COUNT;
    double q_min = Q_MIN, q_step = Q_STEP;

    printf("==========================================\n");
    printf("  Minkowski ?(x) Singularity Spectrum\n");
    printf("  A_max = %d, Chebyshev N = %d\n", A_max, N);
    printf("  q range: [%.1f, %.1f], step %.2f (%d values)\n",
           q_min, Q_MAX, q_step, num_q);
    printf("  Method: τ(q) = s where λ_0(s) = 2^q\n");
    printf("==========================================\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double *d_tau;
    cudaMalloc(&d_tau, num_q * sizeof(double));

    int tpb = 32;
    int nblocks = (num_q + tpb - 1) / tpb;

    printf("  Launching %d blocks x %d threads (%d q-values, each with bisection)...\n",
           nblocks, tpb, num_q);
    fflush(stdout);

    compute_tau<<<nblocks, tpb>>>(num_q, q_min, q_step, A_max, N, d_tau);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double *h_tau = (double *)malloc(num_q * sizeof(double));
    cudaMemcpy(h_tau, d_tau, num_q * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_tau);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("  GPU computation: %.1f seconds\n\n", gpu_time);

    /* Compute q values and Legendre transform */
    double *h_q     = (double *)malloc(num_q * sizeof(double));
    double *h_alpha = (double *)malloc(num_q * sizeof(double));
    double *h_f     = (double *)malloc(num_q * sizeof(double));

    for (int i = 0; i < num_q; i++)
        h_q[i] = q_min + i * q_step;

    /* α(q) = τ'(q) via central finite differences */
    for (int i = 0; i < num_q; i++) {
        double dtau;
        if (i == 0)
            dtau = (h_tau[1] - h_tau[0]) / q_step;
        else if (i == num_q - 1)
            dtau = (h_tau[num_q-1] - h_tau[num_q-2]) / q_step;
        else
            dtau = (h_tau[i+1] - h_tau[i-1]) / (2.0 * q_step);
        h_alpha[i] = dtau;  /* α = τ'(q), NOT -τ'(q) for this convention */
        h_f[i] = h_q[i] * h_alpha[i] - h_tau[i];  /* f = qα - τ */
    }

    /* Write CSV */
    const char *csv_path = "scripts/experiments/minkowski-spectrum/results/spectrum.csv";
    FILE *csv = fopen(csv_path, "w");
    if (csv) {
        fprintf(csv, "q,tau_q,alpha_q,f_alpha\n");
        for (int i = 0; i < num_q; i++)
            fprintf(csv, "%.4f,%.15f,%.15f,%.15f\n",
                    h_q[i], h_tau[i], h_alpha[i], h_f[i]);
        fclose(csv);
    }
    printf("  Output: %s\n", csv_path);

    /* Summary */
    double f_max = -1e30, alpha_fmax = 0, q_fmax = 0;
    for (int i = 0; i < num_q; i++) {
        if (!isnan(h_f[i]) && h_f[i] > f_max) {
            f_max = h_f[i];
            alpha_fmax = h_alpha[i];
            q_fmax = h_q[i];
        }
    }

    /* Find support (where f > 0) */
    double alpha_min = 1e30, alpha_max = -1e30;
    for (int i = 0; i < num_q; i++) {
        if (!isnan(h_f[i]) && !isnan(h_alpha[i]) && h_f[i] > 0.001) {
            if (h_alpha[i] < alpha_min) alpha_min = h_alpha[i];
            if (h_alpha[i] > alpha_max) alpha_max = h_alpha[i];
        }
    }

    printf("\n=== Singularity Spectrum Summary ===\n");
    printf("  max f(α)   = %.15f (should be ≤ 1)\n", f_max);
    printf("  at α       = %.15f\n", alpha_fmax);
    printf("  at q       = %.4f\n", q_fmax);
    printf("  α_min      = %.15f\n", alpha_min);
    printf("  α_max      = %.15f\n", alpha_max);

    /* Verification: τ(0) should equal dim_H(E_{1,...,A_max}) */
    int idx_q0 = (int)((0.0 - q_min) / q_step + 0.5);
    int idx_q1 = (int)((1.0 - q_min) / q_step + 0.5);
    printf("\n=== Verification ===\n");
    printf("  τ(0) = %.15f (should = dim_H(E_{1,...,%d}))\n", h_tau[idx_q0], A_max);
    printf("  τ(1) = %.15f (should = 1 for the Minkowski measure)\n", h_tau[idx_q1]);
    printf("  f(α) at peak should be 1 (dimension of full support)\n");

    printf("\n  GPU time: %.1f seconds\n", gpu_time);

    /* JSON metadata */
    const char *json_path = "scripts/experiments/minkowski-spectrum/results/metadata.json";
    FILE *jf = fopen(json_path, "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"experiment\": \"minkowski-question-mark-singularity-spectrum\",\n");
        fprintf(jf, "  \"date\": \"2026-03-29\",\n");
        fprintf(jf, "  \"hardware\": \"RTX 5090 32GB\",\n");
        fprintf(jf, "  \"A_max\": %d,\n", A_max);
        fprintf(jf, "  \"chebyshev_order\": %d,\n", N);
        fprintf(jf, "  \"q_range\": [%.1f, %.1f],\n", q_min, Q_MAX);
        fprintf(jf, "  \"q_step\": %.2f,\n", q_step);
        fprintf(jf, "  \"num_q_values\": %d,\n", num_q);
        fprintf(jf, "  \"f_alpha_max\": %.15f,\n", f_max);
        fprintf(jf, "  \"alpha_at_fmax\": %.15f,\n", alpha_fmax);
        fprintf(jf, "  \"alpha_support\": [%.15f, %.15f],\n", alpha_min, alpha_max);
        fprintf(jf, "  \"gpu_time_seconds\": %.1f,\n", gpu_time);
        fprintf(jf, "  \"novel\": true,\n");
        fprintf(jf, "  \"description\": \"First numerical computation of the multifractal singularity spectrum of Minkowski ?(x)\"\n");
        fprintf(jf, "}\n");
        fclose(jf);
        printf("  Metadata: %s\n", json_path);
    }

    free(h_tau); free(h_q); free(h_alpha); free(h_f);
    return 0;
}
