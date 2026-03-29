/*
 * FLAT Spectral Gap: permutation-only, no Chebyshev weights
 *
 * For each prime p, compute eigenvalues of the operator
 *   T = Σ_{a=1}^5 P_a
 * where P_a is the permutation matrix of g_a on P^1(F_p).
 *
 * This is a (p+1)×(p+1) sparse matrix with exactly 5 nonzeros per row.
 * Power iteration is O(5·(p+1)) per step — trivially fast.
 *
 * The flat gap σ_flat ≤ σ_weighted (heuristically), so proving the
 * flat gap gives a lower bound on the weighted gap we need.
 *
 * More importantly: the flat eigenvalues are related to Kloosterman
 * sums over F_p, which satisfy the Weil bound |K(a,b;p)| ≤ 2√p.
 * If we can show |λ_2| ≤ C/√p for explicit C, then σ_flat ≥ 0.498
 * for p > (C/0.502)², reducing the conjecture to finite verification.
 *
 * ALL primes processed in ONE kernel launch (one block per prime).
 * Pure GPU, zero CPU in the loop. FP64.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o flat_gap flat_spectral_gap.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define BOUND 5
#define MAX_ITER 500

// Modular inverse via Fermat
__device__ int mod_inv(int x, int p) {
    long long r = 1, b = x % p;
    if (b < 0) b += p;
    int e = p - 2;
    while (e > 0) {
        if (e & 1) r = r * b % p;
        b = b * b % p;
        e >>= 1;
    }
    return (int)r;
}

// Sparse matvec: v_out = T · v_in where T = Σ_a P_a
// P_a(k) computed on-the-fly
__device__ void apply_T(int p, int p1, double *v_in, double *v_out, int tid, int nthreads) {
    for (int k = tid; k < p1; k += nthreads) {
        v_out[k] = 0;
    }
    __syncthreads();

    for (int a = 1; a <= BOUND; a++) {
        for (int k = tid; k < p1; k += nthreads) {
            int pk;
            if (k == p) pk = a % p;        // ∞ → a
            else if (k == 0) pk = p;         // 0 → ∞
            else {
                int kinv = mod_inv(k, p);
                pk = (int)(((long long)a * k + 1) % p * kinv % p);
            }
            atomicAdd(&v_out[pk], v_in[k]);
        }
        __syncthreads();
    }
}

__global__ void flat_gap_kernel(
    int *d_primes, int num_primes,
    long long *d_offsets,
    double *d_workspace,
    double *d_gaps,
    double *d_lambda2s  // also output |λ₂|
) {
    int pidx = blockIdx.x;
    if (pidx >= num_primes) return;

    int p = d_primes[pidx];
    int p1 = p + 1;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    double *v = d_workspace + d_offsets[pidx];
    double *w = v + p1;
    double *v1 = w + p1; // stored leading eigenvector

    // Initialize
    for (int k = tid; k < p1; k += nt) v[k] = 1.0;
    __syncthreads();

    // Leading eigenvector (eigenvalue = 5, eigenvector = constant)
    // T · (1,1,...,1) = 5 · (1,1,...,1) since each P_a is a permutation
    // So λ₁ = 5 exactly, v₁ = (1,...,1)/√(p+1)
    double inv_sqrt = 1.0 / sqrt((double)p1);
    for (int k = tid; k < p1; k += nt) v1[k] = inv_sqrt;
    __syncthreads();

    // Initialize v orthogonal to v1
    for (int k = tid; k < p1; k += nt) {
        v[k] = sin(k * 1.618 + pidx * 3.14 + 0.5);
    }
    __syncthreads();

    // Project out v1
    __shared__ double reduce[256];
    double local_dot = 0;
    for (int k = tid; k < p1; k += nt) local_dot += v[k] * v1[k];
    reduce[tid] = local_dot;
    __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    double dot = reduce[0];
    for (int k = tid; k < p1; k += nt) v[k] -= dot * v1[k];
    __syncthreads();

    double eigenvalue = 0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // w = T · v
        apply_T(p, p1, v, w, tid, nt);

        // Project out v1
        local_dot = 0;
        for (int k = tid; k < p1; k += nt) local_dot += w[k] * v1[k];
        reduce[tid] = local_dot;
        __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        dot = reduce[0];
        for (int k = tid; k < p1; k += nt) w[k] -= dot * v1[k];
        __syncthreads();

        // Rayleigh quotient: λ = (v·w)/(v·v)
        double local_vw = 0, local_vv = 0;
        for (int k = tid; k < p1; k += nt) {
            local_vw += v[k] * w[k];
            local_vv += v[k] * v[k];
        }
        reduce[tid] = local_vw;
        __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        double vw = reduce[0];

        reduce[tid] = local_vv;
        __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        double vv = reduce[0];

        eigenvalue = vw / vv;

        // Normalize w
        double local_ww = 0;
        for (int k = tid; k < p1; k += nt) local_ww += w[k] * w[k];
        reduce[tid] = local_ww;
        __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        double norm = sqrt(reduce[0]);
        if (norm > 1e-30) {
            double inv = 1.0 / norm;
            for (int k = tid; k < p1; k += nt) w[k] *= inv;
        }
        __syncthreads();

        // Swap
        double *tmp = v; v = w; w = tmp;
    }

    if (tid == 0) {
        // λ₁ = 5 (exact for permutation sum)
        // σ = 1 - |λ₂|/λ₁ = 1 - |eigenvalue|/5
        d_lambda2s[pidx] = eigenvalue;
        d_gaps[pidx] = 1.0 - fabs(eigenvalue) / 5.0;
    }
}

int main(int argc, char **argv) {
    int max_p = argc > 1 ? atoi(argv[1]) : 100000;

    printf("Flat Spectral Gap (permutation-only) for primes to %d\n", max_p);
    printf("FP64, one block per prime, ONE kernel launch\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Sieve
    char *sieve = (char*)calloc(max_p + 1, 1);
    memset(sieve, 1, max_p + 1);
    sieve[0] = sieve[1] = 0;
    for (int i = 2; (long long)i*i <= max_p; i++)
        if (sieve[i]) for (int j = i*i; j <= max_p; j += i) sieve[j] = 0;

    int np = 0;
    for (int p = 2; p <= max_p; p++) if (sieve[p]) np++;

    int *h_primes = (int*)malloc(np * sizeof(int));
    long long *h_offsets = (long long*)malloc(np * sizeof(long long));
    int idx = 0;
    long long total = 0;
    for (int p = 2; p <= max_p; p++) {
        if (!sieve[p]) continue;
        h_primes[idx] = p;
        h_offsets[idx] = total;
        total += 3LL * (p + 1); // v, w, v1
        idx++;
    }

    printf("Primes: %d, workspace: %.2f GB\n\n", np, total * 8.0 / 1e9);

    int *d_primes; long long *d_offsets;
    double *d_ws, *d_gaps, *d_lam2;
    cudaMalloc(&d_primes, np * sizeof(int));
    cudaMalloc(&d_offsets, np * sizeof(long long));
    cudaMalloc(&d_ws, total * sizeof(double));
    cudaMalloc(&d_gaps, np * sizeof(double));
    cudaMalloc(&d_lam2, np * sizeof(double));
    cudaMemcpy(d_primes, h_primes, np * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, np * sizeof(long long), cudaMemcpyHostToDevice);

    struct timespec tk0, tk1;
    clock_gettime(CLOCK_MONOTONIC, &tk0);

    flat_gap_kernel<<<np, 256>>>(d_primes, np, d_offsets, d_ws, d_gaps, d_lam2);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &tk1);
    double kt = (tk1.tv_sec - tk0.tv_sec) + (tk1.tv_nsec - tk0.tv_nsec) / 1e9;

    double *h_gaps = (double*)malloc(np * sizeof(double));
    double *h_lam2 = (double*)malloc(np * sizeof(double));
    cudaMemcpy(h_gaps, d_gaps, np * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lam2, d_lam2, np * sizeof(double), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Analysis
    double min_gap = 999; int min_gap_p = 0;
    double max_lam2_norm = 0; int max_lam2_p = 0;

    printf("%8s  %12s  %12s  %12s\n", "p", "λ₂", "|λ₂|/√p", "σ_flat");
    printf("--------  ------------  ------------  ------------\n");

    for (int i = 0; i < np; i++) {
        double gap = h_gaps[i];
        double lam2 = h_lam2[i];
        double lam2_norm = fabs(lam2) / sqrt((double)h_primes[i]);

        if (gap < min_gap) { min_gap = gap; min_gap_p = h_primes[i]; }
        if (lam2_norm > max_lam2_norm) { max_lam2_norm = lam2_norm; max_lam2_p = h_primes[i]; }

        // Print small primes and tight gaps
        if (h_primes[i] <= 100 || gap < 0.50 ||
            h_primes[i] % 10000 < 50 || i == np - 1) {
            printf("%8d  %12.6f  %12.6f  %12.6f", h_primes[i], lam2, lam2_norm, gap);
            if (gap < 0.50) printf("  <-- tight");
            printf("\n");
        }
    }

    printf("\n========================================\n");
    printf("Primes: %d (to p=%d)\n", np, max_p);
    printf("Kernel time: %.2fs\n", kt);
    printf("Total time: %.2fs\n", tt);
    printf("Min flat gap: %.6f at p=%d\n", min_gap, min_gap_p);
    printf("Max |λ₂|/√p: %.6f at p=%d\n", max_lam2_norm, max_lam2_p);
    printf("\nWeil bound test: if |λ₂| ≤ C·√p for all p,\n");
    printf("then C ≤ %.6f (from data).\n", max_lam2_norm);
    printf("For σ_flat ≥ 0.498: need |λ₂| < 0.502×5 = 2.51\n");
    printf("This holds for p > (C·√p < 2.51) → p > (%.2f/2.51)² = %.0f\n",
           max_lam2_norm * sqrt((double)max_lam2_p),
           pow(max_lam2_norm * sqrt((double)max_lam2_p) / 2.51, 2));
    printf("========================================\n");

    cudaFree(d_primes); cudaFree(d_offsets);
    cudaFree(d_ws); cudaFree(d_gaps); cudaFree(d_lam2);
    free(h_primes); free(h_offsets); free(h_gaps); free(h_lam2); free(sieve);
    return 0;
}
