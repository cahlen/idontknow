/*
 * FP64/N=40 spectral gaps via implicit Kronecker — NO atomicAdd
 *
 * Each thread owns OUTPUT points (not input). For each output point k:
 *   w[i*p1 + k] = Σ_a Σ_j Ma[i][j] * v[j*p1 + Pa_inv(k)]
 *
 * where Pa_inv(k) is the PREIMAGE of k under Pa.
 * Since Pa is a permutation, Pa_inv is also a permutation.
 * We precompute Pa_inv for each a.
 *
 * No atomicAdd needed — each thread writes to its own output locations.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o verify_v2 verify_gaps_v2.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define BOUND 5
#define NC 40
#define MAX_ITER 500
#define DELTA 0.836829443681208

static double h_nodes[NC], h_bary[NC];
static double h_Ma[BOUND][NC * NC];

void init_chebyshev() {
    for (int j = 0; j < NC; j++) {
        h_nodes[j] = 0.5 * (1.0 + cos(M_PI * (2.0*j + 1.0) / (2.0*NC)));
        h_bary[j] = pow(-1.0, j) * sin(M_PI * (2.0*j + 1.0) / (2.0*NC));
    }
    for (int a = 0; a < BOUND; a++) {
        int digit = a + 1;
        memset(h_Ma[a], 0, NC * NC * sizeof(double));
        for (int i = 0; i < NC; i++) {
            double y = 1.0 / (digit + h_nodes[i]);
            double ws = pow(digit + h_nodes[i], -2.0 * DELTA);
            int exact = -1;
            for (int k = 0; k < NC; k++)
                if (fabs(y - h_nodes[k]) < 1e-15) { exact = k; break; }
            if (exact >= 0) {
                h_Ma[a][i * NC + exact] = ws;
            } else {
                double den = 0, num[NC];
                for (int j = 0; j < NC; j++) { num[j] = h_bary[j] / (y - h_nodes[j]); den += num[j]; }
                for (int j = 0; j < NC; j++) h_Ma[a][i * NC + j] = ws * num[j] / den;
            }
        }
    }
}

__constant__ double d_Ma[BOUND * NC * NC];

__device__ int mod_inv_dev(int x, int p) {
    long long r = 1, b = x % p;
    if (b < 0) b += p;
    int e = p - 2;
    while (e > 0) { if (e & 1) r = r * b % p; b = b * b % p; e >>= 1; }
    return (int)r;
}

// Compute Pa_inv(k): find x such that Pa(x) = k
// Pa(x) = (a*x+1)/x for x≠0, Pa(0)=∞, Pa(∞)=a
// Pa_inv(k) = x where (a*x+1)/x = k → a*x+1 = k*x → x(k-a) = 1 → x = 1/(k-a)
// Special: Pa_inv(∞) = 0, Pa_inv(a%p) = ∞(=p)
__device__ int perm_inv(int a, int k, int p) {
    if (k == p) return 0;           // ∞ ← 0
    if (k == a % p) return p;       // a ← ∞
    int diff = (k - a % p + p) % p;
    if (diff == 0) return p;        // shouldn't happen if k≠a%p
    return mod_inv_dev(diff, p);    // 1/(k-a) mod p
}

__global__ void gap_kernel(
    int *d_primes, int num_primes,
    long long *d_offsets,
    double *d_workspace,
    double *d_gaps
) {
    int pidx = blockIdx.x;
    if (pidx >= num_primes) return;

    int p = d_primes[pidx];
    int p1 = p + 1;
    int vec_sz = NC * p1;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    double *v = d_workspace + d_offsets[pidx];
    double *w = v + vec_sz;
    double *v1 = w + vec_sz;

    __shared__ double reduce[256];

    // === Find leading eigenvector v1 ===
    for (int idx = tid; idx < vec_sz; idx += nt) v1[idx] = 1.0;
    __syncthreads();

    for (int iter = 0; iter < 200; iter++) {
        // w = L * v1 using INVERSE permutations (no atomicAdd!)
        // w[i*p1 + k] = Σ_a Σ_j Ma[i][j] * v1[j*p1 + Pa_inv(k)]
        for (int k = tid; k < p1; k += nt) {
            for (int i = 0; i < NC; i++) {
                double sum = 0;
                for (int a = 0; a < BOUND; a++) {
                    int src_k = perm_inv(a + 1, k, p);
                    for (int j = 0; j < NC; j++)
                        sum += d_Ma[a * NC * NC + i * NC + j] * v1[j * p1 + src_k];
                }
                w[i * p1 + k] = sum;
            }
        }
        __syncthreads();

        // Normalize
        double ln = 0;
        for (int idx = tid; idx < vec_sz; idx += nt) ln += w[idx] * w[idx];
        reduce[tid] = ln; __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] += reduce[tid+s]; __syncthreads(); }
        double norm = sqrt(reduce[0]);
        if (norm > 1e-30) { double inv = 1.0/norm; for (int idx = tid; idx < vec_sz; idx += nt) v1[idx] = w[idx] * inv; }
        __syncthreads();
    }

    // === Deflated power iteration for λ₂ ===
    for (int idx = tid; idx < vec_sz; idx += nt) v[idx] = sin(idx * 1.618 + pidx * 3.14 + 0.5);
    __syncthreads();

    // Project out v1
    double ld = 0, ln1 = 0;
    for (int idx = tid; idx < vec_sz; idx += nt) { ld += v[idx]*v1[idx]; ln1 += v1[idx]*v1[idx]; }
    reduce[tid] = ld; __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] += reduce[tid+s]; __syncthreads(); }
    double dot = reduce[0];
    reduce[tid] = ln1; __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] += reduce[tid+s]; __syncthreads(); }
    double n1 = reduce[0];
    for (int idx = tid; idx < vec_sz; idx += nt) v[idx] -= (dot/n1) * v1[idx];
    __syncthreads();

    double eigenvalue = 0;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // w = L * v (inverse perm, no atomicAdd)
        for (int k = tid; k < p1; k += nt) {
            for (int i = 0; i < NC; i++) {
                double sum = 0;
                for (int a = 0; a < BOUND; a++) {
                    int src_k = perm_inv(a + 1, k, p);
                    for (int j = 0; j < NC; j++)
                        sum += d_Ma[a * NC * NC + i * NC + j] * v[j * p1 + src_k];
                }
                w[i * p1 + k] = sum;
            }
        }
        __syncthreads();

        // Project out v1
        ld = 0; ln1 = 0;
        for (int idx = tid; idx < vec_sz; idx += nt) { ld += w[idx]*v1[idx]; ln1 += v1[idx]*v1[idx]; }
        reduce[tid] = ld; __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] += reduce[tid+s]; __syncthreads(); }
        dot = reduce[0];
        reduce[tid] = ln1; __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] += reduce[tid+s]; __syncthreads(); }
        n1 = reduce[0];
        for (int idx = tid; idx < vec_sz; idx += nt) w[idx] -= (dot/n1) * v1[idx];
        __syncthreads();

        // Rayleigh quotient
        double lv = 0, lw = 0;
        for (int idx = tid; idx < vec_sz; idx += nt) { lv += v[idx]*w[idx]; lw += v[idx]*v[idx]; }
        reduce[tid] = lv; __syncthreads();
        for (int s=nt/2;s>0;s>>=1){if(tid<s)reduce[tid]+=reduce[tid+s];__syncthreads();}
        double vw = reduce[0];
        reduce[tid] = lw; __syncthreads();
        for (int s=nt/2;s>0;s>>=1){if(tid<s)reduce[tid]+=reduce[tid+s];__syncthreads();}
        double vv = reduce[0];
        eigenvalue = vw / vv;

        // Normalize
        double ln = 0;
        for (int idx = tid; idx < vec_sz; idx += nt) ln += w[idx]*w[idx];
        reduce[tid] = ln; __syncthreads();
        for (int s=nt/2;s>0;s>>=1){if(tid<s)reduce[tid]+=reduce[tid+s];__syncthreads();}
        double norm = sqrt(reduce[0]);
        if (norm > 1e-30) { double inv = 1.0/norm; for (int idx = tid; idx < vec_sz; idx += nt) w[idx] *= inv; }
        __syncthreads();
        double *tmp = v; v = w; w = tmp;
    }

    if (tid == 0) d_gaps[pidx] = 1.0 - fabs(eigenvalue);
}

int main(int argc, char **argv) {
    int lo_p = argc > 1 ? atoi(argv[1]) : 2;
    int hi_p = argc > 2 ? atoi(argv[2]) : 3500;

    printf("FP64/N=%d gaps for primes %d to %d (implicit Kronecker v2)\n\n", NC, lo_p, hi_p);
    init_chebyshev();
    cudaMemcpyToSymbol(d_Ma, h_Ma, sizeof(h_Ma));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    char *sieve = (char*)calloc(hi_p + 1, 1);
    memset(sieve, 1, hi_p + 1); sieve[0] = sieve[1] = 0;
    for (int i = 2; (long long)i*i <= hi_p; i++)
        if (sieve[i]) for (int j = i*i; j <= hi_p; j += i) sieve[j] = 0;

    int np = 0;
    for (int p = lo_p; p <= hi_p; p++) if (sieve[p]) np++;

    int *h_primes = (int*)malloc(np * sizeof(int));
    long long *h_offsets = (long long*)malloc(np * sizeof(long long));
    int idx = 0; long long total = 0;
    for (int p = lo_p; p <= hi_p; p++) {
        if (!sieve[p]) continue;
        h_primes[idx] = p;
        h_offsets[idx] = total;
        total += 3LL * NC * (p + 1);
        idx++;
    }
    printf("Primes: %d, workspace: %.2f GB\n", np, total * 8.0 / 1e9);

    int *d_primes; long long *d_offsets; double *d_ws, *d_gaps;
    cudaMalloc(&d_primes, np * sizeof(int));
    cudaMalloc(&d_offsets, np * sizeof(long long));
    cudaMalloc(&d_ws, total * sizeof(double));
    cudaMalloc(&d_gaps, np * sizeof(double));
    cudaMemcpy(d_primes, h_primes, np * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, np * sizeof(long long), cudaMemcpyHostToDevice);

    // Use 64 threads — balance between parallelism and register pressure
    gap_kernel<<<np, 64>>>(d_primes, np, d_offsets, d_ws, d_gaps);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }

    double *h_gaps = (double*)malloc(np * sizeof(double));
    cudaMemcpy(h_gaps, d_gaps, np * sizeof(double), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    double min_gap = 999; int min_p = 0, fails = 0;
    for (int i = 0; i < np; i++) {
        if (h_gaps[i] < min_gap) { min_gap = h_gaps[i]; min_p = h_primes[i]; }
        if (h_gaps[i] < 0.277) fails++;
        if (h_primes[i] <= 100 || h_gaps[i] < 0.40 || i == np-1 || i % 50 == 0)
            printf("p=%5d  σ=%.6f  %s%s\n", h_primes[i], h_gaps[i],
                h_gaps[i] >= 0.277 ? "PASS" : "FAIL",
                h_gaps[i] < 0.35 ? "  <-- TIGHT" : "");
    }
    printf("\n%d primes, min σ=%.6f at p=%d, fails(σ<0.277): %d, time: %.1fs\n",
        np, min_gap, min_p, fails, tt);
    if (fails == 0) printf("ALL PASS.\n");

    cudaFree(d_primes); cudaFree(d_offsets); cudaFree(d_ws); cudaFree(d_gaps);
    free(h_primes); free(h_offsets); free(h_gaps); free(sieve);
    return fails;
}
