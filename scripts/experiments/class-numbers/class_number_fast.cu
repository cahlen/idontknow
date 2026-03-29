/*
 * Fast class number computation via Euler product
 *
 * Instead of summing sqrt(d) terms of the Dirichlet series,
 * compute L(1, χ_d) via the Euler product over primes:
 *   L(1, χ_d) = product_{p prime} (1 - χ_d(p)/p)^{-1}
 *
 * Only need primes up to ~10000 for sufficient accuracy.
 * That's ~1200 primes vs ~10^6 Dirichlet terms = ~1000× faster.
 *
 * For h(d), we also need the regulator R(d) = log(ε_d) from the
 * CF expansion of √d. This is O(sqrt(d)) steps but the constant
 * is small (just integer arithmetic, no Kronecker symbols).
 *
 * The class number is: h(d) = round(sqrt(d) * L(1,χ_d) / (2*R(d)))
 *
 * One GPU thread per discriminant. Batched across millions of d.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o class_fast scripts/experiments/class-numbers/class_number_fast.cu -lm
 * Run:     ./class_fast <start_d> <end_d>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define NUM_PRIMES 1229  // primes up to 10000

typedef unsigned long long uint64;

// Primes stored in constant memory (fast access for all threads)
__constant__ int d_primes[NUM_PRIMES];
__constant__ int d_num_primes;

// Kronecker symbol (d/p) for prime p
// For odd prime p: this is the Legendre symbol = d^((p-1)/2) mod p
__device__ int kronecker(long long d, int p) {
    if (p == 2) {
        int dm8 = ((int)(d % 8) + 8) % 8;
        if (dm8 == 1 || dm8 == 7) return 1;
        if (dm8 == 3 || dm8 == 5) return -1;
        return 0;
    }
    // Legendre symbol via Euler's criterion: d^((p-1)/2) mod p
    long long a = ((d % p) + p) % p;
    if (a == 0) return 0;
    long long result = 1;
    long long exp = (p - 1) / 2;
    long long base = a;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % p;
        base = (base * base) % p;
        exp >>= 1;
    }
    return (result == 1) ? 1 : -1;
}

// Compute L(1, χ_d) via Euler product over preloaded primes
__device__ double euler_L1(long long d) {
    double product = 1.0;
    for (int i = 0; i < d_num_primes; i++) {
        int p = d_primes[i];
        int chi = kronecker(d, p);
        if (chi == 0) continue;  // p | d
        double term = 1.0 / (1.0 - (double)chi / (double)p);
        product *= term;
    }
    return product;
}

// Check if d is a fundamental discriminant
__device__ bool is_fundamental(uint64 d) {
    if (d <= 1) return false;
    uint64 dm4 = d % 4;
    if (dm4 == 1) {
        // Must be squarefree
        for (uint64 p = 2; p * p <= d && p < 100000; p++) {
            if (d % (p * p) == 0) return false;
        }
        return true;
    } else if (dm4 == 0) {
        uint64 m = d / 4;
        uint64 mm4 = m % 4;
        if (mm4 != 2 && mm4 != 3) return false;
        for (uint64 p = 2; p * p <= m && p < 100000; p++) {
            if (m % (p * p) == 0) return false;
        }
        return true;
    }
    return false;
}

// Compute regulator R(d) = log(fundamental unit) via CF of √d
__device__ double compute_regulator(uint64 d) {
    uint64 a0 = (uint64)sqrt((double)d);
    if (a0 * a0 == d) return 0.0;
    // Fix sqrt precision
    while ((a0+1)*(a0+1) <= d) a0++;
    while (a0*a0 > d) a0--;

    uint64 m = 0, dd = 1, a = a0;
    double P_prev = 1.0, P_curr = (double)a0;
    double Q_prev = 0.0, Q_curr = 1.0;
    double sqrtd = sqrt((double)d);

    for (int i = 0; i < 100000; i++) {
        m = dd * a - m;
        dd = (d - m * m) / dd;
        if (dd == 0) break;
        a = (a0 + m) / dd;

        double P_next = a * P_curr + P_prev;
        double Q_next = a * Q_curr + Q_prev;
        P_prev = P_curr; P_curr = P_next;
        Q_prev = Q_curr; Q_curr = Q_next;

        if (a == 2 * a0) {
            return log(P_curr + Q_curr * sqrtd);
        }
    }
    // Period didn't close — use current approximation
    return log(P_curr + Q_curr * sqrtd);
}

__global__ void compute_class_numbers(
    uint64 start_d, uint64 count,
    uint64 *h1_count, uint64 *total_count,
    uint64 *max_h_val, uint64 *max_h_d)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64 d = start_d + idx;
    if (!is_fundamental(d)) return;

    atomicAdd((unsigned long long*)total_count, 1ULL);

    double R = compute_regulator(d);
    if (R <= 0.0) return;

    double L1 = euler_L1((long long)d);
    double h_approx = sqrt((double)d) * L1 / (2.0 * R);
    uint64 h = (uint64)(h_approx + 0.5);
    if (h == 0) h = 1;

    if (h == 1) atomicAdd((unsigned long long*)h1_count, 1ULL);

    // Track max h
    // (Race condition acceptable — we just want approximate max)
    if (h > *max_h_val) {
        *max_h_val = h;
        *max_h_d = d;
    }
}

// CPU sieve for primes
void sieve_primes(int limit, int *primes, int *count) {
    char *is_p = (char*)calloc(limit + 1, 1);
    memset(is_p, 1, limit + 1);
    is_p[0] = is_p[1] = 0;
    for (int i = 2; (long long)i * i <= limit; i++)
        if (is_p[i]) for (int j = i * i; j <= limit; j += i) is_p[j] = 0;
    *count = 0;
    for (int i = 2; i <= limit && *count < NUM_PRIMES; i++)
        if (is_p[i]) primes[(*count)++] = i;
    free(is_p);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <start_d> <end_d> [gpu_id]\n", argv[0]);
        return 1;
    }

    uint64 start_d = (uint64)atoll(argv[1]);
    uint64 end_d = (uint64)atoll(argv[2]);
    int gpu_id = argc > 3 ? atoi(argv[3]) : 0;
    uint64 count = end_d - start_d + 1;

    printf("Fast Class Number Computation (Euler product)\n");
    printf("Range: d = %llu to %llu (%llu values)\n",
           (unsigned long long)start_d, (unsigned long long)end_d,
           (unsigned long long)count);
    printf("GPU: %d\n\n", gpu_id);

    cudaSetDevice(gpu_id);

    // Generate and upload primes
    int h_primes[NUM_PRIMES];
    int num_primes;
    sieve_primes(10000, h_primes, &num_primes);
    printf("Primes loaded: %d (up to %d)\n\n", num_primes, h_primes[num_primes-1]);

    cudaMemcpyToSymbol(d_primes, h_primes, num_primes * sizeof(int));
    cudaMemcpyToSymbol(d_num_primes, &num_primes, sizeof(int));

    uint64 *d_h1, *d_total, *d_max_h, *d_max_d;
    cudaMalloc(&d_h1, sizeof(uint64));
    cudaMalloc(&d_total, sizeof(uint64));
    cudaMalloc(&d_max_h, sizeof(uint64));
    cudaMalloc(&d_max_d, sizeof(uint64));
    cudaMemset(d_h1, 0, sizeof(uint64));
    cudaMemset(d_total, 0, sizeof(uint64));
    cudaMemset(d_max_h, 0, sizeof(uint64));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64 chunk = 100000000;  // 100M per launch
    for (uint64 offset = 0; offset < count; offset += chunk) {
        uint64 n = chunk;
        if (offset + n > count) n = count - offset;

        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compute_class_numbers<<<blocks, THREADS_PER_BLOCK>>>(
            start_d + offset, n, d_h1, d_total, d_max_h, d_max_d);
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        double progress = (double)(offset + n) / count * 100;

        uint64 h_total;
        cudaMemcpy(&h_total, d_total, sizeof(uint64), cudaMemcpyDeviceToHost);

        printf("[GPU %d] d=%llu..%llu (%.1f%%, %llu disc, %.1fs)\n",
               gpu_id, (unsigned long long)(start_d + offset),
               (unsigned long long)(start_d + offset + n),
               progress, (unsigned long long)h_total, elapsed);
        fflush(stdout);
    }

    uint64 h_h1, h_total, h_max_h, h_max_d;
    cudaMemcpy(&h_h1, d_h1, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total, d_total, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_h, d_max_h, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_d, d_max_d, sizeof(uint64), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    double h1_ratio = h_total > 0 ? (double)h_h1 / h_total : 0;
    double cl_prediction = 0.75446;

    printf("\n========================================\n");
    printf("Class Numbers: d = %llu to %llu\n",
           (unsigned long long)start_d, (unsigned long long)end_d);
    printf("Fundamental discriminants: %llu\n", (unsigned long long)h_total);
    printf("h=1 count: %llu (%.4f%%)\n", (unsigned long long)h_h1, 100.0 * h1_ratio);
    printf("Cohen-Lenstra prediction: %.4f%%\n", 100.0 * cl_prediction);
    printf("Ratio observed/predicted: %.6f\n", h1_ratio / cl_prediction);
    printf("Largest h: %llu (d=%llu)\n", (unsigned long long)h_max_h, (unsigned long long)h_max_d);
    printf("Time: %.1fs (%.0f disc/sec)\n", elapsed, h_total / elapsed);
    printf("========================================\n");

    cudaFree(d_h1); cudaFree(d_total);
    cudaFree(d_max_h); cudaFree(d_max_d);
    return 0;
}
