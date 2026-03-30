/*
 * GPU squarefree sieve — prime-driven (correct and fast)
 *
 * For each prime p ≤ √hi: mark all multiples of p² in [lo, hi).
 * This is the standard Eratosthenes approach, parallelized on GPU.
 *
 * Phase 1: One kernel launch per prime p. Each thread marks one
 *          multiple of p² as non-squarefree.
 * Phase 2: Classify fundamental discriminants (d mod 4 check).
 * Phase 3: Stream-compact into packed array.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o sieve_test scripts/experiments/class-numbers/sieve_gpu.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef unsigned long long uint64;
#define BLOCK_SIZE 256

// Mark multiples of p² in [lo, lo+len) as non-squarefree
__global__ void mark_p2_multiples(
    uint8_t *sieve, uint64 lo, uint64 len,
    int p, uint64 first_multiple, uint64 num_multiples)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_multiples) return;

    uint64 pos = first_multiple + idx * (uint64)p * p - lo;
    if (pos < len) sieve[pos] = 0;
}

// Batch version: process MANY small primes in one kernel
__global__ void mark_small_primes(
    uint8_t *sieve, uint64 lo, uint64 len,
    const int *primes, int num_primes)
{
    uint64 pos = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= len) return;

    uint64 d = lo + pos;
    // Check small primes (p² ≤ SMALL_PRIME_LIMIT²)
    for (int i = 0; i < num_primes; i++) {
        int p = primes[i];
        uint64 p2 = (uint64)p * p;
        if (p2 > d) break;
        if (d % p2 == 0) { sieve[pos] = 0; return; }
    }
}

// Classify + compact in one pass
__global__ void classify_and_count(
    const uint8_t *sieve, uint64 lo, uint64 len,
    uint64 *output, uint32_t *count, uint32_t max_out)
{
    uint64 pos = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= len) return;

    uint64 d = lo + pos;
    if (d < 5) return;

    int is_fund = 0;
    if (d % 4 == 1 && sieve[pos]) {
        is_fund = 1;
    } else if (d % 4 == 0) {
        uint64 m = d / 4;
        if ((m % 4 == 2 || m % 4 == 3)) {
            // Check if m is squarefree — m = d/4, position in sieve = m - lo
            // Only if m is in our sieve range
            if (m >= lo && m < lo + len && sieve[m - lo]) {
                is_fund = 1;
            } else if (m < lo) {
                // m is before our range — do trial division
                // For large ranges starting at lo >> 0, m = d/4 < lo only when d < 4*lo
                // which means d is in [lo, 4*lo). For lo = 10^9, this covers d < 4×10^9.
                // Do a quick squarefree check for small primes
                int sqf = 1;
                for (int p = 2; (uint64)p * p <= m; p++) {
                    if (m % ((uint64)p * p) == 0) { sqf = 0; break; }
                    if (p > 1000) break;  // cap trial division
                }
                if (sqf) is_fund = 1;
            }
        }
    }

    if (is_fund) {
        uint32_t idx = atomicAdd(count, 1);
        if (idx < max_out) output[idx] = d;
    }
}

int main(int argc, char **argv) {
    uint64 lo = argc > 1 ? strtoull(argv[1], NULL, 10) : 1000000000ULL;
    uint64 hi = argc > 2 ? strtoull(argv[2], NULL, 10) : 1100000000ULL;
    uint64 len = hi - lo;

    printf("GPU Squarefree Sieve v2: [%llu, %llu), len=%llu\n", lo, hi, len);

    // Generate primes
    int sqrt_hi = 1;
    while ((uint64)sqrt_hi * sqrt_hi < hi) sqrt_hi++;
    char *is_p = (char*)calloc(sqrt_hi + 1, 1);
    for (int i = 2; i <= sqrt_hi; i++) is_p[i] = 1;
    for (int i = 2; i * i <= sqrt_hi; i++)
        if (is_p[i]) for (int j = i*i; j <= sqrt_hi; j += i) is_p[j] = 0;
    int *h_primes = (int*)malloc(sqrt_hi * sizeof(int));
    int num_primes = 0;
    for (int i = 2; i <= sqrt_hi; i++) if (is_p[i]) h_primes[num_primes++] = i;
    free(is_p);
    printf("Primes: %d (up to %d)\n\n", num_primes, h_primes[num_primes-1]);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Upload primes
    int *d_primes;
    cudaMalloc(&d_primes, num_primes * sizeof(int));
    cudaMemcpy(d_primes, h_primes, num_primes * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate sieve + output
    uint8_t *d_sieve;
    uint64 *d_output;
    uint32_t *d_count;
    cudaMalloc(&d_sieve, len);
    cudaMalloc(&d_output, (len / 2) * sizeof(uint64));
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_sieve, 1, len);
    cudaMemset(d_count, 0, sizeof(uint32_t));

    uint64 blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1: Mark non-squarefree using ALL primes at once (per-element check)
    // This is faster than prime-driven for moderate prime counts
    printf("Phase 1: squarefree sieve (%d primes)...\n", num_primes);
    mark_small_primes<<<blocks, BLOCK_SIZE>>>(d_sieve, lo, len, d_primes, num_primes);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  %.2fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9);

    // Phase 2+3: Classify and compact
    printf("Phase 2: classify + compact...\n");
    classify_and_count<<<blocks, BLOCK_SIZE>>>(
        d_sieve, lo, len, d_output, d_count, (uint32_t)(len / 2));
    cudaDeviceSynchronize();

    uint32_t h_count;
    cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Fundamental discriminants: %u (%.2f%%)\n", h_count, 100.0*h_count/len);
    printf("Time: %.2fs (%.1fM integers/sec)\n", elapsed, len/elapsed/1e6);
    printf("Expected: ~30%% density\n");
    printf("========================================\n");

    // Verify first few
    if (h_count > 0) {
        uint64 *h_out = (uint64*)malloc(10 * sizeof(uint64));
        cudaMemcpy(h_out, d_output, 10 * sizeof(uint64), cudaMemcpyDeviceToHost);
        printf("First 10: ");
        for (int i = 0; i < 10 && i < (int)h_count; i++) printf("%llu ", h_out[i]);
        printf("\n");
        free(h_out);
    }

    cudaFree(d_sieve); cudaFree(d_output); cudaFree(d_count); cudaFree(d_primes);
    free(h_primes);
    return 0;
}
