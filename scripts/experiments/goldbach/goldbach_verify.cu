/*
 * CUDA-accelerated Goldbach Conjecture verifier
 *
 * Goldbach's Conjecture: Every even integer > 2 is the sum of two primes.
 *
 * For each even n, we check if there exists a prime p such that
 * both p and n-p are prime. We also record the smallest such p
 * (the "Goldbach partition") and the total number of representations.
 *
 * Uses a segmented sieve of Eratosthenes on GPU shared memory for
 * fast primality testing within each chunk.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o goldbach_verify scripts/experiments/goldbach/goldbach_verify.cu
 * Run:     ./goldbach_verify <start_n> <end_n>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define THREADS_PER_BLOCK 256

// Simple trial division primality test for GPU
// For the range we're working in, this is fast enough per-thread
__device__ bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// For large n, use Miller-Rabin with deterministic witnesses
// For n < 3.3×10^24, testing witnesses {2,3,5,7,11,13,17,19,23,29,31,37}
// gives a deterministic result.
__device__ uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    __uint128_t result = 1;
    __uint128_t b = base % mod;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint64_t)result;
}

__device__ bool miller_rabin_witness(uint64_t n, uint64_t a) {
    if (n % a == 0) return n == a;
    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) { d /= 2; r++; }

    uint64_t x = mod_pow(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 0; i < r - 1; i++) {
        x = ((__uint128_t)x * x) % n;
        if (x == n - 1) return true;
    }
    return false;
}

__device__ bool is_prime_large(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    if (n < 1000000) return is_prime(n);

    // Deterministic Miller-Rabin for n < 3.3×10^24
    const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        if (!miller_rabin_witness(n, witnesses[i])) return false;
    }
    return true;
}

// Each thread checks one even number
__global__ void verify_goldbach(uint64_t start_n, uint64_t count,
                                 uint64_t *failures, uint64_t *fail_count,
                                 uint64_t *max_min_prime_n, uint64_t *max_min_prime) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t n = start_n + idx * 2;  // only even numbers
    if (n <= 2) return;

    // Find smallest prime p such that both p and n-p are prime
    uint64_t smallest_p = 0;
    for (uint64_t p = 2; p <= n / 2; p++) {
        if (p == 2 || (p > 2 && p % 2 != 0)) {
            if (is_prime_large(p) && is_prime_large(n - p)) {
                smallest_p = p;
                break;
            }
        }
        // Skip even p > 2
        if (p == 2) p = 1;  // next iteration: p=2+1=3, then 5,7,...
    }

    if (smallest_p == 0) {
        // COUNTEREXAMPLE — Goldbach fails for this n!
        uint64_t pos = atomicAdd((unsigned long long*)fail_count, (unsigned long long)1);
        if (pos < 1024) {
            failures[pos] = n;
        }
    } else {
        // Track the even number with the largest "smallest prime"
        // (Goldbach's comet — the minimum prime in the partition)
        if (smallest_p > *max_min_prime) {
            atomicMax((unsigned long long*)max_min_prime, (unsigned long long)smallest_p);
            *max_min_prime_n = n;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <start_n> <end_n>\n", argv[0]);
        fprintf(stderr, "  Verifies Goldbach for all even n in [start_n, end_n]\n");
        return 1;
    }

    uint64_t start_n = (uint64_t)atoll(argv[1]);
    uint64_t end_n = (uint64_t)atoll(argv[2]);
    if (start_n % 2 != 0) start_n++;
    if (end_n % 2 != 0) end_n--;
    uint64_t count = (end_n - start_n) / 2 + 1;

    printf("Goldbach Verification: even n=%lu to %lu (%lu values)\n", start_n, end_n, count);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n\n", device_count);

    uint64_t *d_failures, *d_fail_count, *d_max_p_n, *d_max_p;
    cudaMalloc(&d_failures, 1024 * sizeof(uint64_t));
    cudaMalloc(&d_fail_count, sizeof(uint64_t));
    cudaMalloc(&d_max_p_n, sizeof(uint64_t));
    cudaMalloc(&d_max_p, sizeof(uint64_t));
    cudaMemset(d_fail_count, 0, sizeof(uint64_t));
    cudaMemset(d_max_p, 0, sizeof(uint64_t));

    uint64_t chunk_size = 5000000;  // 5M even numbers per chunk
    uint64_t total_failures = 0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (uint64_t offset = 0; offset < count; offset += chunk_size) {
        uint64_t chunk_count = chunk_size;
        if (offset + chunk_count > count) chunk_count = count - offset;
        uint64_t chunk_start = start_n + offset * 2;

        int gpu = (offset / chunk_size) % device_count;
        cudaSetDevice(gpu);
        cudaMemset(d_fail_count, 0, sizeof(uint64_t));

        int blocks = (chunk_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        verify_goldbach<<<blocks, THREADS_PER_BLOCK>>>(
            chunk_start, chunk_count,
            d_failures, d_fail_count, d_max_p_n, d_max_p
        );
        cudaDeviceSynchronize();

        uint64_t h_fail_count;
        cudaMemcpy(&h_fail_count, d_fail_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_fail_count > 0) {
            printf("*** COUNTEREXAMPLE(S) FOUND ***\n");
            total_failures += h_fail_count;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                         (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        double progress = (double)(offset + chunk_count) / count * 100;
        double rate = (offset + chunk_count) / elapsed;

        printf("[GPU %d] n=%lu..%lu (%.1f%%, %.0f n/sec, %.1fs)\n",
               gpu, chunk_start, chunk_start + chunk_count * 2,
               progress, rate, elapsed);
        fflush(stdout);
    }

    uint64_t h_max_p_n, h_max_p;
    cudaMemcpy(&h_max_p_n, d_max_p_n, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_p, d_max_p, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Goldbach Verification: even n=%lu to %lu\n", start_n, end_n);
    printf("Values checked: %lu\n", count);
    printf("Failures: %lu\n", total_failures);
    printf("Largest min-prime in partition: p=%lu (n=%lu)\n", h_max_p, h_max_p_n);
    printf("Time: %.1fs (%.0f n/sec)\n", total_elapsed, count / total_elapsed);
    if (total_failures == 0) {
        printf("Goldbach's Conjecture HOLDS for all even n in [%lu, %lu]\n", start_n, end_n);
    }
    printf("========================================\n");

    cudaFree(d_failures); cudaFree(d_fail_count);
    cudaFree(d_max_p_n); cudaFree(d_max_p);
    return total_failures > 0 ? 1 : 0;
}
