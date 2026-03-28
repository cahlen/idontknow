/*
 * CUDA Mersenne prime candidate sieve using Lucas-Lehmer test
 *
 * A Mersenne prime is a prime of the form M_p = 2^p - 1 where p is prime.
 * The Lucas-Lehmer test: M_p is prime iff S_{p-2} ≡ 0 (mod M_p)
 * where S_0 = 4, S_{n+1} = S_n^2 - 2.
 *
 * For small p (< 2000), we can do the full LL test on GPU with 128-bit
 * arithmetic. For larger p, we do trial factoring to eliminate candidates.
 *
 * Trial factoring rule: any factor of M_p must be ≡ ±1 (mod 8) and ≡ 1 (mod 2p).
 *
 * Compile: nvcc -O3 -arch=sm_100a -o mersenne_check scripts/experiments/mersenne-sieve/mersenne_check.cu
 * Run:     ./mersenne_check <max_exponent>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define THREADS_PER_BLOCK 256

// Modular multiplication for 128-bit: (a * b) mod m
// Using __uint128_t which is supported on CUDA
__device__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
    __uint128_t result = (__uint128_t)a * b;
    return (uint64_t)(result % m);
}

// Modular exponentiation: base^exp mod m
__device__ uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base, m);
        base = mod_mul(base, base, m);
        exp >>= 1;
    }
    return result;
}

// Check if p is prime (simple trial division, p is small)
__device__ bool is_prime_small(uint32_t p) {
    if (p < 2) return false;
    if (p == 2 || p == 3) return true;
    if (p % 2 == 0 || p % 3 == 0) return false;
    for (uint32_t i = 5; i * i <= p; i += 6) {
        if (p % i == 0 || p % (i + 2) == 0) return false;
    }
    return true;
}

// Trial factoring of M_p = 2^p - 1
// Any factor q must satisfy: q ≡ 1 (mod 2p) and q ≡ ±1 (mod 8)
// We check factors up to some limit
__device__ bool trial_factor_mersenne(uint32_t p, uint64_t factor_limit) {
    uint64_t two_p = 2 * (uint64_t)p;

    // Check q = 2kp + 1 for k = 1, 2, 3, ...
    for (uint64_t k = 1; ; k++) {
        uint64_t q = two_p * k + 1;
        if (q > factor_limit) break;

        // q must be ≡ ±1 (mod 8)
        uint64_t q_mod8 = q % 8;
        if (q_mod8 != 1 && q_mod8 != 7) continue;

        // Check if q divides M_p: 2^p ≡ 1 (mod q)
        if (mod_pow(2, p, q) == 1) {
            return true;  // found a factor — M_p is composite
        }
    }
    return false;  // no factor found (doesn't prove primality)
}

// Each thread checks one exponent p
__global__ void check_mersenne_candidates(uint32_t start_p, uint32_t count,
                                           uint32_t *survivors, uint32_t *survivor_count,
                                           uint64_t factor_limit) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint32_t p = start_p + idx;

    // p must be prime for M_p to possibly be prime
    if (!is_prime_small(p)) return;

    // p=2 is known (M_2 = 3 is prime)
    if (p == 2) {
        uint32_t pos = atomicAdd(survivor_count, 1);
        if (pos < 10000) survivors[pos] = p;
        return;
    }

    // Try to find a factor
    if (!trial_factor_mersenne(p, factor_limit)) {
        // No factor found — candidate survives
        uint32_t pos = atomicAdd(survivor_count, 1);
        if (pos < 10000) survivors[pos] = p;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <max_exponent> [factor_limit]\n", argv[0]);
        fprintf(stderr, "  Sieve Mersenne candidates M_p for prime p up to max_exponent\n");
        fprintf(stderr, "  factor_limit: trial division limit (default: 10^12)\n");
        return 1;
    }

    uint32_t max_p = (uint32_t)atol(argv[1]);
    uint64_t factor_limit = argc > 2 ? (uint64_t)atoll(argv[2]) : 1000000000000ULL;

    printf("Mersenne Prime Sieve\n");
    printf("Exponent range: p = 2 to %u\n", max_p);
    printf("Trial factor limit: %lu\n", factor_limit);
    printf("\n");

    int device_count;
    cudaGetDeviceCount(&device_count);

    uint32_t *d_survivors, *d_count;
    cudaMalloc(&d_survivors, 10000 * sizeof(uint32_t));
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int blocks = (max_p + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    check_mersenne_candidates<<<blocks, THREADS_PER_BLOCK>>>(
        2, max_p - 1, d_survivors, d_count, factor_limit
    );
    cudaDeviceSynchronize();

    uint32_t h_count;
    cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t *h_survivors = (uint32_t *)malloc(h_count * sizeof(uint32_t));
    cudaMemcpy(h_survivors, d_survivors, h_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                    (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    // Sort survivors
    for (uint32_t i = 0; i < h_count; i++)
        for (uint32_t j = i + 1; j < h_count; j++)
            if (h_survivors[j] < h_survivors[i]) {
                uint32_t tmp = h_survivors[i];
                h_survivors[i] = h_survivors[j];
                h_survivors[j] = tmp;
            }

    printf("========================================\n");
    printf("Mersenne candidates surviving trial factoring:\n");
    printf("(These need full Lucas-Lehmer test to confirm primality)\n\n");

    // Known Mersenne primes for comparison
    const uint32_t known[] = {2,3,5,7,13,17,19,31,61,89,107,127,521,607,1279,
        2203,2281,3217,4253,4423,9689,9941,11213,19937,21701,23209,44497,
        86243,110503,132049,216091,756839,859433,1257787,1398269,2976221,
        3021377,6972593,13466917,20996011,24036583,25964951,30402457,
        32582657,37156667,42643801,43112609,57885161,74207281,77232917,82589933};
    int num_known = sizeof(known) / sizeof(known[0]);

    int confirmed = 0, new_candidates = 0;
    for (uint32_t i = 0; i < h_count; i++) {
        bool is_known = false;
        for (int j = 0; j < num_known; j++) {
            if (h_survivors[i] == known[j]) { is_known = true; break; }
        }
        if (is_known) {
            printf("  p = %u  [KNOWN MERSENNE PRIME]\n", h_survivors[i]);
            confirmed++;
        } else {
            printf("  p = %u  [CANDIDATE — needs LL test]\n", h_survivors[i]);
            new_candidates++;
        }
    }

    printf("\nSurvivors: %u (known: %d, candidates: %d)\n", h_count, confirmed, new_candidates);
    printf("Time: %.1fs\n", elapsed);
    printf("========================================\n");

    free(h_survivors);
    cudaFree(d_survivors);
    cudaFree(d_count);
    return 0;
}
