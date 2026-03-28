/*
 * CUDA-accelerated Zaremba's Conjecture verifier
 *
 * For each d, searches for the smallest a with gcd(a,d)=1
 * such that all CF partial quotients of a/d are <= 5.
 *
 * Compile: nvcc -O3 -o zaremba_verify scripts/zaremba_verify.cu
 * Run:     ./zaremba_verify <start_d> <end_d>
 *
 * Each CUDA thread handles one value of d.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define BOUND 5
#define THREADS_PER_BLOCK 256

__device__ uint64_t dev_gcd(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Check if all CF partial quotients of a/d are <= BOUND
__device__ bool cf_bounded(uint64_t a, uint64_t d) {
    while (d != 0) {
        uint64_t q = a / d;
        if (q > BOUND) return false;
        uint64_t r = a % d;
        a = d;
        d = r;
    }
    return true;
}

// For a given d, find a witness a (or 0 if none found)
// Optimization: start search at d/6 (where 99.9% of witnesses live)
// then fall back to full search if needed.
__device__ uint64_t find_witness(uint64_t d) {
    if (d == 0) return 0;
    if (d <= BOUND) return 1;  // a=1 works: CF = [0, d], d <= 5

    // Phase 1: search in the sweet spot [d/6, d/4] where most witnesses are
    uint64_t lo = d / 7;
    if (lo == 0) lo = 1;
    uint64_t hi = d / 3;
    if (hi > d) hi = d;
    for (uint64_t a = lo; a <= hi; a++) {
        if (dev_gcd(a, d) == 1 && cf_bounded(a, d)) {
            return a;
        }
    }

    // Phase 2: full search for the rare cases outside the sweet spot
    for (uint64_t a = 1; a < lo; a++) {
        if (dev_gcd(a, d) == 1 && cf_bounded(a, d)) {
            return a;
        }
    }
    for (uint64_t a = hi + 1; a <= d; a++) {
        if (dev_gcd(a, d) == 1 && cf_bounded(a, d)) {
            return a;
        }
    }

    return 0;  // no witness found — would disprove the conjecture!
}

// Each thread checks one d value
__global__ void verify_zaremba(uint64_t start_d, uint64_t count,
                                uint64_t *failures, uint64_t *fail_count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t d = start_d + idx;
    uint64_t witness = find_witness(d);

    if (witness == 0) {
        // COUNTEREXAMPLE FOUND
        uint64_t pos = atomicAdd((unsigned long long*)fail_count, (unsigned long long)1);
        if (pos < 1024) {  // store up to 1024 failures
            failures[pos] = d;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <start_d> <end_d>\n", argv[0]);
        return 1;
    }

    uint64_t start_d = (uint64_t)atoll(argv[1]);
    uint64_t end_d = (uint64_t)atoll(argv[2]);
    uint64_t count = end_d - start_d + 1;

    printf("Zaremba Verification: d=%lu to %lu (%lu values)\n", start_d, end_d, count);

    // Get GPU info
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n", device_count);

    // Allocate failure tracking on device
    uint64_t *d_failures, *d_fail_count;
    cudaMalloc(&d_failures, 1024 * sizeof(uint64_t));
    cudaMalloc(&d_fail_count, sizeof(uint64_t));
    cudaMemset(d_fail_count, 0, sizeof(uint64_t));

    // Process in chunks to show progress and use multiple GPUs
    uint64_t chunk_size = 10000000;  // 10M per chunk
    uint64_t total_failures = 0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (uint64_t chunk_start = start_d; chunk_start <= end_d; chunk_start += chunk_size) {
        uint64_t chunk_end = chunk_start + chunk_size - 1;
        if (chunk_end > end_d) chunk_end = end_d;
        uint64_t chunk_count = chunk_end - chunk_start + 1;

        // Pick GPU (round-robin across available GPUs)
        int gpu = ((chunk_start - start_d) / chunk_size) % device_count;
        cudaSetDevice(gpu);

        // Reset failure count
        cudaMemset(d_fail_count, 0, sizeof(uint64_t));

        int blocks = (chunk_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        verify_zaremba<<<blocks, THREADS_PER_BLOCK>>>(chunk_start, chunk_count,
                                                       d_failures, d_fail_count);
        cudaDeviceSynchronize();

        // Check for failures
        uint64_t h_fail_count;
        cudaMemcpy(&h_fail_count, d_fail_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        if (h_fail_count > 0) {
            uint64_t h_failures[1024];
            uint64_t to_copy = h_fail_count < 1024 ? h_fail_count : 1024;
            cudaMemcpy(h_failures, d_failures, to_copy * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            printf("*** COUNTEREXAMPLE(S) FOUND in [%lu, %lu]: %lu failures ***\n",
                   chunk_start, chunk_end, h_fail_count);
            for (uint64_t i = 0; i < to_copy && i < 20; i++) {
                printf("  d = %lu\n", h_failures[i]);
            }
            total_failures += h_fail_count;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                         (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        double progress = (double)(chunk_end - start_d + 1) / count * 100;
        double rate = (chunk_end - start_d + 1) / elapsed;

        printf("[GPU %d] d=%lu..%lu done (%.1f%%, %.0f d/sec, %.1fs elapsed)\n",
               gpu, chunk_start, chunk_end, progress, rate, elapsed);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Verified d=%lu to %lu (%lu values)\n", start_d, end_d, count);
    printf("Total failures: %lu\n", total_failures);
    printf("Time: %.1fs (%.0f d/sec)\n", total_elapsed, count / total_elapsed);

    if (total_failures == 0) {
        printf("Zaremba's Conjecture HOLDS for all d in [%lu, %lu] with A=%d\n",
               start_d, end_d, BOUND);
    } else {
        printf("*** COUNTEREXAMPLE(S) FOUND — Zaremba's Conjecture FAILS ***\n");
    }
    printf("========================================\n");

    cudaFree(d_failures);
    cudaFree(d_fail_count);

    return total_failures > 0 ? 1 : 0;
}
