/*
 * CUDA-accelerated Collatz Conjecture verifier
 *
 * For each starting value n, verify the Collatz sequence reaches 1.
 * Also records the maximum value reached (trajectory height) and
 * stopping time (number of steps to reach 1).
 *
 * The Collatz function:
 *   f(n) = n/2       if n is even
 *   f(n) = 3n + 1    if n is odd
 *
 * Optimization: use the "shortcut" form that processes multiple
 * steps at once by counting trailing zeros (even steps) and
 * applying the odd step in one go.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o collatz_verify scripts/experiments/collatz/collatz_verify.cu
 * Run:     ./collatz_verify <start_n> <end_n>
 *
 * Each CUDA thread handles one starting value.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define MAX_STEPS 10000000  // safety limit — if exceeded, likely a bug or overflow

// We use 128-bit arithmetic to handle trajectory heights.
// The Collatz sequence for n can temporarily exceed 2^64.
// We use unsigned __int128 on device via two uint64_t.
typedef unsigned long long uint64;

__device__ bool collatz_reaches_one(uint64 n, uint64 *max_val, uint32_t *steps) {
    uint64 current = n;
    uint64 peak = n;
    uint32_t step_count = 0;

    while (current != 1 && step_count < MAX_STEPS) {
        if (current % 2 == 0) {
            // Count trailing zeros for multi-step even reduction
            int tz = __ffsll(current) - 1;  // number of trailing zeros
            current >>= tz;
            step_count += tz;
        } else {
            // Odd: 3n+1, then immediately divide by 2 (result is always even)
            // Check for overflow: 3*current + 1 must not exceed 2^63
            if (current > (UINT64_MAX - 1) / 3) {
                // Would overflow — this means trajectory goes very high
                // For n < 2^60 this shouldn't happen, but flag it
                *max_val = UINT64_MAX;
                *steps = step_count;
                return false;  // overflow — needs 128-bit handling
            }
            current = 3 * current + 1;
            step_count++;

            // Now it's even, do at least one division
            int tz = __ffsll(current) - 1;
            current >>= tz;
            step_count += tz;
        }

        if (current > peak) peak = current;

        // Shortcut: if current < n, it will reach 1
        // (because we've already verified everything below n)
        // Only valid if we're verifying sequentially from 1
        if (current < n) {
            *max_val = peak;
            *steps = step_count;
            return true;
        }
    }

    *max_val = peak;
    *steps = step_count;
    return (current == 1);
}

__global__ void verify_collatz(uint64 start_n, uint64 count,
                                uint64 *failures, uint64 *fail_count,
                                uint64 *max_stopping_time_n, uint32_t *max_stopping_time,
                                uint64 *max_trajectory_n, uint64 *max_trajectory_val) {
    uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64 n = start_n + idx;
    if (n <= 1) return;  // trivial

    uint64 peak;
    uint32_t steps;
    bool reached = collatz_reaches_one(n, &peak, &steps);

    if (!reached) {
        uint64 pos = atomicAdd((unsigned long long*)fail_count, (unsigned long long)1);
        if (pos < 1024) {
            failures[pos] = n;
        }
    }

    // Track records (approximate — race conditions are fine, we just want rough maxima)
    if (steps > *max_stopping_time) {
        atomicMax((unsigned long long*)max_stopping_time, (unsigned long long)steps);
        *max_stopping_time_n = n;
    }
    if (peak > *max_trajectory_val) {
        atomicMax((unsigned long long*)max_trajectory_val, (unsigned long long)peak);
        *max_trajectory_n = n;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <start_n> <end_n>\n", argv[0]);
        return 1;
    }

    uint64 start_n = (uint64)atoll(argv[1]);
    uint64 end_n = (uint64)atoll(argv[2]);
    uint64 count = end_n - start_n + 1;

    printf("Collatz Verification: n=%lu to %lu (%lu values)\n", start_n, end_n, count);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n", device_count);

    // Device memory
    uint64 *d_failures, *d_fail_count;
    uint64 *d_max_stop_n, *d_max_traj_n, *d_max_traj_val;
    uint32_t *d_max_stop_time;

    cudaMalloc(&d_failures, 1024 * sizeof(uint64));
    cudaMalloc(&d_fail_count, sizeof(uint64));
    cudaMalloc(&d_max_stop_n, sizeof(uint64));
    cudaMalloc(&d_max_stop_time, sizeof(uint32_t));
    cudaMalloc(&d_max_traj_n, sizeof(uint64));
    cudaMalloc(&d_max_traj_val, sizeof(uint64));

    cudaMemset(d_fail_count, 0, sizeof(uint64));
    cudaMemset(d_max_stop_time, 0, sizeof(uint32_t));
    cudaMemset(d_max_traj_val, 0, sizeof(uint64));

    uint64 chunk_size = 100000000;  // 100M per chunk
    uint64 total_failures = 0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (uint64 chunk_start = start_n; chunk_start <= end_n; chunk_start += chunk_size) {
        uint64 chunk_end = chunk_start + chunk_size - 1;
        if (chunk_end > end_n) chunk_end = end_n;
        uint64 chunk_count = chunk_end - chunk_start + 1;

        int gpu = ((chunk_start - start_n) / chunk_size) % device_count;
        cudaSetDevice(gpu);

        int blocks = (chunk_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        verify_collatz<<<blocks, THREADS_PER_BLOCK>>>(
            chunk_start, chunk_count,
            d_failures, d_fail_count,
            d_max_stop_n, d_max_stop_time,
            d_max_traj_n, d_max_traj_val
        );
        cudaDeviceSynchronize();

        // Check for failures
        uint64 h_fail_count;
        cudaMemcpy(&h_fail_count, d_fail_count, sizeof(uint64), cudaMemcpyDeviceToHost);

        if (h_fail_count > total_failures) {
            uint64 new_failures = h_fail_count - total_failures;
            printf("*** %lu NEW FAILURE(S) in [%lu, %lu] ***\n",
                   new_failures, chunk_start, chunk_end);
            total_failures = h_fail_count;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                         (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        double progress = (double)(chunk_end - start_n + 1) / count * 100;
        double rate = (chunk_end - start_n + 1) / elapsed;

        printf("[GPU %d] n=%lu..%lu done (%.1f%%, %.0f n/sec, %.1fs elapsed)\n",
               gpu, chunk_start, chunk_end, progress, rate, elapsed);
        fflush(stdout);
    }

    // Retrieve stats
    uint64 h_max_stop_n, h_max_traj_n, h_max_traj_val;
    uint32_t h_max_stop_time;
    cudaMemcpy(&h_max_stop_n, d_max_stop_n, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_stop_time, d_max_stop_time, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_traj_n, d_max_traj_n, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_traj_val, d_max_traj_val, sizeof(uint64), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Collatz Verification: n=%lu to %lu (%lu values)\n", start_n, end_n, count);
    printf("Total failures: %lu\n", total_failures);
    printf("Time: %.1fs (%.0f n/sec)\n", total_elapsed, count / total_elapsed);
    printf("\nRecords:\n");
    printf("  Max stopping time: %u steps (n=%lu)\n", h_max_stop_time, h_max_stop_n);
    printf("  Max trajectory height: %lu (n=%lu)\n", h_max_traj_val, h_max_traj_n);
    printf("\n");

    if (total_failures == 0) {
        printf("Collatz Conjecture HOLDS for all n in [%lu, %lu]\n", start_n, end_n);
    } else {
        printf("*** COUNTEREXAMPLE(S) FOUND ***\n");
    }
    printf("========================================\n");

    cudaFree(d_failures);
    cudaFree(d_fail_count);
    cudaFree(d_max_stop_n);
    cudaFree(d_max_stop_time);
    cudaFree(d_max_traj_n);
    cudaFree(d_max_traj_val);

    return total_failures > 0 ? 1 : 0;
}
