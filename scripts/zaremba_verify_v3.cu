/*
 * Zaremba verifier v3 — block-parallel witness search
 *
 * v2 used 1 thread per d. For large d, the witness search within
 * each thread is the bottleneck (~100M candidates at d=5B).
 *
 * v3 uses 1 BLOCK (256 threads) per d. All threads in the block
 * search different slices of the candidate range in parallel.
 * First thread to find a witness writes it to shared memory and
 * all threads exit. This gives ~256× speedup on the inner loop.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o zaremba_v3 scripts/zaremba_verify_v3.cu
 * Run:     ./zaremba_v3 <start_d> <end_d>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define BOUND 5
#define BLOCK_SIZE 256

typedef unsigned long long uint64;

__device__ uint64 dev_gcd(uint64 a, uint64 b) {
    while (b) { uint64 t = b; b = a % b; a = t; }
    return a;
}

__device__ bool cf_bounded(uint64 a, uint64 d) {
    while (d != 0) {
        uint64 q = a / d;
        if (q > BOUND) return false;
        uint64 r = a % d;
        a = d;
        d = r;
    }
    return true;
}

__device__ bool quick_coprime(uint64 a, uint64 d) {
    if ((d & 1) == 0 && (a & 1) == 0) return false;
    if (d % 3 == 0 && a % 3 == 0) return false;
    if (d % 5 == 0 && a % 5 == 0) return false;
    if (d % 7 == 0 && a % 7 == 0) return false;
    if (d % 11 == 0 && a % 11 == 0) return false;
    if (d % 13 == 0 && a % 13 == 0) return false;
    return true;
}

/*
 * One block = one d value.
 * 256 threads split the search range and race to find the witness.
 */
__global__ void verify_zaremba_v3(uint64 start_d, uint64 count,
                                   uint64 *failures, uint64 *fail_count) {
    // Each block handles one d
    uint64 d_idx = (uint64)blockIdx.x;
    if (d_idx >= count) return;

    uint64 d = start_d + d_idx;
    int tid = threadIdx.x;

    // Shared flag: set to 1 when any thread finds a witness
    __shared__ int found;
    if (tid == 0) found = 0;
    __syncthreads();

    // Trivial cases
    if (d <= BOUND) {
        // a=1 always works for d <= 5
        if (tid == 0) found = 1;
        __syncthreads();
        return;
    }

    // Phase 1: Targeted search around 0.170*d
    // Search band: [0.160d, 0.190d] — wider than v2 to be safe
    uint64 lo = (uint64)(0.160 * (double)d);
    uint64 hi = (uint64)(0.190 * (double)d);
    if (lo < 1) lo = 1;
    if (hi > d) hi = d;

    uint64 band_size = hi - lo + 1;
    uint64 per_thread = (band_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64 my_start = lo + (uint64)tid * per_thread;
    uint64 my_end = my_start + per_thread;
    if (my_end > hi + 1) my_end = hi + 1;

    for (uint64 a = my_start; a < my_end; a++) {
        if (found) return;  // another thread found it
        if (!quick_coprime(a, d)) continue;
        if (dev_gcd(a, d) != 1) continue;
        uint64 first_q = d / a;
        if (first_q > BOUND) continue;
        if (cf_bounded(a, d)) {
            found = 1;  // signal other threads
            return;
        }
    }

    __syncthreads();
    if (found) return;

    // Phase 2: Wider search [d/7, lo) and (hi, d/3]
    // Split between threads the same way
    uint64 lo2 = d / 7;
    uint64 hi2 = d / 3;
    if (lo2 < 1) lo2 = 1;

    // Below the band
    if (lo > lo2) {
        uint64 range = lo - lo2;
        uint64 per_t = (range + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64 s = lo2 + (uint64)tid * per_t;
        uint64 e = s + per_t;
        if (e > lo) e = lo;

        for (uint64 a = s; a < e; a++) {
            if (found) return;
            if (!quick_coprime(a, d)) continue;
            if (dev_gcd(a, d) != 1) continue;
            if (cf_bounded(a, d)) { found = 1; return; }
        }
    }

    // Above the band
    if (hi2 > hi) {
        uint64 range = hi2 - hi;
        uint64 per_t = (range + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64 s = hi + 1 + (uint64)tid * per_t;
        uint64 e = s + per_t;
        if (e > hi2 + 1) e = hi2 + 1;

        for (uint64 a = s; a < e; a++) {
            if (found) return;
            if (!quick_coprime(a, d)) continue;
            if (dev_gcd(a, d) != 1) continue;
            if (cf_bounded(a, d)) { found = 1; return; }
        }
    }

    __syncthreads();
    if (found) return;

    // Phase 3: Full search outside [d/7, d/3] — almost never needed
    // Only thread 0 does this to avoid massive divergence
    if (tid == 0) {
        for (uint64 a = 1; a < lo2; a++) {
            if (quick_coprime(a, d) && dev_gcd(a, d) == 1 && cf_bounded(a, d)) {
                found = 1; return;
            }
        }
        for (uint64 a = hi2 + 1; a <= d; a++) {
            if (quick_coprime(a, d) && dev_gcd(a, d) == 1 && cf_bounded(a, d)) {
                found = 1; return;
            }
        }
    }

    __syncthreads();

    // If still not found, record failure
    if (tid == 0 && !found) {
        uint64 pos = atomicAdd((unsigned long long*)fail_count, (unsigned long long)1);
        if (pos < 1024) failures[pos] = d;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <start_d> <end_d>\n", argv[0]);
        return 1;
    }

    uint64 start_d = (uint64)atoll(argv[1]);
    uint64 end_d = (uint64)atoll(argv[2]);
    uint64 count = end_d - start_d + 1;

    printf("Zaremba v3 (block-parallel): d=%llu to %llu (%llu values)\n",
           (unsigned long long)start_d, (unsigned long long)end_d,
           (unsigned long long)count);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n", device_count);

    uint64 *d_failures, *d_fail_count;
    cudaMalloc(&d_failures, 1024 * sizeof(uint64));
    cudaMalloc(&d_fail_count, sizeof(uint64));
    cudaMemset(d_fail_count, 0, sizeof(uint64));

    // Process in chunks — each chunk launches count_chunk blocks,
    // each block = 256 threads = 1 value of d
    uint64 chunk_size = 100000;  // 100K d values per kernel launch
    // (each launch = 100K blocks × 256 threads = 25.6M threads)
    uint64 total_failures = 0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (uint64 chunk_start = start_d; chunk_start <= end_d; chunk_start += chunk_size) {
        uint64 chunk_end = chunk_start + chunk_size - 1;
        if (chunk_end > end_d) chunk_end = end_d;
        uint64 chunk_count = chunk_end - chunk_start + 1;

        int gpu = ((chunk_start - start_d) / chunk_size) % device_count;
        cudaSetDevice(gpu);
        cudaMemset(d_fail_count, 0, sizeof(uint64));

        // 1 block per d value, BLOCK_SIZE threads per block
        verify_zaremba_v3<<<chunk_count, BLOCK_SIZE>>>(
            chunk_start, chunk_count, d_failures, d_fail_count
        );
        cudaDeviceSynchronize();

        uint64 h_fail_count;
        cudaMemcpy(&h_fail_count, d_fail_count, sizeof(uint64), cudaMemcpyDeviceToHost);
        if (h_fail_count > 0) {
            uint64 h_failures[1024];
            uint64 to_copy = h_fail_count < 1024 ? h_fail_count : 1024;
            cudaMemcpy(h_failures, d_failures, to_copy * sizeof(uint64), cudaMemcpyDeviceToHost);
            printf("*** COUNTEREXAMPLE(S) in [%llu, %llu]: %llu failures ***\n",
                   (unsigned long long)chunk_start, (unsigned long long)chunk_end,
                   (unsigned long long)h_fail_count);
            for (uint64 i = 0; i < to_copy && i < 20; i++)
                printf("  d = %llu\n", (unsigned long long)h_failures[i]);
            total_failures += h_fail_count;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                         (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        double progress = (double)(chunk_end - start_d + 1) / count * 100;
        double rate = (chunk_end - start_d + 1) / elapsed;

        printf("[GPU %d] d=%llu..%llu done (%.1f%%, %.0f d/sec, %.1fs elapsed)\n",
               gpu, (unsigned long long)chunk_start, (unsigned long long)chunk_end,
               progress, rate, elapsed);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Zaremba v3: d=%llu to %llu (%llu values)\n",
           (unsigned long long)start_d, (unsigned long long)end_d,
           (unsigned long long)count);
    printf("Total failures: %llu\n", (unsigned long long)total_failures);
    printf("Time: %.1fs (%.0f d/sec)\n", total_elapsed, count / total_elapsed);
    if (total_failures == 0) {
        printf("Zaremba's Conjecture HOLDS for all d in [%llu, %llu] with A=%d\n",
               (unsigned long long)start_d, (unsigned long long)end_d, BOUND);
    } else {
        printf("*** COUNTEREXAMPLE(S) FOUND ***\n");
    }
    printf("========================================\n");

    cudaFree(d_failures);
    cudaFree(d_fail_count);
    return total_failures > 0 ? 1 : 0;
}
