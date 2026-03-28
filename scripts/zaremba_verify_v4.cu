/*
 * Zaremba verifier v4 — GPU-parallel inverse CF construction
 *
 * Enumerate ALL continued fractions [0; a1, a2, ..., ak] with ai ∈ {1,...,5}
 * on GPU. Each CF gives a denominator d — mark it in a bitset. Any unmarked
 * d is a potential counterexample.
 *
 * Parallelization: we expand the CF tree to a fixed depth on CPU (generating
 * millions of "seed" states), then launch one CUDA thread per seed. Each
 * thread continues the tree walk from its seed, marking denominators it finds.
 *
 * With seed depth 8: 5^8 = 390,625 seeds. Each thread walks the remaining
 * tree below its seed. This gives perfect GPU occupancy.
 *
 * With seed depth 10: 5^10 ≈ 10M seeds — even better occupancy.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o zaremba_v4 scripts/zaremba_verify_v4.cu
 * Run:     ./zaremba_v4 <max_d>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BOUND 5
#define THREADS_PER_BLOCK 256
#define MAX_DEPTH 60  // max CF length per thread (plenty for d up to 10^12)

typedef unsigned long long uint64;

// Seed state: the convergent matrix state at the seed depth
typedef struct {
    uint64 p_prev, p;  // numerator state
    uint64 q_prev, q;  // denominator state (q = current denominator)
} SeedState;

// GPU bitset — one bit per d value
// Use atomicOr on 32-bit words for thread safety
__global__ void gpu_enumerate(SeedState *seeds, uint64 num_seeds,
                               uint32_t *bitset, uint64 max_d) {
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    SeedState s = seeds[idx];

    // Mark the seed's denominator
    if (s.q >= 1 && s.q <= max_d) {
        uint64 word = s.q / 32;
        uint32_t bit = 1u << (s.q % 32);
        atomicOr(&bitset[word], bit);
    }

    // Iterative DFS from this seed using an explicit stack
    // Stack stores: (p_prev, p, q_prev, q)
    struct { uint64 pp, p, qp, q; int next_a; } stack[MAX_DEPTH];
    int sp = 0;

    stack[0].pp = s.p_prev;
    stack[0].p = s.p;
    stack[0].qp = s.q_prev;
    stack[0].q = s.q;
    stack[0].next_a = 1;

    while (sp >= 0) {
        int a = stack[sp].next_a;
        if (a > BOUND) {
            sp--;  // backtrack
            continue;
        }

        stack[sp].next_a = a + 1;  // advance for next iteration

        uint64 q_new = (uint64)a * stack[sp].q + stack[sp].qp;
        if (q_new > max_d) continue;  // prune — denominators only grow

        uint64 p_new = (uint64)a * stack[sp].p + stack[sp].pp;

        // Mark this denominator
        uint64 word = q_new / 32;
        uint32_t bit = 1u << (q_new % 32);
        atomicOr(&bitset[word], bit);

        // Push child onto stack
        if (sp + 1 < MAX_DEPTH) {
            sp++;
            stack[sp].pp = stack[sp-1].p;
            stack[sp].p = p_new;
            stack[sp].qp = stack[sp-1].q;
            stack[sp].q = q_new;
            stack[sp].next_a = 1;
        }
    }
}

// Count uncovered d values on GPU
__global__ void count_uncovered(uint32_t *bitset, uint64 max_d,
                                 uint64 *uncovered_count) {
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    uint64 d = idx + 1;  // d starts at 1
    if (d > max_d) return;

    uint64 word = d / 32;
    uint32_t bit = 1u << (d % 32);
    if (!(bitset[word] & bit)) {
        atomicAdd((unsigned long long*)uncovered_count, 1ULL);
    }
}

// CPU: generate seeds by expanding CF tree to a fixed depth
// CPU bitset for marking denominators found during seed generation
static uint32_t *h_seed_marks = NULL;

static inline void cpu_mark(uint64 d, uint64 max_d) {
    if (d >= 1 && d <= max_d) {
        h_seed_marks[d / 32] |= (1u << (d % 32));
    }
}

void generate_seeds(uint64 p_prev, uint64 p, uint64 q_prev, uint64 q,
                    int depth, int seed_depth, uint64 max_d,
                    SeedState *seeds, uint64 *count, uint64 max_seeds) {
    // Mark EVERY denominator encountered during seed generation
    // This catches all CFs shorter than seed_depth
    cpu_mark(q, max_d);

    if (depth == seed_depth) {
        if (*count < max_seeds) {
            seeds[*count].p_prev = p_prev;
            seeds[*count].p = p;
            seeds[*count].q_prev = q_prev;
            seeds[*count].q = q;
            (*count)++;
        }
        return;
    }

    for (int a = 1; a <= BOUND; a++) {
        uint64 q_new = (uint64)a * q + q_prev;
        if (q_new > max_d) break;
        uint64 p_new = (uint64)a * p + p_prev;
        generate_seeds(p, p_new, q, q_new, depth + 1, seed_depth, max_d,
                       seeds, count, max_seeds);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <max_d> [seed_depth]\n", argv[0]);
        fprintf(stderr, "  seed_depth: CF tree expansion depth before GPU launch (default: 10)\n");
        return 1;
    }

    uint64 max_d = (uint64)atoll(argv[1]);
    int seed_depth = argc > 2 ? atoi(argv[2]) : 10;

    printf("Zaremba v4 — GPU-parallel inverse CF construction\n");
    printf("Target: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Seed depth: %d (up to 5^%d = %llu seeds)\n",
           seed_depth, seed_depth, (unsigned long long)1);  // computed below
    printf("Bound A = %d\n\n", BOUND);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n", device_count);

    // Allocate GPU bitset
    uint64 bitset_words = (max_d + 32) / 32;
    uint64 bitset_bytes = bitset_words * sizeof(uint32_t);
    printf("Bitset: %llu bytes (%.2f GB)\n",
           (unsigned long long)bitset_bytes, bitset_bytes / 1e9);

    uint32_t *d_bitset;
    cudaMalloc(&d_bitset, bitset_bytes);
    cudaMemset(d_bitset, 0, bitset_bytes);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Generate seeds on CPU
    uint64 max_seeds = 100000000;  // 100M max
    SeedState *h_seeds = (SeedState*)malloc(max_seeds * sizeof(SeedState));
    uint64 num_seeds = 0;

    // CPU bitset for short CFs found during seed generation
    h_seed_marks = (uint32_t*)calloc(bitset_words, sizeof(uint32_t));

    printf("\nGenerating seeds (depth %d)...\n", seed_depth);
    // Mark d=1 (trivial case)
    cpu_mark(1, max_d);

    for (int a1 = 1; a1 <= BOUND; a1++) {
        uint64 before = num_seeds;
        generate_seeds(0, 1, 1, (uint64)a1, 1, seed_depth, max_d,
                       h_seeds, &num_seeds, max_seeds);
        printf("  a1=%d: +%llu seeds (total: %llu)\n",
               a1, (unsigned long long)(num_seeds - before),
               (unsigned long long)num_seeds);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double seed_time = (t_end.tv_sec - t_start.tv_sec) +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    printf("Seeds generated: %llu in %.1fs\n\n", (unsigned long long)num_seeds, seed_time);

    // Upload seeds to GPU and launch
    SeedState *d_seeds;
    cudaMalloc(&d_seeds, num_seeds * sizeof(SeedState));
    cudaMemcpy(d_seeds, h_seeds, num_seeds * sizeof(SeedState), cudaMemcpyHostToDevice);

    printf("Launching GPU enumeration across %d GPUs (%llu seeds total)...\n",
           device_count, (unsigned long long)num_seeds);

    // Each GPU gets its own bitset and a slice of the seeds
    uint32_t *d_bitsets[8];
    SeedState *d_seeds_per_gpu[8];
    uint64 seeds_per_gpu = (num_seeds + device_count - 1) / device_count;

    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_bitsets[gpu], bitset_bytes);
        // Upload the CPU seed marks to every GPU's bitset
        cudaMemcpy(d_bitsets[gpu], h_seed_marks, bitset_bytes, cudaMemcpyHostToDevice);

        // Upload this GPU's slice of seeds
        uint64 offset = gpu * seeds_per_gpu;
        uint64 n = seeds_per_gpu;
        if (offset + n > num_seeds) n = num_seeds - offset;
        if (offset >= num_seeds) n = 0;

        if (n > 0) {
            cudaMalloc(&d_seeds_per_gpu[gpu], n * sizeof(SeedState));
            cudaMemcpy(d_seeds_per_gpu[gpu], h_seeds + offset,
                       n * sizeof(SeedState), cudaMemcpyHostToDevice);

            int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            gpu_enumerate<<<blocks, THREADS_PER_BLOCK>>>(
                d_seeds_per_gpu[gpu], n, d_bitsets[gpu], max_d);
            printf("  GPU %d: %llu seeds launched\n", gpu, (unsigned long long)n);
        } else {
            d_seeds_per_gpu[gpu] = NULL;
        }
    }

    // Wait for all GPUs
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    }
    free(h_seed_marks);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double enum_time = (t_end.tv_sec - t_start.tv_sec) +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    printf("GPU enumeration complete: %.1fs\n\n", enum_time - seed_time);

    // Merge all GPU bitsets into GPU 0's bitset via OR
    printf("Merging bitsets from %d GPUs...\n", device_count);
    // Download all to CPU, OR together, upload to GPU 0
    uint32_t *h_merged = (uint32_t*)calloc(bitset_words, sizeof(uint32_t));
    for (int gpu = 0; gpu < device_count; gpu++) {
        uint32_t *h_tmp = (uint32_t*)malloc(bitset_bytes);
        cudaSetDevice(gpu);
        cudaMemcpy(h_tmp, d_bitsets[gpu], bitset_bytes, cudaMemcpyDeviceToHost);
        for (uint64 i = 0; i < bitset_words; i++) {
            h_merged[i] |= h_tmp[i];
        }
        free(h_tmp);
        cudaFree(d_bitsets[gpu]);
        if (d_seeds_per_gpu[gpu]) cudaFree(d_seeds_per_gpu[gpu]);
    }
    cudaSetDevice(0);
    cudaMemcpy(d_bitset, h_merged, bitset_bytes, cudaMemcpyHostToDevice);
    free(h_merged);

    // Count uncovered on GPU
    printf("Counting uncovered denominators...\n");
    uint64 *d_uncovered;
    cudaMalloc(&d_uncovered, sizeof(uint64));
    cudaMemset(d_uncovered, 0, sizeof(uint64));

    int count_blocks = (max_d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    count_uncovered<<<count_blocks, THREADS_PER_BLOCK>>>(d_bitset, max_d, d_uncovered);
    cudaDeviceSynchronize();

    uint64 h_uncovered;
    cudaMemcpy(&h_uncovered, d_uncovered, sizeof(uint64), cudaMemcpyDeviceToHost);

    // If there are uncovered values, download bitset and find them
    uint64 first_uncovered[100];
    int n_first = 0;
    if (h_uncovered > 0 && h_uncovered <= 10000) {
        uint32_t *h_bitset = (uint32_t*)malloc(bitset_bytes);
        cudaMemcpy(h_bitset, d_bitset, bitset_bytes, cudaMemcpyDeviceToHost);
        for (uint64 d = 1; d <= max_d && n_first < 100; d++) {
            uint64 word = d / 32;
            uint32_t bit = 1u << (d % 32);
            if (!(h_bitset[word] & bit)) {
                first_uncovered[n_first++] = d;
            }
        }
        free(h_bitset);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Zaremba v4: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Seeds: %llu (depth %d)\n", (unsigned long long)num_seeds, seed_depth);
    printf("Uncovered: %llu\n", (unsigned long long)h_uncovered);

    if (n_first > 0) {
        printf("First uncovered:\n");
        for (int i = 0; i < n_first && i < 20; i++)
            printf("  d = %llu\n", (unsigned long long)first_uncovered[i]);
    }

    printf("Time: %.1fs (seeds: %.1fs, GPU: %.1fs, scan: %.1fs)\n",
           total_elapsed, seed_time, enum_time - seed_time,
           total_elapsed - enum_time);

    if (h_uncovered == 0) {
        printf("\nZaremba's Conjecture HOLDS for all d in [1, %llu] with A=%d\n",
               (unsigned long long)max_d, BOUND);
    } else {
        printf("\n*** %llu UNCOVERED — need longer CFs or investigation ***\n",
               (unsigned long long)h_uncovered);
    }
    printf("========================================\n");

    free(h_seeds);
    cudaFree(d_seeds);
    cudaFree(d_bitset);
    cudaFree(d_uncovered);
    return (h_uncovered > 0) ? 1 : 0;
}
