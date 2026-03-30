/*
 * Ramsey R(5,5) — Full-Recount SA on GPU
 *
 * Every step: flip random edge, recount ALL monochromatic K₅.
 * No incremental tricks — correctness first.
 *
 * K₅ counting uses bitmask operations: for n ≤ 64, each row of the
 * adjacency matrix fits in a uint64. Counting K₅ is 5 nested loops
 * with bitmask intersection + popcount.
 *
 * For n=44: C(44,5) = 1,086,008 candidate 5-subsets, but the bitmask
 * approach prunes aggressively via neighborhood intersection.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_full scripts/experiments/ramsey-r55/ramsey_fullcount.cu -lcurand
 * Run:     ./ramsey_full <n> <walkers_per_gpu> <steps>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define MAX_N 64
#define BLOCK_SIZE 128

typedef unsigned long long uint64;

// Count ALL monochromatic K₅ in the graph defined by adj
__device__ int count_mono_k5(uint64 *adj, int n) {
    int count = 0;
    for (int a = 0; a < n; a++) {
        uint64 na = adj[a];
        for (int b = a + 1; b < n; b++) {
            if (!((na >> b) & 1)) continue;
            // a-b connected. Find common neighbors > b
            uint64 nab = na & adj[b] & ~((1ULL << (b+1)) - 1);
            while (nab) {
                int c = __ffsll(nab) - 1;
                nab &= nab - 1;
                // a-b-c all connected. Common neighbors > c
                uint64 nabc = nab & adj[c];
                while (nabc) {
                    int d = __ffsll(nabc) - 1;
                    nabc &= nabc - 1;
                    // a-b-c-d all connected. Count neighbors > d in nabc
                    count += __popcll(nabc & adj[d]);
                }
            }
        }
    }
    return count;
}

// Total fitness = red K₅ + blue K₅
__device__ int fitness(uint64 *adj, int n) {
    int red = count_mono_k5(adj, n);
    uint64 comp[MAX_N];
    uint64 mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
    for (int i = 0; i < n; i++)
        comp[i] = (~adj[i]) & mask & ~(1ULL << i);
    int blue = count_mono_k5(comp, n);
    return red + blue;
}

__global__ void ramsey_sa(
    int n, int num_walkers, int max_steps,
    int *global_best, uint64 *best_adj_out,
    int *solution_count, uint64 seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_walkers) return;

    curandState rng;
    curand_init(seed + idx * 7919ULL, 0, 0, &rng);

    uint64 adj[MAX_N];
    uint64 mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;

    // Random initial coloring
    for (int i = 0; i < n; i++) adj[i] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (curand(&rng) % 2) {
                adj[i] |= (1ULL << j);
                adj[j] |= (1ULL << i);
            }
        }
    }

    int cur_fit = fitness(adj, n);
    int best_fit = cur_fit;

    for (int step = 0; step < max_steps && cur_fit > 0; step++) {
        // Temperature schedule: start hot, cool exponentially
        float temp = 5.0f * expf(-5.0f * step / max_steps);

        // Pick random edge
        int u = curand(&rng) % n;
        int v = curand(&rng) % (n - 1);
        if (v >= u) v++;
        if (u > v) { int t = u; u = v; v = t; }

        // Flip edge color
        adj[u] ^= (1ULL << v);
        adj[v] ^= (1ULL << u);

        int new_fit = fitness(adj, n);
        int delta = new_fit - cur_fit;

        if (delta <= 0) {
            // Accept improvement (or equal)
            cur_fit = new_fit;
        } else {
            // Accept worse with Boltzmann probability
            float prob = expf(-(float)delta / (temp + 1e-10f));
            if (curand_uniform(&rng) < prob) {
                cur_fit = new_fit;
            } else {
                // Reject: undo flip
                adj[u] ^= (1ULL << v);
                adj[v] ^= (1ULL << u);
            }
        }

        if (cur_fit < best_fit) {
            best_fit = cur_fit;
            atomicMin(global_best, best_fit);
        }
    }

    // Output solution
    if (cur_fit == 0) {
        int sol_idx = atomicAdd(solution_count, 1);
        if (sol_idx < 100) {
            for (int i = 0; i < n; i++)
                best_adj_out[(uint64)sol_idx * MAX_N + i] = adj[i];
        }
        printf("*** SOLUTION: Walker %d found Ramsey-good K_%d ***\n", idx, n);
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 43;
    int walkers_per_gpu = argc > 2 ? atoi(argv[2]) : 10000;
    int max_steps = argc > 3 ? atoi(argv[3]) : 500000;

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    printf("Ramsey R(5,5) Full-Recount SA\n");
    printf("n=%d, walkers=%d/GPU × %d GPUs = %d total\n",
           n, walkers_per_gpu, num_gpus, walkers_per_gpu * num_gpus);
    printf("Steps: %d per walker\n", max_steps);
    printf("Total flips: %.2e\n\n", (double)walkers_per_gpu * num_gpus * max_steps);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int *d_best[8], *d_sol_count[8];
    uint64 *d_adj[8];
    int h_best = INT_MAX;

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_best[g], sizeof(int));
        cudaMalloc(&d_sol_count[g], sizeof(int));
        int init_best = INT_MAX;
        cudaMemcpy(d_best[g], &init_best, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_sol_count[g], 0, sizeof(int));
        cudaMalloc(&d_adj[g], 100ULL * MAX_N * sizeof(uint64));

        int blocks = (walkers_per_gpu + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64 seed = time(NULL) + g * 1000003ULL;
        ramsey_sa<<<blocks, BLOCK_SIZE>>>(
            n, walkers_per_gpu, max_steps,
            d_best[g], d_adj[g], d_sol_count[g], seed);
        printf("[GPU %d] launched\n", g);
    }

    int total_solutions = 0;
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();

        int g_best, g_sol;
        cudaMemcpy(&g_best, d_best[g], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&g_sol, d_sol_count[g], sizeof(int), cudaMemcpyDeviceToHost);
        printf("[GPU %d] best fitness = %d, solutions = %d\n", g, g_best, g_sol);
        if (g_best < h_best) h_best = g_best;
        total_solutions += g_sol;

        if (g_sol > 0) {
            uint64 *h_adj = (uint64*)malloc((g_sol < 100 ? g_sol : 100) * MAX_N * sizeof(uint64));
            cudaMemcpy(h_adj, d_adj[g], (g_sol < 100 ? g_sol : 100) * MAX_N * sizeof(uint64), cudaMemcpyDeviceToHost);
            for (int s = 0; s < g_sol && s < 3; s++) {
                printf("\n=== SOLUTION %d (GPU %d) ===\n", s, g);
                for (int i = 0; i < n; i++)
                    printf("  %2d: %016llx\n", i, h_adj[s * MAX_N + i]);
            }
            free(h_adj);
        }

        cudaFree(d_best[g]);
        cudaFree(d_sol_count[g]);
        cudaFree(d_adj[g]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Ramsey R(5,5): n=%d\n", n);
    printf("Best fitness: %d\n", h_best);
    printf("Solutions: %d\n", total_solutions);
    printf("Time: %.1fs (%.0f flips/s)\n", elapsed,
           (double)walkers_per_gpu * num_gpus * max_steps / elapsed);
    if (total_solutions > 0)
        printf("*** R(5,5) > %d ***\n", n);
    printf("========================================\n");

    return total_solutions > 0 ? 0 : 1;
}
