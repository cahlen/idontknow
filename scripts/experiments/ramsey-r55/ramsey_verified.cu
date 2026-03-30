/*
 * Ramsey R(5,5) — Verified Incremental SA on GPU
 *
 * Fixes from the previous incremental version:
 * 1. Periodic full recount every SYNC_INTERVAL steps to prevent fitness drift
 * 2. Any claimed solution is INDEPENDENTLY VERIFIED by full_fitness()
 * 3. Verified solutions output their full adjacency matrix
 *
 * The incremental K₅ counter can accumulate off-by-one drift over
 * millions of steps. Syncing every 1000 steps prevents this.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_v2 scripts/experiments/ramsey-r55/ramsey_verified.cu -lcurand
 * Run:     ./ramsey_v2 <n> <walkers_per_gpu> <steps>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define MAX_N 64
#define BLOCK_SIZE 128
#define SYNC_INTERVAL 1000   // Full recount every N steps

typedef unsigned long long uint64;

// Count K₅ containing edge (u,v) in the color given by adj
__device__ int count_k5_through_edge(uint64 *adj, int n, int u, int v) {
    uint64 common = adj[u] & adj[v];
    common &= ~(1ULL << u);
    common &= ~(1ULL << v);

    int count = 0;
    uint64 c1 = common;
    while (c1) {
        int a = __ffsll(c1) - 1;
        c1 &= c1 - 1;

        uint64 c2 = c1 & adj[a];
        while (c2) {
            int b = __ffsll(c2) - 1;
            c2 &= c2 - 1;

            uint64 c3 = c2 & adj[b];
            count += __popcll(c3);
        }
    }
    return count;
}

// Full K₅ count
__device__ int full_k5_count(uint64 *adj, int n) {
    int count = 0;
    for (int a = 0; a < n; a++) {
        uint64 na = adj[a];
        for (int b = a + 1; b < n; b++) {
            if (!((na >> b) & 1)) continue;
            uint64 nab = na & adj[b] & ~((1ULL << (b+1)) - 1);
            while (nab) {
                int c = __ffsll(nab) - 1;
                nab &= nab - 1;
                uint64 nabc = nab & adj[c];
                while (nabc) {
                    int d = __ffsll(nabc) - 1;
                    nabc &= nabc - 1;
                    count += __popcll(nabc & adj[d]);
                }
            }
        }
    }
    return count;
}

__device__ int full_fitness(uint64 *adj, int n) {
    int red = full_k5_count(adj, n);
    uint64 comp[MAX_N];
    uint64 mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
    for (int i = 0; i < n; i++)
        comp[i] = (~adj[i]) & mask & ~(1ULL << i);
    int blue = full_k5_count(comp, n);
    return red + blue;
}

__global__ void ramsey_sa_verified(
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

    int cur_fit = full_fitness(adj, n);
    int best_fit = cur_fit;

    for (int step = 0; step < max_steps && cur_fit > 0; step++) {
        float temp = 3.0f * expf(-4.0f * step / max_steps);

        // Pick random edge
        int u = curand(&rng) % n;
        int v = curand(&rng) % (n - 1);
        if (v >= u) v++;
        if (u > v) { int t = u; u = v; v = t; }

        int was_red = (adj[u] >> v) & 1;
        uint64 comp[MAX_N];

        // Before flip: count K₅ through (u,v) in its current color
        int before_k5;
        if (was_red) {
            before_k5 = count_k5_through_edge(adj, n, u, v);
        } else {
            for (int i = 0; i < n; i++)
                comp[i] = (~adj[i]) & mask & ~(1ULL << i);
            before_k5 = count_k5_through_edge(comp, n, u, v);
        }

        // Flip
        adj[u] ^= (1ULL << v);
        adj[v] ^= (1ULL << u);

        // After flip: count K₅ through (u,v) in its new color
        int after_k5;
        if (was_red) {
            for (int i = 0; i < n; i++)
                comp[i] = (~adj[i]) & mask & ~(1ULL << i);
            after_k5 = count_k5_through_edge(comp, n, u, v);
        } else {
            after_k5 = count_k5_through_edge(adj, n, u, v);
        }

        int delta = after_k5 - before_k5;
        int new_fit = cur_fit + delta;

        if (new_fit <= cur_fit) {
            cur_fit = new_fit;
        } else {
            float prob = expf(-(float)delta / (temp + 1e-10f));
            if (curand_uniform(&rng) < prob) {
                cur_fit = new_fit;
            } else {
                // Undo flip
                adj[u] ^= (1ULL << v);
                adj[v] ^= (1ULL << u);
            }
        }

        // SYNC: periodic full recount to prevent drift
        if ((step + 1) % SYNC_INTERVAL == 0) {
            cur_fit = full_fitness(adj, n);
        }

        if (cur_fit < best_fit) {
            best_fit = cur_fit;
            atomicMin(global_best, best_fit);
        }
    }

    // INDEPENDENT VERIFICATION: if incremental says 0, verify with full recount
    if (cur_fit == 0) {
        int verified_fit = full_fitness(adj, n);
        if (verified_fit == 0) {
            int sol_idx = atomicAdd(solution_count, 1);
            for (int i = 0; i < n; i++)
                best_adj_out[(uint64)sol_idx * MAX_N + i] = adj[i];
            printf("*** VERIFIED: Walker %d found Ramsey-good K_%d (fitness=0, double-checked) ***\n", idx, n);
        } else {
            printf("    Walker %d: FALSE POSITIVE (incremental=0, verified=%d)\n", idx, verified_fit);
        }
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 43;
    int walkers_per_gpu = argc > 2 ? atoi(argv[2]) : 50000;
    int max_steps = argc > 3 ? atoi(argv[3]) : 1000000;

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    printf("Ramsey R(5,5) Verified Incremental SA\n");
    printf("n=%d, walkers=%d/GPU × %d GPUs = %d total\n",
           n, walkers_per_gpu, num_gpus, walkers_per_gpu * num_gpus);
    printf("Steps: %d per walker, sync every %d\n", max_steps, SYNC_INTERVAL);
    printf("Total flips: %.2e\n\n", (double)walkers_per_gpu * num_gpus * max_steps);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int *d_best[8], *d_sol_count[8];
    uint64 *d_adj[8];
    int h_best = INT_MAX;
    int h_sol_count = 0;

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_best[g], sizeof(int));
        cudaMalloc(&d_sol_count[g], sizeof(int));
        cudaMemcpy(d_best[g], &h_best, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_sol_count[g], 0, sizeof(int));
        // Allocate space for up to 100 solutions
        cudaMalloc(&d_adj[g], 100ULL * MAX_N * sizeof(uint64));
        cudaMemset(d_adj[g], 0, 100ULL * MAX_N * sizeof(uint64));

        int blocks = (walkers_per_gpu + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64 seed = time(NULL) + g * 1000003ULL;
        ramsey_sa_verified<<<blocks, BLOCK_SIZE>>>(
            n, walkers_per_gpu, max_steps,
            d_best[g], d_adj[g], d_sol_count[g], seed);
        printf("[GPU %d] launched %d walkers\n", g, walkers_per_gpu);
    }

    // Wait for all GPUs
    int total_solutions = 0;
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();

        int g_best, g_sol;
        cudaMemcpy(&g_best, d_best[g], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&g_sol, d_sol_count[g], sizeof(int), cudaMemcpyDeviceToHost);
        printf("[GPU %d] best fitness = %d, verified solutions = %d\n", g, g_best, g_sol);

        if (g_best < h_best) h_best = g_best;
        total_solutions += g_sol;

        // Print verified solutions
        if (g_sol > 0) {
            uint64 *h_adj = (uint64*)malloc(g_sol * MAX_N * sizeof(uint64));
            cudaMemcpy(h_adj, d_adj[g], g_sol * MAX_N * sizeof(uint64), cudaMemcpyDeviceToHost);
            for (int s = 0; s < g_sol && s < 3; s++) {
                printf("\n=== VERIFIED SOLUTION %d (GPU %d) ===\n", s, g);
                printf("Adjacency (hex, row i = red neighbors of i):\n");
                for (int i = 0; i < n; i++)
                    printf("  row %2d: %016llx\n", i, h_adj[s * MAX_N + i]);
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
    printf("Ramsey R(5,5) Search: n=%d\n", n);
    printf("Best fitness: %d\n", h_best);
    printf("Verified solutions: %d\n", total_solutions);
    printf("Time: %.1fs\n", elapsed);
    if (total_solutions > 0)
        printf("*** R(5,5) > %d CONFIRMED ***\n", n);
    else if (h_best > 0)
        printf("No solution found. Best = %d monochromatic K₅\n", h_best);
    printf("========================================\n");

    return total_solutions > 0 ? 0 : 1;
}
