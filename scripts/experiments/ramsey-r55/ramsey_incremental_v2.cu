/*
 * Ramsey R(5,5) — Fixed Incremental SA on GPU
 *
 * Uses explicit-loop K₅ counter (proven correct on GPU) instead of
 * the bitmask version that had a drift bug in the SA loop context.
 *
 * The bitmask count_k5_through_edge passes unit tests on GPU but
 * produces systematic drift when used inside the SA loop with local
 * arrays (suspected register spilling / local memory corruption).
 * The explicit-loop version avoids this by not using intermediate
 * bitmask variables that could be corrupted.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_inc2 scripts/experiments/ramsey-r55/ramsey_incremental_v2.cu -lcurand
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define MAX_N 48
#define BLOCK_SIZE 128

typedef unsigned long long uint64;

// Correct K₅-through-edge counter using explicit loops (GPU-verified)
__device__ int count_k5_through_edge(uint64 *adj, int n, int u, int v) {
    // Build common neighbor list
    int cn[MAX_N], ncn = 0;
    for (int w = 0; w < n; w++) {
        if (w == u || w == v) continue;
        if ((adj[u] >> w) & 1 && (adj[v] >> w) & 1)
            cn[ncn++] = w;
    }
    // Count triangles in common-neighbor subgraph
    int count = 0;
    for (int i = 0; i < ncn; i++)
        for (int j = i+1; j < ncn; j++) {
            if (!((adj[cn[i]] >> cn[j]) & 1)) continue;
            for (int k = j+1; k < ncn; k++)
                if ((adj[cn[i]] >> cn[k]) & 1 && (adj[cn[j]] >> cn[k]) & 1)
                    count++;
        }
    return count;
}

// Full K₅ count (for initial fitness + periodic sync)
__device__ int full_k5_count(uint64 *adj, int n) {
    int count = 0;
    for (int a = 0; a < n; a++) {
        uint64 na = adj[a];
        for (int b = a+1; b < n; b++) {
            if (!((na >> b) & 1)) continue;
            uint64 nab = na & adj[b] & ~((1ULL << (b+1)) - 1);
            while (nab) {
                int c = __ffsll(nab) - 1; nab &= nab - 1;
                uint64 nabc = nab & adj[c];
                while (nabc) {
                    int d = __ffsll(nabc) - 1; nabc &= nabc - 1;
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
    return red + full_k5_count(comp, n);
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

    int cur_fit = full_fitness(adj, n);
    int best_fit = cur_fit;

    for (int step = 0; step < max_steps && cur_fit > 0; step++) {
        float temp = 5.0f * expf(-5.0f * step / max_steps);

        int u = curand(&rng) % n;
        int v = curand(&rng) % (n - 1);
        if (v >= u) v++;
        if (u > v) { int t = u; u = v; v = t; }

        int was_red = (adj[u] >> v) & 1;

        // Before: K₅ through (u,v) in current color
        int before_k5;
        if (was_red) {
            before_k5 = count_k5_through_edge(adj, n, u, v);
        } else {
            uint64 comp[MAX_N];
            for (int i = 0; i < n; i++)
                comp[i] = (~adj[i]) & mask & ~(1ULL << i);
            before_k5 = count_k5_through_edge(comp, n, u, v);
        }

        // Flip
        adj[u] ^= (1ULL << v);
        adj[v] ^= (1ULL << u);

        // After: K₅ through (u,v) in new color
        int after_k5;
        if (was_red) {
            uint64 comp[MAX_N];
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
                adj[u] ^= (1ULL << v);
                adj[v] ^= (1ULL << u);
            }
        }

        // Periodic sync to catch any remaining drift
        if ((step + 1) % 10000 == 0) {
            int true_fit = full_fitness(adj, n);
            if (cur_fit != true_fit) {
                cur_fit = true_fit;  // resync
            }
        }

        if (cur_fit < best_fit) {
            best_fit = cur_fit;
            atomicMin(global_best, best_fit);
        }
    }

    // Verify solution
    if (cur_fit == 0) {
        int verified = full_fitness(adj, n);
        if (verified == 0) {
            int sol_idx = atomicAdd(solution_count, 1);
            if (sol_idx < 100) {
                for (int i = 0; i < n; i++)
                    best_adj_out[(uint64)sol_idx * MAX_N + i] = adj[i];
            }
            printf("*** VERIFIED SOLUTION: Walker %d, K_%d ***\n", idx, n);
        } else {
            printf("    Walker %d: false positive (inc=0, verified=%d)\n", idx, verified);
        }
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 43;
    int walkers_per_gpu = argc > 2 ? atoi(argv[2]) : 50000;
    int max_steps = argc > 3 ? atoi(argv[3]) : 5000000;

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    printf("Ramsey R(5,5) Incremental v2 (explicit-loop counter)\n");
    printf("n=%d, walkers=%d/GPU × %d GPUs = %d total\n",
           n, walkers_per_gpu, num_gpus, walkers_per_gpu * num_gpus);
    printf("Steps: %d per walker, sync every 10000\n", max_steps);
    printf("Total flips: %.2e\n\n", (double)walkers_per_gpu * num_gpus * max_steps);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int *d_best[8], *d_sol_count[8];
    uint64 *d_adj[8];

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_best[g], sizeof(int));
        cudaMalloc(&d_sol_count[g], sizeof(int));
        int init = 0x7FFFFFFF;
        cudaMemcpy(d_best[g], &init, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_sol_count[g], 0, sizeof(int));
        cudaMalloc(&d_adj[g], 100ULL * MAX_N * sizeof(uint64));

        int blocks = (walkers_per_gpu + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ramsey_sa<<<blocks, BLOCK_SIZE>>>(
            n, walkers_per_gpu, max_steps,
            d_best[g], d_adj[g], d_sol_count[g],
            time(NULL) + g * 1000003ULL);
        printf("[GPU %d] launched\n", g);
    }

    int total_solutions = 0;
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
        int g_best, g_sol;
        cudaMemcpy(&g_best, d_best[g], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&g_sol, d_sol_count[g], sizeof(int), cudaMemcpyDeviceToHost);
        printf("[GPU %d] best=%d, verified_solutions=%d\n", g, g_best, g_sol);
        if (g_sol > 0) total_solutions += g_sol;

        if (g_sol > 0) {
            uint64 *h = (uint64*)malloc(MAX_N * sizeof(uint64));
            cudaMemcpy(h, d_adj[g], MAX_N * sizeof(uint64), cudaMemcpyDeviceToHost);
            printf("  Solution adjacency (first):\n");
            for (int i = 0; i < n; i++)
                printf("    %2d: %012llx\n", i, h[i]);
            free(h);
        }
        cudaFree(d_best[g]); cudaFree(d_sol_count[g]); cudaFree(d_adj[g]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Ramsey R(5,5): n=%d\n", n);
    printf("Verified solutions: %d\n", total_solutions);
    printf("Time: %.1fs\n", elapsed);
    if (total_solutions > 0) printf("*** R(5,5) > %d ***\n", n);
    printf("========================================\n");

    return total_solutions > 0 ? 0 : 1;
}
