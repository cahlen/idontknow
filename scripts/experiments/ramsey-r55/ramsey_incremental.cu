/*
 * Ramsey R(5,5) — Incremental Fitness SA on GPU
 *
 * Key optimization: when flipping edge (u,v), only recount K₅
 * subgraphs that contain BOTH u and v. This is O(n²) per step
 * instead of O(n³) for full recount — ~43× faster for n=43.
 *
 * For edge (u,v), a monochromatic K₅ containing both u,v requires
 * 3 more vertices {a,b,c} all mutually connected and all connected
 * to both u and v in the same color.
 *
 * Before flip: count K₅ containing (u,v) as a RED edge
 * After flip: count K₅ containing (u,v) as a BLUE edge
 * delta = (after_blue_k5 - before_red_k5) for the (u,v) subgraphs
 *       + (after_red_k5 - before_blue_k5) for the complement
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_inc scripts/experiments/ramsey-r55/ramsey_incremental.cu -lcurand
 * Run:     ./ramsey_inc <n> <walkers> <steps>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define MAX_N 64
#define BLOCK_SIZE 128

typedef unsigned long long uint64;

// Count K₅ containing edge (u,v) in the color given by adj
// A K₅ through (u,v) needs 3 vertices {a,b,c} where:
//   - a,b,c are all neighbors of u AND v in this color
//   - a,b,c are mutually connected in this color
__device__ int count_k5_through_edge(uint64 *adj, int n, int u, int v) {
    // Common neighbors of u and v (same color)
    uint64 common = adj[u] & adj[v];
    // Remove u and v themselves
    common &= ~(1ULL << u);
    common &= ~(1ULL << v);

    int count = 0;
    // For each triple (a,b,c) in common that forms a triangle
    uint64 c1 = common;
    while (c1) {
        int a = __ffsll(c1) - 1;
        c1 &= c1 - 1;

        uint64 c2 = c1 & adj[a]; // neighbors of a that are also in common, > a
        while (c2) {
            int b = __ffsll(c2) - 1;
            c2 &= c2 - 1;

            // How many vertices in common are connected to both a and b?
            uint64 c3 = c2 & adj[b]; // common neighbors of a,b that are > b and in common
            count += __popcll(c3);
        }
    }
    return count;
}

// Full K₅ count (for initial fitness)
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

// SA walker with incremental fitness
__global__ void ramsey_sa_incremental(
    int n, int num_walkers, int max_steps,
    int *global_best, uint64 *best_adj_out,
    uint64 seed)
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

        // Compute delta fitness incrementally
        // Before flip: count K₅ through (u,v) in current color
        int was_red = (adj[u] >> v) & 1;

        int before_k5;
        uint64 comp[MAX_N];
        if (was_red) {
            before_k5 = count_k5_through_edge(adj, n, u, v);
            // Also count blue K₅ NOT through this edge — unchanged
            // But we need blue K₅ through (u,v) after flip
            for (int i = 0; i < n; i++)
                comp[i] = (~adj[i]) & mask & ~(1ULL << i);
        } else {
            for (int i = 0; i < n; i++)
                comp[i] = (~adj[i]) & mask & ~(1ULL << i);
            before_k5 = count_k5_through_edge(comp, n, u, v);
        }

        // Flip
        adj[u] ^= (1ULL << v);
        adj[v] ^= (1ULL << u);

        // After flip
        int after_k5;
        if (was_red) {
            // (u,v) was red, now blue. Count blue K₅ through (u,v)
            for (int i = 0; i < n; i++)
                comp[i] = (~adj[i]) & mask & ~(1ULL << i);
            after_k5 = count_k5_through_edge(comp, n, u, v);
        } else {
            // (u,v) was blue, now red. Count red K₅ through (u,v)
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

        if (cur_fit < best_fit) {
            best_fit = cur_fit;
            atomicMin(global_best, best_fit);
        }
    }

    if (cur_fit == 0) {
        for (int i = 0; i < n; i++)
            best_adj_out[(uint64)idx * MAX_N + i] = adj[i];
        printf("*** GPU WALKER %d: FOUND RAMSEY-GOOD COLORING OF K_%d ***\n", idx, n);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <n> <walkers> <steps>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int walkers = atoi(argv[2]);
    int steps = atoi(argv[3]);

    printf("Ramsey R(5,5) Incremental SA — GPU\n");
    printf("n=%d, walkers=%d, steps=%d\n", n, walkers, steps);
    printf("Total flips: %llu\n\n", (uint64)walkers * steps);

    int ngpus;
    cudaGetDeviceCount(&ngpus);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int h_best = INT_MAX;
    int *d_best[8];
    uint64 *d_adj[8];
    int per_gpu = (walkers + ngpus - 1) / ngpus;

    for (int g = 0; g < ngpus; g++) {
        cudaSetDevice(g);
        int gw = per_gpu;
        if (g == ngpus - 1) gw = walkers - per_gpu * (ngpus - 1);
        if (gw <= 0) continue;

        cudaMalloc(&d_best[g], sizeof(int));
        cudaMemcpy(d_best[g], &h_best, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_adj[g], (uint64)gw * MAX_N * sizeof(uint64));

        int blocks = (gw + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("[GPU %d] %d walkers\n", g, gw);
        ramsey_sa_incremental<<<blocks, BLOCK_SIZE>>>(
            n, gw, steps, d_best[g], d_adj[g],
            (uint64)time(NULL) + g * 999983ULL);
    }

    for (int g = 0; g < ngpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
        int gb;
        cudaMemcpy(&gb, d_best[g], sizeof(int), cudaMemcpyDeviceToHost);
        if (gb < h_best) h_best = gb;
        cudaFree(d_best[g]);
        cudaFree(d_adj[g]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Ramsey R(5,5): n=%d\n", n);
    printf("Walkers: %d, Steps: %d\n", walkers, steps);
    printf("Best fitness: %d\n", h_best);
    printf("Time: %.1fs\n", elapsed);
    if (h_best == 0)
        printf("\n*** RAMSEY-GOOD COLORING FOUND! R(5,5) > %d ***\n", n);
    else
        printf("\nNo Ramsey-good coloring found (best had %d monochromatic K₅)\n", h_best);
    printf("========================================\n");

    return h_best == 0 ? 0 : 1;
}
