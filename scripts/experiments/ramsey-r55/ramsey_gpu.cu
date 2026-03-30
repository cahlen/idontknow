/*
 * GPU-native Ramsey R(5,5) search
 *
 * Everything on GPU. No CPU loops.
 *
 * Adjacency matrix: n uint64 bitmasks (n ≤ 64).
 * K₅ detection: nested bitmask AND + popcount.
 * Simulated annealing: each thread is an independent walker.
 * Random numbers: curand per thread.
 *
 * Fitness (count monochromatic K₅):
 *   For each ordered triple (a,b,c) with a<b<c:
 *     common = A[a] & A[b] & A[c]  (red common neighbors of a,b,c)
 *     For each pair (d,e) in common with d<e:
 *       if A[d] & (1<<e): found red K₅ {a,b,c,d,e}
 *   Same for blue (complement graph).
 *
 * All operations are bitmask AND + popcount on uint64.
 * For n=43: each fitness evaluation is ~43^3 / 6 ≈ 13K triples,
 * each doing 3 AND + popcount ops = ~40K ops. Trivial for GPU.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_gpu scripts/experiments/ramsey-r55/ramsey_gpu.cu -lcurand
 * Run:     ./ramsey_gpu <n> <walkers> <steps>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define MAX_N 64
#define BLOCK_SIZE 128

typedef unsigned long long uint64;

// Count monochromatic K₅ in color given by adjacency bitmasks
__device__ int count_k5(uint64 *adj, int n) {
    int count = 0;
    for (int a = 0; a < n; a++) {
        uint64 na = adj[a];
        for (int b = a + 1; b < n; b++) {
            if (!((na >> b) & 1)) continue;
            uint64 nab = na & adj[b];
            nab &= ~((1ULL << (b + 1)) - 1); // only c > b

            while (nab) {
                int c = __ffsll(nab) - 1;
                nab &= nab - 1;
                uint64 nabc = nab & adj[c]; // common neighbors > c

                // Count K₅: each pair (d,e) in nabc where d-e connected
                // Actually nabc already ensures d,e connected to a,b,c
                // Just need d-e connected
                uint64 temp = nabc;
                while (temp) {
                    int d = __ffsll(temp) - 1;
                    temp &= temp - 1;
                    count += __popcll(temp & adj[d]);
                }
            }
        }
    }
    return count;
}

__device__ int fitness(uint64 *adj, int n) {
    int red = count_k5(adj, n);
    // Blue = complement
    uint64 comp[MAX_N];
    uint64 mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
    for (int i = 0; i < n; i++)
        comp[i] = (~adj[i]) & mask & ~(1ULL << i);
    int blue = count_k5(comp, n);
    return red + blue;
}

// Each thread: independent SA walker
__global__ void ramsey_sa(
    int n, int num_walkers, int max_steps,
    int *best_fitness_out, uint64 *best_adj_out,
    uint64 seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_walkers) return;

    curandState rng;
    curand_init(seed + idx, 0, 0, &rng);

    // Random initial coloring
    uint64 adj[MAX_N];
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

    for (int step = 0; step < max_steps; step++) {
        if (cur_fit == 0) break;

        // Temperature
        float temp = 5.0f * expf(-6.0f * step / max_steps);

        // Pick random edge
        int u = curand(&rng) % n;
        int v = curand(&rng) % n;
        if (u == v) continue;
        if (u > v) { int t = u; u = v; v = t; }

        // Flip
        adj[u] ^= (1ULL << v);
        adj[v] ^= (1ULL << u);

        int new_fit = fitness(adj, n);

        if (new_fit <= cur_fit) {
            cur_fit = new_fit;
        } else {
            float delta = (float)(new_fit - cur_fit);
            float prob = expf(-delta / (temp + 1e-10f));
            if (curand_uniform(&rng) < prob) {
                cur_fit = new_fit;
            } else {
                adj[u] ^= (1ULL << v);
                adj[v] ^= (1ULL << u);
            }
        }

        if (cur_fit < best_fit) best_fit = cur_fit;
    }

    atomicMin(best_fitness_out, best_fit);

    if (cur_fit == 0) {
        // Save winning adjacency
        for (int i = 0; i < n; i++)
            best_adj_out[(uint64)idx * MAX_N + i] = adj[i];
        printf("*** WALKER %d FOUND RAMSEY-GOOD COLORING (fitness=0) ***\n", idx);
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

    printf("Ramsey R(5,5) GPU Search\n");
    printf("Vertices: %d, Walkers: %d, Steps: %d\n", n, walkers, steps);
    printf("Total edge flips: %llu\n\n", (uint64)walkers * steps);

    int ngpus;
    cudaGetDeviceCount(&ngpus);
    printf("GPUs: %d\n\n", ngpus);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Split walkers across GPUs
    int per_gpu = (walkers + ngpus - 1) / ngpus;
    int global_best = INT_MAX;

    for (int g = 0; g < ngpus; g++) {
        cudaSetDevice(g);

        int gw = per_gpu;
        if (g == ngpus - 1) gw = walkers - per_gpu * (ngpus - 1);
        if (gw <= 0) continue;

        int *d_best;
        uint64 *d_adj;
        cudaMalloc(&d_best, sizeof(int));
        cudaMemcpy(d_best, &global_best, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_adj, (uint64)gw * MAX_N * sizeof(uint64));

        int blocks = (gw + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("[GPU %d] Launching %d walkers...\n", g, gw);

        ramsey_sa<<<blocks, BLOCK_SIZE>>>(
            n, gw, steps, d_best, d_adj,
            (uint64)time(NULL) + g * 1000000);
    }

    // Sync all
    for (int g = 0; g < ngpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }

    // Collect best
    for (int g = 0; g < ngpus; g++) {
        // Note: we'd need to save d_best pointers to read them
        // For now just report from printf output
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Ramsey R(5,5): n=%d, %d walkers × %d steps\n", n, walkers, steps);
    printf("Time: %.1fs\n", elapsed);
    printf("========================================\n");

    return 0;
}
