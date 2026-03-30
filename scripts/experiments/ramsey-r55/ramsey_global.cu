/*
 * Ramsey R(5,5) — Incremental SA with GLOBAL memory adjacency
 *
 * Fix for the local memory corruption bug: move adj arrays to
 * pre-allocated global memory. Each walker gets a slice of a
 * large global buffer instead of stack-allocated local arrays.
 *
 * This eliminates the stack overflow / corruption that caused
 * systematic fitness drift in the incremental counter.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_global scripts/experiments/ramsey-r55/ramsey_global.cu -lcurand
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define MAX_N 48
#define BLOCK_SIZE 128

typedef unsigned long long uint64;

// K₅ through edge (u,v) — explicit loop version (GPU-verified correct)
__device__ int count_k5_through_edge(uint64 *adj, int n, int u, int v) {
    int cn[MAX_N], ncn = 0;
    for (int w = 0; w < n; w++) {
        if (w == u || w == v) continue;
        if ((adj[u] >> w) & 1 && (adj[v] >> w) & 1)
            cn[ncn++] = w;
    }
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

__device__ int full_fitness(uint64 *adj, uint64 *comp, int n) {
    int red = full_k5_count(adj, n);
    uint64 mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
    for (int i = 0; i < n; i++)
        comp[i] = (~adj[i]) & mask & ~(1ULL << i);
    return red + full_k5_count(comp, n);
}

// Each walker gets adj[MAX_N] and comp[MAX_N] from GLOBAL memory
__global__ void ramsey_sa(
    int n, int num_walkers, int max_steps,
    uint64 *g_adj,    // [num_walkers * MAX_N]
    uint64 *g_comp,   // [num_walkers * MAX_N]
    int *global_best, uint64 *best_adj_out,
    int *solution_count, uint64 seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_walkers) return;

    // Pointers into global memory for this walker
    uint64 *adj = g_adj + (uint64)idx * MAX_N;
    uint64 *comp = g_comp + (uint64)idx * MAX_N;

    curandState rng;
    curand_init(seed + idx * 7919ULL, 0, 0, &rng);

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

    int cur_fit = full_fitness(adj, comp, n);
    int best_fit = cur_fit;

    for (int step = 0; step < max_steps && cur_fit > 0; step++) {
        float progress = (float)step / max_steps;
        float temp = 3.0f * (1.0f - progress * progress);
        if (temp < 0.05f) temp = 0.05f;

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

        // Periodic sync
        if ((step + 1) % 10000 == 0) {
            int true_fit = full_fitness(adj, comp, n);
            if (cur_fit != true_fit) {
                // If there's ANY drift, print warning and resync
                if (cur_fit != true_fit && step < 100000)
                    printf("Walker %d step %d: drift %d (inc=%d true=%d)\n",
                           idx, step, cur_fit - true_fit, cur_fit, true_fit);
                cur_fit = true_fit;
            }
        }

        if (cur_fit < best_fit) {
            best_fit = cur_fit;
            atomicMin(global_best, best_fit);
        }
    }

    // Verify
    if (cur_fit == 0) {
        int verified = full_fitness(adj, comp, n);
        if (verified == 0) {
            int sol_idx = atomicAdd(solution_count, 1);
            if (sol_idx < 100)
                for (int i = 0; i < n; i++)
                    best_adj_out[(uint64)sol_idx * MAX_N + i] = adj[i];
            printf("*** VERIFIED SOLUTION: Walker %d ***\n", idx);
        } else {
            printf("    Walker %d: false positive (%d)\n", idx, verified);
        }
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 43;
    int wpg = argc > 2 ? atoi(argv[2]) : 10000;
    int steps = argc > 3 ? atoi(argv[3]) : 2000000;

    int ngpu; cudaGetDeviceCount(&ngpu);
    printf("Ramsey R(5,5) Global-Memory Incremental SA\n");
    printf("n=%d, %d walkers/GPU × %d GPUs, %d steps\n\n", n, wpg, ngpu, steps);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int *d_best[8], *d_sol[8];
    uint64 *d_adj_buf[8], *d_comp_buf[8], *d_out[8];

    for (int g = 0; g < ngpu; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_best[g], 4);
        cudaMalloc(&d_sol[g], 4);
        int inf = 0x7FFFFFFF;
        cudaMemcpy(d_best[g], &inf, 4, cudaMemcpyHostToDevice);
        cudaMemset(d_sol[g], 0, 4);
        cudaMalloc(&d_adj_buf[g], (uint64)wpg * MAX_N * 8);
        cudaMalloc(&d_comp_buf[g], (uint64)wpg * MAX_N * 8);
        cudaMalloc(&d_out[g], 100ULL * MAX_N * 8);

        ramsey_sa<<<(wpg+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            n, wpg, steps,
            d_adj_buf[g], d_comp_buf[g],
            d_best[g], d_out[g], d_sol[g],
            time(NULL) + g * 1000003ULL);
        printf("[GPU %d] launched (%llu MB adj + %llu MB comp)\n",
               g, (uint64)wpg*MAX_N*8/1048576, (uint64)wpg*MAX_N*8/1048576);
    }

    int total_sol = 0;
    for (int g = 0; g < ngpu; g++) {
        cudaSetDevice(g); cudaDeviceSynchronize();
        int gb, gs;
        cudaMemcpy(&gb, d_best[g], 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&gs, d_sol[g], 4, cudaMemcpyDeviceToHost);
        printf("[GPU %d] best=%d solutions=%d\n", g, gb, gs);
        total_sol += gs;
        if (gs > 0) {
            uint64 h[MAX_N];
            cudaMemcpy(h, d_out[g], MAX_N*8, cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++) printf("  %2d: %012llx\n", i, h[i]);
        }
        cudaFree(d_best[g]); cudaFree(d_sol[g]);
        cudaFree(d_adj_buf[g]); cudaFree(d_comp_buf[g]); cudaFree(d_out[g]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n== n=%d, solutions=%d, time=%.1fs ==\n", n, total_sol, elapsed);
    return total_sol > 0 ? 0 : 1;
}
