/*
 * CUDA-accelerated Ramsey R(5,5) lower bound search
 *
 * R(5,5) is the smallest n such that every 2-coloring of edges of K_n
 * contains a monochromatic K_5. Known: 43 ≤ R(5,5) ≤ 48.
 *
 * We search for Ramsey(5,5)-good graphs on n=43 vertices: 2-colorings
 * of K_43 with no monochromatic K_5 in either color. Finding one on
 * n=44 would improve the lower bound.
 *
 * Method: massively parallel simulated annealing over adjacency matrices.
 * The fitness function counts monochromatic K_5 subgraphs. A coloring
 * with fitness 0 is Ramsey-good.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_search scripts/experiments/ramsey-r55/ramsey_search.cu
 * Run:     ./ramsey_search <num_vertices> <num_walkers> <max_steps>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 128
#define MAX_VERTICES 48
// Adjacency matrix stored as bitmask: adj[i] has bit j set if edge (i,j) is "red"
// Unset = "blue". We need to avoid monochromatic K_5 in both colors.

// Count monochromatic K_5 in color given by adjacency bitmasks
// For n ≤ 48, each adj[i] fits in a uint64_t
__device__ uint32_t count_monochromatic_k5(uint64_t *adj, int n) {
    uint32_t count = 0;

    // Enumerate all 5-subsets by iterating over ordered 5-tuples
    // and checking complete subgraph in one color.
    // Optimization: use bitmask intersection.
    // For each pair (a,b) with edge, compute the common neighbors
    // in that color, then look for K_3 within those.

    for (int a = 0; a < n; a++) {
        uint64_t na = adj[a];  // red neighbors of a
        for (int b = a + 1; b < n; b++) {
            if (!((na >> b) & 1)) continue;  // a-b must be red

            uint64_t nab = na & adj[b];  // common red neighbors of a,b
            // Remove bits ≤ b to avoid double counting
            nab &= ~((1ULL << (b + 1)) - 1);

            while (nab) {
                int c = __ffsll(nab) - 1;
                nab &= nab - 1;

                uint64_t nabc = nab & adj[c];  // common red neighbors of a,b,c (> c)

                while (nabc) {
                    int d = __ffsll(nabc) - 1;
                    nabc &= nabc - 1;

                    // Check if d connects to all of {a,b,c} in red — already guaranteed
                    // Now find e > d that connects to all of {a,b,c,d} in red
                    uint64_t nabcd = nabc & adj[d];

                    count += __popcll(nabcd);
                }
            }
        }
    }
    return count;
}

// Compute fitness = total monochromatic K_5 count (red + blue)
__device__ uint32_t fitness(uint64_t *adj, int n) {
    // Count red K_5
    uint32_t red_k5 = count_monochromatic_k5(adj, n);

    // Build complement (blue) adjacency
    uint64_t comp[MAX_VERTICES];
    uint64_t mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
    for (int i = 0; i < n; i++) {
        comp[i] = (~adj[i]) & mask & ~(1ULL << i);  // complement, exclude self-loop
    }

    uint32_t blue_k5 = count_monochromatic_k5(comp, n);
    return red_k5 + blue_k5;
}

// Simulated annealing walker
__global__ void sa_walkers(int n, uint64_t num_walkers, uint64_t max_steps,
                            uint64_t *best_adj_out, uint32_t *best_fitness_out,
                            uint64_t seed) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_walkers) return;

    // Initialize RNG
    curandState rng;
    curand_init(seed + idx, 0, 0, &rng);

    // Random initial coloring
    uint64_t adj[MAX_VERTICES];
    for (int i = 0; i < n; i++) {
        adj[i] = 0;
        for (int j = i + 1; j < n; j++) {
            if (curand(&rng) % 2) {
                adj[i] |= (1ULL << j);
                adj[j] |= (1ULL << i);
            }
        }
    }

    uint32_t current_fitness = fitness(adj, n);
    uint32_t best_fitness_local = current_fitness;

    for (uint64_t step = 0; step < max_steps; step++) {
        if (current_fitness == 0) break;  // FOUND a Ramsey-good coloring!

        // Temperature schedule
        double temp = 5.0 * exp(-6.0 * step / max_steps);

        // Pick a random edge and flip it
        int u = curand(&rng) % n;
        int v = curand(&rng) % n;
        if (u == v) continue;
        if (u > v) { int t = u; u = v; v = t; }

        // Flip edge (u,v)
        adj[u] ^= (1ULL << v);
        adj[v] ^= (1ULL << u);

        uint32_t new_fitness = fitness(adj, n);

        // Accept or reject
        if (new_fitness <= current_fitness) {
            current_fitness = new_fitness;
        } else {
            double delta = (double)(new_fitness - current_fitness);
            double accept_prob = exp(-delta / (temp + 1e-10));
            double r = (double)curand(&rng) / (double)UINT32_MAX;
            if (r < accept_prob) {
                current_fitness = new_fitness;
            } else {
                // Reject: flip back
                adj[u] ^= (1ULL << v);
                adj[v] ^= (1ULL << u);
            }
        }

        if (current_fitness < best_fitness_local) {
            best_fitness_local = current_fitness;
        }
    }

    // Report best fitness via atomic min
    atomicMin(best_fitness_out, best_fitness_local);

    // If this walker found fitness 0, save the adjacency matrix
    if (current_fitness == 0) {
        for (int i = 0; i < n; i++) {
            best_adj_out[idx * MAX_VERTICES + i] = adj[i];
        }
        printf("*** WALKER %lu FOUND RAMSEY-GOOD COLORING ON K_%d (fitness=0) ***\n", idx, n);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <num_vertices> <num_walkers> <max_steps_per_walker>\n", argv[0]);
        fprintf(stderr, "\nExample: %s 43 100000 1000000\n", argv[0]);
        fprintf(stderr, "  Search for R(5,5)-good colorings of K_43\n");
        fprintf(stderr, "  Known: R(5,5) >= 43, so K_43 colorings should exist\n");
        fprintf(stderr, "  Try n=44 to attempt improving the lower bound\n");
        return 1;
    }

    int n = atoi(argv[1]);
    uint64_t num_walkers = (uint64_t)atoll(argv[2]);
    uint64_t max_steps = (uint64_t)atoll(argv[3]);

    printf("Ramsey R(5,5) Search\n");
    printf("Vertices: %d\n", n);
    printf("Walkers: %lu\n", num_walkers);
    printf("Steps per walker: %lu\n", max_steps);
    printf("Total edge flips: %lu\n", num_walkers * max_steps);
    printf("\n");

    if (n > MAX_VERTICES) {
        fprintf(stderr, "Error: max vertices = %d\n", MAX_VERTICES);
        return 1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n\n", device_count);

    uint64_t *d_adj;
    uint32_t *d_best_fitness;
    cudaMalloc(&d_adj, num_walkers * MAX_VERTICES * sizeof(uint64_t));
    cudaMalloc(&d_best_fitness, sizeof(uint32_t));

    uint32_t init_fitness = UINT32_MAX;
    cudaMemcpy(d_best_fitness, &init_fitness, sizeof(uint32_t), cudaMemcpyHostToDevice);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Launch across all GPUs
    uint64_t walkers_per_gpu = num_walkers / device_count;
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);

        uint64_t gpu_walkers = walkers_per_gpu;
        if (gpu == device_count - 1) gpu_walkers = num_walkers - walkers_per_gpu * (device_count - 1);

        int blocks = (gpu_walkers + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        printf("[GPU %d] Launching %lu walkers...\n", gpu, gpu_walkers);
        sa_walkers<<<blocks, THREADS_PER_BLOCK>>>(
            n, gpu_walkers, max_steps,
            d_adj + gpu * walkers_per_gpu * MAX_VERTICES,
            d_best_fitness,
            (uint64_t)time(NULL) + gpu * 1000000
        );
    }

    // Sync all GPUs
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                    (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    uint32_t h_best_fitness;
    cudaMemcpy(&h_best_fitness, d_best_fitness, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("\n========================================\n");
    printf("Ramsey R(5,5) Search Results\n");
    printf("Vertices: %d\n", n);
    printf("Total walkers: %lu\n", num_walkers);
    printf("Steps per walker: %lu\n", max_steps);
    printf("Best fitness (monochromatic K_5 count): %u\n", h_best_fitness);
    printf("Time: %.1fs\n", elapsed);

    if (h_best_fitness == 0) {
        printf("\n*** SUCCESS: Found a 2-coloring of K_%d with no monochromatic K_5! ***\n", n);
        printf("This proves R(5,5) > %d\n", n);
        if (n >= 44) {
            printf("*** THIS IMPROVES THE KNOWN LOWER BOUND ***\n");
        }
    } else {
        printf("\nNo Ramsey-good coloring found (best had %u monochromatic K_5)\n", h_best_fitness);
        printf("Try: more walkers, more steps, or different search strategy\n");
    }
    printf("========================================\n");

    cudaFree(d_adj);
    cudaFree(d_best_fitness);
    return (h_best_fitness == 0) ? 0 : 1;
}
