/*
 * Ramsey R(5,5) — Exhaustive Extension of ALL 656 K₄₂ Colorings → K₄₃
 *
 * For each of McKay's 656 (5,5)-good K₄₂ colorings, exhaustively check
 * all 2^42 ways to add a 43rd vertex. Total: 656 × 4.4T ≈ 2.9×10¹⁵.
 *
 * Optimization: process multiple colorings per GPU in sequence.
 * Each coloring takes ~130s. 656 colorings / 8 GPUs = 82 per GPU ≈ 3 hours.
 *
 * If ANY coloring has a valid extension → R(5,5) ≥ 44.
 * If NONE → strongest computational evidence ever that R(5,5) = 43.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_extend_all \
 *          scripts/experiments/ramsey-r55/ramsey_extend_all.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

typedef unsigned long long uint64;
#define BLOCK_SIZE 256
#define N 42

__global__ void check_extensions(
    uint64 start, uint64 count,
    const uint64 *red_k4, int num_red_k4,
    const uint64 *blue_k4, int num_blue_k4,
    int *num_solutions, int coloring_id)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64 ext = start + idx;
    uint64 blue_ext = (~ext) & ((1ULL << N) - 1);

    // Check red K₅
    for (int k = 0; k < num_red_k4; k++)
        if ((ext & red_k4[k]) == red_k4[k]) return;

    // Check blue K₅
    for (int k = 0; k < num_blue_k4; k++)
        if ((blue_ext & blue_k4[k]) == blue_k4[k]) return;

    atomicAdd(num_solutions, 1);
    printf("*** SOLUTION: coloring %d, ext=0x%011llx ***\n", coloring_id, ext);
}

int main() {
    printf("========================================\n");
    printf("Ramsey R(5,5) — ALL 656 K₄₂ Extensions\n");
    printf("========================================\n\n");

    // Load binary data
    FILE *f = fopen("scripts/experiments/ramsey-r55/mckay_k42_all.bin", "rb");
    if (!f) { printf("Cannot open data file\n"); return 1; }

    unsigned int num_colorings;
    fread(&num_colorings, sizeof(unsigned int), 1, f);
    printf("Colorings: %u\n", num_colorings);

    // Read all K₄ data into host memory
    typedef struct {
        int num_red, num_blue;
        uint64 *red_k4, *blue_k4;
    } ColoringData;

    ColoringData *colorings = (ColoringData*)malloc(num_colorings * sizeof(ColoringData));
    for (unsigned int i = 0; i < num_colorings; i++) {
        unsigned int nr, nb;
        fread(&nr, sizeof(unsigned int), 1, f);
        fread(&nb, sizeof(unsigned int), 1, f);
        colorings[i].num_red = nr;
        colorings[i].num_blue = nb;
        colorings[i].red_k4 = (uint64*)malloc(nr * sizeof(uint64));
        colorings[i].blue_k4 = (uint64*)malloc(nb * sizeof(uint64));
        fread(colorings[i].red_k4, sizeof(uint64), nr, f);
        fread(colorings[i].blue_k4, sizeof(uint64), nb, f);
    }
    fclose(f);
    printf("Loaded all K₄ data\n\n");

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    uint64 total_per_coloring = 1ULL << N;
    uint64 chunk_size = 1ULL << 30;

    printf("Plan: %u colorings × 2^%d extensions = %.2e total checks\n",
           num_colorings, N, (double)num_colorings * total_per_coloring);
    printf("Using %d GPUs, ~%.0f min estimated\n\n",
           num_gpus, (double)num_colorings * 130.0 / num_gpus / 60.0);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // GPU buffers (reused per coloring)
    uint64 *d_red[8], *d_blue[8];
    int *d_nsol[8];
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_red[g], 5000 * sizeof(uint64));
        cudaMalloc(&d_blue[g], 5000 * sizeof(uint64));
        cudaMalloc(&d_nsol[g], sizeof(int));
    }

    int grand_total_solutions = 0;

    // Process colorings round-robin across GPUs
    // Launch one coloring per GPU at a time, process all chunks
    for (unsigned int ci = 0; ci < num_colorings; ci += num_gpus) {
        // Launch one coloring on each GPU
        for (int g = 0; g < num_gpus && ci + g < num_colorings; g++) {
            int c = ci + g;
            cudaSetDevice(g);

            // Upload this coloring's K₄ data
            cudaMemcpy(d_red[g], colorings[c].red_k4,
                       colorings[c].num_red * sizeof(uint64), cudaMemcpyHostToDevice);
            cudaMemcpy(d_blue[g], colorings[c].blue_k4,
                       colorings[c].num_blue * sizeof(uint64), cudaMemcpyHostToDevice);
            cudaMemset(d_nsol[g], 0, sizeof(int));

            // Launch all chunks for this coloring
            for (uint64 start = 0; start < total_per_coloring; start += chunk_size) {
                uint64 count = (start + chunk_size > total_per_coloring) ?
                               (total_per_coloring - start) : chunk_size;
                uint64 blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                check_extensions<<<blocks, BLOCK_SIZE>>>(
                    start, count,
                    d_red[g], colorings[c].num_red,
                    d_blue[g], colorings[c].num_blue,
                    d_nsol[g], c);
            }
        }

        // Sync all GPUs
        for (int g = 0; g < num_gpus && ci + g < num_colorings; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();

            int ns;
            cudaMemcpy(&ns, d_nsol[g], sizeof(int), cudaMemcpyDeviceToHost);
            if (ns > 0) {
                printf("*** COLORING %d HAS %d VALID EXTENSIONS! ***\n", ci + g, ns);
                grand_total_solutions += ns;
            }
        }

        // Progress
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;
        int done = (ci + num_gpus < num_colorings) ? ci + num_gpus : num_colorings;
        double rate = done / elapsed;
        double eta = (num_colorings - done) / rate;
        printf("[%.0fs] %d/%u colorings done (%.1f col/min) | ETA %.0fs | solutions: %d\n",
               elapsed, done, num_colorings, rate * 60, eta, grand_total_solutions);
        fflush(stdout);

        if (grand_total_solutions > 0) {
            printf("\n*** SOLUTION FOUND — R(5,5) >= 44! ***\n");
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("ALL %u K₄₂ colorings checked\n", num_colorings);
    printf("Total extensions: %.2e\n", (double)num_colorings * total_per_coloring);
    printf("Solutions: %d\n", grand_total_solutions);
    printf("Time: %.1fs (%.1f min)\n", elapsed, elapsed / 60);
    if (grand_total_solutions > 0) {
        printf("\n*** R(5,5) >= 44 ***\n");
    } else {
        printf("\nNONE of the 656 known K₄₂ colorings can be extended to K₄₃.\n");
        printf("This is the strongest computational evidence that R(5,5) = 43.\n");
    }
    printf("========================================\n");

    // Cleanup
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(d_red[g]); cudaFree(d_blue[g]); cudaFree(d_nsol[g]);
    }
    for (unsigned int i = 0; i < num_colorings; i++) {
        free(colorings[i].red_k4);
        free(colorings[i].blue_k4);
    }
    free(colorings);

    return grand_total_solutions > 0 ? 0 : 1;
}
