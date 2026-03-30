/*
 * Ramsey R(5,5) — Exhaustive Extension of Exoo's K₄₂ → K₄₃
 *
 * Exoo (1989) proved R(5,5) ≥ 43 by constructing a (5,5)-good
 * 2-coloring of K₄₂. This kernel exhaustively checks ALL 2^42
 * ways to add a 43rd vertex to determine if R(5,5) ≥ 44.
 *
 * Method: precompute all 2,318 monochromatic K₄ in Exoo's K₄₂.
 * For each extension pattern (bitmask of 42 edge colors from the
 * new vertex to existing vertices), check if it completes any K₄
 * into a K₅. A pattern is valid iff it avoids ALL constraints.
 *
 * Complexity: 2^42 ≈ 4.4×10¹² extensions × 2,318 checks each.
 * Each check is a single bitmask AND+compare (1 cycle on GPU).
 * Estimated time: ~73 minutes on 8×B200.
 *
 * If ANY extension is valid → R(5,5) ≥ 44 (first improvement since 1989).
 * If NONE valid → Exoo's K₄₂ cannot be extended (but other K₄₂ colorings
 * from McKay's database of 656 could still work).
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_extend \
 *          scripts/experiments/ramsey-r55/ramsey_extend.cu
 * Run:     ./ramsey_extend
 *
 * Data source: arXiv:2212.12630 (Study of Exoo's Lower Bound)
 * Verified: 0 monochromatic K₅, 1148 red K₄, 1170 blue K₄
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef unsigned long long uint64;
#define BLOCK_SIZE 256

#include "exoo_k42_data.h"

__global__ void check_extensions(
    uint64 start, uint64 count,
    const uint64 *red_k4, int num_red_k4,
    const uint64 *blue_k4, int num_blue_k4,
    uint64 *solutions, int *num_solutions,
    uint64 *progress)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64 ext = start + idx;

    // Check red K₅: need a red K₄ where ALL 4 vertices are red-connected to new vertex
    for (int k = 0; k < num_red_k4; k++) {
        if ((ext & red_k4[k]) == red_k4[k]) return;
    }

    // Check blue K₅: need a blue K₄ where ALL 4 vertices are blue-connected to new vertex
    uint64 blue_ext = (~ext) & ((1ULL << EXOO_N) - 1);
    for (int k = 0; k < num_blue_k4; k++) {
        if ((blue_ext & blue_k4[k]) == blue_k4[k]) return;
    }

    // VALID EXTENSION — no monochromatic K₅!
    int si = atomicAdd(num_solutions, 1);
    if (si < 10000) solutions[si] = ext;
    printf("*** R(5,5) >= 44: extension 0x%011llx ***\n", ext);
}

// Progress reporting kernel — runs on one thread, reads atomics
__global__ void report_progress(uint64 total_checked, uint64 total, int *num_solutions, int gpu_id) {
    printf("[GPU %d] %.2f%% done (%llu / %llu), solutions so far: %d\n",
           gpu_id, 100.0 * total_checked / total, total_checked, total, *num_solutions);
}

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("Ramsey R(5,5) Exhaustive Extension\n");
    printf("Base: Exoo's K₄₂ (verified K₅-free)\n");
    printf("Target: K₄₃ (would prove R(5,5) ≥ 44)\n");
    printf("========================================\n\n");

    printf("Constraints: %d red K₄ + %d blue K₄ = %d total\n",
           NUM_RED_K4, NUM_BLUE_K4, NUM_RED_K4 + NUM_BLUE_K4);

    uint64 total = 1ULL << EXOO_N;  // 2^42
    printf("Extensions to check: 2^%d = %.2e\n\n", EXOO_N, (double)total);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    // Chunk the work across GPUs
    // Use smaller chunks for progress reporting
    uint64 chunk_size = 1ULL << 30;  // ~1 billion per chunk
    uint64 num_chunks = (total + chunk_size - 1) / chunk_size;

    printf("Using %d GPUs, %llu chunks of %llu each\n\n", num_gpus, num_chunks, chunk_size);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Upload K₄ data to each GPU
    uint64 *d_red[8], *d_blue[8], *d_sol[8];
    int *d_nsol[8];
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_red[g], NUM_RED_K4 * sizeof(uint64));
        cudaMalloc(&d_blue[g], NUM_BLUE_K4 * sizeof(uint64));
        cudaMalloc(&d_sol[g], 10000 * sizeof(uint64));
        cudaMalloc(&d_nsol[g], sizeof(int));
        cudaMemcpy(d_red[g], RED_K4, NUM_RED_K4 * sizeof(uint64), cudaMemcpyHostToDevice);
        cudaMemcpy(d_blue[g], BLUE_K4, NUM_BLUE_K4 * sizeof(uint64), cudaMemcpyHostToDevice);
        cudaMemset(d_nsol[g], 0, sizeof(int));
    }

    int total_solutions = 0;
    uint64 total_checked = 0;

    // Process chunks round-robin across GPUs
    for (uint64 chunk = 0; chunk < num_chunks; chunk++) {
        int g = chunk % num_gpus;
        cudaSetDevice(g);

        uint64 start = chunk * chunk_size;
        uint64 count = (start + chunk_size > total) ? (total - start) : chunk_size;

        uint64 blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        check_extensions<<<blocks, BLOCK_SIZE>>>(
            start, count,
            d_red[g], NUM_RED_K4,
            d_blue[g], NUM_BLUE_K4,
            d_sol[g], d_nsol[g], NULL);

        // Sync and report progress every num_gpus chunks
        if ((chunk + 1) % num_gpus == 0 || chunk == num_chunks - 1) {
            for (int gg = 0; gg < num_gpus; gg++) {
                cudaSetDevice(gg);
                cudaDeviceSynchronize();
            }

            total_checked = (chunk + 1) * chunk_size;
            if (total_checked > total) total_checked = total;

            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;
            double rate = total_checked / elapsed;
            double eta = (total - total_checked) / rate;

            // Check solutions
            int batch_sol = 0;
            for (int gg = 0; gg < num_gpus; gg++) {
                int ns;
                cudaSetDevice(gg);
                cudaMemcpy(&ns, d_nsol[gg], sizeof(int), cudaMemcpyDeviceToHost);
                batch_sol += ns;
            }

            printf("[%.0fs] %.2f%% (%llu / %llu) | %.2e ext/s | ETA %.0fs | solutions: %d\n",
                   elapsed, 100.0 * total_checked / total,
                   total_checked, total, rate, eta, batch_sol);
            fflush(stdout);

            if (batch_sol > 0) {
                total_solutions = batch_sol;
                printf("\n*** SOLUTIONS FOUND — stopping early ***\n");
                break;
            }
        }
    }

    // Final results
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    // Collect all solutions
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        int ns;
        cudaMemcpy(&ns, d_nsol[g], sizeof(int), cudaMemcpyDeviceToHost);
        if (ns > 0) {
            uint64 *h_sol = (uint64*)malloc(ns * sizeof(uint64));
            cudaMemcpy(h_sol, d_sol[g], (ns < 10000 ? ns : 10000) * sizeof(uint64), cudaMemcpyDeviceToHost);
            printf("\n[GPU %d] %d solutions:\n", g, ns);
            for (int s = 0; s < ns && s < 20; s++)
                printf("  ext[%d] = 0x%011llx\n", s, h_sol[s]);
            free(h_sol);
            total_solutions += ns;
        }
        cudaFree(d_red[g]); cudaFree(d_blue[g]);
        cudaFree(d_sol[g]); cudaFree(d_nsol[g]);
    }

    printf("\n========================================\n");
    printf("Exhaustive extension of Exoo's K₄₂ → K₄₃\n");
    printf("Checked: %llu extensions\n", total_checked);
    printf("Solutions: %d\n", total_solutions);
    printf("Time: %.1fs (%.2e ext/s)\n", elapsed, total_checked / elapsed);
    if (total_solutions > 0) {
        printf("\n*** R(5,5) >= 44 ***\n");
        printf("*** First improvement to Ramsey R(5,5) lower bound since 1989! ***\n");
    } else {
        printf("\nExoo's K₄₂ CANNOT be extended to K₄₃.\n");
        printf("Next: try McKay's other 655 (5,5)-good K₄₂ colorings.\n");
    }
    printf("========================================\n");

    return total_solutions > 0 ? 0 : 1;
}
