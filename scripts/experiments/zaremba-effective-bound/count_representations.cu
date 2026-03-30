/*
 * Count R(d) = representation number for each d ≤ max_d
 *
 * Unlike the v6 kernel (which marks a bitset 0/1), this kernel
 * COUNTS how many CF paths land on each denominator d.
 *
 * R(d) = #{(a₁,...,aₖ) : aᵢ ∈ {1,...,5}, q_k = d}
 *
 * Output: CSV with d, R(d) for all d with R(d) > 0.
 *
 * For d ≤ 10^6: fits in GPU memory easily.
 * Uses the same fused expand+mark kernel but with atomicAdd
 * on a count array instead of atomicOr on a bitset.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o count_reps count_representations.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define BOUND 5
#define BLOCK_SIZE 256
#define MAX_DEPTH 40

typedef unsigned long long uint64;
typedef unsigned int uint32;

__global__ void expand_and_count(
    uint64 *in, uint64 num_in,
    uint64 *out, unsigned long long *out_count,
    uint32 *counts, uint64 max_d,
    unsigned long long max_out)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    uint64 m00 = in[idx*4], m01 = in[idx*4+1];
    uint64 m10 = in[idx*4+2], m11 = in[idx*4+3];

    for (int a = 1; a <= BOUND; a++) {
        uint64 n10 = m10 * a + m11;
        if (n10 > max_d) break;

        uint64 n00 = m00 * a + m01;

        // COUNT (not just mark)
        atomicAdd(&counts[n10], 1u);

        // Compact write for further expansion
        unsigned long long pos = atomicAdd(out_count, 1ULL);
        if (pos < max_out) {
            out[pos*4] = n00; out[pos*4+1] = m00;
            out[pos*4+2] = n10; out[pos*4+3] = m10;
        }
    }
}

int main(int argc, char **argv) {
    uint64 max_d = argc > 1 ? (uint64)atoll(argv[1]) : 1000000;

    printf("Zaremba Representation Counter: R(d) for d ≤ %llu\n\n",
           (unsigned long long)max_d);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Allocate count array on GPU
    uint32 *d_counts;
    cudaMalloc(&d_counts, (max_d + 1) * sizeof(uint32));
    cudaMemset(d_counts, 0, (max_d + 1) * sizeof(uint32));

    // Mark d=1
    uint32 one = 1;
    cudaMemcpy(d_counts + 1, &one, sizeof(uint32), cudaMemcpyHostToDevice);

    // Buffers for tree expansion
    uint64 buf_slots = 200000000ULL; // 200M
    uint64 *d_buf_a, *d_buf_b;
    cudaMalloc(&d_buf_a, buf_slots * 4 * sizeof(uint64));
    cudaMalloc(&d_buf_b, buf_slots * 4 * sizeof(uint64));
    unsigned long long *d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned long long));

    // Init depth 1
    uint64 h_init[5*4];
    for (int a = 1; a <= BOUND; a++) {
        h_init[(a-1)*4] = a; h_init[(a-1)*4+1] = 1;
        h_init[(a-1)*4+2] = 1; h_init[(a-1)*4+3] = 0;
    }
    cudaMemcpy(d_buf_a, h_init, 5*4*sizeof(uint64), cudaMemcpyHostToDevice);
    uint64 num = 5;

    // Count the 5 initial denominators (q₁ = 1 for all a)
    // Actually q₁ = 1 always, already marked above.
    // The depth-1 matrices have m10=1, m11=0, so denominator = 1.
    // We need to mark the depth-1 paths: denominator q₁ = 1 for each a.
    // Already counted (5 paths give d=1, so R(1) should be 5...
    // but actually [0;a] = 1/a, so denominator = a, not 1!
    // Let me fix: the matrix g_a = [[a,1],[1,0]], so q₁ = 1 (bottom-right).
    // Wait: [0;a] = 1/a has denominator a. But g_a = [[a,1],[1,0]]
    // means the convergent is p₁/q₁ = a/1. So q₁ = 1.
    // Hmm, that's the denominator of the CONVERGENT a/1 = a.
    // Actually [0;a₁] = 1/a₁, which has numerator 1, denominator a₁.
    // The matrix product for [0;a₁] is g_{a₁} = [[a₁,1],[1,0]].
    // So p₁ = a₁, q₁ = 1. That means the fraction is a₁/1 = a₁.
    // But we want [0;a₁] = 1/a₁. The convention differs!
    //
    // In Zaremba: b/d = [a₁,...,aₖ] means g_{a₁}...g_{aₖ} = [[pₖ,p_{k-1}],[qₖ,q_{k-1}]]
    // and b/d = pₖ/qₖ.
    // For k=1: g_{a₁} = [[a₁,1],[1,0]], so p₁ = a₁, q₁ = 1.
    // So b/d = a₁/1 ??? That gives d = 1 for all single-digit CFs.
    //
    // For k=2: g_{a₁}g_{a₂} = [[a₁a₂+1, a₁],[a₂, 1]]
    // So q₂ = a₂, and the fraction is (a₁a₂+1)/a₂.
    //
    // So denominators at depth 1 are all 1, at depth 2 are a₂ ∈ {1,...,5}.
    // The expand kernel correctly tracks this via the matrix product.

    for (int depth = 1; depth < MAX_DEPTH && num > 0; depth++) {
        cudaMemset(d_out_count, 0, sizeof(unsigned long long));
        int blocks = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_and_count<<<blocks, BLOCK_SIZE>>>(
            d_buf_a, num, d_buf_b, d_out_count,
            d_counts, max_d, buf_slots);
        cudaDeviceSynchronize();

        unsigned long long h_out;
        cudaMemcpy(&h_out, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        uint64 *tmp = d_buf_a; d_buf_a = d_buf_b; d_buf_b = tmp;
        num = h_out < buf_slots ? h_out : buf_slots;

        if (depth <= 10 || depth % 5 == 0)
            printf("  depth %2d: %llu live matrices\n", depth+1, (unsigned long long)num);
    }

    // Download counts
    uint32 *h_counts = (uint32*)malloc((max_d + 1) * sizeof(uint32));
    cudaMemcpy(h_counts, d_counts, (max_d + 1) * sizeof(uint32), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Output CSV
    char filename[256];
    snprintf(filename, sizeof(filename),
             "scripts/experiments/zaremba-effective-bound/representation_counts_%llu.csv",
             (unsigned long long)max_d);
    FILE *f = fopen(filename, "w");
    fprintf(f, "d,R(d)\n");

    uint64 total_reps = 0;
    uint64 zero_count = 0;
    uint64 min_nonzero_R = UINT64_MAX;
    uint64 min_nonzero_d = 0;
    double sum_log_R = 0;
    int log_count = 0;

    for (uint64 d = 1; d <= max_d; d++) {
        uint32 R = h_counts[d];
        if (R > 0) {
            fprintf(f, "%llu,%u\n", (unsigned long long)d, R);
            total_reps += R;
            if (R < min_nonzero_R) { min_nonzero_R = R; min_nonzero_d = d; }
            if (d >= 100) { sum_log_R += log((double)R) / log((double)d); log_count++; }
        } else {
            zero_count++;
        }
    }
    fclose(f);

    printf("\n========================================\n");
    printf("R(d) counts for d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Time: %.1fs\n", elapsed);
    printf("Total representations: %llu\n", (unsigned long long)total_reps);
    printf("Denominators with R(d) = 0: %llu\n", (unsigned long long)zero_count);
    printf("Min nonzero R(d): %llu at d=%llu\n",
           (unsigned long long)min_nonzero_R, (unsigned long long)min_nonzero_d);
    printf("Average log R(d) / log d (for d ≥ 100): %.6f\n",
           log_count > 0 ? sum_log_R / log_count : 0);
    printf("Expected (2δ-1): %.6f\n", 2*0.836829443681208 - 1);
    printf("Output: %s\n", filename);
    printf("========================================\n");

    cudaFree(d_counts); cudaFree(d_buf_a); cudaFree(d_buf_b); cudaFree(d_out_count);
    free(h_counts);
    return zero_count > 0 ? 1 : 0;
}
