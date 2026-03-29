/*
 * GPU-native CF denominator enumeration via batched matrix multiply
 *
 * NO CPU TREE WALK. The entire enumeration happens on GPU.
 *
 * At each depth k, we have a batch of 2x2 matrices representing
 * all CF paths of length k. To go to depth k+1, we multiply each
 * matrix by 5 generator matrices g_1,...,g_5, giving 5x more matrices.
 *
 * g_a = [[a, 1], [1, 0]]
 *
 * The denominator of CF [a1,...,ak] is the (1,0) entry (row 1, col 0)
 * of the product g_a1 * g_a2 * ... * g_ak.
 *
 * Memory: at depth k we have 5^k matrices of 4 uint64 each = 32 bytes.
 * Depth 12: 5^12 = 244M matrices = 7.6 GB. Fits on one B200 (183 GB).
 * Depth 14: 5^14 = 6.1B matrices = 195 GB. Needs 2 GPUs.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o matrix_enum scripts/experiments/zaremba-effective-bound/matrix_enum.cu
 * Run:     ./matrix_enum <max_d> <max_depth> [gpu_id]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BOUND 5
#define BLOCK_SIZE 256

typedef unsigned long long uint64;

// 2x2 matrix stored as 4 uint64: [a, b, c, d] = [[a,b],[c,d]]
// Denominator = c (row 1, col 0) after product g_a1 * ... * g_ak

// Combined expand + mark + compact kernel
// For each input matrix, produce children with d <= max_d,
// mark them in the bitset, and write to output using atomicAdd for position.
__global__ void expand_mark_compact(
    uint64 *matrices_in, uint64 num_in,
    uint64 *matrices_out, unsigned long long *out_count,
    uint32_t *bitset, uint64 max_d, uint32_t *mark_count,
    unsigned long long max_out)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    uint64 m00 = matrices_in[idx * 4 + 0];
    uint64 m01 = matrices_in[idx * 4 + 1];
    uint64 m10 = matrices_in[idx * 4 + 2];
    uint64 m11 = matrices_in[idx * 4 + 3];

    for (int a = 1; a <= BOUND; a++) {
        uint64 n10 = m10 * a + m11;  // new denominator
        if (n10 > max_d) break;      // denominators only grow with a

        uint64 n00 = m00 * a + m01;
        uint64 n01 = m00;
        uint64 n11 = m10;

        // Mark in bitset
        uint64 word = n10 / 32;
        uint32_t bit = 1u << (n10 % 32);
        atomicOr(&bitset[word], bit);
        atomicAdd(mark_count, 1);

        // Write to output (compacted — only surviving children)
        unsigned long long pos = atomicAdd(out_count, 1ULL);
        if (pos < max_out) {
            matrices_out[pos * 4 + 0] = n00;
            matrices_out[pos * 4 + 1] = n01;
            matrices_out[pos * 4 + 2] = n10;
            matrices_out[pos * 4 + 3] = n11;
        }
    }
}

// Compact: keep only matrices where denominator (entry 2) <= max_d
// Uses atomicAdd for output position — safe because each thread writes
// to a UNIQUE position (no two threads share the same atomicAdd result)
__global__ void compact_matrices(
    uint64 *matrices_in, uint64 num_in,
    uint64 *matrices_out, unsigned long long *out_count,
    uint64 max_d)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    uint64 denom = matrices_in[idx * 4 + 2];
    if (denom >= 1 && denom <= max_d) {
        unsigned long long pos = atomicAdd(out_count, 1ULL);
        if (pos < 1999000000ULL) {  // stay within buffer
            matrices_out[pos * 4 + 0] = matrices_in[idx * 4 + 0];
            matrices_out[pos * 4 + 1] = matrices_in[idx * 4 + 1];
            matrices_out[pos * 4 + 2] = matrices_in[idx * 4 + 2];
            matrices_out[pos * 4 + 3] = matrices_in[idx * 4 + 3];
        }
    }
}

// Count uncovered
__global__ void count_uncovered(uint32_t *bitset, uint64 max_d, uint64 *uncovered) {
    uint64 d = (uint64)blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (d > max_d) return;
    uint64 word = d / 32;
    uint32_t bit = 1u << (d % 32);
    if (!(bitset[word] & bit)) {
        atomicAdd((unsigned long long*)uncovered, 1ULL);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <max_d> <max_depth> [gpu_id]\n", argv[0]);
        return 1;
    }

    uint64 max_d = (uint64)atoll(argv[1]);
    int max_depth = atoi(argv[2]);
    int gpu_id = argc > 3 ? atoi(argv[3]) : 4;

    printf("GPU Matrix Enumeration for Zaremba\n");
    printf("Max d: %llu\n", (unsigned long long)max_d);
    printf("Max depth: %d\n", max_depth);
    printf("GPU: %d\n", gpu_id);

    // Memory estimate
    uint64 max_matrices = 1;
    for (int i = 0; i < max_depth; i++) max_matrices *= BOUND;
    double mem_gb = max_matrices * 32.0 / 1e9;
    printf("Max matrices at depth %d: %llu (%.1f GB)\n\n",
           max_depth, (unsigned long long)max_matrices, mem_gb);

    printf("(With compaction, actual memory usage will be much smaller)\n");

    cudaSetDevice(gpu_id);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Bitset for denominators
    uint64 bitset_words = (max_d + 32) / 32;
    uint32_t *d_bitset;
    cudaMalloc(&d_bitset, bitset_words * sizeof(uint32_t));
    cudaMemset(d_bitset, 0, bitset_words * sizeof(uint32_t));

    // Mark d=1 (identity)
    uint32_t one_bit = 1u << 1;
    cudaMemcpy(d_bitset, &one_bit, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t *d_count;
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    // Initialize depth 1: 5 matrices (g_1 through g_5)
    // g_a = [[a,1],[1,0]]
    uint64 h_init[5 * 4];
    for (int a = 1; a <= BOUND; a++) {
        h_init[(a-1)*4 + 0] = a;  // (0,0)
        h_init[(a-1)*4 + 1] = 1;  // (0,1)
        h_init[(a-1)*4 + 2] = 1;  // (1,0) = denominator
        h_init[(a-1)*4 + 3] = 0;  // (1,1)
    }

    // Mark initial denominators (1,1,1,1,1 = all are d=1, already marked)
    // Actually g_a has denominator entry = 1, so d=1 is marked

    // Double buffer — need space for the expansion step (5x current live)
    // Peak is around depth 11-12 where we have ~50M live, expanding to 250M
    // Allocate 300M slots = 9.6 GB. Fits on B200.
    uint64 buf_matrices = 2000000000ULL;  // 2B slots = 64GB per buffer
    if (buf_matrices > max_matrices) buf_matrices = max_matrices;
    uint64 buf_size = buf_matrices * 4 * sizeof(uint64);
    printf("Allocating %.1f GB per buffer (%llu slots)...\n",
           buf_size / 1e9, (unsigned long long)buf_matrices);

    uint64 *d_buf_a, *d_buf_b;
    cudaMalloc(&d_buf_a, buf_size);
    cudaMalloc(&d_buf_b, buf_size);

    // Upload initial matrices
    cudaMemcpy(d_buf_a, h_init, 5 * 4 * sizeof(uint64), cudaMemcpyHostToDevice);
    uint64 num_matrices = 5;

    // Mark depth-1 denominators (all = 1, already handled)

    unsigned long long *d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned long long));

    printf("Expanding tree on GPU (fused expand+compact)...\n");
    for (int depth = 1; depth < max_depth; depth++) {
        cudaMemset(d_out_count, 0, sizeof(unsigned long long));

        uint64 blocks64 = (num_matrices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocks = (int)(blocks64 > 2147483647 ? 2147483647 : blocks64);
        expand_mark_compact<<<blocks, BLOCK_SIZE>>>(
            d_buf_a, num_matrices,
            d_buf_b, d_out_count,
            d_bitset, max_d, d_count,
            buf_matrices
        );
        cudaDeviceSynchronize();

        unsigned long long h_out;
        cudaMemcpy(&h_out, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // Swap buffers
        uint64 *tmp = d_buf_a; d_buf_a = d_buf_b; d_buf_b = tmp;
        num_matrices = (uint64)h_out;
        if (num_matrices > buf_matrices) num_matrices = buf_matrices;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

        uint32_t h_count;
        cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        printf("  depth %2d: %12llu live, %u marks, %.1fs\n",
               depth + 1, (unsigned long long)num_matrices, h_count, elapsed);
        fflush(stdout);

        if (num_matrices == 0) {
            printf("  (all branches pruned)\n");
            break;
        }
    }

    cudaFree(d_out_count);

    // Count uncovered
    uint64 *d_uncovered;
    cudaMalloc(&d_uncovered, sizeof(uint64));
    cudaMemset(d_uncovered, 0, sizeof(uint64));

    int count_blocks = (max_d + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_uncovered<<<count_blocks, BLOCK_SIZE>>>(d_bitset, max_d, d_uncovered);
    cudaDeviceSynchronize();

    uint64 h_uncovered;
    cudaMemcpy(&h_uncovered, d_uncovered, sizeof(uint64), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("GPU Matrix Enumeration: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Uncovered: %llu\n", (unsigned long long)h_uncovered);
    printf("Time: %.1fs\n", elapsed);
    if (h_uncovered == 0)
        printf("ALL d in [1, %llu] are Zaremba denominators\n", (unsigned long long)max_d);
    printf("========================================\n");

    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaFree(d_bitset); cudaFree(d_count); cudaFree(d_uncovered);
    return h_uncovered > 0 ? 1 : 0;
}
