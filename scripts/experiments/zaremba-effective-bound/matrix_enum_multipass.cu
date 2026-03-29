/*
 * GPU Matrix Enumeration v6 — multi-pass for 1B+ clean verification
 *
 * Problem: at depth 14 for 1B max_d, the live matrix count exceeds
 * the 2B buffer. Solution: run in two phases:
 *
 * Phase A: expand tree to depth 13 (1.2B matrices, fits in buffer)
 *          Mark all denominators found so far in the bitset.
 *          Save the live matrices count.
 *
 * Phase B: process depth-13 matrices in CHUNKS of 400M.
 *          For each chunk, expand from depth 13 to depth 40.
 *          Each chunk is independent — different chunks on different GPUs.
 *
 * This eliminates the buffer cap entirely.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o matrix_v6 scripts/experiments/zaremba-effective-bound/matrix_enum_multipass.cu
 * Run:     ./matrix_v6 <max_d>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define BOUND 5
#define BLOCK_SIZE 256
#define MAX_DEPTH 45
#define BUF_SLOTS 2000000000ULL  // 400M per buffer = 12.8 GB

typedef unsigned long long uint64;
typedef unsigned int uint32;

// Fused expand+mark+compact
__global__ void expand_mark_compact(
    uint64 *in, uint64 num_in,
    uint64 *out, unsigned long long *out_count,
    uint32 *bitset, uint64 max_d, uint32 *marks,
    unsigned long long max_out)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    uint64 m00 = in[idx*4], m01 = in[idx*4+1], m10 = in[idx*4+2], m11 = in[idx*4+3];

    for (int a = 1; a <= BOUND; a++) {
        uint64 n10 = m10 * a + m11;
        if (n10 > max_d) break;

        uint64 n00 = m00 * a + m01;

        // Mark
        atomicOr(&bitset[n10 / 32], 1u << (n10 % 32));
        atomicAdd(marks, 1);

        // Compact write
        unsigned long long pos = atomicAdd(out_count, 1ULL);
        if (pos < max_out) {
            out[pos*4] = n00; out[pos*4+1] = m00;
            out[pos*4+2] = n10; out[pos*4+3] = m10;
        }
    }
}

__global__ void count_uncovered(uint32 *bitset, uint64 max_d, unsigned long long *unc) {
    uint64 d = (uint64)blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (d > max_d) return;
    if (!(bitset[d/32] & (1u << (d%32))))
        atomicAdd(unc, 1ULL);
}

typedef struct {
    int gpu_id;
    uint64 *chunk_data;      // host: matrices for this chunk
    uint64 chunk_size;        // number of matrices
    uint32 *d_bitset;        // shared bitset (on this GPU)
    uint64 max_d;
    uint64 bitset_words;
    double elapsed;
} ChunkArgs;

void *process_chunk(void *arg) {
    ChunkArgs *c = (ChunkArgs*)arg;
    cudaSetDevice(c->gpu_id);

    uint64 *d_buf_a, *d_buf_b;
    cudaMalloc(&d_buf_a, BUF_SLOTS * 4 * sizeof(uint64));
    cudaMalloc(&d_buf_b, BUF_SLOTS * 4 * sizeof(uint64));
    unsigned long long *d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned long long));
    uint32 *d_marks;
    cudaMalloc(&d_marks, sizeof(uint32));
    cudaMemset(d_marks, 0, sizeof(uint32));

    // Upload chunk
    cudaMemcpy(d_buf_a, c->chunk_data, c->chunk_size * 4 * sizeof(uint64), cudaMemcpyHostToDevice);
    uint64 num = c->chunk_size;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int depth = 0; depth < 50 && num > 0; depth++) {
        cudaMemset(d_out_count, 0, sizeof(unsigned long long));
        int blocks = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_mark_compact<<<blocks, BLOCK_SIZE>>>(
            d_buf_a, num, d_buf_b, d_out_count,
            c->d_bitset, c->max_d, d_marks, BUF_SLOTS);
        cudaDeviceSynchronize();

        unsigned long long h_out;
        cudaMemcpy(&h_out, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        uint64 *tmp = d_buf_a; d_buf_a = d_buf_b; d_buf_b = tmp;
        num = h_out < BUF_SLOTS ? h_out : BUF_SLOTS;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    c->elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaFree(d_out_count); cudaFree(d_marks);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <max_d>\n", argv[0]);
        return 1;
    }

    uint64 max_d = (uint64)atoll(argv[1]);
    printf("Zaremba v6 Multi-Pass Verification\n");
    printf("Max d: %llu\n\n", (unsigned long long)max_d);

    int ngpus;
    cudaGetDeviceCount(&ngpus);
    printf("GPUs: %d\n\n", ngpus);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Phase A: build tree to depth 13 on GPU 0
    printf("=== Phase A: tree to depth 13 ===\n");
    cudaSetDevice(0);

    uint64 bitset_words = (max_d + 32) / 32;
    uint32 *d_bitset;
    cudaMalloc(&d_bitset, bitset_words * sizeof(uint32));
    cudaMemset(d_bitset, 0, bitset_words * sizeof(uint32));

    // Mark d=1
    uint32 bit1 = 1u << 1;
    cudaMemcpy(d_bitset, &bit1, sizeof(uint32), cudaMemcpyHostToDevice);

    uint64 *d_buf_a, *d_buf_b;
    cudaMalloc(&d_buf_a, BUF_SLOTS * 4 * sizeof(uint64));
    cudaMalloc(&d_buf_b, BUF_SLOTS * 4 * sizeof(uint64));
    unsigned long long *d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned long long));
    uint32 *d_marks;
    cudaMalloc(&d_marks, sizeof(uint32));
    cudaMemset(d_marks, 0, sizeof(uint32));

    // Init depth 1
    uint64 h_init[5*4];
    for (int a = 1; a <= BOUND; a++) {
        h_init[(a-1)*4] = a; h_init[(a-1)*4+1] = 1;
        h_init[(a-1)*4+2] = 1; h_init[(a-1)*4+3] = 0;
    }
    cudaMemcpy(d_buf_a, h_init, 5*4*sizeof(uint64), cudaMemcpyHostToDevice);
    uint64 num = 5;

    // Expand to depth 13 (stays under 1.22B which fits in buffer... barely)
    // Actually 5^12 = 244M at depth 12, 5^13 = 1.22B > 400M buffer
    // So we go to depth 12 (244M fits in 400M buffer), then chunk depth 12→40
    int phase_a_depth = 12;
    for (int depth = 1; depth < phase_a_depth; depth++) {
        cudaMemset(d_out_count, 0, sizeof(unsigned long long));
        int blocks = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_mark_compact<<<blocks, BLOCK_SIZE>>>(
            d_buf_a, num, d_buf_b, d_out_count,
            d_bitset, max_d, d_marks, BUF_SLOTS);
        cudaDeviceSynchronize();

        unsigned long long h_out;
        cudaMemcpy(&h_out, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        uint64 *tmp = d_buf_a; d_buf_a = d_buf_b; d_buf_b = tmp;
        num = h_out < BUF_SLOTS ? h_out : BUF_SLOTS;

        printf("  depth %2d: %llu live\n", depth+1, (unsigned long long)num);
    }

    // Download depth-12 matrices to host
    printf("\n  Downloading %llu depth-%d matrices...\n",
           (unsigned long long)num, phase_a_depth);
    uint64 *h_matrices = (uint64*)malloc(num * 4 * sizeof(uint64));
    cudaMemcpy(h_matrices, d_buf_a, num * 4 * sizeof(uint64), cudaMemcpyDeviceToHost);
    uint64 total_depth12 = num;

    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaFree(d_out_count); cudaFree(d_marks);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  Phase A done: %.1fs\n\n",
           (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9);

    // Phase B: process depth-12 matrices in chunks across GPUs
    printf("=== Phase B: expand depth %d→40 in chunks ===\n", phase_a_depth);

    // Allocate bitsets on each GPU (copy from GPU 0)
    uint32 *h_bitset = (uint32*)malloc(bitset_words * sizeof(uint32));
    cudaSetDevice(0);
    cudaMemcpy(h_bitset, d_bitset, bitset_words * sizeof(uint32), cudaMemcpyDeviceToHost);

    uint32 *gpu_bitsets[8];
    for (int g = 0; g < ngpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_bitsets[g], bitset_words * sizeof(uint32));
        cudaMemcpy(gpu_bitsets[g], h_bitset, bitset_words * sizeof(uint32), cudaMemcpyHostToDevice);
    }

    // Split matrices into small chunks to prevent buffer overflow
    // With 30M matrices per GPU, frontier can exceed 2B at intermediate depths
    // Solution: process in multiple rounds of smaller chunks
    int num_rounds = (max_d > 1000000000ULL) ? 8 : 1;  // 8 rounds for >1B
    uint64 round_chunk = (total_depth12 + (ngpus * num_rounds) - 1) / (ngpus * num_rounds);
    printf("  Total matrices: %llu, rounds: %d, chunk: %llu, GPUs: %d\n\n",
           (unsigned long long)total_depth12, num_rounds, (unsigned long long)round_chunk, ngpus);

    for (int round = 0; round < num_rounds; round++) {
        printf("  Round %d/%d:\n", round+1, num_rounds);
        ChunkArgs args[8];
        pthread_t threads[8];
        int active = 0;
        for (int g = 0; g < ngpus; g++) {
            uint64 slot = round * ngpus + g;
            uint64 start = slot * round_chunk;
            uint64 end = start + round_chunk;
            if (end > total_depth12) end = total_depth12;
            if (start >= total_depth12) { args[g].chunk_size = 0; continue; }

            args[g].gpu_id = g;
            args[g].chunk_data = h_matrices + start * 4;
            args[g].chunk_size = end - start;
            args[g].d_bitset = gpu_bitsets[g];
            args[g].max_d = max_d;
            args[g].bitset_words = bitset_words;

            printf("    GPU %d: %llu matrices\n", g, (unsigned long long)args[g].chunk_size);
            pthread_create(&threads[g], NULL, process_chunk, &args[g]);
            active++;
        }

        for (int g = 0; g < ngpus; g++) {
            if (args[g].chunk_size > 0) {
                pthread_join(threads[g], NULL);
                printf("    GPU %d done: %.1fs\n", g, args[g].elapsed);
            }
        }
    }

    // Merge bitsets: OR all GPU bitsets into h_bitset
    printf("\n  Merging bitsets...\n");
    for (int g = 0; g < ngpus; g++) {
        uint32 *tmp = (uint32*)malloc(bitset_words * sizeof(uint32));
        cudaSetDevice(g);
        cudaMemcpy(tmp, gpu_bitsets[g], bitset_words * sizeof(uint32), cudaMemcpyDeviceToHost);
        for (uint64 i = 0; i < bitset_words; i++) h_bitset[i] |= tmp[i];
        free(tmp);
        cudaFree(gpu_bitsets[g]);
    }

    // Count uncovered
    uint64 uncovered = 0;
    for (uint64 d = 1; d <= max_d; d++) {
        if (!(h_bitset[d/32] & (1u << (d%32)))) uncovered++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Zaremba v6: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Uncovered: %llu\n", (unsigned long long)uncovered);
    printf("Time: %.1fs\n", total);
    if (uncovered == 0)
        printf("ALL d in [1, %llu] are Zaremba denominators\n", (unsigned long long)max_d);
    printf("========================================\n");

    free(h_matrices); free(h_bitset);
    cudaSetDevice(0); cudaFree(d_bitset);
    return uncovered > 0 ? 1 : 0;
}
