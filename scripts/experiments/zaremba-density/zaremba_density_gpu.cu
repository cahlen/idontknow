/*
 * GPU-accelerated Zaremba density computation.
 *
 * Each GPU thread explores one branch of the CF tree in parallel.
 * The tree root has |A| children (one per digit). Each thread takes
 * one root and does DFS, marking denominators in a shared bitset.
 *
 * For 10^9 with A={1,2,3}: the tree has ~3^60 leaves but most branches
 * terminate early (q > max_d). GPU parallelism over the first few levels
 * of the tree gives millions of independent subtrees.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o zaremba_density_gpu zaremba_density_gpu.cu -lm
 * Run:     ./zaremba_density_gpu <max_d> <digits>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef unsigned long long uint64;

#define MAX_DIGITS 10
#define MAX_DEPTH 200

// Global bitset in device memory
__device__ uint8_t *d_bitset;

__device__ void mark(uint64 d, uint8_t *bitset, uint64 max_d) {
    if (d < 1 || d > max_d) return;
    uint64 byte = d >> 3;
    uint8_t bit = 1 << (d & 7);
    atomicOr((unsigned int*)&bitset[byte & ~3], (unsigned int)bit << (8 * (byte & 3)));
}

// Each thread gets a prefix (p_prev, p, q_prev, q) and does DFS from there
__global__ void enumerate_subtrees(
    uint64 *prefixes,    // [N × 4]: p_prev, p, q_prev, q per prefix
    int num_prefixes,
    int *digits, int num_digits,
    uint8_t *bitset, uint64 max_d)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_prefixes) return;

    uint64 p_prev0 = prefixes[tid * 4 + 0];
    uint64 p0      = prefixes[tid * 4 + 1];
    uint64 q_prev0 = prefixes[tid * 4 + 2];
    uint64 q0      = prefixes[tid * 4 + 3];

    // Mark the prefix denominator
    mark(q0, bitset, max_d);

    // DFS stack (iterative to avoid recursion limits)
    struct { uint64 p_prev, p, q_prev, q; int digit_idx; } stack[MAX_DEPTH];
    int sp = 0;

    // Push initial children
    for (int i = num_digits - 1; i >= 0; i--) {
        uint64 a = digits[i];
        uint64 q_new = a * q0 + q_prev0;
        if (q_new > max_d) continue;
        uint64 p_new = a * p0 + p_prev0;
        stack[sp].p_prev = p0;
        stack[sp].p = p_new;
        stack[sp].q_prev = q0;
        stack[sp].q = q_new;
        stack[sp].digit_idx = 0;
        sp++;
        if (sp >= MAX_DEPTH) break;
    }

    while (sp > 0) {
        sp--;
        uint64 pp = stack[sp].p_prev;
        uint64 p  = stack[sp].p;
        uint64 qp = stack[sp].q_prev;
        uint64 q  = stack[sp].q;

        mark(q, bitset, max_d);

        // Push children
        for (int i = num_digits - 1; i >= 0; i--) {
            uint64 a = digits[i];
            uint64 q_new = a * q + qp;
            if (q_new > max_d) continue;
            if (sp >= MAX_DEPTH) break;
            uint64 p_new = a * p + pp;
            stack[sp].p_prev = p;
            stack[sp].p = p_new;
            stack[sp].q_prev = q;
            stack[sp].q = q_new;
            sp++;
        }
    }
}

// Count marked bits
__global__ void count_marked(uint8_t *bitset, uint64 max_d, uint64 *count) {
    uint64 tid = blockIdx.x * (uint64)blockDim.x + threadIdx.x;
    uint64 byte_idx = tid;
    uint64 max_byte = (max_d + 8) / 8;
    if (byte_idx >= max_byte) return;

    uint8_t b = bitset[byte_idx];
    int bits = __popc((unsigned int)b);
    // Adjust for bits beyond max_d in the last byte
    if (byte_idx == max_byte - 1) {
        int valid_bits = (max_d % 8) + 1;
        uint8_t mask = (1 << valid_bits) - 1;
        bits = __popc((unsigned int)(b & mask));
    }
    if (bits > 0) atomicAdd(count, (uint64)bits);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <max_d> <digits>\n", argv[0]);
        return 1;
    }

    uint64 max_d = (uint64)atoll(argv[1]);

    // Parse digits
    int h_digits[MAX_DIGITS];
    int num_digits = 0;
    char buf[256]; strncpy(buf, argv[2], 255);
    char *tok = strtok(buf, ",");
    while (tok && num_digits < MAX_DIGITS) {
        h_digits[num_digits++] = atoi(tok);
        tok = strtok(NULL, ",");
    }

    printf("========================================\n");
    printf("Zaremba Density (GPU)\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Digits: {");
    for (int i = 0; i < num_digits; i++) printf("%s%d", i?",":"", h_digits[i]);
    printf("}\n");
    printf("========================================\n\n");

    // Generate prefixes: enumerate CF tree to target depth on CPU
    // More prefixes = more GPU threads = better utilization
    int PREFIX_DEPTH = 8;  // 3^8 = 6561 for A={1,2,3}
    if (max_d >= 1000000000ULL) PREFIX_DEPTH = 10;  // 3^10 = 59049
    if (max_d >= 10000000000ULL) PREFIX_DEPTH = 12;

    int max_prefixes = 10000000;
    uint64 *h_prefixes = (uint64*)malloc(max_prefixes * 4 * sizeof(uint64));
    int np = 0;

    // Recursive prefix generation on CPU
    struct PrefixEntry { uint64 pp, p, qp, q; int depth; };
    struct PrefixEntry *stk = (struct PrefixEntry*)malloc(2000000 * sizeof(struct PrefixEntry));
    int ssp = 0;
    // Seed: depth-1 nodes
    for (int i = 0; i < num_digits; i++) {
        stk[ssp].pp = 0; stk[ssp].p = 1;
        stk[ssp].qp = 1; stk[ssp].q = h_digits[i];
        stk[ssp].depth = 1;
        ssp++;
    }
    while (ssp > 0) {
        ssp--;
        uint64 pp = stk[ssp].pp, p = stk[ssp].p;
        uint64 qp = stk[ssp].qp, q = stk[ssp].q;
        int dep = stk[ssp].depth;
        if (q > max_d) continue;
        if (dep >= PREFIX_DEPTH) {
            // Deep enough — hand off to GPU
            if (np < max_prefixes) {
                h_prefixes[np*4+0] = pp;
                h_prefixes[np*4+1] = p;
                h_prefixes[np*4+2] = qp;
                h_prefixes[np*4+3] = q;
                np++;
            }
        } else {
            for (int i = num_digits - 1; i >= 0; i--) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d) continue;
                uint64 pn = (uint64)h_digits[i] * p + pp;
                stk[ssp].pp = p; stk[ssp].p = pn;
                stk[ssp].qp = q; stk[ssp].q = qn;
                stk[ssp].depth = dep + 1;
                ssp++;
            }
        }
    }

    // Also add depth-1, depth-2, depth-3 denominators
    // (the prefixes above start at depth 4)
    // We'll mark these on CPU after

    printf("Prefixes (depth 4): %d\n", np);
    printf("Bitset: %.2f GB\n\n", (max_d + 8) / 8.0 / 1e9);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Allocate GPU
    uint64 bitset_bytes = (max_d + 8) / 8;
    uint8_t *d_bs;
    cudaMalloc(&d_bs, bitset_bytes);
    cudaMemset(d_bs, 0, bitset_bytes);

    int *d_digits;
    cudaMalloc(&d_digits, num_digits * sizeof(int));
    cudaMemcpy(d_digits, h_digits, num_digits * sizeof(int), cudaMemcpyHostToDevice);

    uint64 *d_prefixes;
    cudaMalloc(&d_prefixes, np * 4 * sizeof(uint64));
    cudaMemcpy(d_prefixes, h_prefixes, np * 4 * sizeof(uint64), cudaMemcpyHostToDevice);

    // Launch in batches for progress reporting + checkpoint support
    int BATCH_SIZE = (np < 10000) ? np : 10000;
    int num_batches = (np + BATCH_SIZE - 1) / BATCH_SIZE;

    printf("Launching %d GPU threads in %d batches (batch=%d)...\n", np, num_batches, BATCH_SIZE);
    fflush(stdout);

    int block = 256;
    struct timespec t_batch;
    double last_report = 0;

    // Build checkpoint filename from args
    char ckpt_path[512];
    snprintf(ckpt_path, 512, "scripts/experiments/zaremba-density/results/checkpoint_A%s_%llu.bin",
             argv[2], (unsigned long long)max_d);
    // Replace commas with nothing in checkpoint name
    for (char *c = ckpt_path; *c; c++) if (*c == ',') *c = '_';

    for (int batch = 0; batch < num_batches; batch++) {
        int offset = batch * BATCH_SIZE;
        int count = (batch + 1 == num_batches) ? (np - offset) : BATCH_SIZE;

        int grid = (count + block - 1) / block;
        enumerate_subtrees<<<grid, block>>>(d_prefixes + offset * 4, count, d_digits, num_digits, d_bs, max_d);
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &t_batch);
        double elapsed = (t_batch.tv_sec - t0.tv_sec) + (t_batch.tv_nsec - t0.tv_nsec) / 1e9;

        // Progress report every 60 seconds
        if (elapsed - last_report >= 60.0 || batch == num_batches - 1) {
            double pct = 100.0 * (batch + 1) / num_batches;
            double eta = (batch > 0) ? elapsed * (num_batches - batch - 1) / (batch + 1) : 0;
            printf("  batch %d/%d (%.0f%%) %.0fs elapsed, ETA %.0fs\n",
                   batch + 1, num_batches, pct, elapsed, eta);
            fflush(stdout);
            last_report = elapsed;
        }

        // Checkpoint bitset every 10% of batches
        if ((batch + 1) % (num_batches / 10 + 1) == 0 || batch == num_batches - 1) {
            // Save bitset to disk so partial results survive if killed
            uint8_t *h_ckpt = (uint8_t*)malloc(bitset_bytes);
            cudaMemcpy(h_ckpt, d_bs, bitset_bytes, cudaMemcpyDeviceToHost);
            FILE *fp = fopen(ckpt_path, "wb");
            if (fp) {
                fwrite(&max_d, sizeof(uint64), 1, fp);
                fwrite(&batch, sizeof(int), 1, fp);
                fwrite(&num_batches, sizeof(int), 1, fp);
                fwrite(h_ckpt, 1, bitset_bytes, fp);
                fclose(fp);
            }
            free(h_ckpt);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double enum_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("GPU enumeration: %.1fs\n", enum_time);

    // Clean up checkpoint after successful completion
    remove(ckpt_path);

    // Mark all denominators at depth < PREFIX_DEPTH on CPU (these are few)
    uint8_t *h_bs = (uint8_t*)calloc(bitset_bytes, 1);
    cudaMemcpy(h_bs, d_bs, bitset_bytes, cudaMemcpyDeviceToHost);

    // CPU: mark all CFs at depth 1 through PREFIX_DEPTH-1 (bounded, tiny)
    h_bs[0] |= (1 << 1);  // d=1
    {
        struct { uint64 pp, p, qp, q; int dep; } cstk[200000];
        int csp = 0;
        for (int i = 0; i < num_digits; i++) {
            cstk[csp].pp = 0; cstk[csp].p = 1;
            cstk[csp].qp = 1; cstk[csp].q = h_digits[i];
            cstk[csp].dep = 1;
            csp++;
        }
        while (csp > 0) {
            csp--;
            uint64 q = cstk[csp].q;
            int dep = cstk[csp].dep;
            if (q > max_d) continue;
            h_bs[q>>3] |= (1 << (q&7));
            if (dep >= PREFIX_DEPTH) continue;  // GPU handles deeper
            uint64 pp = cstk[csp].pp, p = cstk[csp].p, qp = cstk[csp].qp;
            for (int i = 0; i < num_digits; i++) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d) continue;
                if (csp < 199999) {
                    cstk[csp].pp = p;
                    cstk[csp].p = (uint64)h_digits[i] * p + pp;
                    cstk[csp].qp = q;
                    cstk[csp].q = qn;
                    cstk[csp].dep = dep + 1;
                    csp++;
                }
            }
        }
    }

    // Count
    uint64 covered = 0;
    for (uint64 d = 1; d <= max_d; d++) {
        if (h_bs[d>>3] & (1 << (d&7))) covered++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    uint64 uncovered = max_d - covered;

    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("Digit set: {");
    for (int i = 0; i < num_digits; i++) printf("%s%d", i?",":"", h_digits[i]);
    printf("}\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Covered: %llu / %llu\n", (unsigned long long)covered, (unsigned long long)max_d);
    printf("Density: %.10f%%\n", 100.0 * covered / max_d);
    printf("Uncovered: %llu\n", (unsigned long long)uncovered);

    if (uncovered > 0 && uncovered <= 100) {
        printf("Uncovered d:");
        for (uint64 d = 1; d <= max_d; d++) {
            if (!(h_bs[d>>3] & (1 << (d&7)))) printf(" %llu", (unsigned long long)d);
        }
        printf("\n");
    }

    printf("Time: %.1fs (enum: %.1fs)\n", total_time, enum_time);
    printf("========================================\n");

    free(h_prefixes); free(h_bs);
    cudaFree(d_bs); cudaFree(d_digits); cudaFree(d_prefixes);
    return 0;
}
