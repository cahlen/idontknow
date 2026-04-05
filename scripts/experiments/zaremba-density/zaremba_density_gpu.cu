/*
 * GPU-accelerated Zaremba density computation — overnight production version.
 *
 * Persistent-thread design with periodic disk checkpointing:
 *   1. CPU generates prefixes at fixed depth, sorts by q descending
 *   2. GPU persistent threads self-schedule via atomic counter
 *   3. Bitset checkpointed to disk every 5 minutes (survives kill)
 *   4. Shallow denominators marked on CPU after GPU enumeration
 *   5. Bit counting on GPU
 *
 * Compile: nvcc -O3 -arch=sm_90 -o zaremba_density_gpu zaremba_density_gpu.cu -lm
 * Run:     ./zaremba_density_gpu <max_d> <digits>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

typedef unsigned long long uint64;

#define MAX_DIGITS 10
#define MAX_DEPTH 200

__device__ void mark(uint64 d, uint8_t *bitset, uint64 max_d) {
    if (d < 1 || d > max_d) return;
    uint64 byte = d >> 3;
    uint8_t bit = 1 << (d & 7);
    atomicOr((unsigned int*)&bitset[byte & ~3], (unsigned int)bit << (8 * (byte & 3)));
}

__global__ void enumerate_persistent(
    uint64 *prefixes, int num_prefixes,
    int *digits, int num_digits,
    uint8_t *bitset, uint64 max_d,
    int *progress)
{
    struct { uint64 p_prev, p, q_prev, q; } stack[MAX_DEPTH];

    while (true) {
        int my_prefix = atomicAdd(progress, 1);
        if (my_prefix >= num_prefixes) return;

        uint64 pp0 = prefixes[my_prefix * 4 + 0];
        uint64 p0  = prefixes[my_prefix * 4 + 1];
        uint64 qp0 = prefixes[my_prefix * 4 + 2];
        uint64 q0  = prefixes[my_prefix * 4 + 3];

        mark(q0, bitset, max_d);

        int sp = 0;
        for (int i = num_digits - 1; i >= 0; i--) {
            uint64 a = digits[i];
            uint64 q_new = a * q0 + qp0;
            if (q_new > max_d || sp >= MAX_DEPTH) continue;
            stack[sp].p_prev = p0; stack[sp].p = a * p0 + pp0;
            stack[sp].q_prev = q0; stack[sp].q = q_new;
            sp++;
        }

        while (sp > 0) {
            sp--;
            uint64 pp = stack[sp].p_prev, p = stack[sp].p;
            uint64 qp = stack[sp].q_prev, q = stack[sp].q;
            mark(q, bitset, max_d);
            for (int i = num_digits - 1; i >= 0; i--) {
                uint64 a = digits[i];
                uint64 q_new = a * q + qp;
                if (q_new > max_d || sp >= MAX_DEPTH) continue;
                stack[sp].p_prev = p; stack[sp].p = a * p + pp;
                stack[sp].q_prev = q; stack[sp].q = q_new;
                sp++;
            }
        }
    }
}

__global__ void count_marked(uint8_t *bitset, uint64 max_d, uint64 *count) {
    uint64 tid = blockIdx.x * (uint64)blockDim.x + threadIdx.x;
    uint64 max_byte = (max_d + 8) / 8;
    if (tid >= max_byte) return;
    uint8_t b = bitset[tid];
    int bits = __popc((unsigned int)b);
    if (tid == max_byte - 1) {
        int valid_bits = (max_d % 8) + 1;
        bits = __popc((unsigned int)(b & ((1 << valid_bits) - 1)));
    }
    if (bits > 0) atomicAdd(count, (uint64)bits);
}

int cmp_by_q_desc(const void *a, const void *b) {
    uint64 qa = ((const uint64*)a)[3], qb = ((const uint64*)b)[3];
    return (qa > qb) ? -1 : (qa < qb) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <max_d> <digits>\n", argv[0]);
        return 1;
    }

    uint64 max_d = (uint64)atoll(argv[1]);

    int h_digits[MAX_DIGITS];
    int num_digits = 0;
    char buf[256]; strncpy(buf, argv[2], 255);
    char *tok = strtok(buf, ",");
    while (tok && num_digits < MAX_DIGITS) {
        h_digits[num_digits++] = atoi(tok);
        tok = strtok(NULL, ",");
    }

    printf("========================================\n");
    printf("Zaremba Density (GPU) — production\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Digits: {");
    for (int i = 0; i < num_digits; i++) printf("%s%d", i?",":"", h_digits[i]);
    printf("}\n");
    printf("========================================\n\n");
    fflush(stdout);

    // Prefix generation — fixed depth, sorted by q descending
    int PREFIX_DEPTH = 8;
    if (max_d >= 1000000000ULL)   PREFIX_DEPTH = 15;
    if (max_d >= 10000000000ULL)  PREFIX_DEPTH = 15;

    int max_prefixes = 20000000;
    uint64 *h_prefixes = (uint64*)malloc((uint64)max_prefixes * 4 * sizeof(uint64));
    int np = 0;

    printf("Generating prefixes (depth=%d)...\n", PREFIX_DEPTH);
    fflush(stdout);

    struct PfxEntry { uint64 pp, p, qp, q; int depth; };
    struct PfxEntry *stk = (struct PfxEntry*)malloc(20000000 * sizeof(struct PfxEntry));
    int ssp = 0;
    for (int i = 0; i < num_digits; i++) {
        stk[ssp].pp = 0; stk[ssp].p = 1;
        stk[ssp].qp = 1; stk[ssp].q = h_digits[i];
        stk[ssp].depth = 1; ssp++;
    }
    while (ssp > 0) {
        ssp--;
        uint64 pp = stk[ssp].pp, p = stk[ssp].p;
        uint64 qp = stk[ssp].qp, q = stk[ssp].q;
        int dep = stk[ssp].depth;
        if (q > max_d) continue;
        if (dep >= PREFIX_DEPTH) {
            if (np < max_prefixes) {
                h_prefixes[np*4+0] = pp; h_prefixes[np*4+1] = p;
                h_prefixes[np*4+2] = qp; h_prefixes[np*4+3] = q;
                np++;
            }
        } else {
            for (int i = num_digits - 1; i >= 0; i--) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d || ssp >= 19999999) continue;
                stk[ssp].pp = p; stk[ssp].p = (uint64)h_digits[i] * p + pp;
                stk[ssp].qp = q; stk[ssp].q = qn;
                stk[ssp].depth = dep + 1; ssp++;
            }
        }
    }
    free(stk);

    printf("Prefixes: %d. Sorting...\n", np);
    fflush(stdout);
    qsort(h_prefixes, np, 4 * sizeof(uint64), cmp_by_q_desc);

    printf("Bitset: %.2f GB\n\n", (max_d + 8) / 8.0 / 1e9);
    fflush(stdout);

    struct timespec t0, t1, t_check;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // GPU alloc
    uint64 bitset_bytes = (max_d + 8) / 8;
    uint8_t *d_bs;
    cudaError_t err = cudaMalloc(&d_bs, bitset_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc bitset (%.2f GB): %s\n",
                bitset_bytes / 1e9, cudaGetErrorString(err));
        return 1;
    }
    cudaMemset(d_bs, 0, bitset_bytes);

    int *d_digits;
    cudaMalloc(&d_digits, num_digits * sizeof(int));
    cudaMemcpy(d_digits, h_digits, num_digits * sizeof(int), cudaMemcpyHostToDevice);

    uint64 *d_prefixes;
    cudaMalloc(&d_prefixes, (uint64)np * 4 * sizeof(uint64));
    cudaMemcpy(d_prefixes, h_prefixes, (uint64)np * 4 * sizeof(uint64), cudaMemcpyHostToDevice);

    // Mapped progress counter
    int *h_progress_mapped, *d_progress;
    cudaHostAlloc(&h_progress_mapped, sizeof(int), cudaHostAllocMapped);
    *h_progress_mapped = 0;
    cudaHostGetDevicePointer(&d_progress, h_progress_mapped, 0);

    // Launch config
    int num_SMs, max_thr_per_SM;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&max_thr_per_SM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    int block_size = 256;
    int use_SMs = num_SMs - 2;
    if (use_SMs < 1) use_SMs = 1;
    int total_threads = use_SMs * max_thr_per_SM;
    if (total_threads > np) total_threads = np;
    int grid_size = (total_threads + block_size - 1) / block_size;

    // Checkpoint path
    char ckpt_path[512];
    snprintf(ckpt_path, 512, "scripts/experiments/zaremba-density/results/checkpoint_A%s_%llu.bin",
             argv[2], (unsigned long long)max_d);
    for (char *c = ckpt_path; *c; c++) if (*c == ',') *c = '_';

    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream);

    printf("Launching %d persistent threads on %d/%d SMs (%d prefixes)...\n",
           grid_size * block_size, use_SMs, num_SMs, np);
    fflush(stdout);

    enumerate_persistent<<<grid_size, block_size, 0, kernel_stream>>>(
        d_prefixes, np, d_digits, num_digits, d_bs, max_d, d_progress);

    // Poll progress + checkpoint
    double last_report = 0;
    int last_progress_val = 0;
    int last_ckpt_min = 0;
    while (true) {
        __sync_synchronize();
        int h_progress = *h_progress_mapped;
        if (h_progress >= np) break;

        clock_gettime(CLOCK_MONOTONIC, &t_check);
        double elapsed = (t_check.tv_sec - t0.tv_sec) + (t_check.tv_nsec - t0.tv_nsec) / 1e9;

        if (elapsed - last_report >= 30.0) {
            double pct = 100.0 * h_progress / np;
            double rate = (elapsed > last_report) ?
                (h_progress - last_progress_val) / (elapsed - last_report) : 0;
            double eta = (rate > 0) ? (np - h_progress) / rate : 0;
            printf("  [%6.0fs] %d/%d (%.1f%%) %.0f pfx/s ETA %.0fs\n",
                   elapsed, h_progress, np, pct, rate, eta);
            fflush(stdout);
            last_report = elapsed;
            last_progress_val = h_progress;
        }

        // Checkpoint every 5 minutes
        int curr_min = (int)(elapsed / 300);
        if (curr_min > last_ckpt_min && elapsed > 60) {
            last_ckpt_min = curr_min;
            // Download bitset from GPU (non-blocking on default stream while kernel runs on kernel_stream)
            uint8_t *h_ckpt = (uint8_t*)malloc(bitset_bytes);
            if (h_ckpt) {
                cudaMemcpy(h_ckpt, d_bs, bitset_bytes, cudaMemcpyDeviceToHost);
                FILE *fp = fopen(ckpt_path, "wb");
                if (fp) {
                    fwrite(&max_d, sizeof(uint64), 1, fp);
                    fwrite(&h_progress, sizeof(int), 1, fp);
                    fwrite(&np, sizeof(int), 1, fp);
                    fwrite(h_ckpt, 1, bitset_bytes, fp);
                    fclose(fp);
                    printf("  [checkpoint saved: %d/%d prefixes, %.1f GB]\n",
                           h_progress, np, bitset_bytes / 1e9);
                    fflush(stdout);
                }
                free(h_ckpt);
            }
        }

        usleep(2000000);
    }

    cudaStreamSynchronize(kernel_stream);
    cudaStreamDestroy(kernel_stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double enum_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("GPU enumeration: %.1fs\n", enum_time);
    fflush(stdout);

    remove(ckpt_path);

    // Mark shallow denominators on CPU
    uint8_t *h_bs = (uint8_t*)malloc(bitset_bytes);
    cudaMemcpy(h_bs, d_bs, bitset_bytes, cudaMemcpyDeviceToHost);
    h_bs[0] |= (1 << 1);  // d=1
    {
        struct ShallowEntry { uint64 pp, p, qp, q; int dep; };
        struct ShallowEntry *cstk = (struct ShallowEntry*)malloc(500000 * sizeof(struct ShallowEntry));
        int csp = 0;
        for (int i = 0; i < num_digits; i++) {
            cstk[csp].pp = 0; cstk[csp].p = 1;
            cstk[csp].qp = 1; cstk[csp].q = h_digits[i];
            cstk[csp].dep = 1; csp++;
        }
        while (csp > 0) {
            csp--;
            uint64 q = cstk[csp].q;
            int dep = cstk[csp].dep;
            if (q > max_d) continue;
            h_bs[q>>3] |= (1 << (q&7));
            if (dep >= PREFIX_DEPTH) continue;
            uint64 pp = cstk[csp].pp, p = cstk[csp].p, qp = cstk[csp].qp;
            for (int i = 0; i < num_digits; i++) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d || csp >= 499999) continue;
                cstk[csp].pp = p;
                cstk[csp].p = (uint64)h_digits[i] * p + pp;
                cstk[csp].qp = q; cstk[csp].q = qn;
                cstk[csp].dep = dep + 1; csp++;
            }
        }
        free(cstk);
    }
    cudaMemcpy(d_bs, h_bs, bitset_bytes, cudaMemcpyHostToDevice);

    // Count on GPU
    uint64 *d_count;
    cudaMalloc(&d_count, sizeof(uint64));
    cudaMemset(d_count, 0, sizeof(uint64));
    {
        uint64 max_byte = (max_d + 8) / 8;
        int gd = (max_byte + 255) / 256;
        count_marked<<<gd, 256>>>(d_bs, max_d, d_count);
        cudaDeviceSynchronize();
    }
    uint64 covered = 0;
    cudaMemcpy(&covered, d_count, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

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
        for (uint64 d = 1; d <= max_d; d++)
            if (!(h_bs[d>>3] & (1 << (d&7)))) printf(" %llu", (unsigned long long)d);
        printf("\n");
    }

    printf("Time: %.1fs (enum: %.1fs)\n", total_time, enum_time);
    printf("========================================\n");

    free(h_prefixes); free(h_bs);
    cudaFree(d_bs); cudaFree(d_digits); cudaFree(d_prefixes);
    cudaFreeHost(h_progress_mapped);
    return 0;
}
