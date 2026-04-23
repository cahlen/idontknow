/*
 * GPU Matrix Enumeration v6.1 — Certified, no-overflow version
 *
 * Hardens the v6 multi-pass kernel against the software-audit concerns raised
 * in external review:
 *
 *   1. The original v6 clipped the next frontier to BUF_SLOTS when a chunk
 *      generated more children than would fit in the output buffer. This was
 *      silent truncation: the dropped children were not expanded, so their
 *      descendant denominators would never be marked, and a false "Uncovered: 0"
 *      result was in principle possible.
 *
 *   2. v6.1 replaces the silent clip with a HARD ABORT. The GPU kernel writes
 *      a per-round overflow flag that the host checks after every kernel
 *      launch. If any child would have been dropped, the program exits with
 *      error code 2 and an explicit message naming the GPU, round, depth,
 *      overflow count, and buffer capacity.
 *
 *   3. v6.1 also tracks, and prints to the log, the PEAK LIVE FRONTIER per
 *      (round, gpu, depth) level. This gives an independent post-hoc
 *      no-clipping certificate: if every printed peak is < BUF_SLOTS, the
 *      computation was buffer-safe even if the abort had never fired.
 *
 *   4. v6.1 fixes a cosmetic depth-labeling bug in v6: the depth-print loop
 *      used (depth+1) as a label while the expansion index was `depth`, so
 *      "depth 12" in the old log was actually the output of the 12th
 *      expansion (i.e. level 12 from the 5 seeds at level 1). v6.1 prints
 *      levels unambiguously as "level N (after N expansions)" and reconciles
 *      phase_a_depth (the exclusive upper bound of the expansion loop) with
 *      the level labels in the comments and the printed output.
 *
 * The frontier-max certificate is reported both per-round and globally at
 * the end of the run, so the run log itself becomes machine-checkable:
 *
 *   PEAK FRONTIER CERTIFICATE
 *   phase A max: <x>   (buffer BUF_SLOTS = ...)
 *   phase B max: <y>   (buffer BUF_SLOTS = ...)
 *   all peaks < BUF_SLOTS: YES/NO
 *   no-overflow abort triggered: YES/NO
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120a -o matrix_v6_1 \
 *       scripts/experiments/zaremba-effective-bound/matrix_enum_multipass_v6_1.cu \
 *       -lpthread
 *   (use -arch=sm_100a for B200, -arch=sm_120a for RTX 5090, -arch=sm_89 for 4090)
 *
 * Run:
 *   ./matrix_v6_1 <max_d>
 *
 * Return code:
 *   0  Uncovered == 0 AND no-overflow abort never triggered
 *   1  Uncovered > 0 (genuine counterexample, shouldn't happen for known ranges)
 *   2  Buffer overflow detected — result cannot be trusted, rerun with more rounds
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
#define MAX_DEPTH 62

/*
 * Canonical buffer size for the B200 210B run: 2B slots * 4 u64 = 64 GB per
 * buffer, so 128 GB of per-GPU VRAM used by the two ping-pong buffers.
 * On smaller GPUs (e.g. RTX 5090 with 32 GB), you must build with a smaller
 * BUF_SLOTS; the no-overflow certificate still works, just the safety ratio
 * changes. The kernel REFUSES to silently clip regardless of the buffer size.
 */
#ifndef BUF_SLOTS
#define BUF_SLOTS 2000000000ULL
#endif

typedef unsigned long long uint64;
typedef unsigned int uint32;

/*
 * expand_mark_compact_safe
 *
 * Unlike v6, this kernel *refuses* to increment out_count beyond max_out.
 * Every generated child increments either:
 *   - out_count            (if pos < max_out, the child is actually written)
 *   - overflow_count       (if pos >= max_out, the child is NOT written, and
 *                           the host will treat this as a fatal condition)
 *
 * The bitset mark always fires (the denominator n10 is covered regardless of
 * whether we save the matrix for further expansion). But any child we fail
 * to propagate is a silent truncation risk, so overflow_count > 0 must kill
 * the run.
 */
__global__ void expand_mark_compact_safe(
    uint64 *in, uint64 num_in,
    uint64 *out, unsigned long long *out_count,
    unsigned long long *overflow_count,
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

        atomicOr(&bitset[n10 / 32], 1u << (n10 % 32));
        atomicAdd(marks, 1);

        unsigned long long pos = atomicAdd(out_count, 1ULL);
        if (pos < max_out) {
            out[pos*4]   = n00; out[pos*4+1] = m00;
            out[pos*4+2] = n10; out[pos*4+3] = m10;
        } else {
            atomicAdd(overflow_count, 1ULL);
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
    uint64 *chunk_data;
    uint64 chunk_size;
    uint32 *d_bitset;
    uint64 max_d;
    uint64 bitset_words;
    double elapsed;
    int probe_mode;

    unsigned long long chunk_overflow;
    unsigned long long chunk_peak_frontier;
    int overflow_depth;
} ChunkArgs;

void *process_chunk(void *arg) {
    ChunkArgs *c = (ChunkArgs*)arg;
    cudaSetDevice(c->gpu_id);

    uint64 *d_buf_a, *d_buf_b;
    cudaMalloc(&d_buf_a, BUF_SLOTS * 4 * sizeof(uint64));
    cudaMalloc(&d_buf_b, BUF_SLOTS * 4 * sizeof(uint64));
    unsigned long long *d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned long long));
    unsigned long long *d_overflow;
    cudaMalloc(&d_overflow, sizeof(unsigned long long));
    cudaMemset(d_overflow, 0, sizeof(unsigned long long));
    uint32 *d_marks;
    cudaMalloc(&d_marks, sizeof(uint32));
    cudaMemset(d_marks, 0, sizeof(uint32));

    cudaMemcpy(d_buf_a, c->chunk_data, c->chunk_size * 4 * sizeof(uint64), cudaMemcpyHostToDevice);
    uint64 num = c->chunk_size;

    c->chunk_overflow = 0;
    c->chunk_peak_frontier = num;
    c->overflow_depth = -1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int depth = 0; depth < MAX_DEPTH && num > 0; depth++) {
        cudaMemset(d_out_count, 0, sizeof(unsigned long long));
        int blocks = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_mark_compact_safe<<<blocks, BLOCK_SIZE>>>(
            d_buf_a, num, d_buf_b, d_out_count, d_overflow,
            c->d_bitset, c->max_d, d_marks, BUF_SLOTS);
        cudaDeviceSynchronize();

        unsigned long long h_out, h_over;
        cudaMemcpy(&h_out,  d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_over, d_overflow,  sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        if (h_over > c->chunk_overflow) {
            c->chunk_overflow = h_over;
            c->overflow_depth = depth;
        }
        if (h_out > c->chunk_peak_frontier) {
            c->chunk_peak_frontier = h_out;
        }

        if (h_over > 0 && !c->probe_mode) {
            /* Fail-fast: do not continue past a clipped round. */
            break;
        }

        uint64 *tmp = d_buf_a; d_buf_a = d_buf_b; d_buf_b = tmp;
        num = h_out < BUF_SLOTS ? h_out : BUF_SLOTS;  /* only clips in probe mode */
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    c->elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaFree(d_out_count); cudaFree(d_overflow); cudaFree(d_marks);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <max_d>\n", argv[0]);
        return 1;
    }

    uint64 max_d = (uint64)atoll(argv[1]);
    const char *probe_env = getenv("ZAREMBA_PROBE");
    int probe_mode = (probe_env && atoi(probe_env) > 0) ? 1 : 0;

    printf("Zaremba v6.1 Certified Multi-Pass Verification\n");
    printf("Max d: %llu\n", (unsigned long long)max_d);
    printf("BUF_SLOTS (per-GPU frontier cap): %llu\n", (unsigned long long)BUF_SLOTS);
    if (probe_mode) {
        printf("ZAREMBA_PROBE=1: peak-frontier PROBE mode. Overflow will NOT abort;\n");
        printf("                Uncovered count is INVALID in this mode (matrices dropped).\n");
        printf("                Use for scaling studies only.\n");
    }
    printf("\n");

    int ngpus;
    cudaGetDeviceCount(&ngpus);
    printf("GPUs: %d\n\n", ngpus);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /*
     * Phase A: we expand from the 5 depth-1 seeds, and the loop runs for
     * (phase_a_depth - 1) expansions. That produces matrices of length
     * phase_a_depth (i.e. 12 partial quotients).
     */
    int phase_a_depth = 12;
    printf("=== Phase A: build seed frontier to level %d (%d expansions from 5 depth-1 seeds) ===\n",
           phase_a_depth, phase_a_depth - 1);
    cudaSetDevice(0);

    uint64 bitset_words = (max_d + 32) / 32;
    uint32 *d_bitset;
    cudaMalloc(&d_bitset, bitset_words * sizeof(uint32));
    cudaMemset(d_bitset, 0, bitset_words * sizeof(uint32));

    uint32 bit1 = 1u << 1;
    cudaMemcpy(d_bitset, &bit1, sizeof(uint32), cudaMemcpyHostToDevice);

    uint64 *d_buf_a, *d_buf_b;
    cudaMalloc(&d_buf_a, BUF_SLOTS * 4 * sizeof(uint64));
    cudaMalloc(&d_buf_b, BUF_SLOTS * 4 * sizeof(uint64));
    unsigned long long *d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned long long));
    unsigned long long *d_overflow;
    cudaMalloc(&d_overflow, sizeof(unsigned long long));
    cudaMemset(d_overflow, 0, sizeof(unsigned long long));
    uint32 *d_marks;
    cudaMalloc(&d_marks, sizeof(uint32));
    cudaMemset(d_marks, 0, sizeof(uint32));

    uint64 h_init[5*4];
    for (int a = 1; a <= BOUND; a++) {
        h_init[(a-1)*4]   = a; h_init[(a-1)*4+1] = 1;
        h_init[(a-1)*4+2] = 1; h_init[(a-1)*4+3] = 0;
    }
    cudaMemcpy(d_buf_a, h_init, 5*4*sizeof(uint64), cudaMemcpyHostToDevice);
    uint64 num = 5;

    unsigned long long phase_a_peak = num;
    unsigned long long phase_a_overflow = 0;

    /*
     * After level 1 we already have 5 seeds. Each iteration of this loop
     * expands one level, producing levels 2, 3, ..., phase_a_depth.
     */
    for (int level = 2; level <= phase_a_depth; level++) {
        cudaMemset(d_out_count, 0, sizeof(unsigned long long));
        int blocks = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_mark_compact_safe<<<blocks, BLOCK_SIZE>>>(
            d_buf_a, num, d_buf_b, d_out_count, d_overflow,
            d_bitset, max_d, d_marks, BUF_SLOTS);
        cudaDeviceSynchronize();

        unsigned long long h_out, h_over;
        cudaMemcpy(&h_out,  d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_over, d_overflow,  sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        if (h_out > phase_a_peak) phase_a_peak = h_out;
        phase_a_overflow += h_over;

        if (h_over > 0 && !probe_mode) {
            fprintf(stderr,
                "FATAL: buffer overflow in Phase A at level %d. "
                "Overflow count: %llu (BUF_SLOTS = %llu). "
                "Rerun with a larger buffer or smaller phase_a_depth.\n",
                level, (unsigned long long)h_over, (unsigned long long)BUF_SLOTS);
            return 2;
        }

        uint64 *tmp = d_buf_a; d_buf_a = d_buf_b; d_buf_b = tmp;
        num = h_out < BUF_SLOTS ? h_out : BUF_SLOTS;  /* only clips in probe mode */

        printf("  level %2d: %llu attempted (kept %llu, overflow %llu, peak so far %llu)\n",
               level, (unsigned long long)h_out, (unsigned long long)num,
               (unsigned long long)h_over, (unsigned long long)phase_a_peak);
    }

    printf("\n  Downloading %llu level-%d matrices...\n",
           (unsigned long long)num, phase_a_depth);
    uint64 *h_matrices = (uint64*)malloc(num * 4 * sizeof(uint64));
    cudaMemcpy(h_matrices, d_buf_a, num * 4 * sizeof(uint64), cudaMemcpyDeviceToHost);
    uint64 total_seed = num;

    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaFree(d_out_count); cudaFree(d_overflow); cudaFree(d_marks);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  Phase A done: %.1fs. Phase A peak frontier: %llu / %llu (safety ratio %.3f).\n\n",
           (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9,
           (unsigned long long)phase_a_peak, (unsigned long long)BUF_SLOTS,
           (double)phase_a_peak / (double)BUF_SLOTS);

    /*
     * Phase B: distribute level-12 seeds into chunks, expand each chunk to
     * depth 62 on its assigned GPU. Rounds are sized so that per-chunk
     * peak frontier stays under BUF_SLOTS; if not, the kernel aborts.
     */
    printf("=== Phase B: expand level %d -> depth %d in chunks ===\n",
           phase_a_depth, MAX_DEPTH);

    uint32 *h_bitset = (uint32*)malloc(bitset_words * sizeof(uint32));
    cudaSetDevice(0);
    cudaMemcpy(h_bitset, d_bitset, bitset_words * sizeof(uint32), cudaMemcpyDeviceToHost);

    uint32 *gpu_bitsets[8];
    for (int g = 0; g < ngpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_bitsets[g], bitset_words * sizeof(uint32));
        cudaMemcpy(gpu_bitsets[g], h_bitset, bitset_words * sizeof(uint32), cudaMemcpyHostToDevice);
    }

    /*
     * Rounds schedule. Empirically we observe Phase B peak frontier ~ 5x
     * the seed count for max_d in [1e9, 3e11]. With BUF_SLOTS = 2e9 this
     * means chunk_per_GPU must be kept below ~4e8 to be buffer-safe. For
     * BUF_SLOTS other than 2e9, scale num_rounds proportionally so the
     * no-overflow abort never fires. The override via $ZAREMBA_ROUNDS is
     * provided for reproducing exact historical B200 runs.
     */
    int num_rounds;
    const char *rounds_env = getenv("ZAREMBA_ROUNDS");
    if (rounds_env && atoi(rounds_env) > 0) {
        num_rounds = atoi(rounds_env);
        printf("  ZAREMBA_ROUNDS override: %d\n", num_rounds);
    } else {
        double scale = 2000000000.0 / (double)BUF_SLOTS;
        if      (max_d <=   1000000000ULL) num_rounds = (int)(1  * scale);
        else if (max_d <=  10000000000ULL) num_rounds = (int)(8  * scale);
        else if (max_d <= 100000000000ULL) num_rounds = (int)(64 * scale);
        else                               num_rounds = (int)(256 * scale);
        if (num_rounds < 1) num_rounds = 1;
    }
    uint64 round_chunk = (total_seed + (ngpus * num_rounds) - 1) / (ngpus * num_rounds);
    printf("  Total seeds: %llu, rounds: %d, chunk per GPU: %llu, GPUs: %d\n\n",
           (unsigned long long)total_seed, num_rounds,
           (unsigned long long)round_chunk, ngpus);

    unsigned long long phase_b_peak = 0;
    unsigned long long phase_b_overflow = 0;

    for (int round = 0; round < num_rounds; round++) {
        printf("  Round %d/%d:\n", round+1, num_rounds);
        ChunkArgs args[8];
        pthread_t threads[8];
        for (int g = 0; g < ngpus; g++) {
            uint64 slot = round * ngpus + g;
            uint64 start = slot * round_chunk;
            uint64 end = start + round_chunk;
            if (end > total_seed) end = total_seed;
            if (start >= total_seed) { args[g].chunk_size = 0; continue; }

            args[g].gpu_id = g;
            args[g].chunk_data = h_matrices + start * 4;
            args[g].chunk_size = end - start;
            args[g].d_bitset = gpu_bitsets[g];
            args[g].max_d = max_d;
            args[g].bitset_words = bitset_words;
            args[g].probe_mode = probe_mode;

            printf("    GPU %d: %llu matrices\n", g, (unsigned long long)args[g].chunk_size);
            pthread_create(&threads[g], NULL, process_chunk, &args[g]);
        }

        for (int g = 0; g < ngpus; g++) {
            if (args[g].chunk_size > 0) {
                pthread_join(threads[g], NULL);
                printf("    GPU %d done: %.1fs (peak frontier %llu, overflow %llu)\n",
                       g, args[g].elapsed,
                       (unsigned long long)args[g].chunk_peak_frontier,
                       (unsigned long long)args[g].chunk_overflow);
                if (args[g].chunk_peak_frontier > phase_b_peak)
                    phase_b_peak = args[g].chunk_peak_frontier;
                phase_b_overflow += args[g].chunk_overflow;

                if (args[g].chunk_overflow > 0 && !probe_mode) {
                    fprintf(stderr,
                        "FATAL: buffer overflow in Phase B, round %d, GPU %d, depth %d.\n"
                        "  Overflow count: %llu (BUF_SLOTS = %llu).\n"
                        "  Peak frontier this chunk: %llu\n"
                        "  Rerun with more rounds (current: %d).\n",
                        round+1, g, args[g].overflow_depth,
                        (unsigned long long)args[g].chunk_overflow,
                        (unsigned long long)BUF_SLOTS,
                        (unsigned long long)args[g].chunk_peak_frontier,
                        num_rounds);
                    return 2;
                }
            }
        }
    }

    printf("\n  Merging bitsets...\n");
    for (int g = 0; g < ngpus; g++) {
        uint32 *tmp = (uint32*)malloc(bitset_words * sizeof(uint32));
        cudaSetDevice(g);
        cudaMemcpy(tmp, gpu_bitsets[g], bitset_words * sizeof(uint32), cudaMemcpyDeviceToHost);
        for (uint64 i = 0; i < bitset_words; i++) h_bitset[i] |= tmp[i];
        free(tmp);
        cudaFree(gpu_bitsets[g]);
    }

    uint64 uncovered = 0;
    for (uint64 d = 1; d <= max_d; d++) {
        if (!(h_bitset[d/32] & (1u << (d%32)))) uncovered++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("Zaremba v6.1: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Uncovered: %llu\n", (unsigned long long)uncovered);
    printf("Time: %.1fs\n", total);

    printf("\n--- NO-OVERFLOW CERTIFICATE ---\n");
    printf("BUF_SLOTS                 : %llu\n", (unsigned long long)BUF_SLOTS);
    printf("Phase A peak frontier     : %llu  (%.4f of BUF_SLOTS)\n",
           (unsigned long long)phase_a_peak, (double)phase_a_peak / (double)BUF_SLOTS);
    printf("Phase A overflow events   : %llu\n", (unsigned long long)phase_a_overflow);
    printf("Phase B peak frontier     : %llu  (%.4f of BUF_SLOTS)\n",
           (unsigned long long)phase_b_peak, (double)phase_b_peak / (double)BUF_SLOTS);
    printf("Phase B overflow events   : %llu\n", (unsigned long long)phase_b_overflow);
    printf("All peaks < BUF_SLOTS     : %s\n",
           (phase_a_peak < BUF_SLOTS && phase_b_peak < BUF_SLOTS) ? "YES" : "NO");
    printf("No-overflow abort fired   : %s\n",
           (phase_a_overflow == 0 && phase_b_overflow == 0) ? "NO" : "YES");

    if (probe_mode) {
        printf("\nRESULT: PROBE MODE — uncovered count is INVALID (silently clipped).\n");
        printf("        Use the peak-frontier data above to choose BUF_SLOTS/num_rounds\n");
        printf("        for a real certified run.\n");
    } else if (uncovered == 0 && phase_a_overflow == 0 && phase_b_overflow == 0) {
        printf("\nRESULT: ALL d in [1, %llu] are Zaremba denominators (A=5).\n",
               (unsigned long long)max_d);
        printf("        Run is buffer-safe (no frontier ever reached BUF_SLOTS).\n");
    } else if (phase_a_overflow > 0 || phase_b_overflow > 0) {
        printf("\nRESULT: COMPUTATION INVALIDATED by buffer overflow. Rerun with more rounds.\n");
    } else {
        printf("\nRESULT: %llu uncovered denominators remain (genuine miss).\n",
               (unsigned long long)uncovered);
    }
    printf("========================================\n");

    free(h_matrices); free(h_bitset);
    cudaSetDevice(0); cudaFree(d_bitset);

    if (phase_a_overflow > 0 || phase_b_overflow > 0) return 2;
    return uncovered > 0 ? 1 : 0;
}
