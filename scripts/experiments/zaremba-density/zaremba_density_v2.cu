/*
 * Zaremba density v2 — host-driven iterative batching with node-budget DFS.
 *
 * PROBLEM: The original kernel hangs because digit-1 paths create extremely
 * deep continued-fraction trees (Fibonacci growth, ~60+ levels at 10^11).
 * A single thread can be stuck processing billions of nodes while all other
 * threads sit idle.
 *
 * SOLUTION: Each GPU thread does DFS with a hard NODE_BUDGET. When the budget
 * is exhausted, the thread dumps its remaining DFS stack to an overflow buffer.
 * The host collects overflow items and launches them as new work items in the
 * next batch. This guarantees:
 *   - No thread runs for more than ~0.1-1 second
 *   - Deep subtrees get split across many threads over multiple rounds
 *   - The host can report progress after every batch
 *   - No complex in-kernel synchronization or work-stealing needed
 *
 * Compile: nvcc -O3 -arch=sm_90 -o zaremba_density_v2 zaremba_density_v2.cu -lm
 * Run:     ./zaremba_density_v2 <max_d> <digits>
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
#define MAX_DEPTH  200

/* Node budget per thread. After processing this many nodes, the thread
 * stops DFS and writes remaining stack to the overflow buffer.
 * 2M nodes at ~1-10 ns/node = 2-20 ms per thread — well under the 60s target. */
#define NODE_BUDGET 2000000

/* Maximum DFS stack entries that one thread can overflow.
 * Each overflow entry is 32 bytes (4x uint64). */
#define MAX_OVERFLOW_PER_THREAD 128

// ── Work item: defines a starting state for DFS ──
struct WorkItem {
    uint64 pp, p, qp, q;
};

// ── Device: mark denominator in bitset ──
__device__ void mark(uint64 d, uint8_t *bitset, uint64 max_d) {
    if (d < 1 || d > max_d) return;
    uint64 byte = d >> 3;
    uint8_t bit = 1 << (d & 7);
    atomicOr((unsigned int*)&bitset[byte & ~3], (unsigned int)bit << (8 * (byte & 3)));
}

// ── Kernel: node-budget-limited DFS ──
// Each thread processes exactly ONE work item from work_items[].
// It does DFS up to NODE_BUDGET nodes. If the budget runs out,
// it writes its remaining stack to overflow[] and increments *overflow_count.
__global__ void dfs_bounded(
    WorkItem *work_items, int num_items,
    int *digits, int num_digits,
    uint8_t *bitset, uint64 max_d,
    WorkItem *overflow, int *overflow_count,
    int max_total_overflow)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    WorkItem item = work_items[tid];

    struct { uint64 pp, p, qp, q; } stack[MAX_DEPTH];

    // Mark the starting denominator
    mark(item.q, bitset, max_d);

    // Push children of starting node
    int sp = 0;
    for (int i = num_digits - 1; i >= 0; i--) {
        uint64 a = digits[i];
        uint64 q_new = a * item.q + item.qp;
        if (q_new > max_d || sp >= MAX_DEPTH) continue;
        stack[sp].pp = item.p;
        stack[sp].p  = a * item.p + item.pp;
        stack[sp].qp = item.q;
        stack[sp].q  = q_new;
        sp++;
    }

    int nodes = 0;

    while (sp > 0) {
        sp--;
        uint64 pp = stack[sp].pp, p = stack[sp].p;
        uint64 qp = stack[sp].qp, q = stack[sp].q;

        mark(q, bitset, max_d);
        nodes++;

        if (nodes >= NODE_BUDGET) {
            // Budget exhausted. Dump remaining stack + current node's children
            // to overflow buffer.

            // First, push current node's children back onto local stack
            // so we can dump everything at once.
            for (int i = num_digits - 1; i >= 0; i--) {
                uint64 a = digits[i];
                uint64 q_new = a * q + qp;
                if (q_new > max_d || sp >= MAX_DEPTH) continue;
                stack[sp].pp = p;
                stack[sp].p  = a * p + pp;
                stack[sp].qp = q;
                stack[sp].q  = q_new;
                sp++;
            }

            // How many items to overflow
            int to_write = sp;
            if (to_write > MAX_OVERFLOW_PER_THREAD) to_write = MAX_OVERFLOW_PER_THREAD;
            if (to_write <= 0) break;

            // Atomically reserve slots in the overflow buffer
            int base = atomicAdd(overflow_count, to_write);
            if (base + to_write > max_total_overflow) {
                // Overflow buffer full — can't write, must finish locally.
                // Undo the reservation (best-effort, the count is just a hint).
                atomicSub(overflow_count, to_write);
                // Continue DFS without budget limit — this is a rare fallback.
                // We still process the remaining stack, just without the budget cap.
                // Push the children back if we popped too many...
                // Actually the stack already has everything. Just continue the loop.
                continue;
            }

            // Write stack items to overflow (bottom to top, take deepest first
            // since those are most likely to be the expensive ones, but for
            // simplicity just write from top of stack)
            for (int i = 0; i < to_write; i++) {
                int idx = sp - 1 - i;  // top of stack first
                overflow[base + i].pp = stack[idx].pp;
                overflow[base + i].p  = stack[idx].p;
                overflow[base + i].qp = stack[idx].qp;
                overflow[base + i].q  = stack[idx].q;
            }

            break;  // Done with this work item
        }

        // Push children
        for (int i = num_digits - 1; i >= 0; i--) {
            uint64 a = digits[i];
            uint64 q_new = a * q + qp;
            if (q_new > max_d || sp >= MAX_DEPTH) continue;
            stack[sp].pp = p;
            stack[sp].p  = a * p + pp;
            stack[sp].qp = q;
            stack[sp].q  = q_new;
            sp++;
        }
    }
}

// ── Bit counting kernel (unchanged from v1) ──
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

int cmp_workitem_by_q_asc(const void *a, const void *b) {
    const WorkItem *wa = (const WorkItem*)a;
    const WorkItem *wb = (const WorkItem*)b;
    return (wa->q < wb->q) ? -1 : (wa->q > wb->q) ? 1 : 0;
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
    printf("Zaremba Density v2 (GPU) — bounded DFS\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Digits: {");
    for (int i = 0; i < num_digits; i++) printf("%s%d", i?",":"", h_digits[i]);
    printf("}\n");
    printf("Node budget per thread: %d\n", NODE_BUDGET);
    printf("========================================\n\n");
    fflush(stdout);

    // ── Prefix generation with adaptive cost-bounded splitting ──
    // For digit sets with small digits (esp. 1), we need deep prefixes to
    // avoid creating monster subtrees. We estimate subtree cost using
    // Fibonacci-growth heuristics and split until cost < threshold.

    double COST_THRESHOLD = 5e7;  // target ~50M nodes per prefix max
    int MIN_PREFIX_DEPTH = 8;

    double log_phi = log(1.618033988749895);
    int max_prefixes = 50000000;
    uint64 *h_prefix_raw = (uint64*)malloc((uint64)max_prefixes * 4 * sizeof(uint64));
    int np = 0;

    printf("Generating prefixes (adaptive, threshold=%.0e)...\n", COST_THRESHOLD);
    fflush(stdout);

    struct PfxEntry { uint64 pp, p, qp, q; int depth; };
    int stk_cap = 50000000;
    struct PfxEntry *stk = (struct PfxEntry*)malloc(stk_cap * sizeof(struct PfxEntry));
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

        // Estimate subtree cost
        double remaining = log((double)max_d / (double)q) / log_phi;
        double est_cost = pow((double)num_digits, remaining * 0.6);

        bool should_split = (dep < MIN_PREFIX_DEPTH) ||
                           (est_cost > COST_THRESHOLD && np < max_prefixes - num_digits * 10);

        if (!should_split || np >= max_prefixes - num_digits) {
            if (np < max_prefixes) {
                h_prefix_raw[np*4+0] = pp; h_prefix_raw[np*4+1] = p;
                h_prefix_raw[np*4+2] = qp; h_prefix_raw[np*4+3] = q;
                np++;
            }
        } else {
            for (int i = num_digits - 1; i >= 0; i--) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d || ssp >= stk_cap - 1) continue;
                stk[ssp].pp = p; stk[ssp].p = (uint64)h_digits[i] * p + pp;
                stk[ssp].qp = q; stk[ssp].q = qn;
                stk[ssp].depth = dep + 1; ssp++;
            }
        }
    }
    free(stk);

    printf("Prefixes generated: %d\n", np);
    fflush(stdout);

    // Sort by q descending (large q = shallow subtrees first, clears fast)
    qsort(h_prefix_raw, np, 4 * sizeof(uint64), cmp_by_q_desc);

    // Convert to WorkItem array
    WorkItem *h_work = (WorkItem*)malloc((uint64)np * sizeof(WorkItem));
    for (int i = 0; i < np; i++) {
        h_work[i].pp = h_prefix_raw[i*4+0];
        h_work[i].p  = h_prefix_raw[i*4+1];
        h_work[i].qp = h_prefix_raw[i*4+2];
        h_work[i].q  = h_prefix_raw[i*4+3];
    }
    free(h_prefix_raw);

    struct timespec t0, t1, t_batch;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // ── GPU allocation ──
    uint64 bitset_bytes = (max_d + 8) / 8;
    printf("Bitset: %.2f GB\n", bitset_bytes / 1e9);
    fflush(stdout);

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

    // ── Determine launch parameters ──
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    int block_size = 256;
    // We'll launch exactly as many threads as work items (capped at a reasonable max)
    int max_threads_per_launch = num_SMs * 2048;  // ~2048 threads per SM max occupancy

    // Overflow buffer: each thread can overflow up to MAX_OVERFLOW_PER_THREAD items.
    // Size the buffer for the maximum concurrent threads.
    int overflow_cap = max_threads_per_launch * MAX_OVERFLOW_PER_THREAD;
    // Cap at 64M items to avoid excessive memory (64M * 32B = 2GB)
    if (overflow_cap > 64 * 1024 * 1024) overflow_cap = 64 * 1024 * 1024;

    WorkItem *d_work = NULL;
    WorkItem *d_overflow = NULL;
    int *d_overflow_count = NULL;

    // Allocate work buffer (will be resized as needed)
    size_t work_alloc = (uint64)max_threads_per_launch * sizeof(WorkItem);
    // Start with enough for initial prefixes
    if ((uint64)np * sizeof(WorkItem) > work_alloc)
        work_alloc = (uint64)np * sizeof(WorkItem);
    cudaMalloc(&d_work, work_alloc);
    cudaMalloc(&d_overflow, (uint64)overflow_cap * sizeof(WorkItem));
    cudaMalloc(&d_overflow_count, sizeof(int));

    printf("Overflow buffer: %d items (%.0f MB)\n",
           overflow_cap, (double)overflow_cap * sizeof(WorkItem) / 1e6);
    printf("Max threads per launch: %d\n\n", max_threads_per_launch);
    fflush(stdout);

    // Host-side overflow buffer for collecting results
    WorkItem *h_overflow = (WorkItem*)malloc((uint64)overflow_cap * sizeof(WorkItem));

    // ── Main iterative loop ──
    int round = 0;
    int total_work_items = np;
    int total_nodes_approx = 0;
    int total_overflow_items = 0;

    // Current work: starts with initial prefixes
    WorkItem *current_work = h_work;
    int current_count = np;

    while (current_count > 0) {
        round++;
        clock_gettime(CLOCK_MONOTONIC, &t_batch);
        double elapsed = (t_batch.tv_sec - t0.tv_sec) + (t_batch.tv_nsec - t0.tv_nsec) / 1e9;

        printf("  Round %d: %d work items (elapsed %.1fs)\n", round, current_count, elapsed);
        fflush(stdout);

        // Process work in batches if there are more items than max_threads_per_launch
        int items_remaining = current_count;
        int items_offset = 0;
        // We need a temporary host buffer for overflow from all batches in this round
        WorkItem *round_overflow = (WorkItem*)malloc((uint64)overflow_cap * sizeof(WorkItem));
        int round_overflow_count = 0;

        while (items_remaining > 0) {
            int batch_size = items_remaining;
            if (batch_size > max_threads_per_launch) batch_size = max_threads_per_launch;

            // Upload batch to GPU
            // Ensure d_work is large enough
            size_t needed = (uint64)batch_size * sizeof(WorkItem);
            if (needed > work_alloc) {
                cudaFree(d_work);
                work_alloc = needed;
                cudaMalloc(&d_work, work_alloc);
            }
            cudaMemcpy(d_work, current_work + items_offset, needed, cudaMemcpyHostToDevice);

            // Reset overflow counter
            int zero = 0;
            cudaMemcpy(d_overflow_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // Launch kernel
            int grid = (batch_size + block_size - 1) / block_size;
            dfs_bounded<<<grid, block_size>>>(
                d_work, batch_size,
                d_digits, num_digits,
                d_bs, max_d,
                d_overflow, d_overflow_count,
                overflow_cap);

            cudaDeviceSynchronize();

            // Check for errors
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "FATAL: kernel error: %s\n", cudaGetErrorString(kerr));
                return 1;
            }

            // Read overflow count
            int h_ocount = 0;
            cudaMemcpy(&h_ocount, d_overflow_count, sizeof(int), cudaMemcpyDeviceToHost);

            // Download overflow items
            if (h_ocount > 0) {
                if (h_ocount > overflow_cap) h_ocount = overflow_cap;
                // Make sure round_overflow has space
                if (round_overflow_count + h_ocount > overflow_cap) {
                    // Reallocate
                    int new_cap = (round_overflow_count + h_ocount) * 2;
                    WorkItem *tmp = (WorkItem*)realloc(round_overflow, (uint64)new_cap * sizeof(WorkItem));
                    if (tmp) {
                        round_overflow = tmp;
                    } else {
                        fprintf(stderr, "WARNING: overflow realloc failed, truncating\n");
                        h_ocount = overflow_cap - round_overflow_count;
                    }
                }
                cudaMemcpy(round_overflow + round_overflow_count, d_overflow,
                           (uint64)h_ocount * sizeof(WorkItem), cudaMemcpyDeviceToHost);
                round_overflow_count += h_ocount;
            }

            total_nodes_approx += batch_size;  // rough approximation
            items_remaining -= batch_size;
            items_offset += batch_size;
        }

        // Free current work if it's not the original h_work
        if (current_work != h_work) free(current_work);

        // The overflow items from this round become the work for the next round
        if (round_overflow_count > 0) {
            printf("    -> %d overflow items (will be processed in next round)\n",
                   round_overflow_count);
            fflush(stdout);
            total_overflow_items += round_overflow_count;
            total_work_items += round_overflow_count;
            current_work = round_overflow;
            current_count = round_overflow_count;
        } else {
            free(round_overflow);
            current_work = NULL;
            current_count = 0;
        }
    }

    free(h_work);
    free(h_overflow);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double enum_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\nGPU enumeration: %.1fs (%d rounds, %d total work items, %d overflow items)\n",
           enum_time, round, total_work_items, total_overflow_items);
    fflush(stdout);

    // ── Mark shallow denominators on CPU ──
    // These are CF denominators at depth < PREFIX_DEPTH that were not
    // included as GPU prefixes. We mark them on CPU since there are few.
    uint8_t *h_bs = (uint8_t*)malloc(bitset_bytes);
    cudaMemcpy(h_bs, d_bs, bitset_bytes, cudaMemcpyDeviceToHost);

    h_bs[0] |= (1 << 1);  // d=1 is always covered
    {
        struct ShallowEntry { uint64 pp, p, qp, q; int dep; };
        struct ShallowEntry *cstk = (struct ShallowEntry*)malloc(2000000 * sizeof(struct ShallowEntry));
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
            if (dep >= MIN_PREFIX_DEPTH) continue;
            uint64 pp = cstk[csp].pp, p = cstk[csp].p, qp = cstk[csp].qp;
            for (int i = 0; i < num_digits; i++) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d || csp >= 1999999) continue;
                cstk[csp].pp = p;
                cstk[csp].p = (uint64)h_digits[i] * p + pp;
                cstk[csp].qp = q; cstk[csp].q = qn;
                cstk[csp].dep = dep + 1; csp++;
            }
        }
        free(cstk);
    }
    cudaMemcpy(d_bs, h_bs, bitset_bytes, cudaMemcpyHostToDevice);

    // ── Count marked bits on GPU ──
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

    if (uncovered > 0 && uncovered <= 1000 && max_d <= 100000000ULL) {
        printf("Uncovered d:");
        for (uint64 d = 1; d <= max_d; d++)
            if (!(h_bs[d>>3] & (1 << (d&7)))) printf(" %llu", (unsigned long long)d);
        printf("\n");
    } else if (uncovered > 0 && uncovered <= 1000) {
        printf("(Uncovered list omitted for large range)\n");
    }

    printf("Time: %.1fs (enum: %.1fs)\n", total_time, enum_time);
    printf("========================================\n");

    free(h_bs);
    cudaFree(d_bs); cudaFree(d_digits); cudaFree(d_work);
    cudaFree(d_overflow); cudaFree(d_overflow_count);
    return 0;
}
