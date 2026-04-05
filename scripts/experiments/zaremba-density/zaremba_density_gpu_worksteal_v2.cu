/*
 * GPU-accelerated Zaremba density computation — work-stealing edition.
 *
 * Architecture:
 *   1. CPU generates prefixes at fixed depth (as before)
 *   2. GPU launches persistent threads that self-schedule via atomic counter
 *   3. Each thread does DFS. After DONATE_THRESHOLD nodes, it donates
 *      all-but-one children at each branch point to a global work queue.
 *   4. When a thread finishes its subtree, it grabs from the work queue.
 *   5. Termination: atomic active-thread counter reaches 0 with empty queue.
 *
 * The donation mechanism is THE key innovation: it dynamically redistributes
 * work from the deepest subtrees (digit-1 Fibonacci paths) to idle threads.
 * Without it, a single thread can be stuck for hours on one subtree while
 * 300K threads sit idle. With it, deep subtrees get split across all SMs.
 *
 * Memory budget (B200, 183 GB):
 *   Bitset:   max_d/8        (12.5 GB for 10^11, 125 GB for 10^12)
 *   Prefixes: N * 32 bytes   (531K * 32 = 17 MB at depth 12)
 *   Queue:    Q * 32 bytes   (16M * 32 = 512 MB)
 *   Total:    ~13-126 GB — fits comfortably
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
#define MAX_DEPTH  128   // DFS stack depth per thread (enough for q up to 10^15)

// ── Work queue item: same as a prefix (the 4 values defining a CF state) ──
struct WorkItem {
    uint64 pp, p, qp, q;
};

// ── Device-side mark function ──
__device__ void mark(uint64 d, uint8_t *bitset, uint64 max_d) {
    if (d < 1 || d > max_d) return;
    uint64 byte = d >> 3;
    uint8_t bit = 1 << (d & 7);
    atomicOr((unsigned int*)&bitset[byte & ~3], (unsigned int)bit << (8 * (byte & 3)));
}

// ── Work-stealing kernel v2: depth-limited DFS with re-enqueueing ──
//
// Key improvements over v1:
//   1. QUEUE-FIRST work acquisition: check donation queue before prefix list.
//      This ensures donated items (partially-explored deep subtrees) get
//      picked up immediately instead of starving while prefixes remain.
//   2. DEPTH-LIMITED DFS: each work item runs DFS to at most DFS_DEPTH_LIMIT
//      additional levels. At the limit, remaining children are pushed to the
//      queue. This prevents any thread from owning a trillion-node subtree.
//   3. ALWAYS DONATE at branch points after the threshold, regardless of
//      queue fullness (the depth limit prevents queue explosion).
//
__global__ void enumerate_worksteal(
    uint64 *prefixes, int num_prefixes,
    int *digits, int num_digits,
    uint8_t *bitset, uint64 max_d,
    int *prefix_counter,
    WorkItem *queue, int queue_capacity,
    int *queue_head, int *queue_tail,
    int *active_threads,
    int *total_donated,
    int *total_dequeued)
{
    // DFS depth limit per work item. After this many levels, re-enqueue
    // remaining children. At ~phi^50 ~ 10^10 denominators reachable in 50
    // Fibonacci-growth levels, this bounds per-thread work to ~10^10 nodes
    // in the absolute worst case (all digit-1 path), but typically much less
    // since non-1 digits prune quickly.
    // Depth limit: after this many DFS levels, re-enqueue remaining children.
    // 30 levels with digit 1 gives q growth of phi^30 ~ 2M, so a thread
    // starting at q=1 would reach q~2M before re-enqueueing. The re-enqueued
    // items start at q~2M and go another 30 levels to q~4B, etc.
    // This creates a cascade of bounded-work items.
    const int DFS_DEPTH_LIMIT = 30;

    // Donation threshold: after this many nodes, donate children at the
    // next branch point. High value = rely on depth-limit re-enqueueing
    // as the primary redistribution mechanism, with donation as backup.
    const int DONATE_THRESHOLD = 10000000;

    struct { uint64 pp, p, qp, q; int depth; } stack[MAX_DEPTH];

    while (true) {
        // ── Get work: try QUEUE first, then prefix list ──
        uint64 start_pp, start_p, start_qp, start_q;
        bool got_work = false;

        // Queue first (donated items = partially-explored deep subtrees)
        if (*queue_tail > *queue_head) {
            int my_slot = atomicAdd(queue_head, 1);
            if (my_slot < *queue_tail) {
                WorkItem item = queue[my_slot % queue_capacity];
                start_pp = item.pp; start_p = item.p;
                start_qp = item.qp; start_q = item.q;
                got_work = true;
                atomicAdd(total_dequeued, 1);
            } else {
                atomicSub(queue_head, 1);
            }
        }

        // Then prefix list
        if (!got_work) {
            int my_prefix = atomicAdd(prefix_counter, 1);
            if (my_prefix < num_prefixes) {
                start_pp = prefixes[my_prefix * 4 + 0];
                start_p  = prefixes[my_prefix * 4 + 1];
                start_qp = prefixes[my_prefix * 4 + 2];
                start_q  = prefixes[my_prefix * 4 + 3];
                got_work = true;
            } else {
                atomicSub(prefix_counter, 1);
            }
        }

        // Try queue again (in case something was donated while we checked prefixes)
        if (!got_work && *queue_tail > *queue_head) {
            int my_slot = atomicAdd(queue_head, 1);
            if (my_slot < *queue_tail) {
                WorkItem item = queue[my_slot % queue_capacity];
                start_pp = item.pp; start_p = item.p;
                start_qp = item.qp; start_q = item.q;
                got_work = true;
                atomicAdd(total_dequeued, 1);
            } else {
                atomicSub(queue_head, 1);
            }
        }

        if (!got_work) {
            // No work. Spin waiting for donations.
            atomicSub(active_threads, 1);

            for (int spin = 0; spin < 200000; spin++) {
                // Try queue
                if (*queue_tail > *queue_head) {
                    int my_slot = atomicAdd(queue_head, 1);
                    if (my_slot < *queue_tail) {
                        WorkItem item = queue[my_slot % queue_capacity];
                        start_pp = item.pp; start_p = item.p;
                        start_qp = item.qp; start_q = item.q;
                        got_work = true;
                        atomicAdd(active_threads, 1);
                        atomicAdd(total_dequeued, 1);
                        break;
                    }
                    atomicSub(queue_head, 1);
                }
                // Try prefixes
                if (*prefix_counter < num_prefixes) {
                    int my_pfx = atomicAdd(prefix_counter, 1);
                    if (my_pfx < num_prefixes) {
                        start_pp = prefixes[my_pfx * 4 + 0];
                        start_p  = prefixes[my_pfx * 4 + 1];
                        start_qp = prefixes[my_pfx * 4 + 2];
                        start_q  = prefixes[my_pfx * 4 + 3];
                        got_work = true;
                        atomicAdd(active_threads, 1);
                        break;
                    }
                    atomicSub(prefix_counter, 1);
                }
                // Termination check
                if (*active_threads <= 0 && *queue_head >= *queue_tail
                    && *prefix_counter >= num_prefixes) return;
                __nanosleep(5000);  // 5 microseconds
            }
            if (!got_work) return;
        }

        // ── Depth-limited DFS with donation ──
        mark(start_q, bitset, max_d);

        int sp = 0;
        for (int i = num_digits - 1; i >= 0; i--) {
            uint64 a = digits[i];
            uint64 q_new = a * start_q + start_qp;
            if (q_new > max_d || sp >= MAX_DEPTH) continue;
            stack[sp].pp = start_p;
            stack[sp].p  = a * start_p + start_pp;
            stack[sp].qp = start_q;
            stack[sp].q  = q_new;
            stack[sp].depth = 0;
            sp++;
        }

        int nodes_processed = 0;

        while (sp > 0) {
            sp--;
            uint64 pp = stack[sp].pp;
            uint64 p  = stack[sp].p;
            uint64 qp = stack[sp].qp;
            uint64 q  = stack[sp].q;
            int depth  = stack[sp].depth;

            mark(q, bitset, max_d);
            nodes_processed++;

            // Count viable children
            int nchildren = 0;
            WorkItem children[MAX_DIGITS];
            for (int i = 0; i < num_digits; i++) {
                uint64 a = digits[i];
                uint64 q_new = a * q + qp;
                if (q_new > max_d) continue;
                children[nchildren].pp = p;
                children[nchildren].p  = a * p + pp;
                children[nchildren].qp = q;
                children[nchildren].q  = q_new;
                nchildren++;
            }
            if (nchildren == 0) continue;

            // ── Depth limit: YIELD this DFS, push everything to queue ──
            // When we hit the depth limit, dump ALL remaining work (children
            // + entire local stack) to the queue and break out of the DFS
            // loop. The thread then goes back to the main loop and picks up
            // queue items. This forces threads to cycle through work items
            // instead of being stuck on one deep subtree forever.
            //
            // Back pressure: if queue > 75% full, skip the yield and keep
            // grinding locally. This prevents queue overflow.
            int q_pending = *queue_tail - *queue_head;
            bool queue_accepting = (q_pending < (queue_capacity * 3 / 4));

            if (depth >= DFS_DEPTH_LIMIT && queue_accepting) {
                // Enqueue current children
                int total_to_enqueue = nchildren + sp;  // children + remaining stack
                if (total_to_enqueue > 0 && q_pending + total_to_enqueue < queue_capacity) {
                    int base = atomicAdd(queue_tail, total_to_enqueue);
                    // First: current children
                    for (int j = 0; j < nchildren; j++) {
                        queue[(base + j) % queue_capacity] = children[j];
                    }
                    // Then: remaining stack items (convert to WorkItem)
                    for (int j = 0; j < sp; j++) {
                        WorkItem w;
                        w.pp = stack[j].pp; w.p = stack[j].p;
                        w.qp = stack[j].qp; w.q = stack[j].q;
                        queue[(base + nchildren + j) % queue_capacity] = w;
                    }
                    atomicAdd(total_donated, total_to_enqueue);
                    sp = 0;  // stack is now empty
                    break;   // EXIT DFS loop — go back to main work acquisition
                }
                // Queue can't fit everything — fall through to local processing
            }

            // ── Normal: donate at threshold OR push to local stack ──
            if (nchildren > 1 && nodes_processed >= DONATE_THRESHOLD && queue_accepting) {
                int to_donate = nchildren - 1;
                int base = atomicAdd(queue_tail, to_donate);
                for (int j = 0; j < to_donate; j++) {
                    queue[(base + j) % queue_capacity] = children[1 + j];
                }
                atomicAdd(total_donated, to_donate);
                if (sp < MAX_DEPTH) {
                    stack[sp].pp = children[0].pp;
                    stack[sp].p  = children[0].p;
                    stack[sp].qp = children[0].qp;
                    stack[sp].q  = children[0].q;
                    stack[sp].depth = depth + 1;
                    sp++;
                }
                nodes_processed = 0;
            } else {
                for (int i = nchildren - 1; i >= 0; i--) {
                    if (sp >= MAX_DEPTH) break;
                    stack[sp].pp = children[i].pp;
                    stack[sp].p  = children[i].p;
                    stack[sp].qp = children[i].qp;
                    stack[sp].q  = children[i].q;
                    stack[sp].depth = depth + 1;
                    sp++;
                }
            }
        }
    }
}

// ── Bit counting kernel (unchanged) ──
__global__ void count_marked(uint8_t *bitset, uint64 max_d, uint64 *count) {
    uint64 tid = blockIdx.x * (uint64)blockDim.x + threadIdx.x;
    uint64 byte_idx = tid;
    uint64 max_byte = (max_d + 8) / 8;
    if (byte_idx >= max_byte) return;

    uint8_t b = bitset[byte_idx];
    int bits = __popc((unsigned int)b);
    if (byte_idx == max_byte - 1) {
        int valid_bits = (max_d % 8) + 1;
        uint8_t mask = (1 << valid_bits) - 1;
        bits = __popc((unsigned int)(b & mask));
    }
    if (bits > 0) atomicAdd(count, (uint64)bits);
}

// Sort comparator: descending by q (4th element of each 4-uint64 record)
int cmp_by_q_desc(const void *a, const void *b) {
    uint64 qa = ((const uint64*)a)[3];
    uint64 qb = ((const uint64*)b)[3];
    return (qa > qb) ? -1 : (qa < qb) ? 1 : 0;
}

// ── Merge mode: combine partial bitset files from multi-GPU shards ──
int do_merge(int argc, char **argv) {
    // Usage: zaremba_density_gpu --merge <max_d> <digits> <num_shards> <bitset_prefix>
    if (argc < 6) {
        fprintf(stderr, "Usage: %s --merge <max_d> <digits> <num_shards> <bitset_prefix>\n", argv[0]);
        return 1;
    }
    uint64 max_d = (uint64)atoll(argv[2]);
    char *digits_str = argv[3];
    int num_shards = atoi(argv[4]);
    char *prefix = argv[5];

    uint64 bitset_bytes = (max_d + 8) / 8;
    uint8_t *merged = (uint8_t*)calloc(bitset_bytes, 1);

    printf("Merging %d shard bitsets (%.2f GB each)...\n", num_shards, bitset_bytes / 1e9);
    fflush(stdout);

    for (int s = 0; s < num_shards; s++) {
        char path[512];
        snprintf(path, 512, "%s.shard%d.bin", prefix, s);
        FILE *fp = fopen(path, "rb");
        if (!fp) { fprintf(stderr, "FATAL: cannot open %s\n", path); return 1; }
        uint8_t *shard = (uint8_t*)malloc(bitset_bytes);
        size_t rd = fread(shard, 1, bitset_bytes, fp);
        fclose(fp);
        if (rd != bitset_bytes) {
            fprintf(stderr, "FATAL: %s: expected %llu bytes, got %zu\n",
                    path, (unsigned long long)bitset_bytes, rd);
            return 1;
        }
        // OR into merged
        for (uint64 i = 0; i < bitset_bytes; i++)
            merged[i] |= shard[i];
        free(shard);
        printf("  merged shard %d/%d\n", s + 1, num_shards);
        fflush(stdout);
    }

    // Also mark shallow denominators (depth < PREFIX_DEPTH) — same as single-GPU
    int h_digits[MAX_DIGITS];
    int num_digits = 0;
    char buf[256]; strncpy(buf, digits_str, 255);
    char *tok = strtok(buf, ",");
    while (tok && num_digits < MAX_DIGITS) {
        h_digits[num_digits++] = atoi(tok);
        tok = strtok(NULL, ",");
    }

    int PREFIX_DEPTH = 8;
    if (max_d >= 1000000000ULL)   PREFIX_DEPTH = 15;
    if (max_d >= 10000000000ULL)  PREFIX_DEPTH = 18;
    if (max_d >= 100000000000ULL) PREFIX_DEPTH = 20;
    if (max_d >= 1000000000000ULL) PREFIX_DEPTH = 22;

    merged[0] |= (1 << 1);  // d=1
    {
        struct ShallowEntry { uint64 pp, p, qp, q; int dep; };
        struct ShallowEntry *cstk = (struct ShallowEntry*)malloc(500000 * sizeof(struct ShallowEntry));
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
            merged[q>>3] |= (1 << (q&7));
            if (dep >= PREFIX_DEPTH) continue;
            uint64 pp = cstk[csp].pp, p = cstk[csp].p, qp = cstk[csp].qp;
            for (int i = 0; i < num_digits; i++) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d) continue;
                if (csp < 499999) {
                    cstk[csp].pp = p;
                    cstk[csp].p = (uint64)h_digits[i] * p + pp;
                    cstk[csp].qp = q;
                    cstk[csp].q = qn;
                    cstk[csp].dep = dep + 1;
                    csp++;
                }
            }
        }
        free(cstk);
    }

    // Count
    uint64 covered = 0;
    for (uint64 d = 1; d <= max_d; d++)
        if (merged[d>>3] & (1 << (d&7))) covered++;

    uint64 uncovered = max_d - covered;

    printf("\n========================================\n");
    printf("RESULTS (merged %d shards)\n", num_shards);
    printf("========================================\n");
    printf("Digit set: {%s}\n", digits_str);
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Covered: %llu / %llu\n", (unsigned long long)covered, (unsigned long long)max_d);
    printf("Density: %.10f%%\n", 100.0 * covered / max_d);
    printf("Uncovered: %llu\n", (unsigned long long)uncovered);

    if (uncovered > 0 && uncovered <= 100) {
        printf("Uncovered d:");
        for (uint64 d = 1; d <= max_d; d++)
            if (!(merged[d>>3] & (1 << (d&7)))) printf(" %llu", (unsigned long long)d);
        printf("\n");
    }
    printf("========================================\n");

    // Clean up shard files
    for (int s = 0; s < num_shards; s++) {
        char path[512];
        snprintf(path, 512, "%s.shard%d.bin", prefix, s);
        remove(path);
    }

    free(merged);
    return 0;
}

int main(int argc, char **argv) {
    // Check for --merge mode
    if (argc >= 2 && strcmp(argv[1], "--merge") == 0)
        return do_merge(argc, argv);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <max_d> <digits> [--shard K N]\n", argv[0]);
        fprintf(stderr, "       %s --merge <max_d> <digits> <num_shards> <bitset_prefix>\n", argv[0]);
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

    // Parse optional --shard K N
    int shard_id = 0, num_shards = 1;
    char *bitset_output = NULL;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--shard") == 0 && i + 2 < argc) {
            shard_id = atoi(argv[i+1]);
            num_shards = atoi(argv[i+2]);
            i += 2;
        }
        if (strcmp(argv[i], "--bitset-out") == 0 && i + 1 < argc) {
            bitset_output = argv[i+1];
            i += 1;
        }
    }

    printf("========================================\n");
    if (num_shards > 1)
        printf("Zaremba Density (GPU) — shard %d/%d\n", shard_id, num_shards);
    else
        printf("Zaremba Density (GPU) — work-stealing\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d);
    printf("Digits: {");
    for (int i = 0; i < num_digits; i++) printf("%s%d", i?",":"", h_digits[i]);
    printf("}\n");
    printf("========================================\n\n");
    fflush(stdout);

    // ── Prefix generation (fixed depth, same as before) ──
    // Adaptive prefix generation: split until each prefix's estimated
    // subtree cost is below a threshold. Cost estimate for a node with
    // denominator q: remaining depth ≈ log(max_d/q) / log(phi) for
    // digit-1-heavy paths, total nodes ≈ |A|^remaining_depth.
    // We split until estimated nodes per prefix < COST_THRESHOLD.
    //
    // This replaces fixed PREFIX_DEPTH and ensures balanced work per prefix
    // regardless of digit set composition.
    double COST_THRESHOLD = 1e8;  // target ~100M nodes per prefix max
    int PREFIX_DEPTH = 8;  // minimum depth before cost check kicks in

    // Adaptive prefix generation with cost-bounded splitting.
    // Estimate subtree cost for each node: log(max_d/q) / log(phi) gives
    // remaining Fibonacci-depth, then |A|^depth gives estimated nodes.
    // Split until estimated cost < COST_THRESHOLD.
    double log_phi = log(1.618033988749895);
    int max_prefixes = 50000000;  // 50M max
    uint64 *all_prefixes = (uint64*)malloc((uint64)max_prefixes * 4 * sizeof(uint64));
    int total_prefixes = 0;

    printf("Generating prefixes (adaptive, cost_threshold=%.0e)...\n", COST_THRESHOLD);
    fflush(stdout);

    struct PfxEntry { uint64 pp, p, qp, q; int depth; };
    int stk_size = 50000000;
    struct PfxEntry *stk = (struct PfxEntry*)malloc(stk_size * sizeof(struct PfxEntry));
    int ssp = 0;
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

        // Estimate subtree cost: remaining depth * branching
        double remaining_depth = log((double)max_d / (double)q) / log_phi;
        double est_cost = pow((double)num_digits, remaining_depth * 0.6);
        // The 0.6 factor accounts for pruning (not all branches survive)

        bool should_split = (dep < PREFIX_DEPTH) ||
                           (est_cost > COST_THRESHOLD && total_prefixes < max_prefixes - num_digits * 10);

        if (!should_split || total_prefixes >= max_prefixes - num_digits) {
            // Emit as a prefix
            if (total_prefixes < max_prefixes) {
                all_prefixes[total_prefixes*4+0] = pp;
                all_prefixes[total_prefixes*4+1] = p;
                all_prefixes[total_prefixes*4+2] = qp;
                all_prefixes[total_prefixes*4+3] = q;
                total_prefixes++;
            }
        } else {
            // Split further
            for (int i = num_digits - 1; i >= 0; i--) {
                uint64 qn = (uint64)h_digits[i] * q + qp;
                if (qn > max_d) continue;
                uint64 pn = (uint64)h_digits[i] * p + pp;
                if (ssp >= stk_size - 1) break;
                stk[ssp].pp = p; stk[ssp].p = pn;
                stk[ssp].qp = q; stk[ssp].q = qn;
                stk[ssp].depth = dep + 1;
                ssp++;
            }
        }
    }
    free(stk);

    // Sort by q descending and extract shard
    printf("Total prefixes: %d. Sorting by q descending...\n", total_prefixes);
    fflush(stdout);
    qsort(all_prefixes, total_prefixes, 4 * sizeof(uint64), cmp_by_q_desc);

    uint64 *h_prefixes = (uint64*)malloc((uint64)max_prefixes * 4 * sizeof(uint64));
    int np = 0;
    for (int i = shard_id; i < total_prefixes; i += num_shards) {
        if (np >= max_prefixes) break;
        h_prefixes[np*4+0] = all_prefixes[i*4+0];
        h_prefixes[np*4+1] = all_prefixes[i*4+1];
        h_prefixes[np*4+2] = all_prefixes[i*4+2];
        h_prefixes[np*4+3] = all_prefixes[i*4+3];
        np++;
    }
    free(all_prefixes);

    printf("Prefixes: %d (shard %d/%d, total %d)\nBitset: %.2f GB\n",
           np, shard_id, num_shards, total_prefixes, (max_d + 8) / 8.0 / 1e9);
    fflush(stdout);

    struct timespec t0, t1, t_check;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // ── Allocate GPU memory ──
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

    // ── Donation queue ──
    // Size: 16M items = 512 MB. This is a circular buffer.
    // With persistent threads donating 1-9 children at a time, this provides
    // ample headroom. The queue wraps around, so head and tail can grow without
    // bound (we use modular indexing).
    int queue_capacity = 256 * 1024 * 1024;  // 256M items = 8 GB
    WorkItem *d_queue;
    err = cudaMalloc(&d_queue, (uint64)queue_capacity * sizeof(WorkItem));
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc queue (%.0f MB): %s\n",
                (double)queue_capacity * sizeof(WorkItem) / 1e6, cudaGetErrorString(err));
        return 1;
    }
    printf("Work queue: %d items (%.0f MB)\n", queue_capacity,
           (double)queue_capacity * sizeof(WorkItem) / 1e6);
    fflush(stdout);

    // ── Mapped pinned memory for atomic counters (CPU-readable without memcpy) ──
    int *h_mapped;  // array of 6 ints: [prefix_ctr, q_head, q_tail, active, donated, dequeued]
    int *d_mapped;
    cudaHostAlloc(&h_mapped, 6 * sizeof(int), cudaHostAllocMapped);
    memset(h_mapped, 0, 6 * sizeof(int));
    cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);

    int *d_prefix_counter = &d_mapped[0];
    int *d_queue_head     = &d_mapped[1];
    int *d_queue_tail     = &d_mapped[2];
    int *d_active_threads = &d_mapped[3];
    int *d_total_donated  = &d_mapped[4];
    int *d_total_dequeued = &d_mapped[5];

    // ── Launch config ──
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    int max_threads_per_SM;
    cudaDeviceGetAttribute(&max_threads_per_SM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    int block_size = 256;
    int use_SMs = num_SMs - 2;  // leave 2 SMs free for progress polling
    if (use_SMs < 1) use_SMs = 1;
    int total_threads = use_SMs * max_threads_per_SM;
    int grid_size = (total_threads + block_size - 1) / block_size;

    // Initialize active thread count to total threads
    h_mapped[3] = grid_size * block_size;

    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream);

    printf("\nLaunching %d persistent threads on %d/%d SMs (%d initial prefixes)...\n",
           grid_size * block_size, use_SMs, num_SMs, np);
    fflush(stdout);

    enumerate_worksteal<<<grid_size, block_size, 0, kernel_stream>>>(
        d_prefixes, np, d_digits, num_digits, d_bs, max_d,
        d_prefix_counter, d_queue, queue_capacity,
        d_queue_head, d_queue_tail,
        d_active_threads, d_total_donated, d_total_dequeued);

    // ── Poll progress via mapped memory ──
    double last_report = 0;
    while (true) {
        __sync_synchronize();
        int pfx_done   = h_mapped[0];  // prefixes grabbed
        int q_head     = h_mapped[1];  // queue dequeue pointer
        int q_tail     = h_mapped[2];  // queue enqueue pointer
        int active     = h_mapped[3];  // threads currently doing work
        int donated    = h_mapped[4];  // total items ever donated
        int dequeued   = h_mapped[5];  // total items ever dequeued

        // Check termination: kernel sets active_threads to 0 and returns
        if (active <= 0 && pfx_done >= np && q_head >= q_tail) break;

        clock_gettime(CLOCK_MONOTONIC, &t_check);
        double elapsed = (t_check.tv_sec - t0.tv_sec) + (t_check.tv_nsec - t0.tv_nsec) / 1e9;

        if (elapsed - last_report >= 15.0) {
            int queue_pending = q_tail - q_head;
            if (queue_pending < 0) queue_pending = 0;
            int pfx_capped = pfx_done > np ? np : pfx_done;
            printf("  [%6.0fs] prefixes: %d/%d | queue: %d pending (%d donated, %d dequeued) | active: %d\n",
                   elapsed, pfx_capped, np, queue_pending, donated, dequeued, active);
            fflush(stdout);
            last_report = elapsed;
        }

        usleep(2000000);  // 2s poll
    }

    cudaStreamSynchronize(kernel_stream);
    cudaStreamDestroy(kernel_stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double enum_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    int final_donated  = h_mapped[4];
    int final_dequeued = h_mapped[5];
    printf("GPU enumeration: %.1fs (%d donated, %d dequeued)\n",
           enum_time, final_donated, final_dequeued);
    fflush(stdout);

    // ── Save bitset if in shard mode ──
    if (bitset_output) {
        printf("Saving bitset to %s (%.2f GB)...\n", bitset_output, bitset_bytes / 1e9);
        fflush(stdout);
        uint8_t *h_bs = (uint8_t*)malloc(bitset_bytes);
        cudaMemcpy(h_bs, d_bs, bitset_bytes, cudaMemcpyDeviceToHost);
        FILE *fp = fopen(bitset_output, "wb");
        if (fp) {
            fwrite(h_bs, 1, bitset_bytes, fp);
            fclose(fp);
            printf("Shard %d complete. Bitset saved.\n", shard_id);
        } else {
            fprintf(stderr, "FATAL: cannot write %s\n", bitset_output);
        }
        free(h_bs);
        free(h_prefixes);
        cudaFree(d_bs); cudaFree(d_digits); cudaFree(d_prefixes); cudaFree(d_queue);
        cudaFreeHost(h_mapped);
        return 0;
    }

    // ── Single-GPU mode: mark shallow + count + print results ──
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
            cstk[csp].dep = 1;
            csp++;
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
                if (qn > max_d) continue;
                if (csp < 499999) {
                    cstk[csp].pp = p;
                    cstk[csp].p = (uint64)h_digits[i] * p + pp;
                    cstk[csp].qp = q;
                    cstk[csp].q = qn;
                    cstk[csp].dep = dep + 1;
                    csp++;
                }
            }
        }
        free(cstk);
    }
    cudaMemcpy(d_bs, h_bs, bitset_bytes, cudaMemcpyHostToDevice);

    uint64 *d_count;
    cudaMalloc(&d_count, sizeof(uint64));
    cudaMemset(d_count, 0, sizeof(uint64));
    {
        uint64 max_byte = (max_d + 8) / 8;
        int bk = 256;
        int gd = (max_byte + bk - 1) / bk;
        count_marked<<<gd, bk>>>(d_bs, max_d, d_count);
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
        for (uint64 d = 1; d <= max_d; d++) {
            if (!(h_bs[d>>3] & (1 << (d&7)))) printf(" %llu", (unsigned long long)d);
        }
        printf("\n");
    }

    printf("Time: %.1fs (enum: %.1fs)\n", total_time, enum_time);
    printf("========================================\n");

    free(h_prefixes); free(h_bs);
    cudaFree(d_bs); cudaFree(d_digits); cudaFree(d_prefixes); cudaFree(d_queue);
    cudaFreeHost(h_mapped);
    return 0;
}
