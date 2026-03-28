/*
 * Zaremba verifier v4 — inverse CF construction
 *
 * Instead of searching for witness per d, enumerate ALL bounded CFs and
 * mark which denominators they produce. Then check for gaps.
 *
 * A CF [0; a1, a2, ..., ak] with ai ∈ {1,...,5} gives fraction p/q
 * via the convergent recurrence. We enumerate the tree of all such CFs,
 * marking each q in a bitset. Any unmarked q is a potential counterexample.
 *
 * Memory: 1 bit per d → 1.25 GB for d up to 10^10. Fits in CPU RAM easily.
 * The enumeration is CPU-bound (recursive tree walk), but it's O(total_CFs)
 * not O(max_d), which is fundamentally faster for large d.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o zaremba_v4 scripts/zaremba_verify_v4.cu
 *   (or just gcc — this doesn't actually need CUDA)
 * Run:     ./zaremba_v4 <max_d>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BOUND 5

typedef unsigned long long uint64;

static uint8_t *bitset = NULL;
static uint64 max_d_global = 0;
static uint64 total_marked = 0;

static inline void mark(uint64 d) {
    if (d < 1 || d > max_d_global) return;
    uint64 byte = d >> 3;
    uint8_t bit = 1 << (d & 7);
    if (!(bitset[byte] & bit)) {
        bitset[byte] |= bit;
        total_marked++;
    }
}

static inline int is_marked(uint64 d) {
    if (d < 1 || d > max_d_global) return 0;
    return (bitset[d >> 3] >> (d & 7)) & 1;
}

/*
 * Recursive enumeration of all CFs [0; a1, a2, ..., ak] with ai ∈ {1,...,5}.
 *
 * We track the convergent matrix:
 *   p_prev, p  (numerator history)
 *   q_prev, q  (denominator history)
 *
 * Recurrence: when appending quotient a:
 *   p_new = a * p + p_prev
 *   q_new = a * q + q_prev
 *
 * Every q encountered is a valid Zaremba denominator (gcd(p,q)=1 by CF property).
 */
void enumerate(uint64 p_prev, uint64 p, uint64 q_prev, uint64 q,
               int depth, int max_depth) {
    // Mark current denominator
    mark(q);

    if (depth >= max_depth) return;

    // Try each next partial quotient
    for (int a = 1; a <= BOUND; a++) {
        uint64 q_new = (uint64)a * q + q_prev;
        if (q_new > max_d_global) break;  // all larger a give even larger q

        uint64 p_new = (uint64)a * p + p_prev;
        enumerate(p, p_new, q, q_new, depth + 1, max_depth);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <max_d> [max_cf_length]\n", argv[0]);
        return 1;
    }

    max_d_global = (uint64)atoll(argv[1]);
    int max_len = argc > 2 ? atoi(argv[2]) : 100;

    printf("Zaremba v4 (inverse CF construction)\n");
    printf("Target: all d from 1 to %llu\n", (unsigned long long)max_d_global);
    printf("Max CF length: %d\n", max_len);
    printf("Bound A = %d\n\n", BOUND);

    // Allocate bitset
    uint64 bitset_bytes = (max_d_global + 8) / 8;
    printf("Bitset: %llu bytes (%.2f GB)\n\n",
           (unsigned long long)bitset_bytes, bitset_bytes / 1e9);

    bitset = (uint8_t*)calloc(bitset_bytes, 1);
    if (!bitset) {
        printf("ERROR: cannot allocate %.2f GB for bitset\n", bitset_bytes / 1e9);
        return 1;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Enumerate all CFs starting with [0; a1, ...]
    // Initial state: p_prev=1, p=0, q_prev=0, q=1 (before any quotient)
    // After first quotient a1: p=1, q=a1, p_prev=0, q_prev=1
    printf("Enumerating all CFs with quotients in {1,...,%d}...\n", BOUND);

    // Also mark d=1 (trivially, a=1 works)
    mark(1);

    for (int a1 = 1; a1 <= BOUND; a1++) {
        printf("  Tree a1=%d... ", a1);
        fflush(stdout);
        uint64 before = total_marked;
        enumerate(0, 1, 1, (uint64)a1, 1, max_len);
        printf("(+%llu new, %llu total)\n",
               (unsigned long long)(total_marked - before),
               (unsigned long long)total_marked);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double enum_time = (t_end.tv_sec - t_start.tv_sec) +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\nEnumeration: %.1fs, %llu unique denominators marked\n\n",
           enum_time, (unsigned long long)total_marked);

    // Check for uncovered d values
    printf("Scanning for uncovered denominators...\n");
    uint64 uncovered = 0;
    uint64 first_uncovered[100];

    for (uint64 d = 1; d <= max_d_global; d++) {
        if (!is_marked(d)) {
            if (uncovered < 100) first_uncovered[uncovered] = d;
            uncovered++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Zaremba v4: d=1 to %llu\n", (unsigned long long)max_d_global);
    printf("Unique denominators covered: %llu / %llu\n",
           (unsigned long long)total_marked, (unsigned long long)max_d_global);
    printf("Uncovered: %llu\n", (unsigned long long)uncovered);

    if (uncovered > 0 && uncovered <= 20) {
        printf("Uncovered d values:\n");
        for (uint64 i = 0; i < uncovered; i++)
            printf("  d = %llu\n", (unsigned long long)first_uncovered[i]);
    } else if (uncovered > 20) {
        printf("First 20 uncovered:\n");
        for (uint64 i = 0; i < 20; i++)
            printf("  d = %llu\n", (unsigned long long)first_uncovered[i]);
    }

    printf("Time: %.1fs (enum: %.1fs, scan: %.1fs)\n",
           total_elapsed, enum_time, total_elapsed - enum_time);

    if (uncovered == 0) {
        printf("\nZaremba's Conjecture HOLDS for all d in [1, %llu] with A=%d\n",
               (unsigned long long)max_d_global, BOUND);
    } else {
        printf("\n*** %llu UNCOVERED — investigate (may need longer CFs) ***\n",
               (unsigned long long)uncovered);
    }
    printf("========================================\n");

    free(bitset);
    return (uncovered > 0) ? 1 : 0;
}
