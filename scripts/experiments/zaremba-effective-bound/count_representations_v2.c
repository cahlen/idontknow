/*
 * count_representations_v2.c — corrected R(d) counter (CPU reference).
 *
 * Replaces the defective count_representations.cu, which (a) counted the
 * numerator continuant m10 = K(a2..ak) instead of the denominator
 * K(a1..ak), (b) double-seeded d=1, and (c) silently clipped each BFS
 * level to 200M matrices while the true tree to 1e6 has ~1e10 nodes.
 *
 * This version is a plain depth-first search over CF words with digits in
 * {1,...,5} using the denominator continuant recurrence q_new = a*q + q_prev
 * from (q, q_prev) = (1, 0). No frontier buffers exist, so nothing can clip.
 *
 * Two quantities are counted per denominator d <= max_d:
 *
 *   R_words(d) = #{ words (a1,...,ak), ai in {1..5}, k >= 1 : K(a1..ak) = d }
 *                (continued-fraction expansions, i.e. semigroup elements
 *                with bottom-left continuant d; this is the quantity the
 *                transfer-operator heuristic R(d) ~ C * d^(2*delta-1) counts)
 *
 *   R_fracs(d) = #{ a : gcd(a,d)=1, a/d = [0;a1,...,ak] with all ai <= 5 }
 *                (coprime numerators admitting a bounded expansion).
 *                Computed via the bijection: a rational has an admissible
 *                expansion iff its digit-1-terminated expansion variant is
 *                admissible, so R_fracs(d) = # admissible words ending in 1.
 *
 * Every rational has exactly two CF expansions ([...,ak] with ak>=2 and
 * [...,ak-1,1]), so R_words ~ 2*R_fracs with boundary corrections.
 *
 * Compile: cc -O3 -o count_reps_v2 count_representations_v2.c
 * Run:     ./count_reps_v2 1000000
 * Output:  representation_counts_v2_<max_d>.csv  (d,R_words,R_fracs)
 *          plus a summary block on stdout (redirect to a log file).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static uint32_t *cnt_words;
static uint32_t *cnt_fracs;
static uint64_t maxd;
static uint64_t nodes;

static void dfs(uint64_t q, uint64_t qp) {
    for (int a = 1; a <= 5; a++) {
        uint64_t qn = (uint64_t)a * q + qp;
        if (qn > maxd) break;      /* qn strictly increasing in a */
        nodes++;
        cnt_words[qn]++;
        if (a == 1) cnt_fracs[qn]++;   /* word ends in digit 1 here iff we stop;
                                          counting at every a==1 node counts each
                                          word ending in 1 exactly once, at its
                                          terminal node */
        dfs(qn, q);
    }
}

int main(int argc, char **argv) {
    maxd = argc > 1 ? (uint64_t)atoll(argv[1]) : 1000000ULL;

    printf("Zaremba representation counter v2 (CPU reference, no clipping)\n");
    printf("Digit set {1..5}, denominators d <= %llu\n\n", (unsigned long long)maxd);

    cnt_words = calloc(maxd + 1, sizeof(uint32_t));
    cnt_fracs = calloc(maxd + 1, sizeof(uint32_t));
    if (!cnt_words || !cnt_fracs) { fprintf(stderr, "alloc failed\n"); return 2; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dfs(1, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    char fn[256];
    snprintf(fn, sizeof(fn), "representation_counts_v2_%llu.csv",
             (unsigned long long)maxd);
    FILE *f = fopen(fn, "w");
    fprintf(f, "d,R_words,R_fracs\n");

    uint64_t total_w = 0, total_f = 0, zeros = 0;
    uint64_t min_w = UINT64_MAX, min_wd = 0, max_w = 0, max_wd = 0;
    for (uint64_t d = 1; d <= maxd; d++) {
        uint32_t w = cnt_words[d], fr = cnt_fracs[d];
        fprintf(f, "%llu,%u,%u\n", (unsigned long long)d, w, fr);
        total_w += w; total_f += fr;
        if (w == 0) zeros++;
        if (w > 0 && w < min_w) { min_w = w; min_wd = d; }
        if (w > max_w) { max_w = w; max_wd = d; }
    }
    fclose(f);

    printf("========================================\n");
    printf("RESULTS\n");
    printf("max_d:            %llu\n", (unsigned long long)maxd);
    printf("DFS nodes:        %llu\n", (unsigned long long)nodes);
    printf("Time:             %.1f s (%.0f nodes/s)\n", el, nodes / el);
    printf("Total words:      %llu\n", (unsigned long long)total_w);
    printf("Total fractions:  %llu\n", (unsigned long long)total_f);
    printf("Zero R_words:     %llu   (Zaremba A=5: expect 0)\n", (unsigned long long)zeros);
    printf("Min R_words:      %llu at d=%llu\n", (unsigned long long)min_w, (unsigned long long)min_wd);
    printf("Max R_words:      %llu at d=%llu\n", (unsigned long long)max_w, (unsigned long long)max_wd);
    printf("Spot values (d: R_words, R_fracs):\n");
    uint64_t spots[] = {1, 2, 5, 13, 100, 1000, 10000, 100000, 1000000};
    for (int i = 0; i < 9; i++) {
        uint64_t d = spots[i];
        if (d <= maxd)
            printf("  %8llu: %u, %u\n", (unsigned long long)d, cnt_words[d], cnt_fracs[d]);
    }
    printf("Output: %s\n", fn);
    printf("========================================\n");

    free(cnt_words); free(cnt_fracs);
    return zeros > 0 ? 1 : 0;
}
