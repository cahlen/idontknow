/*
 * witness_minimal.c — minimal Zaremba witnesses for A = {1..5}.
 *
 * For every denominator d <= max_d, finds the minimal numerator a with
 * gcd(a,d) = 1 such that a/d = [0;a1,...,ak] with all ai <= 5, by a single
 * DFS over admissible CF words tracking both convergent continuants:
 *
 *   q_new = a*q + q_prev   (denominator, from (q,q_prev) = (1,0))
 *   p_new = a*p + p_prev   (numerator,   from (p,p_prev) = (0,1))
 *
 * Consecutive convergents are automatically coprime. For each d we record:
 *   - alpha(d): the minimal admissible numerator
 *   - the first two digits (a1,a2) of a word achieving it
 *   - the minimal max-digit over words achieving (alpha(d), d)
 *
 * Backs the "witness distribution" finding: this is the raw artifact whose
 * absence was flagged in the 2026-07-21 audit.
 *
 * Compile: cc -O3 -o witness_minimal witness_minimal.c
 * Run:     ./witness_minimal 100000
 * Output:  witnesses_<max_d>.csv  (d,alpha,ratio,a1,a2,max_digit)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t *best_a;      /* minimal numerator per d */
static uint8_t  *best_d1, *best_d2, *best_maxdig;
static uint64_t maxd, nodes;

static void dfs(uint64_t q, uint64_t qp, uint64_t p, uint64_t pp,
                int d1, int d2, int mx, int depth) {
    for (int a = 1; a <= 5; a++) {
        uint64_t qn = (uint64_t)a * q + qp;
        if (qn > maxd) break;
        uint64_t pn = (uint64_t)a * p + pp;
        int nd1 = depth == 0 ? a : d1;
        int nd2 = depth == 1 ? a : d2;
        int nmx = a > mx ? a : mx;
        nodes++;
        if (pn < best_a[qn] ||
            (pn == best_a[qn] && nmx < best_maxdig[qn])) {
            best_a[qn] = (uint32_t)pn;
            best_d1[qn] = (uint8_t)nd1;
            best_d2[qn] = (uint8_t)(depth >= 1 ? nd2 : 0);
            best_maxdig[qn] = (uint8_t)nmx;
        }
        dfs(qn, q, pn, p, nd1, nd2, nmx, depth + 1);
    }
}

int main(int argc, char **argv) {
    maxd = argc > 1 ? (uint64_t)atoll(argv[1]) : 100000ULL;

    printf("Minimal Zaremba witnesses, digit set {1..5}, d <= %llu\n\n",
           (unsigned long long)maxd);

    best_a = malloc((maxd + 1) * sizeof(uint32_t));
    best_d1 = calloc(maxd + 1, 1);
    best_d2 = calloc(maxd + 1, 1);
    best_maxdig = calloc(maxd + 1, 1);
    if (!best_a || !best_d1 || !best_d2 || !best_maxdig) return 2;
    memset(best_a, 0xff, (maxd + 1) * sizeof(uint32_t));
    memset(best_maxdig, 0xff, maxd + 1);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dfs(1, 0, 0, 1, 0, 0, 0, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    char fn[256];
    snprintf(fn, sizeof(fn), "witnesses_%llu.csv", (unsigned long long)maxd);
    FILE *f = fopen(fn, "w");
    fprintf(f, "d,alpha,ratio,a1,a2,max_digit\n");

    uint64_t missing = 0, n = 0;
    double sum_ratio = 0;
    uint64_t prefix51 = 0, gt1000 = 0, prefix51_gt1000 = 0;
    uint64_t maxdig_hist[6] = {0};
    for (uint64_t d = 1; d <= maxd; d++) {
        if (best_a[d] == UINT32_MAX) { missing++; continue; }
        double r = (double)best_a[d] / (double)d;
        fprintf(f, "%llu,%u,%.8f,%u,%u,%u\n", (unsigned long long)d,
                best_a[d], r, best_d1[d], best_d2[d], best_maxdig[d]);
        n++; sum_ratio += r;
        maxdig_hist[best_maxdig[d]]++;
        if (d > 1000) {
            gt1000++;
            if (best_d1[d] == 5 && best_d2[d] == 1) prefix51_gt1000++;
        }
        if (best_d1[d] == 5 && best_d2[d] == 1) prefix51++;
    }
    fclose(f);

    printf("========================================\n");
    printf("RESULTS\n");
    printf("DFS nodes:              %llu\n", (unsigned long long)nodes);
    printf("Time:                   %.1f s\n", el);
    printf("Denominators covered:   %llu / %llu (missing: %llu)\n",
           (unsigned long long)n, (unsigned long long)maxd,
           (unsigned long long)missing);
    printf("Mean alpha(d)/d:        %.6f\n", sum_ratio / n);
    printf("Prefix [0;5,1,...]:     %llu / %llu = %.4f%% (all d)\n",
           (unsigned long long)prefix51, (unsigned long long)n,
           100.0 * prefix51 / n);
    printf("Prefix [0;5,1,...]:     %llu / %llu = %.4f%% (d > 1000)\n",
           (unsigned long long)prefix51_gt1000, (unsigned long long)gt1000,
           100.0 * prefix51_gt1000 / gt1000);
    printf("Max digit of minimal witness:\n");
    for (int m = 1; m <= 5; m++)
        printf("  max_digit = %d: %llu (%.4f%%)\n", m,
               (unsigned long long)maxdig_hist[m], 100.0 * maxdig_hist[m] / n);
    printf("Output: %s\n", fn);
    printf("========================================\n");

    return missing > 0 ? 1 : 0;
}
