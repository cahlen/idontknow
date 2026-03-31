/*
 * Zaremba density computation for arbitrary digit sets
 *
 * For digit set A ⊆ {1,...,K}, enumerate all CFs [0; a1, a2, ...]
 * with ai ∈ A. Mark each denominator in a bitset. Count the fraction
 * of integers d ≤ N that are covered.
 *
 * This answers: for what fraction of integers d does there exist a
 * coprime a/d with all partial quotients in A?
 *
 * Key question: does A = {1,2,3} give full density?
 * Zaremba's conjecture says A = {1,...,5} gives density 1.
 * What about smaller sets?
 *
 * Based on zaremba_verify_v4.c (inverse CF construction).
 *
 * Compile: gcc -O3 -o zaremba_density zaremba_density.c -lm
 * Run:     ./zaremba_density <max_d> <digits>
 * Example: ./zaremba_density 1000000000 1,2,3
 *          ./zaremba_density 1000000000 1,2,3,4,5
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long long uint64;

#define MAX_DIGITS 20

static uint8_t *bitset = NULL;
static uint64 max_d_global = 0;
static uint64 total_marked = 0;

static int digits[MAX_DIGITS];
static int num_digits = 0;

static inline void mark(uint64 d) {
    if (d < 1 || d > max_d_global) return;
    uint64 byte = d >> 3;
    uint8_t bit = 1 << (d & 7);
    if (!(bitset[byte] & bit)) {
        __sync_fetch_and_or(&bitset[byte], bit);
        __sync_fetch_and_add(&total_marked, 1);
    }
}

static inline int is_marked(uint64 d) {
    if (d < 1 || d > max_d_global) return 0;
    return (bitset[d >> 3] >> (d & 7)) & 1;
}

void enumerate(uint64 p_prev, uint64 p, uint64 q_prev, uint64 q,
               int depth, int max_depth) {
    mark(q);

    if (depth >= max_depth) return;

    for (int i = 0; i < num_digits; i++) {
        int a = digits[i];
        uint64 q_new = (uint64)a * q + q_prev;
        if (q_new > max_d_global) break;  // digits are sorted, larger ones give larger q

        uint64 p_new = (uint64)a * p + p_prev;
        enumerate(p, p_new, q, q_new, depth + 1, max_depth);
    }
}

void parse_digits(const char *s) {
    num_digits = 0;
    char buf[256];
    strncpy(buf, s, 255);
    char *tok = strtok(buf, ",");
    while (tok && num_digits < MAX_DIGITS) {
        digits[num_digits++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    // Sort ascending
    for (int i = 0; i < num_digits - 1; i++)
        for (int j = i + 1; j < num_digits; j++)
            if (digits[i] > digits[j]) {
                int t = digits[i]; digits[i] = digits[j]; digits[j] = t;
            }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <max_d> <digits>\n", argv[0]);
        fprintf(stderr, "  max_d: scan denominators from 1 to max_d\n");
        fprintf(stderr, "  digits: comma-separated digit set (e.g., 1,2,3)\n");
        fprintf(stderr, "\nExamples:\n");
        fprintf(stderr, "  %s 1000000 1,2,3        # Zaremba density for A={1,2,3}\n", argv[0]);
        fprintf(stderr, "  %s 1000000000 1,2,3,4,5  # Zaremba density for A={1,...,5}\n", argv[0]);
        return 1;
    }

    max_d_global = (uint64)atoll(argv[1]);
    parse_digits(argv[2]);
    int max_len = argc > 3 ? atoi(argv[3]) : 200;

    printf("========================================\n");
    printf("Zaremba Density Computation\n");
    printf("========================================\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d_global);
    printf("Digit set A = {");
    for (int i = 0; i < num_digits; i++)
        printf("%s%d", i ? "," : "", digits[i]);
    printf("}\n");
    printf("Max CF length: %d\n", max_len);

    uint64 bitset_bytes = (max_d_global + 8) / 8;
    printf("Bitset: %.2f GB\n\n", bitset_bytes / 1e9);

    bitset = (uint8_t *)calloc(bitset_bytes, 1);
    if (!bitset) {
        printf("ERROR: cannot allocate %.2f GB\n", bitset_bytes / 1e9);
        return 1;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    mark(1);  // d=1 is trivially covered

    printf("Enumerating CFs...\n");
    for (int i = 0; i < num_digits; i++) {
        int a1 = digits[i];
        printf("  Tree a1=%d...", a1);
        fflush(stdout);
        uint64 before = total_marked;
        enumerate(0, 1, 1, (uint64)a1, 1, max_len);
        printf(" (+%llu, total %llu)\n",
               (unsigned long long)(total_marked - before),
               (unsigned long long)total_marked);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double enum_time = (t_end.tv_sec - t_start.tv_sec) +
                       (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    printf("Enumeration: %.1fs\n\n", enum_time);

    // Compute density at various checkpoints
    printf("Density by range:\n");
    printf("%-20s %-15s %-15s %-10s\n", "Range", "Covered", "Total", "Density");

    uint64 covered = 0;
    uint64 checkpoints[] = {
        100, 1000, 10000, 100000, 1000000,
        10000000, 100000000, 1000000000,
        10000000000ULL, 100000000000ULL, 0
    };

    int cp_idx = 0;
    for (uint64 d = 1; d <= max_d_global; d++) {
        if (is_marked(d)) covered++;

        if (checkpoints[cp_idx] && d == checkpoints[cp_idx]) {
            printf("d ≤ %-16llu %-15llu %-15llu %.6f%%\n",
                   (unsigned long long)d,
                   (unsigned long long)covered,
                   (unsigned long long)d,
                   100.0 * covered / d);
            cp_idx++;
        }
    }

    // Final
    double density = 100.0 * covered / max_d_global;

    // Count uncovered
    uint64 uncovered = max_d_global - covered;
    uint64 first_uncovered[20];
    int uc_count = 0;
    if (uncovered > 0 && uncovered <= 10000) {
        for (uint64 d = 1; d <= max_d_global && uc_count < 20; d++) {
            if (!is_marked(d))
                first_uncovered[uc_count++] = d;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_time = (t_end.tv_sec - t_start.tv_sec) +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("Digit set: A = {");
    for (int i = 0; i < num_digits; i++)
        printf("%s%d", i ? "," : "", digits[i]);
    printf("}\n");
    printf("Range: d = 1 to %llu\n", (unsigned long long)max_d_global);
    printf("Covered: %llu / %llu\n",
           (unsigned long long)covered, (unsigned long long)max_d_global);
    printf("Density: %.8f%%\n", density);
    printf("Uncovered: %llu\n", (unsigned long long)uncovered);

    if (uc_count > 0) {
        printf("First uncovered d:");
        for (int i = 0; i < uc_count; i++)
            printf(" %llu", (unsigned long long)first_uncovered[i]);
        printf("\n");
    }

    if (uncovered == 0)
        printf("\n*** ALL d covered — full density for this range ***\n");

    printf("Time: %.1fs (enum: %.1fs)\n", total_time, enum_time);
    printf("========================================\n");

    // Save results to log file
    char logpath[256];
    snprintf(logpath, 256,
             "scripts/experiments/zaremba-density/results/density_A");
    for (int i = 0; i < num_digits; i++)
        snprintf(logpath + strlen(logpath), 256 - strlen(logpath), "%d", digits[i]);
    snprintf(logpath + strlen(logpath), 256 - strlen(logpath),
             "_%llu.log", (unsigned long long)max_d_global);

    FILE *log = fopen(logpath, "w");
    if (log) {
        fprintf(log, "digit_set: ");
        for (int i = 0; i < num_digits; i++)
            fprintf(log, "%s%d", i ? "," : "", digits[i]);
        fprintf(log, "\nmax_d: %llu\n", (unsigned long long)max_d_global);
        fprintf(log, "covered: %llu\n", (unsigned long long)covered);
        fprintf(log, "density: %.10f\n", (double)covered / max_d_global);
        fprintf(log, "uncovered: %llu\n", (unsigned long long)uncovered);
        fprintf(log, "enum_time: %.1f\n", enum_time);
        fprintf(log, "total_time: %.1f\n", total_time);
        fclose(log);
        printf("Log: %s\n", logpath);
    }

    free(bitset);
    return 0;
}
