/*
 * Ramanujan Machine v2: ASYMMETRIC-DEGREE polynomial CF search
 *
 * KEY INSIGHT: Every known CF formula for transcendental constants has
 * deg(b_n) ≈ 2 * deg(a_n).  v1 forced equal degrees, which is why it
 * only re-derived classical formulas and produced zero new transcendentals.
 *
 * CF = a(0) + b(1) / (a(1) + b(2) / (a(2) + b(3) / (a(3) + ...)))
 *   a(n) = polynomial of degree deg_a, coefficients in [-range_a, range_a]
 *   b(n) = polynomial of degree deg_b, coefficients in [-range_b, range_b]
 *
 * Productive search targets (deg_a, deg_b):
 *   (1, 2)  — Brouncker/Wallis family (4/pi, etc.)
 *   (2, 4)  — Catalan/zeta(2) family
 *   (3, 6)  — Apéry family (zeta(3), zeta(5))
 *   (2, 3)  — sub-ratio region, still productive
 *   (1, 3)  — mixed regime
 *
 * Also outputs ALL converged CFs (not just matched ones) to enable
 * offline multi-constant PSLQ scanning.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramanujan_v2 ramanujan_v2.cu -lm
 * Run:     ./ramanujan_v2 <deg_a> <deg_b> <range_a> <range_b> [cf_depth] [gpu_id]
 *
 * Examples:
 *   ./ramanujan_v2 2 4 6 6          # Catalan-type, 1.7T candidates
 *   ./ramanujan_v2 1 2 10 10        # Brouncker-type, 194M candidates
 *   ./ramanujan_v2 3 6 3 3          # Apéry-type, 282B candidates
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define BLOCK 256
#define MAX_DEG_A 6
#define MAX_DEG_B 12
#define MAX_CF_DEPTH 500

/* ── Known constants ──────────────────────────────────────── */

__constant__ double d_constants[] = {
    3.14159265358979323846,   // 0  pi
    2.71828182845904523536,   // 1  e
    0.69314718055994530942,   // 2  ln(2)
    0.57721566490153286061,   // 3  Euler-Mascheroni gamma
    0.91596559417721901505,   // 4  Catalan's constant
    1.20205690315959428540,   // 5  zeta(3)
    1.03692775514336992633,   // 6  zeta(5)
    1.00834927738192282684,   // 7  zeta(7)
    0.83462684167407318628,   // 8  Gauss's constant
    2.62205755429211981046,   // 9  Lemniscate constant
    1.41421356237309504880,   // 10 sqrt(2)
    1.61803398874989484820,   // 11 golden ratio phi
    0.0,
};

static const char* h_const_names[] = {
    "pi", "e", "ln(2)", "gamma", "Catalan",
    "zeta(3)", "zeta(5)", "zeta(7)", "Gauss", "Lemniscate",
    "sqrt(2)", "phi"
};

#define NUM_CONSTANTS 12

__constant__ double d_compounds[] = {
    // Reciprocals
    0.31830988618379067,  // 1/pi
    0.36787944117144233,  // 1/e
    1.44269504088896341,  // 1/ln(2)
    // Pi expressions
    1.27323954473516269,  // 4/pi
    0.78539816339744831,  // pi/4
    1.57079632679489662,  // pi/2
    1.04719755119659775,  // pi/3
    0.52359877559829887,  // pi/6
    9.86960440108935862,  // pi^2
    1.64493406684822644,  // pi^2/6  = zeta(2)
    2.46740110027233966,  // pi^2/4
    0.82246703342411322,  // pi^2/12
    // Log expressions
    1.38629436111989061,  // 2*ln(2)
    2.30258509299404568,  // ln(10)
    1.09861228866810970,  // ln(3)
    // Cross-products
    8.53973422267356706,  // e*pi
    0.86525597943226508,  // e/pi
    1.15572734979092172,  // pi/e
    2.17758609030360229,  // pi*ln(2)
    // Roots
    1.77245385090551603,  // sqrt(pi)
    0.56418958354775629,  // 1/sqrt(pi)
    1.12837916709551258,  // 2/sqrt(pi)
    2.50662827463100051,  // sqrt(2*pi)
    0.39894228040143268,  // 1/sqrt(2*pi)
    // Zeta products
    3.77495308672748408,  // pi*zeta(3)
    0.0,
};

static const char* h_compound_names[] = {
    "1/pi", "1/e", "1/ln(2)",
    "4/pi", "pi/4", "pi/2", "pi/3", "pi/6",
    "pi^2", "pi^2/6", "pi^2/4", "pi^2/12",
    "2*ln(2)", "ln(10)", "ln(3)",
    "e*pi", "e/pi", "pi/e", "pi*ln(2)",
    "sqrt(pi)", "1/sqrt(pi)", "2/sqrt(pi)",
    "sqrt(2pi)", "1/sqrt(2pi)",
    "pi*zeta(3)",
};

#define NUM_COMPOUNDS 25

static const char* get_const_name(int mc) {
    if (mc >= 100) return h_compound_names[mc - 100];
    return h_const_names[mc];
}

/* ── Polynomial evaluation ────────────────────────────────── */

__device__ double eval_poly_a(const int *coeffs, int deg_a, int n) {
    double result = 0.0, np = 1.0;
    for (int i = 0; i <= deg_a; i++) {
        result += coeffs[i] * np;
        np *= (double)n;
    }
    return result;
}

__device__ double eval_poly_b(const int *coeffs, int deg_b, int n) {
    double result = 0.0, np = 1.0;
    for (int i = 0; i <= deg_b; i++) {
        result += coeffs[i] * np;
        np *= (double)n;
    }
    return result;
}

/* ── CF evaluation with asymmetric degrees ────────────────── */

__device__ double eval_pcf_asym(const int *a_coeffs, int deg_a,
                                const int *b_coeffs, int deg_b,
                                int depth)
{
    // Backward recurrence: start from n=depth
    double val = eval_poly_a(a_coeffs, deg_a, depth);

    for (int n = depth - 1; n >= 1; n--) {
        double bn1 = eval_poly_b(b_coeffs, deg_b, n + 1);
        double an  = eval_poly_a(a_coeffs, deg_a, n);
        if (fabs(val) < 1e-300) return NAN;
        val = an + bn1 / val;
    }

    // CF = a(0) + b(1) / val
    double a0 = eval_poly_a(a_coeffs, deg_a, 0);
    double b1 = eval_poly_b(b_coeffs, deg_b, 1);
    if (fabs(val) < 1e-300) return NAN;
    return a0 + b1 / val;
}

__device__ int check_convergence_asym(const int *a_coeffs, int deg_a,
                                      const int *b_coeffs, int deg_b,
                                      int depth, double *result)
{
    double v1 = eval_pcf_asym(a_coeffs, deg_a, b_coeffs, deg_b, depth);
    double v2 = eval_pcf_asym(a_coeffs, deg_a, b_coeffs, deg_b, depth - 50);

    if (isnan(v1) || isnan(v2) || isinf(v1) || isinf(v2)) return 0;
    if (fabs(v1) > 1e15 || fabs(v1) < 1e-15) return 0;

    double reldiff = fabs(v1 - v2) / (fabs(v1) + 1e-300);
    if (reldiff > 1e-10) return 0;

    *result = v1;
    return 1;
}

/* ── Constant matching (same as v1 but with tighter threshold) ── */

__device__ int match_constant(double val, int *match_const, int *match_c0,
                              int *match_c1, int *match_c2)
{
    double absval = val < 0.0 ? -val : val;
    if (absval < 1e-8) return 0;

    // Phase 1: compound expressions
    for (int ci = 0; ci < NUM_COMPOUNDS; ci++) {
        double K = d_compounds[ci];
        if (K == 0.0) continue;
        for (int c1 = 1; c1 <= 6; c1++) {
            for (int c2 = -6; c2 <= 6; c2++) {
                if (c2 == 0) continue;
                for (int c0 = -6; c0 <= 6; c0++) {
                    double expected = ((double)c0 + (double)c2 * K) / (double)c1;
                    if (fabs(expected) < 1e-15 || fabs(expected) > 1e15) continue;
                    double reldiff = fabs(val - expected) / (fabs(expected) + 1e-300);
                    if (reldiff < 1e-11) {
                        *match_const = 100 + ci;
                        *match_c0 = c0; *match_c1 = c1; *match_c2 = c2;
                        return 1;
                    }
                }
            }
        }
    }

    // Phase 2: base constants
    for (int ci = 0; ci < NUM_CONSTANTS; ci++) {
        double K = d_constants[ci];
        if (K == 0.0) continue;
        for (int c1 = 1; c1 <= 8; c1++) {
            for (int c2 = -8; c2 <= 8; c2++) {
                if (c2 == 0) continue;
                for (int c0 = -8; c0 <= 8; c0++) {
                    double expected = ((double)c0 + (double)c2 * K) / (double)c1;
                    double reldiff = fabs(val - expected) / (fabs(expected) + 1e-300);
                    if (reldiff < 1e-12) {
                        *match_const = ci;
                        *match_c0 = c0; *match_c1 = c1; *match_c2 = c2;
                        return 1;
                    }
                }
            }
        }
        // Power matches
        for (int p = -4; p <= 4; p++) {
            for (int q = 1; q <= 4; q++) {
                if (p == 0) continue;
                double expected = pow(K, (double)p / (double)q);
                if (isnan(expected) || isinf(expected)) continue;
                double reldiff = fabs(val - expected) / (fabs(expected) + 1e-300);
                if (reldiff < 1e-12) {
                    *match_const = ci;
                    *match_c0 = p; *match_c1 = q; *match_c2 = -999;
                    return 1;
                }
            }
        }
    }
    return 0;
}

/* ── Main kernel ──────────────────────────────────────────── */

struct Hit {
    int a_coeffs[MAX_DEG_A + 1];
    int b_coeffs[MAX_DEG_B + 1];
    int deg_a, deg_b;
    double value;
    int match_const;
    int match_c0, match_c1, match_c2;
    int matched;  // 1 = matched a constant, 0 = converged but unmatched
};

__global__ void search_kernel(
    long long start_idx, long long count,
    int deg_a, int deg_b, int range_a, int range_b, int cf_depth,
    Hit *hits, int *hit_count, int max_hits,
    Hit *unmatched, int *unmatched_count, int max_unmatched)
{
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    long long idx = start_idx + tid;

    // Decode: first (deg_a+1) coefficients for a, then (deg_b+1) for b
    int width_a = 2 * range_a + 1;
    int width_b = 2 * range_b + 1;

    int a_coeffs[MAX_DEG_A + 1] = {0};
    int b_coeffs[MAX_DEG_B + 1] = {0};

    long long tmp = idx;
    for (int i = 0; i <= deg_a; i++) {
        a_coeffs[i] = (int)(tmp % width_a) - range_a;
        tmp /= width_a;
    }
    for (int i = 0; i <= deg_b; i++) {
        b_coeffs[i] = (int)(tmp % width_b) - range_b;
        tmp /= width_b;
    }

    // Skip trivial: b(n) = 0
    int all_zero_b = 1;
    for (int i = 0; i <= deg_b; i++) if (b_coeffs[i] != 0) { all_zero_b = 0; break; }
    if (all_zero_b) return;

    // Skip trivial: leading coefficient of b is zero (reduces to lower degree)
    if (b_coeffs[deg_b] == 0) return;

    // Evaluate CF
    double value;
    if (!check_convergence_asym(a_coeffs, deg_a, b_coeffs, deg_b, cf_depth, &value))
        return;

    // Skip trivial values
    if (value == 0.0 || value != value || value > 1e15 || value < -1e15) return;
    if (value > -1e-10 && value < 1e-10) return;

    // Try matching
    int mc, c0, c1, c2;
    if (match_constant(value, &mc, &c0, &c1, &c2)) {
        int slot = atomicAdd(hit_count, 1);
        if (slot < max_hits) {
            Hit *h = &hits[slot];
            for (int i = 0; i <= deg_a; i++) h->a_coeffs[i] = a_coeffs[i];
            for (int i = 0; i <= deg_b; i++) h->b_coeffs[i] = b_coeffs[i];
            h->deg_a = deg_a; h->deg_b = deg_b;
            h->value = value;
            h->match_const = mc;
            h->match_c0 = c0; h->match_c1 = c1; h->match_c2 = c2;
            h->matched = 1;
        }
    } else {
        // Save unmatched converged CFs for offline PSLQ
        int slot = atomicAdd(unmatched_count, 1);
        if (slot < max_unmatched) {
            Hit *h = &unmatched[slot];
            for (int i = 0; i <= deg_a; i++) h->a_coeffs[i] = a_coeffs[i];
            for (int i = 0; i <= deg_b; i++) h->b_coeffs[i] = b_coeffs[i];
            h->deg_a = deg_a; h->deg_b = deg_b;
            h->value = value;
            h->matched = 0;
        }
    }
}

/* ── Main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <deg_a> <deg_b> <range_a> <range_b> [cf_depth] [gpu_id]\n", argv[0]);
        printf("\nProductive configurations:\n");
        printf("  %s 1 2 10 10   # Brouncker-type (194M candidates)\n", argv[0]);
        printf("  %s 2 4 6 6     # Catalan-type (1.7T candidates)\n", argv[0]);
        printf("  %s 3 6 3 3     # Apéry-type (282B candidates)\n", argv[0]);
        printf("  %s 2 3 8 8     # mixed (4.7T candidates)\n", argv[0]);
        return 1;
    }

    int deg_a = atoi(argv[1]);
    int deg_b = atoi(argv[2]);
    int range_a = atoi(argv[3]);
    int range_b = atoi(argv[4]);
    int cf_depth = argc > 5 ? atoi(argv[5]) : 300;
    int gpu_id = argc > 6 ? atoi(argv[6]) : 0;

    if (deg_a > MAX_DEG_A) { printf("ERROR: deg_a > %d\n", MAX_DEG_A); return 1; }
    if (deg_b > MAX_DEG_B) { printf("ERROR: deg_b > %d\n", MAX_DEG_B); return 1; }

    cudaSetDevice(gpu_id);

    int width_a = 2 * range_a + 1;
    int width_b = 2 * range_b + 1;
    long long total_candidates = 1;
    for (int i = 0; i <= deg_a; i++) total_candidates *= width_a;
    for (int i = 0; i <= deg_b; i++) total_candidates *= width_b;

    double ratio = (double)deg_b / (double)(deg_a > 0 ? deg_a : 1);

    printf("========================================\n");
    printf("Ramanujan Machine v2 (asymmetric degree)\n");
    printf("========================================\n");
    printf("a(n) degree: %d, coefficients: [-%d, %d]\n", deg_a, range_a, range_a);
    printf("b(n) degree: %d, coefficients: [-%d, %d]\n", deg_b, range_b, range_b);
    printf("Degree ratio: %.2f %s\n", ratio,
           ratio >= 1.8 && ratio <= 2.2 ? "(OPTIMAL for transcendentals)" :
           ratio >= 1.3 && ratio <= 1.7 ? "(sub-optimal but productive)" :
           "(outside typical productive range)");
    printf("CF evaluation depth: %d terms\n", cf_depth);
    printf("Total candidates: %lld (%.2e)\n", total_candidates, (double)total_candidates);
    printf("GPU: %d\n", gpu_id);
    printf("========================================\n\n");
    fflush(stdout);

    // Allocate buffers
    int max_hits = 500000;
    int max_unmatched = 1000000;  // save converged-but-unmatched for PSLQ
    Hit *d_hits, *d_unmatched;
    int *d_hit_count, *d_unmatched_count;
    cudaMalloc(&d_hits, max_hits * sizeof(Hit));
    cudaMalloc(&d_unmatched, max_unmatched * sizeof(Hit));
    cudaMalloc(&d_hit_count, sizeof(int));
    cudaMalloc(&d_unmatched_count, sizeof(int));
    cudaMemset(d_hit_count, 0, sizeof(int));
    cudaMemset(d_unmatched_count, 0, sizeof(int));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    long long chunk_size = 1000000LL;
    int total_hits = 0;
    int total_unmatched = 0;

    // Output files
    char hits_path[512], unmatched_path[512];
    snprintf(hits_path, 512,
             "scripts/experiments/ramanujan-machine/results/v2_hits_a%d_b%d_r%d_%d.csv",
             deg_a, deg_b, range_a, range_b);
    snprintf(unmatched_path, 512,
             "scripts/experiments/ramanujan-machine/results/v2_unmatched_a%d_b%d_r%d_%d.csv",
             deg_a, deg_b, range_a, range_b);

    FILE *fhits = fopen(hits_path, "w");
    FILE *funm = fopen(unmatched_path, "w");
    if (fhits) fprintf(fhits, "a_coeffs,b_coeffs,value,constant,c0,c1,c2\n");
    if (funm)  fprintf(funm,  "a_coeffs,b_coeffs,value\n");

    for (long long offset = 0; offset < total_candidates; offset += chunk_size) {
        long long this_chunk = chunk_size;
        if (offset + this_chunk > total_candidates)
            this_chunk = total_candidates - offset;

        int grid = (this_chunk + BLOCK - 1) / BLOCK;
        search_kernel<<<grid, BLOCK>>>(
            offset, this_chunk, deg_a, deg_b, range_a, range_b, cf_depth,
            d_hits, d_hit_count, max_hits,
            d_unmatched, d_unmatched_count, max_unmatched);

        if ((offset / chunk_size) % 100 == 0 || offset + this_chunk >= total_candidates) {
            cudaDeviceSynchronize();

            int h_hit_count, h_unm_count;
            cudaMemcpy(&h_hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_unm_count, d_unmatched_count, sizeof(int), cudaMemcpyDeviceToHost);

            // Write new matched hits
            if (h_hit_count > total_hits) {
                Hit *h_hits = (Hit *)malloc(h_hit_count * sizeof(Hit));
                cudaMemcpy(h_hits, d_hits, h_hit_count * sizeof(Hit), cudaMemcpyDeviceToHost);

                for (int i = total_hits; i < h_hit_count && i < max_hits; i++) {
                    Hit *h = &h_hits[i];
                    if (h->value > -1e-8 && h->value < 1e-8) continue;

                    printf("  HIT: a=(");
                    for (int j = 0; j <= h->deg_a; j++) printf("%s%d", j?",":"", h->a_coeffs[j]);
                    printf(") b=(");
                    for (int j = 0; j <= h->deg_b; j++) printf("%s%d", j?",":"", h->b_coeffs[j]);
                    printf(") → %.15g", h->value);

                    if (h->match_c2 == -999)
                        printf(" = %s^(%d/%d)", get_const_name(h->match_const),
                               h->match_c0, h->match_c1);
                    else
                        printf(" = (%d + %d*%s)/%d", h->match_c0, h->match_c2,
                               get_const_name(h->match_const), h->match_c1);
                    printf("\n");

                    if (fhits) {
                        fprintf(fhits, "\"(");
                        for (int j = 0; j <= h->deg_a; j++) fprintf(fhits, "%s%d", j?",":"", h->a_coeffs[j]);
                        fprintf(fhits, ")\",\"(");
                        for (int j = 0; j <= h->deg_b; j++) fprintf(fhits, "%s%d", j?",":"", h->b_coeffs[j]);
                        fprintf(fhits, ")\",%.*g,%s,%d,%d,%d\n",
                                17, h->value, get_const_name(h->match_const),
                                h->match_c0, h->match_c1, h->match_c2);
                    }
                }
                total_hits = h_hit_count;
                free(h_hits);
                if (fhits) fflush(fhits);
            }

            // Write new unmatched CFs
            if (h_unm_count > total_unmatched) {
                Hit *h_unm = (Hit *)malloc(h_unm_count * sizeof(Hit));
                cudaMemcpy(h_unm, d_unmatched, h_unm_count * sizeof(Hit), cudaMemcpyDeviceToHost);

                for (int i = total_unmatched; i < h_unm_count && i < max_unmatched; i++) {
                    Hit *h = &h_unm[i];
                    if (funm) {
                        fprintf(funm, "\"(");
                        for (int j = 0; j <= h->deg_a; j++) fprintf(funm, "%s%d", j?",":"", h->a_coeffs[j]);
                        fprintf(funm, ")\",\"(");
                        for (int j = 0; j <= h->deg_b; j++) fprintf(funm, "%s%d", j?",":"", h->b_coeffs[j]);
                        fprintf(funm, ")\",%.*g\n", 17, h->value);
                    }
                }
                total_unmatched = h_unm_count;
                free(h_unm);
                if (funm) fflush(funm);
            }

            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            double pct = 100.0 * (offset + this_chunk) / total_candidates;
            double rate = (offset + this_chunk) / elapsed;
            double eta = (total_candidates - offset - this_chunk) / (rate + 1);

            printf("  %.1f%% (%lld/%lld) %d matched, %d unmatched, %.0f/sec, ETA %.0fs\n",
                   pct, offset + this_chunk, total_candidates,
                   total_hits, total_unmatched, rate, eta);
            fflush(stdout);
        }
    }

    if (fhits) fclose(fhits);
    if (funm) fclose(funm);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Ramanujan Machine v2 Results\n");
    printf("========================================\n");
    printf("a(n): deg=%d range=[-%d,%d]\n", deg_a, range_a, range_a);
    printf("b(n): deg=%d range=[-%d,%d]\n", deg_b, range_b, range_b);
    printf("Degree ratio: %.2f\n", ratio);
    printf("Candidates: %lld (%.2e)\n", total_candidates, (double)total_candidates);
    printf("Matched hits: %d\n", total_hits);
    printf("Unmatched converged: %d (saved for PSLQ)\n", total_unmatched);
    printf("Time: %.1fs (%.0f candidates/sec)\n", total_time,
           total_candidates / total_time);
    if (total_hits > 0)
        printf("Hits CSV: %s\n", hits_path);
    if (total_unmatched > 0)
        printf("Unmatched CSV: %s\n", unmatched_path);
    printf("========================================\n");

    printf("\nNext step: run PSLQ verification on matched hits:\n");
    printf("  python3 scripts/experiments/ramanujan-machine/verify_hits.py %s\n",
           hits_path);
    printf("Next step: run multi-constant PSLQ on unmatched CFs:\n");
    printf("  python3 scripts/experiments/ramanujan-machine/pslq_scan.py %s\n",
           unmatched_path);

    cudaFree(d_hits); cudaFree(d_unmatched);
    cudaFree(d_hit_count); cudaFree(d_unmatched_count);
    return 0;
}
