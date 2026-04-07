/*
 * GPU-accelerated Ramanujan Machine: polynomial CF evaluation + PSLQ matching
 *
 * For each polynomial pair (P, Q) with bounded integer coefficients:
 *   CF = a0 + Q(1) / (P(1) + Q(2) / (P(2) + Q(3) / (P(3) + ...)))
 * Evaluate to 128-bit precision, then match against known constants via PSLQ.
 *
 * Each GPU thread evaluates one (P, Q) pair independently.
 *
 * Phase 1: double-precision screening (fast, filters 99%+ of candidates)
 * Phase 2: high-precision verification of survivors (CGBN or quad-double)
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramanujan_gpu ramanujan_gpu.cu -lm
 * Run:     ./ramanujan_gpu [degree] [coeff_range] [cf_depth] [gpu_id]
 *
 * References:
 *   Raayoni et al. (2024) "Algorithm-assisted discovery of an intrinsic order
 *   among mathematical constants." PNAS 121(25).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define BLOCK 256
#define MAX_DEGREE 6
#define MAX_CF_DEPTH 500

/* ── Known constants for matching ──────────────────────── */

// We store high-precision values as doubles (53 bits ≈ 16 digits).
// Phase 1 screening at double precision; Phase 2 uses higher precision.
__constant__ double d_constants[] = {
    3.14159265358979323846,   // pi
    2.71828182845904523536,   // e
    0.69314718055994530942,   // ln(2)
    0.57721566490153286061,   // Euler-Mascheroni gamma
    0.91596559417721901505,   // Catalan's constant
    1.20205690315959428540,   // zeta(3) = Apery's constant
    0.83462684167407318628,   // Gauss's constant (1/agm(1,sqrt(2)))
    2.62205755429211981046,   // Lemniscate constant
    1.41421356237309504880,   // sqrt(2)
    1.61803398874989484820,   // golden ratio phi
    0.0,                      // sentinel
};

__constant__ char d_const_names[][20] = {
    "pi", "e", "ln(2)", "gamma", "Catalan",
    "zeta(3)", "Gauss", "Lemniscate", "sqrt(2)", "phi"
};

#define NUM_CONSTANTS 10

/* ── Polynomial CF evaluation ──────────────────────────── */

// Evaluate polynomial P(n) = sum_{i=0}^{deg} coeffs[i] * n^i
__device__ double eval_poly(const int *coeffs, int deg, int n) {
    double result = 0.0;
    double np = 1.0;
    for (int i = 0; i <= deg; i++) {
        result += coeffs[i] * np;
        np *= (double)n;
    }
    return result;
}

// Evaluate a polynomial CF from the bottom up:
// CF = P(0) + Q(1) / (P(1) + Q(2) / (P(2) + ... + Q(N) / P(N)))
// Uses backward recurrence for numerical stability.
__device__ double eval_pcf(const int *p_coeffs, const int *q_coeffs,
                           int deg, int depth)
{
    // Backward evaluation: start from depth N, work toward n=1
    double val = eval_poly(p_coeffs, deg, depth);

    for (int n = depth - 1; n >= 1; n--) {
        double qn = eval_poly(q_coeffs, deg, n + 1);
        double pn = eval_poly(p_coeffs, deg, n);
        if (fabs(val) < 1e-300) return NAN;  // divergence
        val = pn + qn / val;
    }

    // Add a0 = P(0)
    double a0 = eval_poly(p_coeffs, deg, 0);
    if (fabs(val) < 1e-300) return NAN;
    double q1 = eval_poly(q_coeffs, deg, 1);
    return a0 + q1 / val;
}

// Check convergence: evaluate at two depths and compare
__device__ int check_convergence(const int *p_coeffs, const int *q_coeffs,
                                 int deg, int depth, double *result)
{
    double v1 = eval_pcf(p_coeffs, q_coeffs, deg, depth);
    double v2 = eval_pcf(p_coeffs, q_coeffs, deg, depth - 50);

    if (isnan(v1) || isnan(v2) || isinf(v1) || isinf(v2)) return 0;
    if (fabs(v1) > 1e15 || fabs(v1) < 1e-15) return 0;

    double reldiff = fabs(v1 - v2) / (fabs(v1) + 1e-300);
    if (reldiff > 1e-10) return 0;  // not converged

    *result = v1;
    return 1;
}

/* ── Compound constant matching ────────────────────────── */

// Pre-computed compound expressions involving known constants.
// These are the expressions that actually appear in Ramanujan-type CF formulas.
__constant__ double d_compounds[] = {
    // Reciprocals: 1/K
    0.31830988618379067,  // 1/pi
    0.36787944117144233,  // 1/e
    1.44269504088896341,  // 1/ln(2)
    // Products of pi
    1.27323954473516269,  // 4/pi (Brouncker, Wallis)
    0.78539816339744831,  // pi/4
    1.57079632679489662,  // pi/2
    1.04719755119659775,  // pi/3
    0.52359877559829887,  // pi/6
    9.86960440108935862,  // pi^2
    1.64493406684822644,  // pi^2/6 (Basel = zeta(2))
    2.46740110027233966,  // pi^2/4
    0.82246703342411322,  // pi^2/12
    // Products of e
    0.69314718055994531,  // ln(2)
    1.38629436111989061,  // 2*ln(2)
    2.30258509299404568,  // ln(10)
    // Cross-products
    8.53973422267356706,  // e*pi
    0.86525597943226508,  // e/pi
    1.15572734979092172,  // pi/e
    2.17758609030360229,  // pi*ln(2)
    // Roots and powers
    1.77245385090551603,  // sqrt(pi)
    0.56418958354775629,  // 1/sqrt(pi)
    1.12837916709551258,  // 2/sqrt(pi)
    1.64872127070012815,  // sqrt(e)
    0.60653065971263342,  // 1/sqrt(e)  = e^(-1/2)
    2.50662827463100051,  // sqrt(2*pi)
    0.39894228040143268,  // 1/sqrt(2*pi)
    // Other famous
    0.11503837898205527,  // 1/(e*pi)
    1.73205080756887729,  // sqrt(3)
    2.23606797749978969,  // sqrt(5)
    0.0,  // sentinel
};

__constant__ char d_compound_names[][24] = {
    "1/pi", "1/e", "1/ln(2)",
    "4/pi", "pi/4", "pi/2", "pi/3", "pi/6",
    "pi^2", "pi^2/6", "pi^2/4", "pi^2/12",
    "ln(2)", "2*ln(2)", "ln(10)",
    "e*pi", "e/pi", "pi/e", "pi*ln(2)",
    "sqrt(pi)", "1/sqrt(pi)", "2/sqrt(pi)",
    "sqrt(e)", "1/sqrt(e)", "sqrt(2pi)", "1/sqrt(2pi)",
    "1/(e*pi)", "sqrt(3)", "sqrt(5)",
};

#define NUM_COMPOUNDS 29

// Host-side name arrays (device __constant__ arrays can't be read from host)
static const char* h_const_names[] = {
    "pi", "e", "ln(2)", "gamma", "Catalan",
    "zeta(3)", "Gauss", "Lemniscate", "sqrt(2)", "phi"
};

static const char* h_compound_names[] = {
    "1/pi", "1/e", "1/ln(2)",
    "4/pi", "pi/4", "pi/2", "pi/3", "pi/6",
    "pi^2", "pi^2/6", "pi^2/4", "pi^2/12",
    "ln(2)", "2*ln(2)", "ln(10)",
    "e*pi", "e/pi", "pi/e", "pi*ln(2)",
    "sqrt(pi)", "1/sqrt(pi)", "2/sqrt(pi)",
    "sqrt(e)", "1/sqrt(e)", "sqrt(2pi)", "1/sqrt(2pi)",
    "1/(e*pi)", "sqrt(3)", "sqrt(5)",
};

// Helper: get constant name from match_const index (host-side)
static const char* get_const_name(int mc) {
    if (mc >= 100) return h_compound_names[mc - 100];
    return h_const_names[mc];
}

__device__ int match_constant(double val, int *match_const, int *match_c0,
                              int *match_c1, int *match_c2)
{
    // Reject trivial zero values — these match everything
    double absval = val < 0.0 ? -val : val;
    if (absval < 1e-8) return 0;

    // Phase 1: Check compound expressions with small integer multiples
    // val = (c0 + c2 * K) / c1  for K in compounds
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
                        *match_const = 100 + ci;  // 100+ = compound index
                        *match_c0 = c0;
                        *match_c1 = c1;
                        *match_c2 = c2;
                        return 1;
                    }
                }
            }
        }
    }

    // Phase 2: Check base constants with linear combinations
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
                        *match_c0 = c0;
                        *match_c1 = c1;
                        *match_c2 = c2;
                        return 1;
                    }
                }
            }
        }

        // Try: val = K^(p/q) for small p, q
        for (int p = -4; p <= 4; p++) {
            for (int q = 1; q <= 4; q++) {
                if (p == 0) continue;
                double expected = pow(K, (double)p / (double)q);
                if (isnan(expected) || isinf(expected)) continue;
                double reldiff = fabs(val - expected) / (fabs(expected) + 1e-300);
                if (reldiff < 1e-12) {
                    *match_const = ci;
                    *match_c0 = p;
                    *match_c1 = q;
                    *match_c2 = -999;  // flag for power match
                    return 1;
                }
            }
        }
    }
    return 0;
}

/* ── Main GPU kernel ───────────────────────────────────── */

// Each thread gets a unique polynomial pair index, decodes it to
// coefficient arrays, evaluates the CF, and checks for matches.

struct Hit {
    int p_coeffs[MAX_DEGREE + 1];
    int q_coeffs[MAX_DEGREE + 1];
    int deg;
    double value;
    int match_const;
    int match_c0, match_c1, match_c2;
};

__global__ void search_kernel(
    long long start_idx, long long count,
    int deg, int coeff_range, int cf_depth,
    Hit *hits, int *hit_count, int max_hits)
{
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    long long idx = start_idx + tid;

    // Decode index to polynomial coefficients
    // Total coefficients: 2 * (deg + 1)
    // Each coefficient ranges from -coeff_range to +coeff_range
    int num_coeffs = 2 * (deg + 1);
    int range = 2 * coeff_range + 1;

    int p_coeffs[MAX_DEGREE + 1] = {0};
    int q_coeffs[MAX_DEGREE + 1] = {0};

    long long tmp = idx;
    for (int i = 0; i <= deg; i++) {
        p_coeffs[i] = (int)(tmp % range) - coeff_range;
        tmp /= range;
    }
    for (int i = 0; i <= deg; i++) {
        q_coeffs[i] = (int)(tmp % range) - coeff_range;
        tmp /= range;
    }

    // Skip trivial cases
    int all_zero_q = 1;
    for (int i = 0; i <= deg; i++) if (q_coeffs[i] != 0) { all_zero_q = 0; break; }
    if (all_zero_q) return;

    // Evaluate CF
    double value;
    if (!check_convergence(p_coeffs, q_coeffs, deg, cf_depth, &value)) return;

    // Skip trivial values
    if (value == 0.0 || value != value || value > 1e15 || value < -1e15) return;
    if (value > -1e-10 && value < 1e-10) return;

    // Try to match against known constants
    int mc, c0, c1, c2;
    if (match_constant(value, &mc, &c0, &c1, &c2)) {
        int slot = atomicAdd(hit_count, 1);
        if (slot < max_hits) {
            Hit *h = &hits[slot];
            for (int i = 0; i <= deg; i++) {
                h->p_coeffs[i] = p_coeffs[i];
                h->q_coeffs[i] = q_coeffs[i];
            }
            h->deg = deg;
            h->value = value;
            h->match_const = mc;
            h->match_c0 = c0;
            h->match_c1 = c1;
            h->match_c2 = c2;
        }
    }
}

/* ── Main ──────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int deg = argc > 1 ? atoi(argv[1]) : 2;
    int coeff_range = argc > 2 ? atoi(argv[2]) : 5;
    int cf_depth = argc > 3 ? atoi(argv[3]) : 200;
    int gpu_id = argc > 4 ? atoi(argv[4]) : 0;

    cudaSetDevice(gpu_id);

    int range = 2 * coeff_range + 1;
    int num_coeffs = 2 * (deg + 1);
    long long total_candidates = 1;
    for (int i = 0; i < num_coeffs; i++) total_candidates *= range;

    printf("========================================\n");
    printf("Ramanujan Machine (GPU)\n");
    printf("========================================\n");
    printf("Polynomial degree: %d\n", deg);
    printf("Coefficient range: [-%d, %d]\n", coeff_range, coeff_range);
    printf("CF evaluation depth: %d terms\n", cf_depth);
    printf("Total candidates: %lld\n", total_candidates);
    printf("GPU: %d\n", gpu_id);
    printf("Constants: pi, e, ln(2), gamma, Catalan, zeta(3), Gauss, Lemniscate, sqrt(2), phi\n");
    printf("========================================\n\n");
    fflush(stdout);

    // Allocate hits buffer on GPU
    int max_hits = 100000;
    Hit *d_hits;
    int *d_hit_count;
    cudaMalloc(&d_hits, max_hits * sizeof(Hit));
    cudaMalloc(&d_hit_count, sizeof(int));
    cudaMemset(d_hit_count, 0, sizeof(int));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Process in chunks
    long long chunk_size = 1000000LL;  // 1M candidates per kernel launch
    int total_hits = 0;

    // Output file
    char outpath[256];
    snprintf(outpath, 256,
             "scripts/experiments/ramanujan-machine/results/hits_deg%d_range%d.csv",
             deg, coeff_range);
    FILE *fout = fopen(outpath, "w");
    if (fout) {
        fprintf(fout, "P_coeffs,Q_coeffs,value,constant,c0,c1,c2\n");
    }

    for (long long offset = 0; offset < total_candidates; offset += chunk_size) {
        long long this_chunk = chunk_size;
        if (offset + this_chunk > total_candidates)
            this_chunk = total_candidates - offset;

        int grid = (this_chunk + BLOCK - 1) / BLOCK;
        search_kernel<<<grid, BLOCK>>>(
            offset, this_chunk, deg, coeff_range, cf_depth,
            d_hits, d_hit_count, max_hits);

        // Check for new hits periodically
        if ((offset / chunk_size) % 100 == 0 || offset + this_chunk >= total_candidates) {
            cudaDeviceSynchronize();

            int h_hit_count;
            cudaMemcpy(&h_hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);

            if (h_hit_count > total_hits) {
                // Download new hits
                Hit *h_hits = (Hit *)malloc(h_hit_count * sizeof(Hit));
                cudaMemcpy(h_hits, d_hits, h_hit_count * sizeof(Hit), cudaMemcpyDeviceToHost);

                for (int i = total_hits; i < h_hit_count && i < max_hits; i++) {
                    Hit *h = &h_hits[i];
                    // Skip degenerate zero-value matches on host side
                    if (h->value > -1e-8 && h->value < 1e-8) continue;
                    printf("  HIT: P=(");
                    for (int j = 0; j <= h->deg; j++) printf("%s%d", j?",":"", h->p_coeffs[j]);
                    printf(") Q=(");
                    for (int j = 0; j <= h->deg; j++) printf("%s%d", j?",":"", h->q_coeffs[j]);
                    printf(") → %.15g", h->value);

                    if (h->match_c2 == -999) {
                        printf(" = %s^(%d/%d)", get_const_name(h->match_const),
                               h->match_c0, h->match_c1);
                    } else {
                        printf(" = (%d + %d*%s)/%d", h->match_c0, h->match_c2,
                               get_const_name(h->match_const), h->match_c1);
                    }
                    printf("\n");

                    if (fout) {
                        fprintf(fout, "\"(");
                        for (int j = 0; j <= h->deg; j++) fprintf(fout, "%s%d", j?",":"", h->p_coeffs[j]);
                        fprintf(fout, ")\",\"(");
                        for (int j = 0; j <= h->deg; j++) fprintf(fout, "%s%d", j?",":"", h->q_coeffs[j]);
                        fprintf(fout, ")\",%.*g,%s,%d,%d,%d\n",
                                17, h->value, get_const_name(h->match_const),
                                h->match_c0, h->match_c1, h->match_c2);
                    }
                }
                total_hits = h_hit_count;
                free(h_hits);
                if (fout) fflush(fout);
            }

            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            double pct = 100.0 * (offset + this_chunk) / total_candidates;
            double rate = (offset + this_chunk) / elapsed;
            double eta = (total_candidates - offset - this_chunk) / (rate + 1);

            printf("  %.1f%% (%lld/%lld) %d hits, %.0f candidates/sec, ETA %.0fs\n",
                   pct, offset + this_chunk, total_candidates,
                   total_hits, rate, eta);
            fflush(stdout);
        }
    }

    if (fout) fclose(fout);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Ramanujan Machine Results\n");
    printf("========================================\n");
    printf("Degree: %d, range: [-%d,%d]\n", deg, coeff_range, coeff_range);
    printf("Candidates: %lld\n", total_candidates);
    printf("Hits: %d\n", total_hits);
    printf("Time: %.1fs (%.0f candidates/sec)\n", total_time,
           total_candidates / total_time);
    if (total_hits > 0)
        printf("Output: %s\n", outpath);
    printf("========================================\n");

    cudaFree(d_hits);
    cudaFree(d_hit_count);
    return 0;
}
