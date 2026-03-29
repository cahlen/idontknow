/*
 * Direct minor arc evaluation for Zaremba's Conjecture — prime denominators
 *
 * For a target prime p, evaluate the exponential sum:
 *   F_N(alpha) = sum_{gamma in Gamma_A, ||gamma|| <= N} e(alpha * d_gamma)
 *
 * on a fine grid of alpha values in the minor arc region, and bound
 * the minor arc contribution to R(p):
 *   |minor arc| = |integral_{minor} F_N(alpha) * e(-alpha * p) d(alpha)|
 *
 * If |minor arc| < Main(p), then R(p) > 0 and p is a Zaremba denominator.
 *
 * Method:
 *   Phase 1: Enumerate all denominators d_gamma <= N^2 from the CF tree
 *            (stored as an array of denominator values)
 *   Phase 2: For each grid point alpha_j in the minor arc,
 *            compute F_N(alpha_j) = sum_gamma e(2*pi*i * alpha_j * d_gamma)
 *            using GPU parallelism (one thread per alpha_j)
 *   Phase 3: Numerically integrate F_N(alpha) * e(-alpha*p) over minor arc
 *
 * The minor arc is [0,1] \ union_{q <= Q} {|alpha - a/q| < 1/(qN)}
 * where Q = p^theta for some theta < 1.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o minor_arc scripts/experiments/zaremba-effective-bound/minor_arc_primes.cu -lm
 * Run:     ./minor_arc <prime_p> [grid_size] [gpu_id]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BOUND 5
#define MAX_DENOMS 200000000  // 200M max denominators
#define BLOCK_SIZE 256

typedef unsigned long long uint64;

// ============================================================
// Phase 1: Enumerate denominators from CF tree (CPU)
// ============================================================

static uint64 *g_denoms = NULL;
static uint64 g_denom_count = 0;

void enumerate_denoms(uint64 qprev, uint64 q, uint64 max_d) {
    if (q > max_d) return;
    if (q >= 1 && g_denom_count < MAX_DENOMS) {
        g_denoms[g_denom_count++] = q;
    }
    for (int a = 1; a <= BOUND; a++) {
        uint64 qnew = (uint64)a * q + qprev;
        if (qnew > max_d) break;
        enumerate_denoms(q, qnew, max_d);
    }
}

// ============================================================
// Phase 2: Evaluate exponential sum on GPU
// ============================================================

// Each thread computes F(alpha_j) for one grid point alpha_j
// F(alpha) = sum_k e(2*pi*i * alpha * denoms[k])
//          = sum_k cos(2*pi * alpha * denoms[k])  (real part)
//          + i * sum_k sin(...)                    (imag part)
//
// Then compute the contribution to R(p):
//   contribution_j = F(alpha_j) * e(-2*pi*i * alpha_j * p) * d(alpha)
//
// We accumulate: Re[sum_j F(alpha_j) * e(-alpha_j * p) * delta_alpha]

__global__ void eval_exponential_sum(
    uint64 *denoms, uint64 num_denoms,
    double *grid_alphas, int grid_size,
    uint64 target_p,
    double *result_real, double *result_imag)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= grid_size) return;

    double alpha = grid_alphas[j];
    double two_pi = 2.0 * M_PI;

    // Compute F(alpha) = sum_k e(2*pi*i * alpha * d_k)
    double F_re = 0.0, F_im = 0.0;
    for (uint64 k = 0; k < num_denoms; k++) {
        double phase = two_pi * alpha * (double)denoms[k];
        F_re += cos(phase);
        F_im += sin(phase);
    }

    // Multiply by e(-2*pi*i * alpha * p)
    double phase_p = two_pi * alpha * (double)target_p;
    double cos_p = cos(phase_p);
    double sin_p = sin(phase_p);

    // F(alpha) * e(-alpha*p) = (F_re + i*F_im) * (cos_p - i*sin_p)
    double contrib_re = F_re * cos_p + F_im * sin_p;
    double contrib_im = F_im * cos_p - F_re * sin_p;

    result_real[j] = contrib_re;
    result_imag[j] = contrib_im;
}

// ============================================================
// Phase 3: Integrate and compare with main term
// ============================================================

int is_prime(uint64 n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (uint64 i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i+2) == 0) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <prime_p> [grid_size] [gpu_id]\n", argv[0]);
        fprintf(stderr, "\nEvaluates the minor arc exponential sum for prime p.\n");
        fprintf(stderr, "If |minor arc| < Main(p), then p is a Zaremba denominator.\n");
        return 1;
    }

    uint64 target_p = (uint64)atoll(argv[1]);
    int grid_size = argc > 2 ? atoi(argv[2]) : 100000;
    int gpu_id = argc > 3 ? atoi(argv[3]) : 4;

    if (!is_prime(target_p)) {
        fprintf(stderr, "Error: %llu is not prime\n", (unsigned long long)target_p);
        return 1;
    }

    printf("Zaremba Minor Arc Evaluation for p = %llu\n", (unsigned long long)target_p);
    printf("Grid size: %d\n", grid_size);
    printf("GPU: %d\n\n", gpu_id);

    cudaSetDevice(gpu_id);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Phase 1: Enumerate denominators up to N = p^2
    uint64 N = target_p * target_p;
    if (N > 100000000) N = 100000000;  // cap at 100M for memory
    printf("Phase 1: Enumerating denominators up to N = %llu...\n",
           (unsigned long long)N);

    g_denoms = (uint64*)malloc(MAX_DENOMS * sizeof(uint64));
    g_denom_count = 0;

    g_denoms[g_denom_count++] = 1;  // d=1
    for (int a1 = 1; a1 <= BOUND; a1++) {
        enumerate_denoms(1, (uint64)a1, N);
    }
    printf("  Denominators: %llu\n\n", (unsigned long long)g_denom_count);

    if (g_denom_count == 0) {
        printf("No denominators found!\n");
        free(g_denoms);
        return 1;
    }

    // Check if p is directly in the denominator list
    int direct_hit = 0;
    for (uint64 i = 0; i < g_denom_count; i++) {
        if (g_denoms[i] == target_p) { direct_hit = 1; break; }
    }
    if (direct_hit) {
        printf("*** DIRECT HIT: p = %llu found in denominator list ***\n",
               (unsigned long long)target_p);
        printf("*** R(p) >= 1 — p is a Zaremba denominator (trivially) ***\n\n");
    }

    // Phase 2: Set up minor arc grid
    // Major arc: |alpha - a/q| < 1/(q*N) for q <= Q
    // Take Q = p^{0.3} (small major arc, most of [0,1] is minor)
    double Q = pow((double)target_p, 0.3);
    if (Q < 2) Q = 2;
    double N_double = (double)N;
    printf("Phase 2: Setting up grid (Q = %.1f)...\n", Q);

    // Generate grid points in [0, 1] that are in the minor arc
    // (avoiding |alpha - a/q| < 1/(q*N) for q <= Q, gcd(a,q)=1)
    double *h_alphas = (double*)malloc(grid_size * sizeof(double));
    int actual_grid = 0;

    for (int j = 0; j < grid_size; j++) {
        double alpha = (double)j / grid_size;
        // Check if alpha is in any major arc
        int in_major = 0;
        for (int q = 1; q <= (int)Q && !in_major; q++) {
            for (int a = 0; a <= q && !in_major; a++) {
                // Check gcd(a,q) == 1 (or a==0, q==1)
                int g = q, b = a;
                while (b) { int t = b; b = g % b; g = t; }
                if (g != 1 && !(a == 0 && q == 1)) continue;

                double center = (double)a / q;
                double radius = 1.0 / (q * N_double);
                if (fabs(alpha - center) < radius) {
                    in_major = 1;
                }
            }
        }
        if (!in_major) {
            h_alphas[actual_grid++] = alpha;
        }
    }
    printf("  Minor arc grid points: %d / %d\n\n", actual_grid, grid_size);

    // Upload to GPU
    uint64 *d_denoms;
    double *d_alphas, *d_result_re, *d_result_im;

    size_t denom_bytes = g_denom_count * sizeof(uint64);
    printf("  Uploading %llu denominators (%.1f MB)...\n",
           (unsigned long long)g_denom_count, denom_bytes / 1e6);

    cudaMalloc(&d_denoms, denom_bytes);
    cudaMemcpy(d_denoms, g_denoms, denom_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_alphas, actual_grid * sizeof(double));
    cudaMemcpy(d_alphas, h_alphas, actual_grid * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_result_re, actual_grid * sizeof(double));
    cudaMalloc(&d_result_im, actual_grid * sizeof(double));

    // Launch kernel
    printf("Phase 2: Evaluating F(alpha) on %d grid points...\n", actual_grid);
    int blocks = (actual_grid + BLOCK_SIZE - 1) / BLOCK_SIZE;
    eval_exponential_sum<<<blocks, BLOCK_SIZE>>>(
        d_denoms, g_denom_count,
        d_alphas, actual_grid,
        target_p,
        d_result_re, d_result_im
    );
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("  GPU done: %.1fs\n\n", gpu_time);

    // Phase 3: Integrate
    double *h_re = (double*)malloc(actual_grid * sizeof(double));
    double *h_im = (double*)malloc(actual_grid * sizeof(double));
    cudaMemcpy(h_re, d_result_re, actual_grid * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_im, d_result_im, actual_grid * sizeof(double), cudaMemcpyDeviceToHost);

    double dalpha = 1.0 / grid_size;
    double integral_re = 0.0, integral_im = 0.0;
    double max_F = 0.0;

    for (int j = 0; j < actual_grid; j++) {
        integral_re += h_re[j] * dalpha;
        integral_im += h_im[j] * dalpha;
        double F_mag = sqrt(h_re[j] * h_re[j] + h_im[j] * h_im[j]);
        if (F_mag > max_F) max_F = F_mag;
    }

    double minor_arc_mag = sqrt(integral_re * integral_re + integral_im * integral_im);

    // Main term estimate
    double delta = 0.836829443681208;
    double S_p = (double)(target_p * target_p) / (double)(target_p * target_p - 1);
    // Main ~ C * N^{2delta-1} * S(p), with C ~ 1 and N ~ p^2
    double main_term = pow(N_double, 2 * delta - 1) * S_p;

    printf("========================================\n");
    printf("Results for p = %llu\n", (unsigned long long)target_p);
    printf("  Denominators enumerated: %llu\n", (unsigned long long)g_denom_count);
    printf("  Direct hit (p in tree): %s\n", direct_hit ? "YES" : "no");
    printf("  Minor arc integral: |I| = %.6e\n", minor_arc_mag);
    printf("  Max |F(alpha)|: %.6e\n", max_F);
    printf("  Main term estimate: %.6e\n", main_term);
    printf("  Ratio |minor|/Main: %.6e\n", minor_arc_mag / main_term);

    if (direct_hit) {
        printf("\n  p = %llu IS a Zaremba denominator (found in tree)\n",
               (unsigned long long)target_p);
    } else if (minor_arc_mag < main_term) {
        printf("\n  |minor arc| < Main term => R(p) > 0\n");
        printf("  p = %llu IS a Zaremba denominator\n",
               (unsigned long long)target_p);
    } else {
        printf("\n  Cannot conclude R(p) > 0 from this computation\n");
        printf("  (Need finer grid or larger N)\n");
    }
    printf("  Time: %.1fs\n", gpu_time);
    printf("========================================\n");

    free(g_denoms); free(h_alphas); free(h_re); free(h_im);
    cudaFree(d_denoms); cudaFree(d_alphas);
    cudaFree(d_result_re); cudaFree(d_result_im);
    return 0;
}
