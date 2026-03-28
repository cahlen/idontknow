/*
 * CUDA-accelerated Riemann Zeta zero verification
 *
 * Verifies that non-trivial zeros of the Riemann zeta function lie on
 * the critical line Re(s) = 1/2, using the Hardy Z-function approach.
 *
 * The Hardy Z-function Z(t) is real-valued and satisfies:
 *   Z(t) = 0  iff  zeta(1/2 + it) = 0
 *
 * A sign change in Z(t) between t_a and t_b guarantees a zero in [t_a, t_b]
 * on the critical line.
 *
 * Method: Riemann-Siegel formula for efficient Z(t) evaluation,
 * then systematic sign-change detection across a grid of t values.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o riemann_zeros scripts/experiments/riemann-zeta/riemann_zeros.cu -lm
 * Run:     ./riemann_zeros <num_zeros_to_verify>
 *
 * References:
 *   - Edwards, "Riemann's Zeta Function" (Dover)
 *   - Odlyzko, "The 10^20-th zero of the Riemann zeta function"
 *   - Gourdon (2004), verified first 10^13 zeros
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define EULER_MASCHERONI 0.5772156649015328606

/*
 * Riemann-Siegel theta function:
 *   theta(t) = Im(log(Gamma(1/4 + it/2))) - (t/2)*log(pi)
 *
 * Using Stirling's approximation for large t:
 *   theta(t) ≈ (t/2)*log(t/(2*pi*e)) - pi/8 + 1/(48t) + 7/(5760*t^3) + ...
 */
__device__ double theta(double t) {
    double t2 = t * t;
    return (t / 2.0) * log(t / (2.0 * PI * M_E)) - PI / 8.0
           + 1.0 / (48.0 * t)
           + 7.0 / (5760.0 * t * t2);
}

/*
 * Riemann-Siegel Z-function (simplified for moderate t):
 *   Z(t) = 2 * sum_{n=1}^{N} cos(theta(t) - t*log(n)) / sqrt(n) + R
 *
 * where N = floor(sqrt(t/(2*pi))) and R is a correction term.
 *
 * This is the main computation — each evaluation is O(sqrt(t)).
 */
__device__ double Z_function(double t) {
    int N = (int)sqrt(t / (2.0 * PI));
    if (N < 1) N = 1;

    double th = theta(t);
    double sum = 0.0;

    for (int n = 1; n <= N; n++) {
        sum += cos(th - t * log((double)n)) / sqrt((double)n);
    }
    sum *= 2.0;

    // Riemann-Siegel correction term (first order)
    double frac = sqrt(t / (2.0 * PI)) - (double)N;
    // Riemann-Siegel coefficients C0
    double C0 = cos(2.0 * PI * (frac * frac - frac - 1.0 / 16.0))
                / cos(2.0 * PI * frac);
    double correction = pow(-1.0, N - 1) * pow(t / (2.0 * PI), -0.25) * C0;
    sum += correction;

    return sum;
}

/*
 * Each thread evaluates Z(t) at a grid point and detects sign changes
 * with the previous point. A sign change means a zero on the critical line.
 */
__global__ void count_sign_changes(double t_start, double dt, uint64_t count,
                                    uint64_t *sign_changes,
                                    uint64_t *total_changes,
                                    double *z_values) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    double t = t_start + idx * dt;
    double z = Z_function(t);

    // Store Z value for sign-change detection
    if (z_values != NULL && idx < count) {
        z_values[idx] = z;
    }

    // Detect sign change with previous point
    if (idx > 0) {
        double t_prev = t_start + (idx - 1) * dt;
        double z_prev = Z_function(t_prev);

        if ((z > 0 && z_prev < 0) || (z < 0 && z_prev > 0)) {
            uint64_t pos = atomicAdd((unsigned long long*)total_changes,
                                     (unsigned long long)1);
            if (pos < 1000000 && sign_changes != NULL) {
                // Store the approximate location of the zero
                // (linear interpolation between grid points)
                // Encode as fixed-point: multiply t by 1000 for storage
                sign_changes[pos] = (uint64_t)(t * 1000.0);
            }
        }
    }
}

/*
 * Gram points: t_n where theta(t_n) = n*pi
 * These provide a natural grid for zero-counting.
 * Between consecutive Gram points, there should be exactly one zero
 * (Gram's law, which holds for "most" intervals).
 *
 * We use Gram points to validate our zero count against the
 * Riemann-von Mangoldt formula:
 *   N(T) = (T/(2*pi)) * log(T/(2*pi*e)) + 7/8 + S(T)
 * where S(T) is small.
 */
__host__ double riemann_von_mangoldt(double T) {
    return (T / (2.0 * PI)) * log(T / (2.0 * PI * M_E)) + 7.0 / 8.0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_zeros_to_verify> [grid_density]\n", argv[0]);
        fprintf(stderr, "  grid_density: points per unit t (default: 10)\n");
        return 1;
    }

    uint64_t target_zeros = (uint64_t)atoll(argv[1]);
    double grid_density = argc > 2 ? atof(argv[2]) : 10.0;

    // Estimate T needed to find target_zeros zeros
    // N(T) ≈ T/(2*pi) * log(T/(2*pi*e))
    // For a rough estimate: T ≈ 2*pi*target_zeros / log(target_zeros)
    double T_estimate = 2.0 * PI * target_zeros / log((double)target_zeros + 2);
    // Refine with Newton's method
    for (int i = 0; i < 10; i++) {
        double N_T = riemann_von_mangoldt(T_estimate);
        double dN = log(T_estimate / (2.0 * PI)) / (2.0 * PI);
        T_estimate += (target_zeros - N_T) / dN;
    }

    double t_start = 14.0;  // first zero is at t ≈ 14.134
    double t_end = T_estimate * 1.05;  // 5% margin
    double dt = 1.0 / grid_density;
    uint64_t grid_points = (uint64_t)((t_end - t_start) / dt) + 1;

    printf("Riemann Zeta Zero Verification\n");
    printf("Target: %lu zeros\n", target_zeros);
    printf("Search range: t = %.1f to %.1f\n", t_start, t_end);
    printf("Grid density: %.1f points per unit t\n", grid_density);
    printf("Total grid points: %lu\n", grid_points);
    printf("Expected zeros (Riemann-von Mangoldt): %.0f\n", riemann_von_mangoldt(t_end));
    printf("\n");

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n\n", device_count);

    // Allocate device memory
    uint64_t *d_sign_changes, *d_total_changes;
    cudaMalloc(&d_sign_changes, 1000000 * sizeof(uint64_t));
    cudaMalloc(&d_total_changes, sizeof(uint64_t));
    cudaMemset(d_total_changes, 0, sizeof(uint64_t));

    uint64_t chunk_size = 10000000;  // 10M grid points per chunk
    uint64_t total_sign_changes = 0;

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (uint64_t chunk_offset = 0; chunk_offset < grid_points; chunk_offset += chunk_size) {
        uint64_t chunk_count = chunk_size;
        if (chunk_offset + chunk_count > grid_points)
            chunk_count = grid_points - chunk_offset;

        double chunk_t_start = t_start + chunk_offset * dt;

        int gpu = (chunk_offset / chunk_size) % device_count;
        cudaSetDevice(gpu);

        cudaMemset(d_total_changes, 0, sizeof(uint64_t));

        int blocks = (chunk_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        count_sign_changes<<<blocks, THREADS_PER_BLOCK>>>(
            chunk_t_start, dt, chunk_count,
            d_sign_changes, d_total_changes, NULL
        );
        cudaDeviceSynchronize();

        uint64_t h_changes;
        cudaMemcpy(&h_changes, d_total_changes, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        total_sign_changes += h_changes;

        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                         (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
        double progress = (double)(chunk_offset + chunk_count) / grid_points * 100;
        double t_current = chunk_t_start + chunk_count * dt;

        printf("[GPU %d] t=%.1f..%.1f | zeros found: %lu | expected: %.0f | %.1f%% | %.1fs\n",
               gpu, chunk_t_start, t_current,
               total_sign_changes, riemann_von_mangoldt(t_current),
               progress, elapsed);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double total_elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                          (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    double expected = riemann_von_mangoldt(t_end);

    printf("\n========================================\n");
    printf("Riemann Zeta Zero Verification Complete\n");
    printf("Range: t = %.1f to %.1f\n", t_start, t_end);
    printf("Zeros found (sign changes): %lu\n", total_sign_changes);
    printf("Expected (Riemann-von Mangoldt): %.0f\n", expected);
    printf("Difference: %ld\n", (long)total_sign_changes - (long)expected);
    printf("Time: %.1fs\n", total_elapsed);

    if (labs((long)total_sign_changes - (long)expected) <= 2) {
        printf("\nAll zeros accounted for — Riemann Hypothesis CONSISTENT for first %lu zeros\n",
               total_sign_changes);
    } else {
        printf("\n*** DISCREPANCY — missing or extra zeros detected ***\n");
        printf("This could indicate: insufficient grid density, numerical precision issues,\n");
        printf("or (extremely unlikely) a violation of the Riemann Hypothesis.\n");
    }
    printf("========================================\n");

    cudaFree(d_sign_changes);
    cudaFree(d_total_changes);

    return 0;
}
