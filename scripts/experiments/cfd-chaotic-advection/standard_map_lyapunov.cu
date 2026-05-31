/*
 * Chaotic advection: Lyapunov spectrum of the Chirikov standard map
 *
 * Models area-preserving mixing on T^2 — the same phase-space structure as
 * laminar 2D flows with periodic driving (chaotic advection conjectures).
 *
 * Map (mod 2π):
 *   p' = p + K sin(θ)
 *   θ' = θ + p'
 *
 * For each K, estimate the largest Lyapunov exponent Λ(K) by averaging
 * Benettin tangent-vector growth over many initial conditions.
 *
 * Hardware: RTX 5090 (32 GB, compute capability 12.0)
 * Compile: nvcc -O3 -arch=sm_120 -o standard_map_lyapunov \
 *            scripts/experiments/cfd-chaotic-advection/standard_map_lyapunov.cu -lm
 * Run:     ./standard_map_lyapunov [n_k] [n_ic] [n_iters] [k_max]
 *          ./standard_map_lyapunov 512 4096 20000 5.0
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846
#define TWO_PI (2.0 * PI)
#define BLOCK 256

__device__ double d_mod2pi(double x) {
    x = fmod(x, TWO_PI);
    if (x < 0.0) x += TWO_PI;
    return x;
}

__device__ unsigned long long d_splitmix64(unsigned long long *state) {
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ double d_uniform01(unsigned long long *state) {
    return (d_splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ double d_benettin_lyapunov(double K, double theta0, double p0,
                                      int n_iters, unsigned long long seed) {
    double theta = d_mod2pi(theta0);
    double p = d_mod2pi(p0);
    double v0 = 1.0, v1 = 0.0;
    double sum_log = 0.0;
    int count = 0;

    for (int it = 0; it < n_iters; it++) {
        double c = cos(theta);
        double j00 = 1.0 + K * c;
        double j01 = 1.0;
        double j10 = K * c;
        double j11 = 1.0;

        double w0 = j00 * v0 + j01 * v1;
        double w1 = j10 * v0 + j11 * v1;
        double norm = sqrt(w0 * w0 + w1 * w1);
        if (!(norm > 0.0) || isnan(norm) || isinf(norm)) return NAN;

        sum_log += log(norm);
        count++;
        v0 = w0 / norm;
        v1 = w1 / norm;

        double p_new = d_mod2pi(p + K * sin(theta));
        theta = d_mod2pi(theta + p_new);
        p = p_new;
    }
    return sum_log / (double)count;
}

__global__ void lyapunov_kernel(int n_k, int n_ic, int n_iters,
                                double k_max, unsigned long long seed,
                                double *per_ic) {
    int k_idx = blockIdx.x;
    int ic_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (k_idx >= n_k || ic_idx >= n_ic) return;

    double K = (n_k <= 1) ? 0.0 : k_max * (double)k_idx / (double)(n_k - 1);
    unsigned long long rng = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)k_idx)
                             ^ (0xD1B54A32D192ED03ULL * (unsigned long long)ic_idx);

    double theta0 = d_uniform01(&rng) * TWO_PI;
    double p0 = d_uniform01(&rng) * TWO_PI;
    double lam = d_benettin_lyapunov(K, theta0, p0, n_iters, rng);

    per_ic[(size_t)k_idx * (size_t)n_ic + (size_t)ic_idx] = lam;
}

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CERTIFICATE_ERROR: %s: %s\n", msg, cudaGetErrorString(err));
        exit(2);
    }
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int n_k = argc > 1 ? atoi(argv[1]) : 512;
    int n_ic = argc > 2 ? atoi(argv[2]) : 4096;
    int n_iters = argc > 3 ? atoi(argv[3]) : 20000;
    double k_max = argc > 4 ? atof(argv[4]) : 5.0;
    unsigned long long seed = 0xC0FFEEULL;

    if (n_k < 2 || n_ic < 1 || n_iters < 100 || k_max <= 0.0) {
        fprintf(stderr, "Usage: %s [n_k>=2] [n_ic] [n_iters] [k_max]\n", argv[0]);
        return 1;
    }

    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    printf("==========================================\n");
    printf("  CFD Chaotic Advection — Standard Map\n");
    printf("  Device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  K grid: %d points in [0, %.6f]\n", n_k, k_max);
    printf("  ICs per K: %d\n", n_ic);
    printf("  Iterations: %d\n", n_iters);
    printf("  Total trajectories: %lld\n", (long long)n_k * (long long)n_ic);
    printf("==========================================\n\n");

    size_t n_total = (size_t)n_k * (size_t)n_ic;
    size_t bytes = n_total * sizeof(double);
    double *h_per_ic = (double *)malloc(bytes);
    double *d_per_ic = NULL;
    if (!h_per_ic) {
        fprintf(stderr, "CERTIFICATE_ERROR: host alloc failed (%zu bytes)\n", bytes);
        return 2;
    }
    check_cuda(cudaMalloc(&d_per_ic, bytes), "cudaMalloc");

    dim3 grid(n_k, (n_ic + BLOCK - 1) / BLOCK);
    dim3 block(BLOCK);

    double t0 = now_seconds();
    lyapunov_kernel<<<grid, block>>>(n_k, n_ic, n_iters, k_max, seed, d_per_ic);
    check_cuda(cudaDeviceSynchronize(), "kernel sync");
    check_cuda(cudaMemcpy(h_per_ic, d_per_ic, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");

    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path),
             "scripts/experiments/cfd-chaotic-advection/results/lyapunov_k%d_ic%d_iter%d.csv",
             n_k, n_ic, n_iters);

    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "CERTIFICATE_ERROR: cannot open %s\n", csv_path);
        return 2;
    }
    fprintf(csv, "k_index,K,mean_lyapunov,std_lyapunov,min_lyapunov,max_lyapunov,fraction_positive\n");

    int nan_count = 0;
    double k_crit_scan = -1.0;
    int found_transition = 0;

    for (int k_idx = 0; k_idx < n_k; k_idx++) {
        double K = k_max * (double)k_idx / (double)(n_k - 1);
        double sum = 0.0, sum2 = 0.0;
        double mn = INFINITY, mx = -INFINITY;
        int pos = 0, valid = 0;

        for (int ic = 0; ic < n_ic; ic++) {
            double v = h_per_ic[(size_t)k_idx * (size_t)n_ic + (size_t)ic];
            if (isnan(v) || isinf(v)) {
                nan_count++;
                continue;
            }
            valid++;
            sum += v;
            sum2 += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (v > 0.0) pos++;
        }
        if (valid == 0) {
            fprintf(stderr, "CERTIFICATE_ERROR: no valid samples at K=%.6f\n", K);
            return 2;
        }
        double mean = sum / (double)valid;
        double var = sum2 / (double)valid - mean * mean;
        if (var < 0.0) var = 0.0;
        double std = sqrt(var);
        double frac = (double)pos / (double)valid;

        fprintf(csv, "%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.6f\n",
                k_idx, K, mean, std, mn, mx, frac);

        if (!found_transition && K > 0.5 && mean > 0.01 && frac > 0.95) {
            k_crit_scan = K;
            found_transition = 1;
        }
    }
    fclose(csv);

    double elapsed = now_seconds() - t0;
    printf("Wrote %s\n", csv_path);
    printf("Elapsed: %.2f s (%.1f trajectories/s)\n", elapsed,
           (double)n_total / elapsed);
    printf("NaN/Inf samples: %d / %zu\n", nan_count, n_total);

    /* Validation: K=0 should be near-integrable (Λ ≈ 0) */
    double k0_mean = 0.0;
    for (int ic = 0; ic < n_ic; ic++) k0_mean += h_per_ic[ic];
    k0_mean /= (double)n_ic;
    printf("Validation K=0 mean Λ = %.6e (expect ~0)\n", k0_mean);
    if (fabs(k0_mean) > 0.05) {
        fprintf(stderr, "CERTIFICATE_WARN: K=0 Lyapunov unexpectedly large\n");
    }

    if (found_transition) {
        printf("Empirical bulk-chaos onset (mean>0.01, >95%% ICs positive): K ≈ %.4f\n",
               k_crit_scan);
        printf("Literature K_crit (standard map): ≈ 0.971635406\n");
    }

    if (nan_count > 0) {
        fprintf(stderr, "CERTIFICATE_ERROR: numerical failures detected\n");
        cudaFree(d_per_ic);
        free(h_per_ic);
        return 2;
    }

    cudaFree(d_per_ic);
    free(h_per_ic);
    return 0;
}
