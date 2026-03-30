/*
 * DOLGOPYAT SPECTRAL PROFILE: ρ(t) for the transfer operator L_{δ+it}
 *
 * For each t ∈ ℝ, compute the spectral radius of:
 *   (L_s f)(x) = Σ_{a=1}^5 (a+x)^{-2s} f(1/(a+x))
 * at s = δ + it (complex parameter).
 *
 * At t = 0: ρ = 1 (the Perron-Frobenius eigenvalue).
 * For |t| > 0: ρ(t) < 1 (Dolgopyat's theorem for expanding maps).
 * The decay rate ρ_η = sup_{|t|>b₀} ρ(t) determines the power savings ε.
 *
 * The operator L_{δ+it} has COMPLEX matrix entries:
 *   L[i][j] = Σ_a (a+x_j)^{-2δ} × (a+x_j)^{-2it} × B_j(g_a(x_i))
 * where (a+x)^{-2it} = exp(-2it log(a+x)) is the oscillatory factor.
 *
 * Each t value is independent → trivially parallel on GPU.
 * N=40 Chebyshev, FP64 complex arithmetic.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o dolgopyat dolgopyat_profile.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BOUND 5
#define NC 40
#define POWER_ITER 300
#define DELTA 0.836829443681208
#define TWO_PI 6.283185307179586

struct cmplx { double re, im; };
__device__ __host__ cmplx cmul(cmplx a, cmplx b) {
    return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}
__device__ __host__ cmplx cadd(cmplx a, cmplx b) {
    return {a.re + b.re, a.im + b.im};
}
__device__ __host__ double cnorm2(cmplx a) { return a.re*a.re + a.im*a.im; }

__global__ void spectral_profile(
    double *d_tvals, double *d_radii, int num_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_t) return;

    double t = d_tvals[idx];

    // Chebyshev nodes
    double nodes[NC];
    double bary[NC];
    for (int j = 0; j < NC; j++) {
        nodes[j] = 0.5 * (1.0 + cos(M_PI * (2*j + 1) / (2.0 * NC)));
        bary[j] = ((j % 2 == 0) ? 1.0 : -1.0) * sin(M_PI * (2*j + 1) / (2.0 * NC));
    }

    // Build L_{δ+it} matrix (NC × NC complex)
    cmplx L[NC][NC];
    for (int i = 0; i < NC; i++)
        for (int j = 0; j < NC; j++)
            L[i][j] = {0.0, 0.0};

    for (int a = 1; a <= BOUND; a++) {
        for (int i = 0; i < NC; i++) {
            double xi = nodes[i];
            double apx = a + xi;
            double ga = 1.0 / apx;

            // Weight: (a+x)^{-2δ} (real part)
            double weight = pow(apx, -2.0 * DELTA);

            // Oscillatory twist: (a+x)^{-2it} = exp(-2it log(a+x))
            double phase = -2.0 * t * log(apx);
            cmplx twist = {cos(phase), sin(phase)};

            // Combined: weight × twist
            cmplx wt = {weight * twist.re, weight * twist.im};

            // Barycentric interpolation at ga
            int exact = -1;
            for (int k = 0; k < NC; k++)
                if (fabs(ga - nodes[k]) < 1e-12) { exact = k; break; }

            if (exact >= 0) {
                L[i][exact] = cadd(L[i][exact], wt);
            } else {
                double den = 0;
                double num[NC];
                for (int j = 0; j < NC; j++) {
                    num[j] = bary[j] / (ga - nodes[j]);
                    den += num[j];
                }
                for (int j = 0; j < NC; j++) {
                    double b = num[j] / den;
                    cmplx val = {wt.re * b, wt.im * b};
                    L[i][j] = cadd(L[i][j], val);
                }
            }
        }
    }

    // Power iteration for spectral radius
    cmplx v[NC];
    for (int i = 0; i < NC; i++)
        v[i] = {sin(i * 1.618 + 0.5), cos(i * 2.718 + 0.3)};

    double radius = 0;
    for (int iter = 0; iter < POWER_ITER; iter++) {
        cmplx w[NC];
        for (int i = 0; i < NC; i++) {
            w[i] = {0, 0};
            for (int j = 0; j < NC; j++)
                w[i] = cadd(w[i], cmul(L[i][j], v[j]));
        }
        double norm2 = 0;
        for (int i = 0; i < NC; i++) norm2 += cnorm2(w[i]);
        double norm = sqrt(norm2);
        if (norm > 1e-30) {
            double inv = 1.0 / norm;
            for (int i = 0; i < NC; i++)
                v[i] = {w[i].re * inv, w[i].im * inv};
        }
        radius = norm;
    }

    d_radii[idx] = radius;
}

int main(int argc, char **argv) {
    int num_t = argc > 1 ? atoi(argv[1]) : 100000;
    double t_max = argc > 2 ? atof(argv[2]) : 1000.0;

    printf("Dolgopyat Spectral Profile: L_{δ+it} for t ∈ [0, %.0f]\n", t_max);
    printf("Grid: %d points, N=%d Chebyshev, FP64\n\n", num_t, NC);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double *h_t = (double*)malloc(num_t * sizeof(double));
    for (int i = 0; i < num_t; i++)
        h_t[i] = (i + 0.5) * t_max / num_t;

    double *d_t, *d_r;
    cudaMalloc(&d_t, num_t * sizeof(double));
    cudaMalloc(&d_r, num_t * sizeof(double));
    cudaMemcpy(d_t, h_t, num_t * sizeof(double), cudaMemcpyHostToDevice);

    spectral_profile<<<(num_t+255)/256, 256>>>(d_t, d_r, num_t);
    cudaDeviceSynchronize();

    double *h_r = (double*)malloc(num_t * sizeof(double));
    cudaMemcpy(h_r, d_r, num_t * sizeof(double), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Analysis
    double max_rho = 0;
    double max_rho_t = 0;
    double rho_at_1 = 0;
    double b0 = 0; // threshold where ρ drops below 0.99

    for (int i = 0; i < num_t; i++) {
        if (h_r[i] > max_rho) { max_rho = h_r[i]; max_rho_t = h_t[i]; }
        if (fabs(h_t[i] - 1.0) < t_max / num_t) rho_at_1 = h_r[i];
        if (b0 == 0 && h_r[i] < 0.99 && h_t[i] > 0.1) b0 = h_t[i];
    }

    printf("========================================\n");
    printf("Time: %.2fs\n", elapsed);
    printf("Max ρ(t): %.6f at t=%.2f\n", max_rho, max_rho_t);
    printf("ρ(1): %.6f\n", rho_at_1);
    printf("b₀ (where ρ < 0.99): %.2f\n", b0);
    printf("========================================\n\n");

    // Print ρ(t) at key values
    printf("Spectral radius ρ(t) at selected t:\n");
    printf("%12s  %12s\n", "t", "ρ(t)");
    double check_t[] = {0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
    for (int k = 0; k < 13; k++) {
        double target = check_t[k];
        if (target > t_max) break;
        int best = 0;
        for (int i = 0; i < num_t; i++)
            if (fabs(h_t[i] - target) < fabs(h_t[best] - target)) best = i;
        printf("%12.2f  %12.6f\n", h_t[best], h_r[best]);
    }

    // Compute ρ_η = max ρ(t) for |t| > b₀
    double rho_eta = 0;
    for (int i = 0; i < num_t; i++) {
        if (h_t[i] > b0 + 1 && h_r[i] > rho_eta) rho_eta = h_r[i];
    }
    printf("\nρ_η (Dolgopyat bound) = sup_{t > b₀+1} ρ(t) = %.6f\n", rho_eta);
    printf("Dolgopyat contraction: ρ_η = %.6f\n", rho_eta);

    // Compute ε₂ from ρ_η
    double phi = (1 + sqrt(5)) / 2;
    double eps2 = -log(rho_eta) / log(phi);
    printf("ε₂ = -log(ρ_η)/log(φ) = %.6f\n", eps2);

    double eps1 = 0.650 / 1.6539; // σ / |P'(δ)|
    double eps = fmin(eps1, eps2);
    printf("ε₁ (spectral gap) = %.6f\n", eps1);
    printf("ε = min(ε₁, ε₂) = %.6f\n", eps);

    cudaFree(d_t); cudaFree(d_r);
    free(h_t); free(h_r);
    return 0;
}
