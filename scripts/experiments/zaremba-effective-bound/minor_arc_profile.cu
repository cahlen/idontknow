/*
 * Minor Arc Spectral Profile for Zaremba's Circle Method
 *
 * For each α ∈ [0, 1], compute the spectral radius of the TWISTED
 * transfer operator:
 *
 *   L_{δ,α} f(x) = Σ_{a=1}^5 (a+x)^{-2δ} · e(α/(a+x)) · f(1/(a+x))
 *
 * where e(t) = exp(2πit).
 *
 * On the MAJOR arcs (α near a/q with q small), the spectral radius ≈ 1.
 * On the MINOR arcs, the spectral radius < 1.
 * The GAP on the minor arc controls the B-K error term.
 *
 * The twist e(α/(a+x)) encodes the exponential sum F_N(α) structure.
 * No need to enumerate CF denominators — the operator captures everything.
 *
 * Each α is independent → trivially parallel across GPU threads.
 * Operator is N×N complex matrix → fits in registers for N=20.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o minor_arc minor_arc_profile.cu -lm
 * Run:     ./minor_arc <grid_size> [q_max_major]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BOUND 5
#define N_CHEB 20
#define POWER_ITER 150
#define DELTA 0.836829443681208f
#define TWO_PI 6.283185307179586f

// Complex number operations (inline, FP32)
struct cmplx {
    float re, im;
};

__device__ cmplx cmul(cmplx a, cmplx b) {
    return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}
__device__ cmplx cadd(cmplx a, cmplx b) {
    return {a.re + b.re, a.im + b.im};
}
__device__ float cnorm2(cmplx a) { return a.re*a.re + a.im*a.im; }

// Each thread computes the spectral radius at one α value
__global__ void twisted_spectral_radius(
    float *d_alphas,        // input: α values
    float *d_radii,         // output: |λ_1(α)|
    int num_alphas
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_alphas) return;

    float alpha = d_alphas[idx];

    // Precompute Chebyshev nodes on [0,1]
    float nodes[N_CHEB];
    for (int j = 0; j < N_CHEB; j++) {
        nodes[j] = 0.5f * (1.0f + cosf(M_PI * (2*j + 1) / (2.0f * N_CHEB)));
    }

    // Barycentric weights for Chebyshev interpolation
    float bw[N_CHEB];
    for (int j = 0; j < N_CHEB; j++) {
        bw[j] = (j % 2 == 0 ? 1.0f : -1.0f) * sinf(M_PI * (2*j + 1) / (2.0f * N_CHEB));
    }

    // Build the twisted operator matrix L_{δ,α}[i][j] (complex, N×N)
    // Using barycentric interpolation (same as transfer_operator.cu):
    // L_{δ,α}[i][j] = Σ_{a=1}^5 (a+x_i)^{-2δ} · e(α·g_a(x_i)) · B_j(g_a(x_i))
    // where B_j(y) is the j-th barycentric basis function at Chebyshev nodes
    cmplx L[N_CHEB][N_CHEB];
    for (int i = 0; i < N_CHEB; i++)
        for (int j = 0; j < N_CHEB; j++)
            L[i][j] = {0.0f, 0.0f};

    for (int a = 1; a <= BOUND; a++) {
        for (int i = 0; i < N_CHEB; i++) {
            float xi = nodes[i];
            float apx = a + xi;
            float y = 1.0f / apx; // g_a(x_i)

            // Weight: (a+x_i)^{-2δ}
            float weight = powf(apx, -2.0f * DELTA);

            // Phase twist: e(α·g_a(x_i))
            float phase = TWO_PI * alpha * y;
            cmplx twist = {cosf(phase), sinf(phase)};

            // Barycentric interpolation: evaluate at y
            // Check if y coincides with a node
            int exact = -1;
            for (int k = 0; k < N_CHEB; k++) {
                if (fabsf(y - nodes[k]) < 1e-7f) { exact = k; break; }
            }

            if (exact >= 0) {
                cmplx val = {weight, 0.0f};
                val = cmul(val, twist);
                L[i][exact] = cadd(L[i][exact], val);
            } else {
                float denom = 0;
                float num[N_CHEB];
                for (int j = 0; j < N_CHEB; j++) {
                    num[j] = bw[j] / (y - nodes[j]);
                    denom += num[j];
                }
                for (int j = 0; j < N_CHEB; j++) {
                    float bary = num[j] / denom;
                    cmplx val = {weight * bary, 0.0f};
                    val = cmul(val, twist);
                    L[i][j] = cadd(L[i][j], val);
                }
            }
        }
    }

    // Power iteration to find spectral radius
    cmplx v[N_CHEB];
    for (int i = 0; i < N_CHEB; i++) {
        v[i] = {sinf(i * 1.618f + 0.5f), cosf(i * 2.718f + 0.3f)};
    }

    float radius = 0;
    for (int iter = 0; iter < POWER_ITER; iter++) {
        cmplx w[N_CHEB];
        for (int i = 0; i < N_CHEB; i++) {
            w[i] = {0, 0};
            for (int j = 0; j < N_CHEB; j++) {
                w[i] = cadd(w[i], cmul(L[i][j], v[j]));
            }
        }

        // Compute norm
        float norm2 = 0;
        for (int i = 0; i < N_CHEB; i++) norm2 += cnorm2(w[i]);
        float norm = sqrtf(norm2);

        if (norm > 1e-30f) {
            float inv = 1.0f / norm;
            for (int i = 0; i < N_CHEB; i++) {
                v[i] = {w[i].re * inv, w[i].im * inv};
            }
        }
        radius = norm;
    }

    d_radii[idx] = radius;
}

int main(int argc, char **argv) {
    int grid_size = argc > 1 ? atoi(argv[1]) : 1000000;
    int q_max = argc > 2 ? atoi(argv[2]) : 100;  // major arc threshold
    int gpu_id = argc > 3 ? atoi(argv[3]) : 0;
    cudaSetDevice(gpu_id);

    printf("Minor Arc Spectral Profile\n");
    printf("Grid: %d points, Major arc q_max=%d, N=%d Chebyshev\n\n", grid_size, q_max, N_CHEB);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Generate α grid — uniform on [0, 0.5] (symmetry: L_{δ,α} = L_{δ,1-α}*)
    float *h_alphas = (float*)malloc(grid_size * sizeof(float));
    for (int i = 0; i < grid_size; i++) {
        h_alphas[i] = (float)(i + 0.5) / (2.0f * grid_size); // (0, 0.5)
    }

    float *d_alphas, *d_radii;
    cudaMalloc(&d_alphas, grid_size * sizeof(float));
    cudaMalloc(&d_radii, grid_size * sizeof(float));
    cudaMemcpy(d_alphas, h_alphas, grid_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (grid_size + threads - 1) / threads;
    twisted_spectral_radius<<<blocks, threads>>>(d_alphas, d_radii, grid_size);
    cudaDeviceSynchronize();

    float *h_radii = (float*)malloc(grid_size * sizeof(float));
    cudaMemcpy(h_radii, d_radii, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Analyze: find minor arc regions where radius < 1 - ε
    float max_minor_radius = 0;
    float max_radius = 0;
    int max_radius_idx = 0;
    int minor_count = 0;
    float eps = 0.01f; // minor arc threshold

    // Identify major arcs: α near a/q for q ≤ q_max
    // A point α is on the major arc if |α - a/q| < 1/(q*Q) for some Q
    // For now, flag points within 1/(q^2) of any a/q with q ≤ q_max
    for (int i = 0; i < grid_size; i++) {
        float alpha = h_alphas[i];
        float r = h_radii[i];

        if (r > max_radius) {
            max_radius = r;
            max_radius_idx = i;
        }

        // Check if on major arc
        int on_major = 0;
        for (int q = 1; q <= q_max && !on_major; q++) {
            for (int a = 0; a <= q/2; a++) {
                float rational = (float)a / q;
                if (fabsf(alpha - rational) < 1.0f / (q * q)) {
                    on_major = 1;
                    break;
                }
            }
        }

        if (!on_major) {
            minor_count++;
            if (r > max_minor_radius) max_minor_radius = r;
        }
    }

    printf("========================================\n");
    printf("Time: %.2fs\n", elapsed);
    printf("Grid points: %d\n", grid_size);
    printf("Max spectral radius (overall): %.6f at α=%.8f\n",
           max_radius, h_alphas[max_radius_idx]);
    printf("Minor arc points (q_max=%d): %d\n", q_max, minor_count);
    printf("Max spectral radius on MINOR arc: %.6f\n", max_minor_radius);
    printf("Minor arc gap: %.6f\n", 1.0f - max_minor_radius);
    printf("========================================\n\n");

    // Print spectral radius histogram
    printf("Spectral radius histogram:\n");
    int bins[20] = {0};
    for (int i = 0; i < grid_size; i++) {
        int b = (int)(h_radii[i] * 20);
        if (b >= 20) b = 19;
        if (b < 0) b = 0;
        bins[b]++;
    }
    for (int b = 0; b < 20; b++) {
        printf("  [%.2f, %.2f): %d\n", b/20.0f, (b+1)/20.0f, bins[b]);
    }

    // Output top-10 spectral radius values (the "hardest" α values)
    printf("\nTop-10 spectral radii (hardest minor arc points):\n");
    // Simple selection of top 10
    for (int t = 0; t < 10; t++) {
        float best = -1;
        int best_i = -1;
        for (int i = 0; i < grid_size; i++) {
            if (h_radii[i] > best) {
                // Check not already picked
                int skip = 0;
                // (simplified: just pick the top 10 including major arc)
                best = h_radii[i];
                best_i = i;
            }
        }
        if (best_i >= 0) {
            printf("  α = %.10f, |λ| = %.6f\n", h_alphas[best_i], h_radii[best_i]);
            h_radii[best_i] = -1; // mark as picked
        }
    }

    cudaFree(d_alphas);
    cudaFree(d_radii);
    free(h_alphas);
    free(h_radii);
    return 0;
}
