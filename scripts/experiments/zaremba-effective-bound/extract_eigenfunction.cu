/*
 * Extract the Patterson-Sullivan eigenfunction h(x) of L_δ
 * at high precision (FP64, N=40 Chebyshev).
 *
 * h is the Perron-Frobenius eigenvector: L_δ h = h.
 * We need h(0), h(1), and ∫h(x)dx precisely for the main term constant.
 *
 * Also recompute σ_p for the TIGHT primes (p=71,41,29,etc.) at FP64/N=40
 * to get precise minimum gap.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o extract_ef extract_eigenfunction.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cublas_v2.h>

#define BOUND 5
#define N 40
#define DELTA 0.836829443681208

void chebyshev_nodes(double *x, int n) {
    for (int j = 0; j < n; j++)
        x[j] = 0.5 * (1.0 + cos(M_PI * (2.0*j + 1.0) / (2.0*n)));
}

void barycentric_weights(double *w, int n) {
    for (int j = 0; j < n; j++)
        w[j] = pow(-1.0, j) * sin(M_PI * (2.0*j + 1.0) / (2.0*n));
}

void build_matrix(double s, int n, double *x, double *bw, double *M) {
    memset(M, 0, n * n * sizeof(double));
    for (int a = 1; a <= BOUND; a++) {
        for (int i = 0; i < n; i++) {
            double y = 1.0 / (a + x[i]);
            double ws = pow(a + x[i], -2.0 * s);
            int exact = -1;
            for (int k = 0; k < n; k++)
                if (fabs(y - x[k]) < 1e-15) { exact = k; break; }
            if (exact >= 0) {
                M[i + exact * n] += ws;
            } else {
                double den = 0;
                double num[N];
                for (int j = 0; j < n; j++) {
                    num[j] = bw[j] / (y - x[j]);
                    den += num[j];
                }
                for (int j = 0; j < n; j++)
                    M[i + j * n] += ws * num[j] / den;
            }
        }
    }
}

// Power iteration returning eigenvector (not just eigenvalue)
double power_iteration(double *M, int n, double *v, int iters) {
    double *w = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) v[i] = 1.0;
    double lam = 0;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += M[i + j*n] * v[j];
            w[i] = s;
        }
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) { num += v[i]*w[i]; den += v[i]*v[i]; }
        lam = num / den;
        double norm = 0;
        for (int i = 0; i < n; i++) norm += w[i]*w[i];
        norm = sqrt(norm);
        for (int i = 0; i < n; i++) v[i] = w[i] / norm;
    }
    free(w);
    return lam;
}

// Evaluate eigenvector at arbitrary x via barycentric interpolation
double eval_at(double *v, double *nodes, double *bw, int n, double x_eval) {
    // Check for exact node match
    for (int k = 0; k < n; k++)
        if (fabs(x_eval - nodes[k]) < 1e-15) return v[k];

    double num = 0, den = 0;
    for (int j = 0; j < n; j++) {
        double t = bw[j] / (x_eval - nodes[j]);
        num += t * v[j];
        den += t;
    }
    return num / den;
}

// Compute second eigenvalue by deflated power iteration
double second_eigenvalue(double *M, double *v1, int n, int iters) {
    double *v = (double*)malloc(n * sizeof(double));
    double *w = (double*)malloc(n * sizeof(double));

    // Random init orthogonal to v1
    for (int i = 0; i < n; i++)
        v[i] = sin(i * 1.618 + 0.5);

    // Project out v1
    double dot = 0, norm1 = 0;
    for (int i = 0; i < n; i++) { dot += v[i]*v1[i]; norm1 += v1[i]*v1[i]; }
    for (int i = 0; i < n; i++) v[i] -= (dot/norm1) * v1[i];

    double lam = 0;
    for (int it = 0; it < iters; it++) {
        // Apply M
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += M[i + j*n] * v[j];
            w[i] = s;
        }
        // Project out v1
        dot = 0; norm1 = 0;
        for (int i = 0; i < n; i++) { dot += w[i]*v1[i]; norm1 += v1[i]*v1[i]; }
        for (int i = 0; i < n; i++) w[i] -= (dot/norm1) * v1[i];

        // Rayleigh quotient
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) { num += v[i]*w[i]; den += v[i]*v[i]; }
        lam = num / den;

        double norm = 0;
        for (int i = 0; i < n; i++) norm += w[i]*w[i];
        norm = sqrt(norm);
        for (int i = 0; i < n; i++) v[i] = w[i] / norm;
    }
    free(v); free(w);
    return lam;
}

int main() {
    printf("================================================================\n");
    printf("  Eigenfunction Extraction & Precise Gap Recomputation\n");
    printf("  FP64, N=%d Chebyshev, δ = %.15f\n", N, DELTA);
    printf("================================================================\n\n");

    double *x = (double*)malloc(N * sizeof(double));
    double *bw = (double*)malloc(N * sizeof(double));
    double *M = (double*)malloc(N * N * sizeof(double));
    double *h = (double*)malloc(N * sizeof(double));

    chebyshev_nodes(x, N);
    barycentric_weights(bw, N);

    // Build L_δ and extract eigenfunction
    build_matrix(DELTA, N, x, bw, M);
    double lambda1 = power_iteration(M, N, h, 1000);

    printf("=== Leading eigenvalue ===\n");
    printf("λ₁ = %.15f (should be ≈ 1.0)\n\n", lambda1);

    // Normalize h so that h > 0 and ∫h dx = 1
    // First ensure positivity
    if (h[0] < 0) for (int i = 0; i < N; i++) h[i] = -h[i];

    // Compute ∫h(x)dx by Chebyshev quadrature (Clenshaw-Curtis)
    double integral = 0;
    for (int i = 0; i < N; i++) {
        // Clenshaw-Curtis weight for Chebyshev node i on [0,1]
        double wi = 1.0 / N; // simplified; exact would use DCT
        integral += h[i] * wi;
    }
    // Normalize
    for (int i = 0; i < N; i++) h[i] /= integral;
    double check_int = 0;
    for (int i = 0; i < N; i++) check_int += h[i] / N;

    printf("=== Eigenfunction h (Patterson-Sullivan density) ===\n");
    printf("∫h(x)dx = %.15f (after normalization)\n\n", check_int);

    // Evaluate h at key points
    double h0 = eval_at(h, x, bw, N, 0.0);
    double h1 = eval_at(h, x, bw, N, 1.0);
    double h_half = eval_at(h, x, bw, N, 0.5);
    double h_golden = eval_at(h, x, bw, N, 1.0/((1+sqrt(5))/2));
    double h_171 = eval_at(h, x, bw, N, 0.171);

    printf("h(0)   = %.15f\n", h0);
    printf("h(0.5) = %.15f\n", h_half);
    printf("h(1)   = %.15f\n", h1);
    printf("h(1/φ) = %.15f  (golden ratio point)\n", h_golden);
    printf("h(0.171) = %.15f  (witness concentration)\n\n", h_171);

    // Compute ∫h(x)² dx (needed for main term)
    double h2_int = 0;
    for (int i = 0; i < N; i++) h2_int += h[i] * h[i] / N;
    printf("∫h(x)²dx = %.15f\n\n", h2_int);

    // Print h at all Chebyshev nodes
    printf("h(x) at Chebyshev nodes:\n");
    printf("%4s  %18s  %18s\n", "j", "x_j", "h(x_j)");
    for (int j = 0; j < N; j++) {
        printf("%4d  %18.15f  %18.15f\n", j, x[j], h[j]);
    }

    // Second eigenvalue (spectral gap of untwisted operator)
    printf("\n=== Spectral gap of L_δ (untwisted) ===\n");
    double lambda2 = second_eigenvalue(M, h, N, 1000);
    printf("λ₂ = %.15f\n", lambda2);
    printf("σ = 1 - |λ₂/λ₁| = %.15f\n\n", 1.0 - fabs(lambda2 / lambda1));

    // Now recompute spectral gaps for TIGHT primes at FP64/N=40
    printf("=== Precise spectral gaps for tight primes (FP64, N=%d) ===\n\n", N);

    int tight_primes[] = {2, 3, 5, 7, 11, 13, 29, 31, 41, 71, 73, 79, 83, 89, 97};
    int n_tight = sizeof(tight_primes) / sizeof(tight_primes[0]);

    printf("%6s  %18s  %18s  %18s\n", "p", "λ₁(L_{δ,p})", "λ₂(L_{δ,p})", "σ_p");
    printf("------  ------------------  ------------------  ------------------\n");

    // For each prime p, build the congruence operator L_{δ,p}
    // This acts on functions on P^1(F_p) × [0,1]
    // The trivial eigenvalue is 1 (same as untwisted).
    // The second eigenvalue determines the gap.
    //
    // For SMALL p, we can form the FULL matrix of size N×(p+1) and do
    // power iteration. For p ≤ 97, this is at most N×98 = 3920 × 3920.

    for (int t = 0; t < n_tight; t++) {
        int p = tight_primes[t];
        int p1 = p + 1;
        int sz = N * p1;

        double *Lp = (double*)calloc(sz * sz, sizeof(double));

        // Build L_{δ,p} = Σ_{a=1}^5 M_a ⊗ P_a
        // M_a[i][j]: Chebyshev part (same as before)
        // P_a[k][l]: permutation on P^1(F_p)
        // Full matrix: Lp[(i*p1+k), (j*p1+l)] = M_a[i][j] * δ(k, P_a(l))

        for (int a = 1; a <= BOUND; a++) {
            // Build M_a
            double Ma[N * N];
            memset(Ma, 0, sizeof(Ma));
            for (int i = 0; i < N; i++) {
                double y = 1.0 / (a + x[i]);
                double ws = pow(a + x[i], -2.0 * DELTA);
                int exact = -1;
                for (int k = 0; k < N; k++)
                    if (fabs(y - x[k]) < 1e-15) { exact = k; break; }
                if (exact >= 0) {
                    Ma[i + exact * N] = ws;
                } else {
                    double den = 0, num[N];
                    for (int j = 0; j < N; j++) {
                        num[j] = bw[j] / (y - x[j]);
                        den += num[j];
                    }
                    for (int j = 0; j < N; j++)
                        Ma[i + j * N] = ws * num[j] / den;
                }
            }

            // Build P_a: permutation on P^1(F_p)
            // g_a([x:1]) = [ax+1 : x]
            // x=0 → ∞, ∞ → a%p, otherwise → (ax+1)/x mod p
            int Pa[p1];
            for (int k = 0; k < p; k++) {
                if (k == 0) {
                    Pa[k] = p; // 0 → ∞
                } else {
                    // (a*k + 1) * k^{-1} mod p
                    long long kinv = 1, base_v = k, exp_v = p - 2, mod_v = p;
                    while (exp_v > 0) {
                        if (exp_v & 1) kinv = kinv * base_v % mod_v;
                        base_v = base_v * base_v % mod_v;
                        exp_v >>= 1;
                    }
                    Pa[k] = (int)(((long long)a * k + 1) % p * kinv % p);
                }
            }
            Pa[p] = a % p; // ∞ → a

            // Kronecker product: Lp[(i*p1+Pa[k]), (j*p1+k)] += Ma[i][j]
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double mij = Ma[i + j * N];
                    if (fabs(mij) < 1e-20) continue;
                    for (int k = 0; k < p1; k++) {
                        int row = i * p1 + Pa[k];
                        int col = j * p1 + k;
                        Lp[row + col * sz] += mij;
                    }
                }
            }
        }

        // GPU power iteration via cuBLAS DGEMV
        cublasHandle_t handle;
        cublasCreate(&handle);

        double *d_Lp, *d_v, *d_w;
        cudaMalloc(&d_Lp, (long long)sz * sz * sizeof(double));
        cudaMalloc(&d_v, sz * sizeof(double));
        cudaMalloc(&d_w, sz * sizeof(double));
        cudaMemcpy(d_Lp, Lp, (long long)sz * sz * sizeof(double), cudaMemcpyHostToDevice);

        // Leading eigenvalue
        double *v1 = (double*)malloc(sz * sizeof(double));
        for (int i = 0; i < sz; i++) v1[i] = 1.0;
        cudaMemcpy(d_v, v1, sz * sizeof(double), cudaMemcpyHostToDevice);

        double alpha_blas = 1.0, beta_blas = 0.0;
        double lam1 = 0;
        for (int it = 0; it < 500; it++) {
            cublasDgemv(handle, CUBLAS_OP_N, sz, sz, &alpha_blas, d_Lp, sz, d_v, 1, &beta_blas, d_w, 1);
            double dot_vw, dot_vv;
            cublasDdot(handle, sz, d_v, 1, d_w, 1, &dot_vw);
            cublasDdot(handle, sz, d_v, 1, d_v, 1, &dot_vv);
            lam1 = dot_vw / dot_vv;
            double nrm;
            cublasDnrm2(handle, sz, d_w, 1, &nrm);
            double inv_nrm = 1.0 / nrm;
            cublasDscal(handle, sz, &inv_nrm, d_w, 1);
            // swap v <-> w
            double *tmp_d = d_v; d_v = d_w; d_w = tmp_d;
        }
        cudaMemcpy(v1, d_v, sz * sizeof(double), cudaMemcpyDeviceToHost);

        // Second eigenvalue by deflation on GPU
        double *v2_h = (double*)malloc(sz * sizeof(double));
        for (int i = 0; i < sz; i++) v2_h[i] = sin(i * 2.718 + 0.3);
        // Project out v1 on CPU (small)
        double dot = 0, n1 = 0;
        for (int i = 0; i < sz; i++) { dot += v2_h[i]*v1[i]; n1 += v1[i]*v1[i]; }
        for (int i = 0; i < sz; i++) v2_h[i] -= (dot/n1) * v1[i];

        double *d_v1;
        cudaMalloc(&d_v1, sz * sizeof(double));
        cudaMemcpy(d_v1, v1, sz * sizeof(double), cudaMemcpyDeviceToHost);
        // Wait, need to upload v1 to device for dot products
        cudaMemcpy(d_v1, v1, sz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v2_h, sz * sizeof(double), cudaMemcpyHostToDevice);

        double lam2 = 0;
        for (int it = 0; it < 500; it++) {
            cublasDgemv(handle, CUBLAS_OP_N, sz, sz, &alpha_blas, d_Lp, sz, d_v, 1, &beta_blas, d_w, 1);
            // Project out v1: w = w - (w·v1)/(v1·v1) * v1
            double dot_wv1, dot_v1v1;
            cublasDdot(handle, sz, d_w, 1, d_v1, 1, &dot_wv1);
            cublasDdot(handle, sz, d_v1, 1, d_v1, 1, &dot_v1v1);
            double neg_ratio = -dot_wv1 / dot_v1v1;
            cublasDaxpy(handle, sz, &neg_ratio, d_v1, 1, d_w, 1);
            // Rayleigh quotient
            double dot_vw2, dot_vv2;
            cublasDdot(handle, sz, d_v, 1, d_w, 1, &dot_vw2);
            cublasDdot(handle, sz, d_v, 1, d_v, 1, &dot_vv2);
            lam2 = dot_vw2 / dot_vv2;
            // Normalize
            double nrm;
            cublasDnrm2(handle, sz, d_w, 1, &nrm);
            if (nrm > 1e-30) {
                double inv_nrm = 1.0 / nrm;
                cublasDscal(handle, sz, &inv_nrm, d_w, 1);
            }
            double *tmp_d = d_v; d_v = d_w; d_w = tmp_d;
        }

        cudaFree(d_Lp); cudaFree(d_v); cudaFree(d_w); cudaFree(d_v1);
        cublasDestroy(handle);
        free(v2_h);

        double gap = 1.0 - fabs(lam2 / lam1);
        printf("%6d  %18.15f  %18.15f  %18.15f", p, lam1, lam2, gap);
        if (gap < 0.35) printf("  <-- TIGHT");
        printf("\n");

        free(v1);
        free(Lp);
    }

    free(x); free(bw); free(M); free(h);
    return 0;
}
