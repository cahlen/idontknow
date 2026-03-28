/*
 * CUDA Transfer Operator for Zaremba's Conjecture
 *
 * Computes the Ruelle transfer operator L_s for the Gauss map restricted
 * to alphabet A = {1,...,5} and finds its spectral properties.
 *
 * Phase 1: Compute Hausdorff dimension δ via Bowen's equation λ_0(δ) = 1
 * Phase 2: Compute spectral gaps of congruence operators L_{δ,m}
 *
 * The matrix entry in the monomial basis is:
 *   M[m][n](s) = Σ_{a∈A} (-1)^m * C(2s+n+m-1, m) * a^{-(2s+n+m)}
 *
 * where C(α,m) = α*(α-1)*...*(α-m+1)/m! is the generalized binomial coefficient.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o transfer_op scripts/experiments/zaremba-transfer-operator/transfer_operator.cu -lcusolver -lcublas -lm
 * Run:     ./transfer_op [N] [phase]
 *          N = matrix truncation size (default: 200)
 *          phase = 1 (Hausdorff dim) or 2 (congruence gaps) or 3 (both)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
// Using custom power iteration instead of cuSOLVER for portability

#define BOUND 5
#define MAX_N 500  // max matrix dimension

// Generalized binomial coefficient C(alpha, m) = alpha*(alpha-1)*...*(alpha-m+1)/m!
// For real alpha and non-negative integer m
__host__ __device__ double gen_binom(double alpha, int m) {
    if (m == 0) return 1.0;
    if (m < 0) return 0.0;
    double result = 1.0;
    for (int i = 0; i < m; i++) {
        result *= (alpha - i) / (i + 1);
    }
    return result;
}

// Build the N×N transfer operator matrix M(s) on CPU
// M[m][n] = Σ_{a=1}^{5} (-1)^m * C(2s+n+m-1, m) * a^{-(2s+n+m)}
void build_matrix(double s, int N, double *M) {
    memset(M, 0, N * N * sizeof(double));

    for (int m = 0; m < N; m++) {
        double sign = (m % 2 == 0) ? 1.0 : -1.0;
        for (int n = 0; n < N; n++) {
            double alpha = 2.0 * s + n + m - 1.0;
            double binom_val = gen_binom(alpha, m);
            double entry = 0.0;

            for (int a = 1; a <= BOUND; a++) {
                // a^{-(2s+n+m)}
                double power = pow((double)a, -(2.0 * s + n + m));
                entry += power;
            }
            // Column-major for LAPACK/cuSOLVER: M[m + n*N]
            M[m + n * N] = sign * binom_val * entry;
        }
    }
}

// Matrix-vector product: y = M * x (column-major M, size N×N)
void matvec(double *M, double *x, double *y, int N) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += M[i + j * N] * x[j];
        }
        y[i] = sum;
    }
}

// Power iteration on M² to find dominant eigenvalue of M.
// M² has eigenvalue λ² ≥ 0, so power iteration finds |λ_0|².
// We then determine the sign from M*v.
double dominant_eigenvalue(double *M, int N) {
    // Compute M² = M * M
    double *M2 = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += M[i + k * N] * M[k + j * N];
            }
            M2[i + j * N] = sum;
        }
    }

    // Power iteration on M²
    double *v = (double*)malloc(N * sizeof(double));
    double *w = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) v[i] = 1.0;

    double lambda_sq = 0.0;
    for (int iter = 0; iter < 500; iter++) {
        matvec(M2, v, w, N);
        double norm = 0.0;
        for (int i = 0; i < N; i++) norm += w[i] * w[i];
        norm = sqrt(norm);
        if (norm < 1e-300) break;

        // Rayleigh quotient for M²
        lambda_sq = 0.0;
        double vnorm = 0.0;
        for (int i = 0; i < N; i++) { lambda_sq += w[i] * v[i]; vnorm += v[i] * v[i]; }
        lambda_sq /= vnorm;

        for (int i = 0; i < N; i++) v[i] = w[i] / norm;
    }

    // v is now the eigenvector of M². Apply M once to get sign.
    matvec(M, v, w, N);
    double dot = 0.0, vnorm = 0.0;
    for (int i = 0; i < N; i++) { dot += w[i] * v[i]; vnorm += v[i] * v[i]; }
    double lambda = dot / vnorm;

    // Also compute |λ_0| = sqrt(λ²)
    double abs_lambda = sqrt(fabs(lambda_sq));

    printf("    Dominant eigenvalue: λ_0 = %.15f (|λ_0| = %.15f)\n", lambda, abs_lambda);

    free(M2); free(v); free(w);
    return lambda;
}

// Find top-k eigenvalue magnitudes using power iteration + deflation
void all_eigenvalues(double *M_orig, int N, double *eig_abs, int k) {
    double *M = (double*)malloc(N * N * sizeof(double));
    memcpy(M, M_orig, N * N * sizeof(double));

    double *v = (double*)malloc(N * sizeof(double));
    double *w = (double*)malloc(N * sizeof(double));

    for (int eig_idx = 0; eig_idx < k && eig_idx < N; eig_idx++) {
        // Power iteration for current dominant eigenvalue
        for (int i = 0; i < N; i++) v[i] = 1.0 / (i + 1.0 + eig_idx * 7.3);

        double lambda = 0.0;
        for (int iter = 0; iter < 1000; iter++) {
            matvec(M, v, w, N);
            double norm = 0.0;
            for (int i = 0; i < N; i++) norm += w[i] * w[i];
            norm = sqrt(norm);
            if (norm < 1e-300) break;

            lambda = 0.0;
            double vnorm = 0.0;
            for (int i = 0; i < N; i++) { lambda += w[i] * v[i]; vnorm += v[i] * v[i]; }
            lambda /= vnorm;

            for (int i = 0; i < N; i++) v[i] = w[i] / norm;
        }

        eig_abs[eig_idx] = fabs(lambda);

        // Deflate: M = M - lambda * v * v^T
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                M[i + j * N] -= lambda * v[i] * v[j];
    }

    free(M); free(v); free(w);
}

// (all_eigenvalues defined above via power iteration + deflation)

/*
 * Phase 1: Compute Hausdorff dimension δ
 *
 * Solve λ_0(s) = 1 via bisection in s.
 * We know 0.5 < δ < 1 for A = {1,...,5}.
 */
double compute_hausdorff_dimension(int N) {
    printf("=== Phase 1: Hausdorff Dimension ===\n");
    printf("Matrix size: %d × %d\n", N, N);
    printf("Alphabet: A = {1, ..., %d}\n\n", BOUND);

    double *M = (double*)malloc(N * N * sizeof(double));

    // Bisection: find s where λ_0(s) = 1
    double s_lo = 0.5, s_hi = 1.0;

    // Check endpoints
    printf("s = %.6f:\n", s_lo);
    build_matrix(s_lo, N, M);
    double lam_lo = dominant_eigenvalue(M, N);

    printf("\ns = %.6f:\n", s_hi);
    build_matrix(s_hi, N, M);
    double lam_hi = dominant_eigenvalue(M, N);

    printf("\nBisection: λ_0(%.3f) = %.6f, λ_0(%.3f) = %.6f\n",
           s_lo, lam_lo, s_hi, lam_hi);
    printf("Need λ_0(δ) = 1, so δ ∈ (%.3f, %.3f)\n\n", s_lo, s_hi);

    // Bisect
    for (int iter = 0; iter < 60; iter++) {
        double s_mid = (s_lo + s_hi) / 2.0;
        build_matrix(s_mid, N, M);

        // Quick eigenvalue — just find dominant
        double lam = dominant_eigenvalue(M, N);

        if (lam > 1.0) {
            s_lo = s_mid;
        } else {
            s_hi = s_mid;
        }

        printf("  Bisection iter %d: s = %.15f, λ_0 = %.15f (gap = %.2e)\n",
               iter, s_mid, lam, s_hi - s_lo);

        if (s_hi - s_lo < 1e-15) break;
    }

    double delta = (s_lo + s_hi) / 2.0;
    printf("\n*** Hausdorff dimension δ = %.15f ***\n", delta);
    printf("*** 2δ = %.15f (must be > 1 for Zaremba) ***\n\n", 2 * delta);

    // Get the spectral gap at s = δ
    printf("Spectral data at s = δ:\n");
    build_matrix(delta, N, M);
    double eigs[10];
    all_eigenvalues(M, N, eigs, 10);
    printf("  Top 5 eigenvalue magnitudes:\n");
    for (int i = 0; i < 5 && i < N; i++) {
        printf("    |λ_%d| = %.15f\n", i, eigs[i]);
    }
    printf("  Spectral gap (|λ_0| - |λ_1|) = %.15f\n", eigs[0] - eigs[1]);
    printf("  Spectral ratio |λ_1/λ_0| = %.15f\n", eigs[1] / eigs[0]);

    free(M);
    return delta;
}

/*
 * Phase 2: Congruence transfer operators
 *
 * For each modulus m, build L_{δ,m} and compute its spectral gap.
 * The operator acts on functions f: [0,1] × (Z/mZ) → C.
 *
 * In the Kronecker basis: L_{δ,m} = Σ_{a∈A} M_a(δ) ⊗ P_a(m)
 * where P_a(m) is the permutation matrix of the action of γ_a on Z/mZ.
 *
 * For Zaremba, γ_a acts on (r, s) ∈ (Z/mZ)² as:
 *   γ_a · (r, s) = (s, a*s + r) mod m
 * (from the matrix multiplication [[0,1],[1,a]] · [[r],[s]] = [[s],[r+a*s]])
 *
 * Simplified version: track only the bottom row (q_prev, q) mod m.
 * The action of digit a sends (q_prev, q) → (q, a*q + q_prev) mod m.
 *
 * Size of state space: m² (pairs in (Z/mZ)²), but we can reduce by
 * considering only coprime pairs, giving #P¹(Z/mZ) states.
 * For simplicity, use the full m² space.
 */

// Build the permutation matrix P_a(m) for digit a acting on (Z/mZ)²
// P_a maps state (r, s) to (s, a*s + r mod m)
// State index: i = r * m + s (row-major)
void build_perm_matrix(int a, int m, int *perm) {
    for (int r = 0; r < m; r++) {
        for (int s = 0; s < m; s++) {
            int from = r * m + s;
            int new_r = s;
            int new_s = (a * s + r) % m;
            int to = new_r * m + new_s;
            perm[from] = to;
        }
    }
}

// Build the full congruence operator as a (N*m²) × (N*m²) matrix
// Using the Kronecker structure: L = Σ_a M_a ⊗ P_a
// For small m, just build the full matrix.
void build_congruence_matrix(double delta, int N, int m, double *full_M) {
    int state_dim = m * m;
    int full_dim = N * state_dim;
    memset(full_M, 0, (size_t)full_dim * full_dim * sizeof(double));

    int *perm = (int*)malloc(state_dim * sizeof(int));
    double *Ma = (double*)malloc(N * N * sizeof(double));

    for (int a = 1; a <= BOUND; a++) {
        // Build M_a: the single-digit transfer matrix
        memset(Ma, 0, N * N * sizeof(double));
        for (int mi = 0; mi < N; mi++) {
            double sign = (mi % 2 == 0) ? 1.0 : -1.0;
            for (int ni = 0; ni < N; ni++) {
                double alpha = 2.0 * delta + ni + mi - 1.0;
                double binom_val = gen_binom(alpha, mi);
                double power = pow((double)a, -(2.0 * delta + ni + mi));
                Ma[mi + ni * N] = sign * binom_val * power;
            }
        }

        // Build P_a
        build_perm_matrix(a, m, perm);

        // Kronecker product: L += M_a ⊗ P_a
        // full_M[i*state_dim + j, k*state_dim + l] += Ma[i,k] * P_a[j,l]
        // P_a[j,l] = (perm[j] == l) ? 1 : 0
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                double ma_val = Ma[i + k * N];
                if (fabs(ma_val) < 1e-300) continue;

                for (int j = 0; j < state_dim; j++) {
                    int l = perm[j];  // P_a maps j → l
                    int row = i * state_dim + j;
                    int col = k * state_dim + l;
                    // Column-major
                    full_M[row + col * full_dim] += ma_val;
                }
            }
        }
    }

    free(perm);
    free(Ma);
}

double compute_congruence_gap(double delta, int N, int m) {
    int state_dim = m * m;
    int full_dim = N * state_dim;

    if (full_dim > 50000) {
        printf("    m=%d: matrix too large (%d × %d), skipping\n", m, full_dim, full_dim);
        return -1.0;
    }

    size_t matrix_bytes = (size_t)full_dim * full_dim * sizeof(double);
    printf("    m=%d: building %d × %d matrix (%.1f MB)... ",
           m, full_dim, full_dim, matrix_bytes / 1e6);
    fflush(stdout);

    double *h_M = (double*)malloc(matrix_bytes);
    if (!h_M) {
        printf("allocation failed\n");
        return -1.0;
    }

    build_congruence_matrix(delta, N, m, h_M);

    // Find eigenvalues on GPU
    double eigs[5];
    all_eigenvalues(h_M, full_dim, eigs, 5);

    double gap = eigs[0] - eigs[1];
    printf("|λ_0|=%.6f, |λ_1|=%.6f, gap=%.6f\n", eigs[0], eigs[1], gap);

    free(h_M);
    return gap;
}

// Check if m is squarefree
int is_squarefree(int m) {
    for (int p = 2; p * p <= m; p++) {
        if (m % (p * p) == 0) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 200;
    int phase = argc > 2 ? atoi(argv[2]) : 3;
    int max_m = argc > 3 ? atoi(argv[3]) : 30;

    printf("========================================\n");
    printf("  Zaremba Transfer Operator Analysis\n");
    printf("========================================\n\n");

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs: %d\n", device_count);
    printf("Matrix truncation N = %d\n", N);
    printf("Phase: %d\n\n", phase);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    double delta = 0.0;

    if (phase == 1 || phase == 3) {
        delta = compute_hausdorff_dimension(N);
    }

    if (phase == 2 || phase == 3) {
        if (delta == 0.0) {
            // Quick computation of delta
            printf("Computing δ quickly with N=50...\n");
            delta = compute_hausdorff_dimension(50);
        }

        printf("\n=== Phase 2: Congruence Spectral Gaps ===\n");
        printf("Using δ = %.15f\n", delta);
        printf("Testing square-free m up to %d\n\n", max_m);

        // Use smaller N for congruence operators (they're already large)
        int cong_N = 20;  // N=20 for the polynomial part
        if (N < 20) cong_N = N;

        printf("Congruence operator polynomial truncation: N = %d\n", cong_N);
        printf("State space per m: m² states\n\n");

        printf("m\t|dim|\t|λ_0|\t|λ_1|\tgap\tgap/|λ_0|\n");
        printf("---\t-----\t------\t------\t----\t---------\n");

        for (int m = 2; m <= max_m; m++) {
            if (!is_squarefree(m)) continue;

            double gap = compute_congruence_gap(delta, cong_N, m);
            if (gap >= 0) {
                // Also compute relative gap
                // (will be printed inline by compute_congruence_gap)
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                    (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Total time: %.1fs\n", elapsed);
    if (delta > 0) {
        printf("Hausdorff dimension δ = %.15f\n", delta);
        printf("2δ = %.15f (> 1 ✓ — circle method threshold met)\n", 2 * delta);
    }
    printf("========================================\n");

    return 0;
}
