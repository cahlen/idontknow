/*
 * Zaremba Transfer Operator v3 — implicit Kronecker, scales to m=200+
 *
 * KEY OPTIMIZATION: Never form the full (N·m²)×(N·m²) matrix.
 * Instead, compute matrix-vector products implicitly:
 *   (L_{δ,m} · v) = Σ_{a∈A} (M_a ⊗ P_a) · v
 * Each term: permute v's fiber indices by P_a, then multiply by M_a.
 * Memory: O(N·m²) for vectors, O(N²) for M_a. No O(N²·m⁴) matrix.
 *
 * This lets us handle m=200+ on a single B200 (183GB).
 *
 * Compile: nvcc -O3 -arch=sm_100a -o transfer_op scripts/experiments/zaremba-transfer-operator/transfer_operator.cu -lcublas -lm -lpthread
 * Run:     ./transfer_op [N] [phase] [max_m]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <cublas_v2.h>

#define BOUND 5
#define MAX_N 200

// ============================================================
// Phase 1: Hausdorff dimension (CPU, tiny matrix)
// ============================================================

void chebyshev_nodes(double *x, int N) {
    for (int j = 0; j < N; j++)
        x[j] = 0.5 * (1.0 + cos(M_PI * (2.0*j+1.0) / (2.0*N)));
}

void barycentric_weights(double *w, int N) {
    for (int j = 0; j < N; j++)
        w[j] = pow(-1.0, j) * sin(M_PI * (2.0*j+1.0) / (2.0*N));
}

void build_single_digit_matrix(int a, double s, int N, double *x, double *bw, double *Ma) {
    memset(Ma, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double y = 1.0 / (a + x[i]);
        double ws = pow(a + x[i], -2.0 * s);
        int exact = -1;
        for (int k = 0; k < N; k++)
            if (fabs(y - x[k]) < 1e-15) { exact = k; break; }
        if (exact >= 0) { Ma[i + exact * N] = ws; }
        else {
            double den = 0; double num[MAX_N];
            for (int j = 0; j < N; j++) { num[j] = bw[j]/(y-x[j]); den += num[j]; }
            for (int j = 0; j < N; j++) Ma[i + j * N] = ws * num[j] / den;
        }
    }
}

void build_full_matrix(double s, int N, double *x, double *bw, double *M) {
    memset(M, 0, N * N * sizeof(double));
    double *Ma = (double*)malloc(N * N * sizeof(double));
    for (int a = 1; a <= BOUND; a++) {
        build_single_digit_matrix(a, s, N, x, bw, Ma);
        for (int i = 0; i < N*N; i++) M[i] += Ma[i];
    }
    free(Ma);
}

double power_iteration_cpu(double *M, int N, int iters) {
    double *v = (double*)malloc(N * sizeof(double));
    double *w = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) v[i] = 1.0;
    double lam = 0.0;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < N; i++) {
            double s = 0; for (int j = 0; j < N; j++) s += M[i+j*N]*v[j]; w[i]=s;
        }
        double num=0,den=0;
        for (int i=0;i<N;i++){num+=v[i]*w[i];den+=v[i]*v[i];}
        lam=num/den;
        double norm=0; for(int i=0;i<N;i++) norm+=w[i]*w[i]; norm=sqrt(norm);
        for(int i=0;i<N;i++) v[i]=w[i]/norm;
    }
    free(v); free(w);
    return lam;
}

double compute_hausdorff_dimension(int N) {
    printf("=== Phase 1: Hausdorff Dimension (N=%d) ===\n\n", N);
    double *x=(double*)malloc(N*sizeof(double));
    double *bw=(double*)malloc(N*sizeof(double));
    double *M=(double*)malloc(N*N*sizeof(double));
    chebyshev_nodes(x,N); barycentric_weights(bw,N);

    double s_lo=0.5, s_hi=1.0;
    build_full_matrix(s_lo,N,x,bw,M); double l_lo=power_iteration_cpu(M,N,300);
    build_full_matrix(s_hi,N,x,bw,M); double l_hi=power_iteration_cpu(M,N,300);
    printf("λ_0(%.1f)=%.6f, λ_0(%.1f)=%.6f\n\n",s_lo,l_lo,s_hi,l_hi);

    for(int it=0;it<55;it++){
        double s=(s_lo+s_hi)/2;
        build_full_matrix(s,N,x,bw,M);
        double lam=power_iteration_cpu(M,N,300);
        if(lam>1.0) s_lo=s; else s_hi=s;
        if(it%10==0||s_hi-s_lo<1e-14)
            printf("  iter %2d: δ≈%.15f  λ=%.15f  gap=%.2e\n",it,s,lam,s_hi-s_lo);
        if(s_hi-s_lo<1e-15) break;
    }
    double delta=(s_lo+s_hi)/2;
    printf("\n  *** δ = %.15f ***\n  *** 2δ = %.15f %s ***\n\n",
           delta, 2*delta, 2*delta>1?"(>1 ✓)":"(≤1 ✗)");
    free(x);free(bw);free(M);
    return delta;
}

// ============================================================
// Phase 2: Congruence spectral gaps — implicit Kronecker on GPU
// ============================================================

int is_squarefree(int m){for(int p=2;p*p<=m;p++)if(m%(p*p)==0)return 0;return 1;}

int find_orbits(int m, int *orbit_id) {
    int sd = m*m;
    for(int j=0;j<sd;j++) orbit_id[j]=-1;
    int norb=0;
    int *q=(int*)malloc(sd*sizeof(int));
    for(int seed=0;seed<sd;seed++){
        if(orbit_id[seed]>=0) continue;
        int qf=0,qb=0;
        q[qb++]=seed; orbit_id[seed]=norb;
        while(qf<qb){
            int idx=q[qf++]; int r=idx/m, s_val=idx%m;
            for(int a=1;a<=BOUND;a++){
                int nr=s_val, ns=(a*s_val+r)%m, ni=nr*m+ns;
                if(orbit_id[ni]<0){orbit_id[ni]=norb;q[qb++]=ni;}
                nr=((s_val-a*r)%m+m)%m; ns=r; ni=nr*m+ns;
                if(orbit_id[ni]<0){orbit_id[ni]=norb;q[qb++]=ni;}
            }
        }
        norb++;
    }
    free(q);
    return norb;
}

/*
 * Implicit matrix-vector product: w = L_{δ,m} · v
 *
 * v and w are vectors of length full_dim = N * sd (where sd = m²).
 * Layout: v[i * sd + j] = poly index i, fiber state j.
 *
 * L_{δ,m} = Σ_{a} M_a ⊗ P_a
 *
 * For each a:
 *   1. Permute fiber indices of v by P_a: tmp_fiber[j] = v[P_a(j)]
 *   2. Multiply by M_a on the poly indices: w_a = M_a * (reshaped v)
 *   3. Accumulate: w += w_a
 *
 * Using cuBLAS: reshape v as (N × sd), permute columns, dgemm with M_a.
 */

// CUDA kernel: permute columns of a N×sd matrix by perm
__global__ void permute_columns(double *out, const double *in,
                                 const int *perm, int N, int sd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * sd;
    if (idx >= total) return;

    int i = idx / sd;   // poly index
    int j = idx % sd;   // fiber index
    out[i * sd + perm[j]] = in[i * sd + j];
}

// Project out trivial component: v_non = v - Σ_k (v · u_k) u_k
// where u_k is the uniform vector on orbit k
__global__ void project_nontrivial(double *v, const int *orbit_id,
                                     const double *orbit_inv_size,
                                     int N, int sd, int num_orbits) {
    int i = blockIdx.x;  // poly index
    if (i >= N) return;

    int tid = threadIdx.x;

    // For this poly slice i, compute projection
    // v_slice = v + i*sd, length sd
    double *v_slice = v + (size_t)i * sd;

    // Shared memory for orbit sums
    extern __shared__ double shmem[];
    double *orb_sum = shmem;  // [num_orbits]

    // Initialize
    for (int k = tid; k < num_orbits; k += blockDim.x)
        orb_sum[k] = 0.0;
    __syncthreads();

    // Accumulate orbit sums
    for (int j = tid; j < sd; j += blockDim.x)
        atomicAdd(&orb_sum[orbit_id[j]], v_slice[j]);
    __syncthreads();

    // Normalize by orbit size
    for (int k = tid; k < num_orbits; k += blockDim.x)
        orb_sum[k] *= orbit_inv_size[k];
    __syncthreads();

    // Subtract projection
    for (int j = tid; j < sd; j += blockDim.x)
        v_slice[j] -= orb_sum[orbit_id[j]];
}

typedef struct {
    int m;
    int gpu_id;
    int N_poly;
    double delta;
    double *x, *bw;
    double lam_triv, lam_non, gap;
    int num_orbits;
    int status;
} WorkerArgs;

void* congruence_worker(void *arg) {
    WorkerArgs *w = (WorkerArgs*)arg;
    int m = w->m;
    int N = w->N_poly;
    double delta = w->delta;
    int sd = m * m;
    int full_dim = N * sd;

    // Memory check: need ~5 vectors of size full_dim + 5 matrices of N×N
    // Vector: full_dim * 8 bytes. For m=200, N=15: full_dim = 600K, vector = 4.8MB
    // Total: ~25MB. Trivial.
    size_t vec_bytes = (size_t)full_dim * sizeof(double);

    cudaSetDevice(w->gpu_id);

    // Find orbits
    int *h_orbit_id = (int*)malloc(sd * sizeof(int));
    w->num_orbits = find_orbits(m, h_orbit_id);

    // Orbit inverse sizes for projection
    double *h_orbit_inv = (double*)calloc(w->num_orbits, sizeof(double));
    int *orb_count = (int*)calloc(w->num_orbits, sizeof(int));
    for (int j = 0; j < sd; j++) orb_count[h_orbit_id[j]]++;
    for (int k = 0; k < w->num_orbits; k++)
        h_orbit_inv[k] = 1.0 / orb_count[k];
    free(orb_count);

    // Build M_a matrices on CPU (small: N×N each)
    double *h_Ma[BOUND];
    for (int a = 1; a <= BOUND; a++) {
        h_Ma[a-1] = (double*)malloc(N * N * sizeof(double));
        build_single_digit_matrix(a, delta, N, w->x, w->bw, h_Ma[a-1]);
    }

    // Build permutation tables
    int *h_perms[BOUND];
    for (int a = 1; a <= BOUND; a++) {
        h_perms[a-1] = (int*)malloc(sd * sizeof(int));
        for (int r = 0; r < m; r++)
            for (int s = 0; s < m; s++)
                h_perms[a-1][r*m+s] = s*m + ((a*s+r)%m);
    }

    // Upload to GPU
    double *d_Ma[BOUND];
    int *d_perms[BOUND];
    for (int a = 0; a < BOUND; a++) {
        cudaMalloc(&d_Ma[a], N * N * sizeof(double));
        cudaMemcpy(d_Ma[a], h_Ma[a], N * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_perms[a], sd * sizeof(int));
        cudaMemcpy(d_perms[a], h_perms[a], sd * sizeof(int), cudaMemcpyHostToDevice);
        free(h_Ma[a]); free(h_perms[a]);
    }

    int *d_orbit_id;
    double *d_orbit_inv;
    cudaMalloc(&d_orbit_id, sd * sizeof(int));
    cudaMalloc(&d_orbit_inv, w->num_orbits * sizeof(double));
    cudaMemcpy(d_orbit_id, h_orbit_id, sd * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orbit_inv, h_orbit_inv, w->num_orbits * sizeof(double), cudaMemcpyHostToDevice);
    free(h_orbit_id); free(h_orbit_inv);

    // Allocate vectors on GPU
    double *d_v, *d_w, *d_tmp;
    cudaMalloc(&d_v, vec_bytes);
    cudaMalloc(&d_w, vec_bytes);
    cudaMalloc(&d_tmp, vec_bytes);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    double one = 1.0, zero_d = 0.0;
    int perm_blocks = (full_dim + 255) / 256;
    int proj_threads = sd < 256 ? sd : 256;
    size_t shmem_size = w->num_orbits * sizeof(double);

    // ================================================================
    // Power iteration for TRIVIAL eigenvalue (full operator, no projection)
    // ================================================================

    // Initialize v = all ones
    double *h_v = (double*)malloc(vec_bytes);
    for (int i = 0; i < full_dim; i++) h_v[i] = 1.0;
    cudaMemcpy(d_v, h_v, vec_bytes, cudaMemcpyHostToDevice);

    double lam_triv = 0.0;
    for (int it = 0; it < 200; it++) {
        // w = L · v = Σ_a (M_a ⊗ P_a) v
        cudaMemset(d_w, 0, vec_bytes);

        for (int a = 0; a < BOUND; a++) {
            // tmp = permute v by P_a (on fiber indices)
            cudaMemset(d_tmp, 0, vec_bytes);
            permute_columns<<<perm_blocks, 256>>>(d_tmp, d_v, d_perms[a], N, sd);

            // w += M_a * tmp (treat as M_a [N×N] × tmp [N×sd] → contribution [N×sd])
            // tmp is laid out as N rows of sd elements (row-major in the poly index)
            // But cuBLAS expects column-major...
            // Actually our layout is: v[i*sd + j] where i=poly, j=fiber
            // This is a N×sd matrix in ROW-major. For cuBLAS (column-major),
            // it looks like a sd×N matrix. We want M_a * V where V is N×sd.
            // In column-major terms: V^T is sd×N, M_a^T is N×N.
            // (M_a * V)^T = V^T * M_a^T → cublasDgemm(N, sd×N, N×N)
            // Result: sd×N matrix which is (M_a * V)^T
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        sd, N, N,
                        &one,
                        d_tmp, sd,      // sd × N (tmp^T)
                        d_Ma[a], N,     // N × N (Ma^T = Ma since we want Ma * V)
                        &one,           // accumulate into w
                        d_w, sd);       // sd × N (w^T)
        }

        // Rayleigh quotient
        double num_val, den_val;
        cublasDdot(cublas, full_dim, d_v, 1, d_w, 1, &num_val);
        cublasDdot(cublas, full_dim, d_v, 1, d_v, 1, &den_val);
        lam_triv = num_val / den_val;

        // Normalize w → v
        double norm_val;
        cublasDnrm2(cublas, full_dim, d_w, 1, &norm_val);
        double inv_norm = 1.0 / norm_val;
        cublasDscal(cublas, full_dim, &inv_norm, d_w, 1);
        cudaMemcpy(d_v, d_w, vec_bytes, cudaMemcpyDeviceToDevice);
    }

    // ================================================================
    // Power iteration for NON-TRIVIAL eigenvalue (project after each step)
    // ================================================================

    // Initialize with random-ish vector, then project out trivial
    for (int i = 0; i < full_dim; i++) h_v[i] = sin(i * 1.23456 + 0.789);
    cudaMemcpy(d_v, h_v, vec_bytes, cudaMemcpyHostToDevice);

    // Project out trivial component
    project_nontrivial<<<N, proj_threads, shmem_size>>>(
        d_v, d_orbit_id, d_orbit_inv, N, sd, w->num_orbits);

    double lam_non = 0.0;
    for (int it = 0; it < 300; it++) {
        // w = L · v
        cudaMemset(d_w, 0, vec_bytes);
        for (int a = 0; a < BOUND; a++) {
            cudaMemset(d_tmp, 0, vec_bytes);
            permute_columns<<<perm_blocks, 256>>>(d_tmp, d_v, d_perms[a], N, sd);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        sd, N, N, &one, d_tmp, sd, d_Ma[a], N, &one, d_w, sd);
        }

        // Project out trivial component from w
        project_nontrivial<<<N, proj_threads, shmem_size>>>(
            d_w, d_orbit_id, d_orbit_inv, N, sd, w->num_orbits);

        // Rayleigh quotient
        double num_val, den_val;
        cublasDdot(cublas, full_dim, d_v, 1, d_w, 1, &num_val);
        cublasDdot(cublas, full_dim, d_v, 1, d_v, 1, &den_val);
        lam_non = num_val / den_val;

        // Normalize
        double norm_val;
        cublasDnrm2(cublas, full_dim, d_w, 1, &norm_val);
        if (norm_val < 1e-300) break;
        double inv_norm = 1.0 / norm_val;
        cublasDscal(cublas, full_dim, &inv_norm, d_w, 1);
        cudaMemcpy(d_v, d_w, vec_bytes, cudaMemcpyDeviceToDevice);
    }

    w->lam_triv = lam_triv;
    w->lam_non = lam_non;
    w->gap = fabs(lam_triv) - fabs(lam_non);
    w->status = 0;

    // Cleanup
    free(h_v);
    cublasDestroy(cublas);
    for (int a = 0; a < BOUND; a++) { cudaFree(d_Ma[a]); cudaFree(d_perms[a]); }
    cudaFree(d_orbit_id); cudaFree(d_orbit_inv);
    cudaFree(d_v); cudaFree(d_w); cudaFree(d_tmp);

    return NULL;
}

void compute_congruence_gaps(double delta, int N_poly, int max_m) {
    printf("\n=== Phase 2: Congruence Spectral Gaps (implicit Kronecker, multi-GPU) ===\n");
    printf("δ = %.15f, N_poly = %d, max m = %d\n", delta, N_poly, max_m);
    printf("Memory per m: ~%.1f MB (3 vectors of N·m² doubles)\n\n",
           3.0 * N_poly * max_m * max_m * 8.0 / 1e6);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs: %d\n\n", device_count);

    double *x = (double*)malloc(N_poly * sizeof(double));
    double *bw = (double*)malloc(N_poly * sizeof(double));
    chebyshev_nodes(x, N_poly);
    barycentric_weights(bw, N_poly);

    printf("%4s  %10s  %6s  %12s  %12s  %12s  %12s\n",
           "m", "full_dim", "orbits", "|λ_triv|", "|λ_non|", "gap", "gap/triv");
    printf("----  ----------  ------  ------------  ------------  ------------  ------------\n");

    int m_vals[2000];
    int n_m = 0;
    for (int m = 2; m <= max_m && n_m < 2000; m++)
        if (is_squarefree(m)) m_vals[n_m++] = m;

    for (int batch = 0; batch < n_m; batch += device_count) {
        int bsz = device_count;
        if (batch + bsz > n_m) bsz = n_m - batch;

        WorkerArgs args[8];
        pthread_t threads[8];

        for (int i = 0; i < bsz; i++) {
            args[i].m = m_vals[batch + i];
            args[i].gpu_id = i;
            args[i].N_poly = N_poly;
            args[i].delta = delta;
            args[i].x = x;
            args[i].bw = bw;
            args[i].status = -1;
            pthread_create(&threads[i], NULL, congruence_worker, &args[i]);
        }

        for (int i = 0; i < bsz; i++) {
            pthread_join(threads[i], NULL);
            int m_val = args[i].m;
            int fd = args[i].N_poly * m_val * m_val;
            if (args[i].status == 0) {
                printf("%4d  %10d  %6d  %12.6f  %12.6f  %12.6f  %12.6f\n",
                       m_val, fd, args[i].num_orbits,
                       fabs(args[i].lam_triv), fabs(args[i].lam_non),
                       args[i].gap, args[i].gap / fabs(args[i].lam_triv));
                fflush(stdout);
            } else {
                printf("%4d  %10d  %6s  (status=%d)\n", m_val, fd, "-", args[i].status);
            }
        }
    }

    free(x); free(bw);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 40;
    int phase = argc > 2 ? atoi(argv[2]) : 3;
    int max_m = argc > 3 ? atoi(argv[3]) : 100;

    printf("==========================================\n");
    printf("  Zaremba Transfer Operator (implicit GPU)\n");
    printf("==========================================\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double delta = 0.0;
    if (phase == 1 || phase == 3)
        delta = compute_hausdorff_dimension(N);
    if (phase == 2 || phase == 3) {
        if (delta <= 0) delta = 0.836829443681208;
        int cN = N < 15 ? N : 15;
        compute_congruence_gaps(delta, cN, max_m);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("\nTotal: %.1fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9);
    return 0;
}
