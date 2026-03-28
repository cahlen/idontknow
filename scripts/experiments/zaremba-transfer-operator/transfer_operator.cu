/*
 * Zaremba Transfer Operator v2 — fully GPU-accelerated
 *
 * Phase 1: Hausdorff dimension δ (Chebyshev collocation, CPU — tiny matrix)
 * Phase 2: Congruence spectral gaps (cuBLAS/cuSOLVER, multi-GPU)
 *
 * All heavy linear algebra on GPU:
 *   - Matrix construction: GPU kernel
 *   - Projection: cuBLAS dgemm
 *   - Eigensolve: cuSOLVER Xgeev
 *   - Parallelized across 8 GPUs by m
 *
 * Compile: nvcc -O3 -arch=sm_100a -o transfer_op scripts/experiments/zaremba-transfer-operator/transfer_operator.cu -lcusolver -lcublas -lm
 * Run:     ./transfer_op [N] [phase] [max_m]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#define BOUND 5
#define MAX_N 200

// ============================================================
// Phase 1: Hausdorff dimension (CPU — matrix is only 40×40)
// ============================================================

void chebyshev_nodes(double *x, int N) {
    for (int j = 0; j < N; j++)
        x[j] = 0.5 * (1.0 + cos(M_PI * (2.0 * j + 1.0) / (2.0 * N)));
}

void barycentric_weights(double *w, int N) {
    for (int j = 0; j < N; j++)
        w[j] = pow(-1.0, j) * sin(M_PI * (2.0 * j + 1.0) / (2.0 * N));
}

void build_matrix_cpu(double s, int N, double *x, double *bw, double *M) {
    memset(M, 0, N * N * sizeof(double));
    for (int a = 1; a <= BOUND; a++) {
        for (int i = 0; i < N; i++) {
            double y = 1.0 / (a + x[i]);
            double ws = pow(a + x[i], -2.0 * s);
            int exact = -1;
            for (int k = 0; k < N; k++)
                if (fabs(y - x[k]) < 1e-15) { exact = k; break; }
            if (exact >= 0) {
                M[i + exact * N] += ws;
            } else {
                double denom = 0.0;
                double numer[MAX_N];
                for (int j = 0; j < N; j++) {
                    numer[j] = bw[j] / (y - x[j]);
                    denom += numer[j];
                }
                for (int j = 0; j < N; j++)
                    M[i + j * N] += ws * numer[j] / denom;
            }
        }
    }
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
    build_matrix_cpu(s_lo,N,x,bw,M); double l_lo=power_iteration_cpu(M,N,300);
    build_matrix_cpu(s_hi,N,x,bw,M); double l_hi=power_iteration_cpu(M,N,300);
    printf("λ_0(%.1f)=%.6f, λ_0(%.1f)=%.6f\n\n",s_lo,l_lo,s_hi,l_hi);

    for(int it=0;it<55;it++){
        double s=(s_lo+s_hi)/2;
        build_matrix_cpu(s,N,x,bw,M);
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
// Phase 2: Congruence spectral gaps (fully GPU)
// ============================================================

int is_squarefree(int m){for(int p=2;p*p<=m;p++)if(m%(p*p)==0)return 0;return 1;}

// Find orbits of Gamma_A on (Z/mZ)^2
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
            int idx=q[qf++]; int r=idx/m, s=idx%m;
            for(int a=1;a<=BOUND;a++){
                int nr=s, ns=(a*s+r)%m, ni=nr*m+ns;
                if(orbit_id[ni]<0){orbit_id[ni]=norb;q[qb++]=ni;}
                nr=((a*r-s)%m+m)%m; ns=r; ni=nr*m+ns;
                if(orbit_id[ni]<0){orbit_id[ni]=norb;q[qb++]=ni;}
            }
        }
        norb++;
    }
    free(q);
    return norb;
}

// Per-GPU worker for a single m value
typedef struct {
    int m;
    int gpu_id;
    int N_poly;
    double delta;
    double *x, *bw;  // Chebyshev data (shared, read-only)
    // Results
    double lam_triv, lam_non, gap;
    int num_orbits;
    int status;  // 0=ok, -1=skip, -2=error
} WorkerArgs;

void* congruence_worker(void *arg) {
    WorkerArgs *w = (WorkerArgs*)arg;
    int m = w->m;
    int N = w->N_poly;
    double delta = w->delta;
    int sd = m * m;
    int full_dim = N * sd;

    if (full_dim > 30000) { w->status = -1; return NULL; }

    cudaSetDevice(w->gpu_id);

    // Find orbits on CPU (fast, small)
    int *orbit_id = (int*)malloc(sd * sizeof(int));
    w->num_orbits = find_orbits(m, orbit_id);

    // Build full congruence matrix on CPU
    size_t mat_bytes = (size_t)full_dim * full_dim * sizeof(double);
    double *h_M = (double*)calloc((size_t)full_dim * full_dim, sizeof(double));
    if (!h_M) { w->status = -2; free(orbit_id); return NULL; }

    double *Ma = (double*)malloc(N * N * sizeof(double));
    int *perm = (int*)malloc(sd * sizeof(int));

    for (int a = 1; a <= BOUND; a++) {
        // Build single-digit Chebyshev matrix
        memset(Ma, 0, N * N * sizeof(double));
        for (int i = 0; i < N; i++) {
            double y = 1.0 / (a + w->x[i]);
            double ws = pow(a + w->x[i], -2.0 * delta);
            int exact = -1;
            for (int k = 0; k < N; k++)
                if (fabs(y - w->x[k]) < 1e-15) { exact = k; break; }
            if (exact >= 0) { Ma[i + exact * N] = ws; }
            else {
                double den = 0; double num[MAX_N];
                for (int j = 0; j < N; j++) { num[j] = w->bw[j]/(y-w->x[j]); den += num[j]; }
                for (int j = 0; j < N; j++) Ma[i + j * N] = ws * num[j] / den;
            }
        }

        // Build perm
        for (int r = 0; r < m; r++)
            for (int s = 0; s < m; s++)
                perm[r*m+s] = s*m + ((a*s+r)%m);

        // Kronecker: h_M += Ma ⊗ P_a
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                double v = Ma[i + k * N];
                if (fabs(v) < 1e-300) continue;
                for (int j = 0; j < sd; j++) {
                    int row = i*sd+j, col = k*sd+perm[j];
                    h_M[row + (size_t)col * full_dim] += v;
                }
            }
    }
    free(Ma); free(perm);

    // Build non-trivial projector Q_non on CPU (small: sd × sd)
    double *Q_non = (double*)malloc((size_t)sd * sd * sizeof(double));
    int *orb_size = (int*)calloc(w->num_orbits, sizeof(int));
    for (int j = 0; j < sd; j++) orb_size[orbit_id[j]]++;
    for (int i = 0; i < sd; i++)
        for (int j = 0; j < sd; j++)
            Q_non[i + (size_t)j * sd] = ((i==j)?1.0:0.0) -
                ((orbit_id[i]==orbit_id[j]) ? 1.0/orb_size[orbit_id[i]] : 0.0);
    free(orb_size); free(orbit_id);

    // === ALL ON GPU FROM HERE ===

    // Upload M and Q_non to GPU
    double *d_M, *d_Qnon, *d_tmp, *d_Mnon;
    cudaMalloc(&d_M, mat_bytes);
    cudaMalloc(&d_tmp, mat_bytes);
    cudaMalloc(&d_Mnon, mat_bytes);
    cudaMemcpy(d_M, h_M, mat_bytes, cudaMemcpyHostToDevice);

    // Build P_non = I_N ⊗ Q_non on GPU and do projection via cuBLAS
    // Instead of building the full P_non, apply Q_non block-wise:
    // M_non = (I⊗Q) * M * (I⊗Q)

    // Upload Q_non
    double *d_Q;
    cudaMalloc(&d_Q, (size_t)sd * sd * sizeof(double));
    cudaMemcpy(d_Q, Q_non, (size_t)sd * sd * sizeof(double), cudaMemcpyHostToDevice);
    free(Q_non);

    // Build full P_non = I_N ⊗ Q_non on GPU (full_dim × full_dim)
    // P_non[(i*sd+j), (k*sd+l)] = delta_{ik} * Q_non[j,l]
    double *d_Pnon;
    cudaMalloc(&d_Pnon, mat_bytes);
    cudaMemset(d_Pnon, 0, mat_bytes);

    // Fill P_non block-diagonally: for each i, copy Q_non into the (i,i) block
    for (int i = 0; i < N; i++) {
        // Copy Q_non (sd×sd) into d_Pnon at block (i,i)
        // Block starts at row i*sd, col i*sd
        for (int col = 0; col < sd; col++) {
            cudaMemcpy(d_Pnon + (i*sd) + (size_t)(i*sd + col) * full_dim,
                       d_Q + (size_t)col * sd,
                       sd * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }
    cudaFree(d_Q);

    // GPU projection: M_non = P_non * M * P_non using cuBLAS dgemm
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    double one = 1.0, zero = 0.0;

    // tmp = M * P_non
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                full_dim, full_dim, full_dim,
                &one, d_M, full_dim, d_Pnon, full_dim,
                &zero, d_tmp, full_dim);

    // M_non = P_non * tmp
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                full_dim, full_dim, full_dim,
                &one, d_Pnon, full_dim, d_tmp, full_dim,
                &zero, d_Mnon, full_dim);

    cublasDestroy(cublas);
    cudaFree(d_Pnon);
    cudaFree(d_tmp);

    // GPU eigensolve on full_M (trivial eigenvalue)
    cusolverDnHandle_t cusolver;
    cusolverDnCreate(&cusolver);
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    double *d_W_full, *d_W_non;
    cudaMalloc(&d_W_full, 2 * full_dim * sizeof(double));
    cudaMalloc(&d_W_non, 2 * full_dim * sizeof(double));

    size_t dwork_sz=0, hwork_sz=0;
    cusolverDnXgeev_bufferSize(cusolver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR,
        (int64_t)full_dim, CUDA_R_64F, d_M, (int64_t)full_dim,
        CUDA_C_64F, d_W_full, CUDA_R_64F, NULL, (int64_t)full_dim,
        CUDA_R_64F, NULL, (int64_t)full_dim, CUDA_R_64F,
        &dwork_sz, &hwork_sz);

    void *d_work, *h_work;
    cudaMalloc(&d_work, dwork_sz);
    h_work = malloc(hwork_sz > 0 ? hwork_sz : 1);
    int *d_info;
    cudaMalloc(&d_info, sizeof(int));

    // Eigenvalues of full M
    cusolverDnXgeev(cusolver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR,
        (int64_t)full_dim, CUDA_R_64F, d_M, (int64_t)full_dim,
        CUDA_C_64F, d_W_full, CUDA_R_64F, NULL, (int64_t)full_dim,
        CUDA_R_64F, NULL, (int64_t)full_dim, CUDA_R_64F,
        d_work, dwork_sz, h_work, hwork_sz, d_info);
    cudaDeviceSynchronize();

    // Eigenvalues of M_non
    // Need fresh workspace size for M_non (same dim, should be same)
    cusolverDnXgeev(cusolver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR,
        (int64_t)full_dim, CUDA_R_64F, d_Mnon, (int64_t)full_dim,
        CUDA_C_64F, d_W_non, CUDA_R_64F, NULL, (int64_t)full_dim,
        CUDA_R_64F, NULL, (int64_t)full_dim, CUDA_R_64F,
        d_work, dwork_sz, h_work, hwork_sz, d_info);
    cudaDeviceSynchronize();

    // Download eigenvalues
    double *h_W_full = (double*)malloc(2*full_dim*sizeof(double));
    double *h_W_non = (double*)malloc(2*full_dim*sizeof(double));
    cudaMemcpy(h_W_full, d_W_full, 2*full_dim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W_non, d_W_non, 2*full_dim*sizeof(double), cudaMemcpyDeviceToHost);

    // Find max |eigenvalue| for each
    double max_full = 0, max_non = 0;
    for (int i = 0; i < full_dim; i++) {
        double af = sqrt(h_W_full[2*i]*h_W_full[2*i] + h_W_full[2*i+1]*h_W_full[2*i+1]);
        double an = sqrt(h_W_non[2*i]*h_W_non[2*i] + h_W_non[2*i+1]*h_W_non[2*i+1]);
        if (af > max_full) max_full = af;
        if (an > max_non) max_non = an;
    }

    w->lam_triv = max_full;
    w->lam_non = max_non;
    w->gap = max_full - max_non;
    w->status = 0;

    // Cleanup
    free(h_M); free(h_W_full); free(h_W_non); free(h_work);
    cudaFree(d_M); cudaFree(d_Mnon); cudaFree(d_W_full); cudaFree(d_W_non);
    cudaFree(d_work); cudaFree(d_info);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(cusolver);

    return NULL;
}

void compute_congruence_gaps(double delta, int N_poly, int max_m) {
    printf("\n=== Phase 2: Congruence Spectral Gaps (multi-GPU) ===\n");
    printf("δ = %.15f, N_poly = %d, max m = %d\n\n", delta, N_poly, max_m);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs: %d\n\n", device_count);

    // Shared Chebyshev data
    double *x = (double*)malloc(N_poly * sizeof(double));
    double *bw = (double*)malloc(N_poly * sizeof(double));
    chebyshev_nodes(x, N_poly);
    barycentric_weights(bw, N_poly);

    printf("%4s  %8s  %6s  %12s  %12s  %12s  %12s\n",
           "m", "dim", "orbits", "|λ_triv|", "|λ_non|", "gap", "gap/triv");
    printf("----  --------  ------  ------------  ------------  ------------  ------------\n");

    // Collect square-free m values
    int m_vals[500];
    int n_m = 0;
    for (int m = 2; m <= max_m && n_m < 500; m++)
        if (is_squarefree(m)) m_vals[n_m++] = m;

    // Process in batches of device_count (one m per GPU)
    for (int batch = 0; batch < n_m; batch += device_count) {
        int batch_size = device_count;
        if (batch + batch_size > n_m) batch_size = n_m - batch;

        WorkerArgs args[8];
        pthread_t threads[8];

        // Launch workers
        for (int i = 0; i < batch_size; i++) {
            args[i].m = m_vals[batch + i];
            args[i].gpu_id = i;
            args[i].N_poly = N_poly;
            args[i].delta = delta;
            args[i].x = x;
            args[i].bw = bw;
            args[i].status = -1;
            pthread_create(&threads[i], NULL, congruence_worker, &args[i]);
        }

        // Wait and print results
        for (int i = 0; i < batch_size; i++) {
            pthread_join(threads[i], NULL);
            int m = args[i].m;
            int sd = m * m;
            int fd = N_poly * sd;

            if (args[i].status == -1) {
                printf("%4d  %8d  %6s  (skipped)\n", m, fd, "-");
            } else if (args[i].status == -2) {
                printf("%4d  %8d  %6s  (alloc fail)\n", m, fd, "-");
            } else {
                printf("%4d  %8d  %6d  %12.6f  %12.6f  %12.6f  %12.6f\n",
                       m, fd, args[i].num_orbits,
                       args[i].lam_triv, args[i].lam_non,
                       args[i].gap, args[i].gap / args[i].lam_triv);
            }
        }
    }

    free(x); free(bw);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 40;
    int phase = argc > 2 ? atoi(argv[2]) : 3;
    int max_m = argc > 3 ? atoi(argv[3]) : 50;

    printf("========================================\n");
    printf("  Zaremba Transfer Operator (GPU)\n");
    printf("========================================\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double delta = 0.0;
    if (phase == 1 || phase == 3)
        delta = compute_hausdorff_dimension(N);
    if (phase == 2 || phase == 3) {
        if (delta <= 0) delta = 0.836829443681208;
        compute_congruence_gaps(delta, N < 20 ? N : 20, max_m);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\nTotal: %.1fs\n", elapsed);
    return 0;
}
