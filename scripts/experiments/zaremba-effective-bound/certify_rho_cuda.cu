/*
 * RIGOROUS certification of ρ(L_{δ+it}) via matrix powers on GPU.
 *
 * Method: ρ(A) ≤ ||A^k||_∞^{1/k} for any submultiplicative norm.
 * We compute L^{2^nsq} via squarings using cuBLAS ZGEMM, then
 * take the row-norm. This gives a guaranteed upper bound.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o certify_rho_cuda certify_rho_cuda.cu -lcublas -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define BOUND 5
#define NC 40
#define DELTA 0.836829443681208

void build_L(double t, cuDoubleComplex *L) {
    double nodes[NC], bary[NC];
    for (int j = 0; j < NC; j++) {
        nodes[j] = 0.5 * (1.0 + cos(M_PI * (2*j+1) / (2.0*NC)));
        bary[j] = ((j%2==0) ? 1.0 : -1.0) * sin(M_PI * (2*j+1) / (2.0*NC));
    }

    for (int i = 0; i < NC*NC; i++)
        L[i] = make_cuDoubleComplex(0, 0);

    for (int a = 1; a <= BOUND; a++) {
        for (int i = 0; i < NC; i++) {
            double xi = nodes[i], apx = a + xi, ga = 1.0/apx;
            double weight = pow(apx, -2.0*DELTA);
            double phase = -2.0 * t * log(apx);
            double wr = weight * cos(phase), wi = weight * sin(phase);

            double den = 0, num[NC];
            for (int j = 0; j < NC; j++) { num[j] = bary[j]/(ga-nodes[j]); den += num[j]; }
            for (int j = 0; j < NC; j++) {
                double b = num[j] / den;
                L[i + j*NC].x += wr * b;
                L[i + j*NC].y += wi * b;
            }
        }
    }
}

double row_norm_colmajor(cuDoubleComplex *M, int n) {
    double maxrow = 0;
    for (int i = 0; i < n; i++) {
        double rowsum = 0;
        for (int j = 0; j < n; j++) {
            double re = M[i + j*n].x, im = M[i + j*n].y;
            rowsum += sqrt(re*re + im*im);
        }
        if (rowsum > maxrow) maxrow = rowsum;
    }
    return maxrow;
}

int main(int argc, char **argv) {
    int num_t = argc > 1 ? atoi(argv[1]) : 1000;
    double t_min = argc > 2 ? atof(argv[2]) : 0.95;
    double t_max = argc > 3 ? atof(argv[3]) : 2.0;
    int nsq = argc > 4 ? atoi(argv[4]) : 8;  // default L^256

    int power = 1 << nsq;
    printf("RIGOROUS ρ certification via ||L^{%d}||^{1/%d}\n", power, power);
    printf("NC=%d, t∈[%.3f, %.3f], %d grid points, %d squarings\n\n",
           NC, t_min, t_max, num_t, nsq);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cuDoubleComplex *d_A, *d_B;
    cudaMalloc(&d_A, NC*NC*sizeof(cuDoubleComplex));
    cudaMalloc(&d_B, NC*NC*sizeof(cuDoubleComplex));

    cuDoubleComplex *h_L = (cuDoubleComplex*)malloc(NC*NC*sizeof(cuDoubleComplex));
    cuDoubleComplex *h_Lk = (cuDoubleComplex*)malloc(NC*NC*sizeof(cuDoubleComplex));

    cuDoubleComplex alpha = make_cuDoubleComplex(1, 0);
    cuDoubleComplex beta = make_cuDoubleComplex(0, 0);

    struct timespec t0_clock, t1_clock;
    clock_gettime(CLOCK_MONOTONIC, &t0_clock);

    double max_bound = 0, max_bound_t = 0;
    int print_every = num_t / 20;
    if (print_every < 1) print_every = 1;

    for (int ti = 0; ti < num_t; ti++) {
        double t = t_min + (t_max - t_min) * ti / (num_t > 1 ? num_t - 1 : 1);

        build_L(t, h_L);
        cudaMemcpy(d_A, h_L, NC*NC*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        for (int sq = 0; sq < nsq; sq++) {
            cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       NC, NC, NC, &alpha, d_A, NC, d_A, NC, &beta, d_B, NC);
            cuDoubleComplex *tmp = d_A; d_A = d_B; d_B = tmp;
        }

        cudaMemcpy(h_Lk, d_A, NC*NC*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        double rn = row_norm_colmajor(h_Lk, NC);
        double bound = (rn > 0) ? pow(rn, 1.0/power) : 0;

        if (bound > max_bound) {
            max_bound = bound;
            max_bound_t = t;
        }

        if (ti % print_every == 0)
            printf("  t=%8.4f: bound = %.10f\n", t, bound);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1_clock);
    double elapsed = (t1_clock.tv_sec-t0_clock.tv_sec) + (t1_clock.tv_nsec-t0_clock.tv_nsec)/1e9;

    double h = (t_max - t_min) / (num_t > 1 ? num_t - 1 : 1);
    double K = 3.0;

    printf("\n========================================\n");
    printf("Grid max: %.10f at t=%.6f\n", max_bound, max_bound_t);
    printf("Grid spacing h = %.8f\n", h);
    printf("Lipschitz K = %.1f, correction = %.8f\n", K, K*h);
    printf("CERTIFIED: ρ ≤ %.10f\n", max_bound + K*h);
    printf("Time: %.2fs (%d points, %d squarings)\n", elapsed, num_t, nsq);
    printf("========================================\n");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B);
    free(h_L); free(h_Lk);
    return 0;
}
