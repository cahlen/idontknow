/*
 * EXACT Dolgopyat spectral radius via FULL eigendecomposition
 *
 * Power iteration FAILS for the twisted operator at certain t values
 * (multiple eigenvalues of similar magnitude with different phases
 * cause oscillation instead of convergence).
 *
 * Solution: compute ALL eigenvalues of the NC×NC complex matrix
 * using cuSOLVER Xgeev (CUDA 13 API), then take the maximum absolute value.
 * For NC=80: the matrix is 80×80 complex = trivial for cuSOLVER.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o dolgopyat_exact dolgopyat_exact.cu -lcusolver -lcublas -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cusolverDn.h>
#include <cuComplex.h>

#define BOUND 5
#define NC 80
#define DELTA 0.836829443681208

// Build L_{δ+it} on HOST (80×80 complex, trivial size)
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

            int exact = -1;
            for (int k = 0; k < NC; k++)
                if (fabs(ga - nodes[k]) < 1e-14) { exact = k; break; }

            if (exact >= 0) {
                L[i + exact*NC].x += wr;
                L[i + exact*NC].y += wi;
            } else {
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
}

int main(int argc, char **argv) {
    int num_t = argc > 1 ? atoi(argv[1]) : 100000;
    double t_max = argc > 2 ? atof(argv[2]) : 1000.0;

    printf("Dolgopyat EXACT (cuSOLVER Xgeev, CUDA 13): N=%d, %d grid points, t∈[0,%.0f]\n\n",
           NC, num_t, t_max);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // cuSOLVER setup
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    // Device allocations
    cuDoubleComplex *d_A, *d_W;
    int *d_info;

    cudaMalloc(&d_A, NC*NC*sizeof(cuDoubleComplex));
    cudaMalloc(&d_W, NC*sizeof(cuDoubleComplex));
    cudaMalloc(&d_info, sizeof(int));

    // Query workspace sizes
    size_t workDevice = 0, workHost = 0;
    cusolverDnXgeev_bufferSize(
        handle, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR,
        NC,
        CUDA_C_64F, d_A, NC,    // A
        CUDA_C_64F, d_W,        // W (eigenvalues)
        CUDA_C_64F, NULL, NC,   // VL (not computed)
        CUDA_C_64F, NULL, NC,   // VR (not computed)
        CUDA_C_64F,             // compute type
        &workDevice, &workHost);

    void *d_work = NULL, *h_work = NULL;
    if (workDevice > 0) cudaMalloc(&d_work, workDevice);
    if (workHost > 0)   h_work = malloc(workHost);

    printf("Workspace: %zu bytes device, %zu bytes host\n\n", workDevice, workHost);

    cuDoubleComplex *h_L = (cuDoubleComplex*)malloc(NC*NC*sizeof(cuDoubleComplex));
    cuDoubleComplex *h_W = (cuDoubleComplex*)malloc(NC*sizeof(cuDoubleComplex));

    double max_rho = 0;
    double max_rho_t = 0;

    for (int ti = 0; ti < num_t; ti++) {
        double t = (ti + 0.5) * t_max / num_t;
        if (t < 1.0) continue; // skip near-zero

        build_L(t, h_L);
        cudaMemcpy(d_A, h_L, NC*NC*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        cusolverDnXgeev(
            handle, params,
            CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR,
            NC,
            CUDA_C_64F, d_A, NC,
            CUDA_C_64F, d_W,
            CUDA_C_64F, NULL, NC,
            CUDA_C_64F, NULL, NC,
            CUDA_C_64F,
            d_work, workDevice,
            h_work, workHost,
            d_info);
        cudaDeviceSynchronize();

        cudaMemcpy(h_W, d_W, NC*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        // Find max |eigenvalue|
        double rho = 0;
        for (int i = 0; i < NC; i++) {
            double absval = sqrt(h_W[i].x*h_W[i].x + h_W[i].y*h_W[i].y);
            if (absval > rho) rho = absval;
        }

        if (rho > max_rho) {
            max_rho = rho;
            max_rho_t = t;
        }

        if (ti % (num_t/20) == 0)
            printf("  t=%8.2f: ρ = %.8f\n", t, rho);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("sup_{t≥1} ρ(L_{δ+it}) = %.8f at t = %.4f\n", max_rho, max_rho_t);
    printf("Time: %.2fs for %d eigendecompositions\n", elapsed, num_t);
    printf("========================================\n");

    // Print at key t values
    printf("\nKey values:\n");
    double check_t[] = {1, 2, 5, 10, 19.02, 20, 28.6, 50, 100, 500, 1000};
    for (int k = 0; k < 11; k++) {
        build_L(check_t[k], h_L);
        cudaMemcpy(d_A, h_L, NC*NC*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cusolverDnXgeev(
            handle, params,
            CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR,
            NC,
            CUDA_C_64F, d_A, NC,
            CUDA_C_64F, d_W,
            CUDA_C_64F, NULL, NC,
            CUDA_C_64F, NULL, NC,
            CUDA_C_64F,
            d_work, workDevice,
            h_work, workHost,
            d_info);
        cudaDeviceSynchronize();
        cudaMemcpy(h_W, d_W, NC*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        double rho = 0;
        for (int i = 0; i < NC; i++) {
            double absval = sqrt(h_W[i].x*h_W[i].x + h_W[i].y*h_W[i].y);
            if (absval > rho) rho = absval;
        }
        printf("  t=%8.2f: ρ = %.8f\n", check_t[k], rho);
    }

    cusolverDnDestroyParams(params);
    cusolverDnDestroy(handle);
    if (d_work) cudaFree(d_work);
    if (h_work) free(h_work);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_info);
    free(h_L); free(h_W);
    return 0;
}
