#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define BLOCK 256

__global__ void kronecker_slab(
    const int64_t *__restrict__ ct,
    const double  *__restrict__ z,
    int P, int C, int j,
    int64_t *__restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / P;
    int k = tid % P;
    if (i > j || k < j || i >= P) return;
    double sum = 0.0;
    for (int c = 0; c < C; c++)
        sum += z[c] * (double)ct[(int64_t)i*C+c] * (double)ct[(int64_t)j*C+c] * (double)ct[(int64_t)k*C+c];
    out[(int64_t)i*P+k] = llround(sum);
}

__global__ void reduce_stats(const int64_t *slab, int P, int j,
                             unsigned long long *nz, unsigned long long *mx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / P;
    int k = tid % P;
    if (i > j || k < j || i >= P) return;
    int64_t v = slab[(int64_t)i*P+k];
    if (v != 0) {
        atomicAdd(nz, 1ULL);
        unsigned long long av = (unsigned long long)(v > 0 ? v : -v);
        atomicMax(mx, av);
    }
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int gpu = argc > 2 ? atoi(argv[2]) : 0;
    cudaSetDevice(gpu);
    char path[256];
    snprintf(path, 256, "scripts/experiments/kronecker-coefficients/results/char_table_n%d.bin", n);
    FILE *fc = fopen(path, "rb"); fseek(fc, 0, SEEK_END); long ct_sz = ftell(fc); fseek(fc, 0, SEEK_SET);
    snprintf(path, 256, "scripts/experiments/kronecker-coefficients/results/z_inv_n%d.bin", n);
    FILE *fz = fopen(path, "rb"); fseek(fz, 0, SEEK_END); int C = ftell(fz)/sizeof(double); fseek(fz, 0, SEEK_SET);
    int P = ct_sz / (C * sizeof(int64_t));
    int64_t *h_ct = (int64_t*)malloc(ct_sz);
    double *h_z = (double*)malloc(C*sizeof(double));
    fread(h_ct, 1, ct_sz, fc); fclose(fc);
    fread(h_z, sizeof(double), C, fz); fclose(fz);
    printf("S_%d: %d partitions, %d classes — ALL GPU\n", n, P, C);
    fflush(stdout);

    int64_t *d_ct, *d_out; double *d_z;
    unsigned long long *d_nz, *d_mx;
    cudaMalloc(&d_ct, ct_sz);
    cudaMalloc(&d_z, C*sizeof(double));
    cudaMalloc(&d_out, (int64_t)P*P*sizeof(int64_t));
    cudaMalloc(&d_nz, sizeof(unsigned long long));
    cudaMalloc(&d_mx, sizeof(unsigned long long));
    cudaMemcpy(d_ct, h_ct, ct_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, C*sizeof(double), cudaMemcpyHostToDevice);

    unsigned long long total_nz = 0, global_max = 0;
    int blocks = ((int64_t)P*P + BLOCK - 1) / BLOCK;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int j = 0; j < P; j++) {
        cudaMemset(d_out, 0, (int64_t)P*P*sizeof(int64_t));
        kronecker_slab<<<blocks, BLOCK>>>(d_ct, d_z, P, C, j, d_out);
        unsigned long long zero = 0;
        cudaMemcpy(d_nz, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mx, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        reduce_stats<<<blocks, BLOCK>>>(d_out, P, j, d_nz, d_mx);
        unsigned long long slab_nz, slab_mx;
        cudaMemcpy(&slab_nz, d_nz, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&slab_mx, d_mx, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        total_nz += slab_nz;
        if (slab_mx > global_max) global_max = slab_mx;
        if (j % 500 == 0 || j == P-1) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double el = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            double eta = j>0 ? el*(P-j)/j : 0;
            printf("  j=%d/%d (%.0f%%) %llu nz, max=%llu, %.0fs, ETA %.0fs\n",
                   j, P, 100.0*j/P, total_nz, global_max, el, eta);
            fflush(stdout);

            // Checkpoint: save running stats so partial results survive if killed
            char ckpt[256];
            snprintf(ckpt, 256, "scripts/experiments/kronecker-coefficients/results/checkpoint_n%d.txt", n);
            FILE *fc_out = fopen(ckpt, "w");
            if (fc_out) {
                fprintf(fc_out, "n=%d\nP=%d\nslab=%d/%d\nnonzero=%llu\nmax=%llu\nelapsed=%.1f\n",
                        n, P, j+1, P, total_nz, global_max, el);
                fclose(fc_out);
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("S_%d Kronecker (GPU-only)\nP=%d, nonzero=%llu, max=%llu\nTime: %.1fs\n",
           n, P, total_nz, global_max, total);
    printf("========================================\n");

    // Clean up checkpoint
    char ckpt[256];
    snprintf(ckpt, 256, "scripts/experiments/kronecker-coefficients/results/checkpoint_n%d.txt", n);
    remove(ckpt);
    free(h_ct); free(h_z);
    cudaFree(d_ct); cudaFree(d_z); cudaFree(d_out); cudaFree(d_nz); cudaFree(d_mx);
}
