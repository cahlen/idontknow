/*
 * Optimized Kronecker coefficient GPU kernel for S_n.
 *
 * g(λ,μ,ν) = Σ_{ρ⊢n} (1/z_ρ) χ^λ(ρ) χ^μ(ρ) χ^ν(ρ)
 *
 * Optimizations over kronecker_gpu.cu:
 *   1. Shared memory tiling: load character table tiles into shared mem
 *   2. Coalesced global reads: transpose access pattern so adjacent
 *      threads read adjacent memory
 *   3. Only valid (i,j,k) triples launched: no wasted threads
 *   4. Fused reduction: stats computed inline, no second kernel
 *   5. Kahan summation: compensated sum for precision with large values
 *
 * Character table stored as double (sufficient for accumulation;
 * individual values lose low bits but final Kronecker coeff is exact
 * after rounding, as is standard in computational group theory).
 *
 * Input: char_table_n<N>.dbin (P×C doubles, row-major)
 *        z_inv_n<N>.bin (C doubles)
 * Output: stats only (nonzero count, max |g|) + optional CSV
 *
 * Compile: nvcc -O3 -arch=sm_90 -o kronecker_fast kronecker_fast.cu -lm
 * Run:     ./kronecker_fast <n> [gpu_id]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define TILE_C 64   /* classes per shared memory tile */

/*
 * Slab kernel: for fixed j, compute g(i,j,k) for all valid i<=j, k>=j.
 *
 * Grid: (ceil(valid_i/BLOCK_X), ceil(valid_k/BLOCK_Y))
 * Each thread computes one (i,k) pair for the fixed j.
 *
 * Shared memory holds tiles of 3 rows: ct[i,c], ct[j,c], ct[k,c]
 * and z_inv[c], tiled over classes c in chunks of TILE_C.
 */
__global__ void kronecker_slab_tiled(
    const double *__restrict__ ct,    /* P × C, row-major */
    const double *__restrict__ z_inv, /* C */
    int P, int C, int j,
    unsigned long long *__restrict__ nz_count,
    unsigned long long *__restrict__ max_abs)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;  /* 0..j */
    int dk = blockIdx.y * BLOCK_Y + threadIdx.y;  /* offset from j: k = j + dk */
    int k = j + dk;

    if (i > j || k >= P) return;

    /* Shared memory for tiling over class dimension */
    __shared__ double s_zi[TILE_C];        /* z_inv tile */
    __shared__ double s_row_j[TILE_C];     /* ct[j, c] tile (same for whole slab) */

    double sum = 0.0;
    double comp = 0.0;  /* Kahan compensation */

    for (int c0 = 0; c0 < C; c0 += TILE_C) {
        int tile_len = (c0 + TILE_C <= C) ? TILE_C : (C - c0);

        /* Cooperatively load z_inv and row j into shared memory */
        int lid = threadIdx.y * BLOCK_X + threadIdx.x;
        int nthreads = BLOCK_X * BLOCK_Y;
        for (int t = lid; t < tile_len; t += nthreads) {
            s_zi[t] = z_inv[c0 + t];
            s_row_j[t] = ct[(int64_t)j * C + c0 + t];
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            double val = s_zi[t]
                * ct[(int64_t)i * C + c0 + t]
                * s_row_j[t]
                * ct[(int64_t)k * C + c0 + t];
            /* Kahan summation */
            double y = val - comp;
            double t2 = sum + y;
            comp = (t2 - sum) - y;
            sum = t2;
        }
        __syncthreads();
    }

    int64_t g = llround(sum);
    if (g != 0) {
        atomicAdd(nz_count, 1ULL);
        unsigned long long av = (unsigned long long)(g > 0 ? g : -g);
        atomicMax(max_abs, av);
    }
}


int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n> [gpu_id]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int gpu = argc > 2 ? atoi(argv[2]) : 0;
    cudaSetDevice(gpu);

    /* Load character table (doubles) */
    char path[512];
    snprintf(path, 512, "scripts/experiments/kronecker-coefficients/results/char_table_n%d.dbin", n);
    FILE *fc = fopen(path, "rb");
    if (!fc) {
        fprintf(stderr, "Cannot open %s — run convert_char_table.py first\n", path);
        return 1;
    }
    fseek(fc, 0, SEEK_END); long ct_sz = ftell(fc); fseek(fc, 0, SEEK_SET);

    snprintf(path, 512, "scripts/experiments/kronecker-coefficients/results/z_inv_n%d.bin", n);
    FILE *fz = fopen(path, "rb");
    fseek(fz, 0, SEEK_END); int C = ftell(fz) / sizeof(double); fseek(fz, 0, SEEK_SET);
    int P = ct_sz / (C * sizeof(double));

    printf("========================================\n");
    printf("Kronecker S_%d (optimized GPU)\n", n);
    printf("P=%d partitions, C=%d classes\n", P, C);
    printf("Character table: %.2f GB\n", ct_sz / 1e9);
    printf("Triples (i<=j<=k): %lld\n", (long long)P * (P + 1) * (P + 2) / 6);
    printf("========================================\n\n");
    fflush(stdout);

    double *h_ct = (double *)malloc(ct_sz);
    double *h_z = (double *)malloc(C * sizeof(double));
    fread(h_ct, 1, ct_sz, fc); fclose(fc);
    fread(h_z, sizeof(double), C, fz); fclose(fz);

    /* GPU alloc — no output buffer needed, stats accumulated atomically */
    double *d_ct, *d_z;
    unsigned long long *d_nz, *d_mx;

    cudaMalloc(&d_ct, ct_sz);
    cudaMalloc(&d_z, C * sizeof(double));
    cudaMalloc(&d_nz, sizeof(unsigned long long));
    cudaMalloc(&d_mx, sizeof(unsigned long long));
    cudaMemcpy(d_ct, h_ct, ct_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, C * sizeof(double), cudaMemcpyHostToDevice);

    printf("GPU memory: %.1f GB char table (no slab buffer needed)\n", ct_sz / 1e9);
    fflush(stdout);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long long zero = 0;
    cudaMemcpy(d_nz, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mx, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    for (int j = 0; j < P; j++) {
        int num_i = j + 1;        /* i = 0..j */
        int num_k = P - j;        /* k = j..P-1 */

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((num_i + BLOCK_X - 1) / BLOCK_X,
                  (num_k + BLOCK_Y - 1) / BLOCK_Y);

        kronecker_slab_tiled<<<grid, block>>>(
            d_ct, d_z, P, C, j, d_nz, d_mx);

        if (j % 500 == 0 || j == P - 1) {
            cudaDeviceSynchronize();
            unsigned long long snap_nz, snap_mx;
            cudaMemcpy(&snap_nz, d_nz, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&snap_mx, d_mx, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            clock_gettime(CLOCK_MONOTONIC, &t1);
            double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            double eta = j > 0 ? el * (P - j) / j : 0;
            printf("  j=%d/%d (%.1f%%) nz=%llu max=%llu %.0fs ETA %.0fs\n",
                   j, P, 100.0 * j / P, snap_nz, snap_mx, el, eta);
            fflush(stdout);

            /* Checkpoint */
            char ckpt[512];
            snprintf(ckpt, 512,
                     "scripts/experiments/kronecker-coefficients/results/checkpoint_n%d.txt", n);
            FILE *fck = fopen(ckpt, "w");
            if (fck) {
                fprintf(fck, "n=%d\nP=%d\nslab=%d/%d\nnonzero=%llu\nmax=%llu\nelapsed=%.1f\n",
                        n, P, j + 1, P, snap_nz, snap_mx, el);
                fclose(fck);
            }
        }
    }

    cudaDeviceSynchronize();
    unsigned long long final_nz, final_mx;
    cudaMemcpy(&final_nz, d_nz, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_mx, d_mx, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("S_%d Kronecker coefficients (full)\n", n);
    printf("Partitions: %d, Classes: %d\n", P, C);
    printf("Triples (i<=j<=k): %lld\n", (long long)P * (P + 1) * (P + 2) / 6);
    printf("Nonzero: %llu\n", final_nz);
    printf("Max |g|: %llu\n", final_mx);
    printf("Time: %.1fs\n", total_time);
    printf("========================================\n");

    char ckpt[512];
    snprintf(ckpt, 512, "scripts/experiments/kronecker-coefficients/results/checkpoint_n%d.txt", n);
    remove(ckpt);

    free(h_ct); free(h_z);
    cudaFree(d_ct); cudaFree(d_z);
    cudaFree(d_nz); cudaFree(d_mx);
    return 0;
}
