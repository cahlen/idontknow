/*
 * Ramsey R(5,5) — ALL 656 K₄₂ Extensions (TRUE multi-GPU)
 *
 * Each GPU processes its own batch of colorings independently.
 * No cross-GPU synchronization until all done.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o ramsey_extend_all \
 *          scripts/experiments/ramsey-r55/ramsey_extend_all.cu -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

typedef unsigned long long uint64;
#define BLOCK_SIZE 256
#define N 42

__global__ void check_extensions(
    uint64 start, uint64 count,
    const uint64 *red_k4, int num_red_k4,
    const uint64 *blue_k4, int num_blue_k4,
    int *num_solutions, int coloring_id)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64 ext = start + idx;
    uint64 blue_ext = (~ext) & ((1ULL << N) - 1);

    for (int k = 0; k < num_red_k4; k++)
        if ((ext & red_k4[k]) == red_k4[k]) return;
    for (int k = 0; k < num_blue_k4; k++)
        if ((blue_ext & blue_k4[k]) == blue_k4[k]) return;

    atomicAdd(num_solutions, 1);
    printf("*** R(5,5)>=44: coloring %d ext=0x%011llx ***\n", coloring_id, ext);
}

typedef struct {
    int num_red, num_blue;
    uint64 *red_k4, *blue_k4;
} ColoringData;

typedef struct {
    int gpu_id;
    int start_coloring, end_coloring;
    ColoringData *colorings;
    int total_solutions;
} GPUWork;

void *gpu_worker(void *arg) {
    GPUWork *work = (GPUWork*)arg;
    int g = work->gpu_id;
    cudaSetDevice(g);

    uint64 *d_red, *d_blue;
    int *d_nsol;
    cudaMalloc(&d_red, 5000 * sizeof(uint64));
    cudaMalloc(&d_blue, 5000 * sizeof(uint64));
    cudaMalloc(&d_nsol, sizeof(int));

    uint64 total = 1ULL << N;
    uint64 chunk_size = 1ULL << 30;

    work->total_solutions = 0;

    for (int c = work->start_coloring; c < work->end_coloring; c++) {
        ColoringData *cd = &work->colorings[c];

        cudaMemcpy(d_red, cd->red_k4, cd->num_red * sizeof(uint64), cudaMemcpyHostToDevice);
        cudaMemcpy(d_blue, cd->blue_k4, cd->num_blue * sizeof(uint64), cudaMemcpyHostToDevice);
        cudaMemset(d_nsol, 0, sizeof(int));

        for (uint64 start = 0; start < total; start += chunk_size) {
            uint64 count = (start + chunk_size > total) ? (total - start) : chunk_size;
            uint64 blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            check_extensions<<<blocks, BLOCK_SIZE>>>(
                start, count, d_red, cd->num_red, d_blue, cd->num_blue, d_nsol, c);
        }
        cudaDeviceSynchronize();

        int ns;
        cudaMemcpy(&ns, d_nsol, sizeof(int), cudaMemcpyDeviceToHost);
        if (ns > 0) {
            printf("[GPU %d] *** COLORING %d: %d SOLUTIONS! ***\n", g, c, ns);
            work->total_solutions += ns;
        }

        // Progress (every 10 colorings)
        int done = c - work->start_coloring + 1;
        int batch = work->end_coloring - work->start_coloring;
        if (done % 10 == 0 || done == batch)
            printf("[GPU %d] %d/%d colorings done | solutions: %d\n",
                   g, done, batch, work->total_solutions);
    }

    cudaFree(d_red); cudaFree(d_blue); cudaFree(d_nsol);
    return NULL;
}

int main() {
    printf("========================================\n");
    printf("Ramsey R(5,5) — ALL 656 K₄₂ Extensions\n");
    printf("TRUE multi-GPU (pthreads, no sync)\n");
    printf("========================================\n\n");

    FILE *f = fopen("scripts/experiments/ramsey-r55/mckay_k42_all.bin", "rb");
    if (!f) { printf("Cannot open data file\n"); return 1; }

    unsigned int num_colorings;
    fread(&num_colorings, sizeof(unsigned int), 1, f);
    printf("Colorings: %u\n", num_colorings);

    ColoringData *colorings = (ColoringData*)malloc(num_colorings * sizeof(ColoringData));
    for (unsigned int i = 0; i < num_colorings; i++) {
        unsigned int nr, nb;
        fread(&nr, sizeof(unsigned int), 1, f);
        fread(&nb, sizeof(unsigned int), 1, f);
        colorings[i].num_red = nr;
        colorings[i].num_blue = nb;
        colorings[i].red_k4 = (uint64*)malloc(nr * sizeof(uint64));
        colorings[i].blue_k4 = (uint64*)malloc(nb * sizeof(uint64));
        fread(colorings[i].red_k4, sizeof(uint64), nr, f);
        fread(colorings[i].blue_k4, sizeof(uint64), nb, f);
    }
    fclose(f);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    int per_gpu = (num_colorings + num_gpus - 1) / num_gpus;

    printf("Using %d GPUs, ~%d colorings each\n", num_gpus, per_gpu);
    printf("ETA: ~%.0f minutes\n\n", (double)per_gpu * 130.0 / 60.0);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Launch one thread per GPU
    pthread_t threads[8];
    GPUWork works[8];
    for (int g = 0; g < num_gpus; g++) {
        works[g].gpu_id = g;
        works[g].start_coloring = g * per_gpu;
        works[g].end_coloring = (g + 1) * per_gpu;
        if (works[g].end_coloring > (int)num_colorings)
            works[g].end_coloring = num_colorings;
        works[g].colorings = colorings;
        works[g].total_solutions = 0;
        pthread_create(&threads[g], NULL, gpu_worker, &works[g]);
        printf("[GPU %d] colorings %d–%d\n", g, works[g].start_coloring, works[g].end_coloring - 1);
    }

    // Wait for all
    int grand_total = 0;
    for (int g = 0; g < num_gpus; g++) {
        pthread_join(threads[g], NULL);
        grand_total += works[g].total_solutions;
        printf("[GPU %d] finished: %d solutions\n", g, works[g].total_solutions);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("ALL %u K₄₂ colorings exhaustively checked\n", num_colorings);
    printf("Total: %.2e extensions\n", (double)num_colorings * (1ULL << N));
    printf("Solutions: %d\n", grand_total);
    printf("Time: %.1fs (%.1f min)\n", elapsed, elapsed / 60);
    if (grand_total > 0)
        printf("\n*** R(5,5) >= 44! ***\n");
    else
        printf("\nNONE of the 656 K₄₂ colorings extend to K₄₃.\n");
    printf("========================================\n");

    for (unsigned int i = 0; i < num_colorings; i++) {
        free(colorings[i].red_k4); free(colorings[i].blue_k4);
    }
    free(colorings);
    return grand_total > 0 ? 0 : 1;
}
