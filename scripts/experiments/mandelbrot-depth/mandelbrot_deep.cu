/*
 * CUDA deep Mandelbrot set exploration
 *
 * Compute the Mandelbrot set at extreme zoom levels using high-precision
 * arithmetic. At zoom depths beyond 10^15, standard double precision fails
 * and we need 128-bit or arbitrary precision fixed-point.
 *
 * This kernel uses double-double arithmetic (two doubles giving ~31 decimal
 * digits of precision) to push zoom levels to ~10^28.
 *
 * Output: iteration count per pixel, which maps to a color in the
 * final visualization.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o mandelbrot_deep scripts/experiments/mandelbrot-depth/mandelbrot_deep.cu
 * Run:     ./mandelbrot_deep <width> <height> <center_re> <center_im> <zoom> <max_iter> <output.raw>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

// Double-double arithmetic for ~31 digits of precision
// Represents a number as hi + lo where |lo| << |hi|
struct dd {
    double hi, lo;
};

__device__ dd dd_add(dd a, dd b) {
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double t = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    dd r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}

__device__ dd dd_sub(dd a, dd b) {
    b.hi = -b.hi; b.lo = -b.lo;
    return dd_add(a, b);
}

__device__ dd dd_mul(dd a, dd b) {
    double p = a.hi * b.hi;
    double e = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
    dd r;
    r.hi = p + e;
    r.lo = e - (r.hi - p);
    return r;
}

__device__ dd dd_from(double x) {
    dd r; r.hi = x; r.lo = 0.0; return r;
}

__global__ void mandelbrot_kernel(uint32_t *iterations,
                                   double center_re, double center_im,
                                   double pixel_size,
                                   uint32_t width, uint32_t height,
                                   uint32_t max_iter) {
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Map pixel to complex plane
    dd cre = dd_from(center_re + (px - (double)width / 2.0) * pixel_size);
    dd cim = dd_from(center_im + (py - (double)height / 2.0) * pixel_size);

    dd zre = dd_from(0.0);
    dd zim = dd_from(0.0);

    uint32_t iter;
    for (iter = 0; iter < max_iter; iter++) {
        // z = z^2 + c
        dd zre2 = dd_mul(zre, zre);
        dd zim2 = dd_mul(zim, zim);

        // Escape check: |z|^2 > 4
        if (zre2.hi + zim2.hi > 4.0) break;

        dd new_zre = dd_add(dd_sub(zre2, zim2), cre);
        dd new_zim = dd_add(dd_mul(dd_from(2.0), dd_mul(zre, zim)), cim);
        zre = new_zre;
        zim = new_zim;
    }

    iterations[py * width + px] = iter;
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <width> <height> <center_re> <center_im> <zoom> <max_iter> <output.raw>\n", argv[0]);
        fprintf(stderr, "\nExample (full set):  %s 8192 8192 -0.5 0.0 1.0 10000 mandelbrot.raw\n", argv[0]);
        fprintf(stderr, "Example (deep zoom): %s 8192 8192 -0.74364388703 0.13182590421 1e12 100000 deep.raw\n", argv[0]);
        return 1;
    }

    uint32_t width = atoi(argv[1]);
    uint32_t height = atoi(argv[2]);
    double center_re = atof(argv[3]);
    double center_im = atof(argv[4]);
    double zoom = atof(argv[5]);
    uint32_t max_iter = atoi(argv[6]);
    const char *output_path = argv[7];

    double pixel_size = 4.0 / (zoom * width);

    printf("Mandelbrot Deep Zoom\n");
    printf("Resolution: %u x %u\n", width, height);
    printf("Center: (%.15f, %.15f)\n", center_re, center_im);
    printf("Zoom: %.2e\n", zoom);
    printf("Pixel size: %.2e\n", pixel_size);
    printf("Max iterations: %u\n", max_iter);
    printf("Output: %s\n\n", output_path);

    uint64_t total_pixels = (uint64_t)width * height;
    size_t buf_size = total_pixels * sizeof(uint32_t);

    uint32_t *d_iters;
    cudaMalloc(&d_iters, buf_size);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    mandelbrot_kernel<<<grid, block>>>(d_iters, center_re, center_im, pixel_size,
                                        width, height, max_iter);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                    (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("Computed in %.2fs (%.0f Mpix/sec)\n", elapsed, total_pixels / elapsed / 1e6);

    // Copy back and write
    uint32_t *h_iters = (uint32_t *)malloc(buf_size);
    cudaMemcpy(h_iters, d_iters, buf_size, cudaMemcpyDeviceToHost);

    FILE *f = fopen(output_path, "wb");
    fwrite(&width, sizeof(uint32_t), 1, f);
    fwrite(&height, sizeof(uint32_t), 1, f);
    fwrite(&max_iter, sizeof(uint32_t), 1, f);
    fwrite(h_iters, sizeof(uint32_t), total_pixels, f);
    fclose(f);

    // Stats
    uint64_t escaped = 0, max_iter_count = 0;
    uint32_t actual_max = 0;
    for (uint64_t i = 0; i < total_pixels; i++) {
        if (h_iters[i] < max_iter) escaped++;
        else max_iter_count++;
        if (h_iters[i] > actual_max) actual_max = h_iters[i];
    }

    printf("Pixels escaped: %lu (%.1f%%)\n", escaped, 100.0 * escaped / total_pixels);
    printf("Pixels in set: %lu (%.1f%%)\n", max_iter_count, 100.0 * max_iter_count / total_pixels);
    printf("Max iteration reached: %u\n", actual_max);
    printf("Output written: %s (%lu bytes)\n", output_path, buf_size + 12);

    free(h_iters);
    cudaFree(d_iters);
    return 0;
}
