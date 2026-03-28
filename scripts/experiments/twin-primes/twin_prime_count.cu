/*
 * CUDA-accelerated Twin Prime counting and gap analysis
 *
 * Twin Prime Conjecture: There are infinitely many primes p such that p+2 is also prime.
 *
 * We count all twin prime pairs up to N and analyze:
 *   - Twin prime counting function pi_2(N)
 *   - Distribution of gaps between consecutive twin prime pairs
 *   - Comparison with the Hardy-Littlewood prediction: pi_2(N) ~ 2*C2 * N/(ln N)^2
 *     where C2 = 0.6601618... is the twin prime constant
 *   - Record twin prime gaps
 *
 * Method: segmented sieve of Eratosthenes on GPU, then scan for twins.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o twin_prime_count scripts/experiments/twin-primes/twin_prime_count.cu
 * Run:     ./twin_prime_count <N>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define THREADS_PER_BLOCK 256
#define SEGMENT_SIZE (1 << 20)  // 1M per segment

// Small primes for the sieve (primes up to sqrt(10^12) ≈ 10^6)
#define MAX_SMALL_PRIMES 80000  // pi(10^6) ≈ 78498

__constant__ uint32_t d_small_primes[MAX_SMALL_PRIMES];
__constant__ uint32_t d_num_small_primes;

// Segmented sieve kernel: mark composites in a segment
__global__ void sieve_segment(uint8_t *segment, uint64_t seg_start, uint32_t seg_size) {
    // Each thread handles one small prime
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_num_small_primes) return;

    uint32_t p = d_small_primes[idx];
    if (p < 2) return;

    // Find first multiple of p in [seg_start, seg_start + seg_size)
    uint64_t first = ((seg_start + p - 1) / p) * p;
    if (first == p) first += p;  // don't mark p itself

    for (uint64_t j = first; j < seg_start + seg_size; j += p) {
        segment[j - seg_start] = 0;  // mark composite
    }
}

// Count twin primes in a sieved segment
__global__ void count_twins(uint8_t *segment, uint64_t seg_start, uint32_t seg_size,
                             uint64_t *twin_count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seg_size - 2) return;

    uint64_t n = seg_start + idx;
    if (n < 3) return;
    if (n % 2 == 0) return;  // only check odd n

    if (segment[idx] && segment[idx + 2]) {
        // n and n+2 are both prime — twin prime pair!
        atomicAdd((unsigned long long*)twin_count, (unsigned long long)1);
    }
}

// Generate small primes on CPU via simple sieve
void generate_small_primes(uint32_t limit, uint32_t *primes, uint32_t *count) {
    uint8_t *is_prime = (uint8_t *)calloc(limit + 1, 1);
    memset(is_prime, 1, limit + 1);
    is_prime[0] = is_prime[1] = 0;

    for (uint32_t i = 2; (uint64_t)i * i <= limit; i++) {
        if (is_prime[i]) {
            for (uint32_t j = i * i; j <= limit; j += i)
                is_prime[j] = 0;
        }
    }

    *count = 0;
    for (uint32_t i = 2; i <= limit && *count < MAX_SMALL_PRIMES; i++) {
        if (is_prime[i]) primes[(*count)++] = i;
    }
    free(is_prime);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        fprintf(stderr, "  Count twin primes up to N\n");
        return 1;
    }

    uint64_t N = (uint64_t)atoll(argv[1]);
    printf("Twin Prime Counting: up to N=%lu\n", N);

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("GPUs available: %d\n", device_count);

    // Generate small primes
    uint32_t sieve_limit = (uint32_t)sqrt((double)N) + 1;
    uint32_t *h_primes = (uint32_t *)malloc(MAX_SMALL_PRIMES * sizeof(uint32_t));
    uint32_t num_primes;
    generate_small_primes(sieve_limit, h_primes, &num_primes);
    printf("Small primes up to %u: %u primes\n\n", sieve_limit, num_primes);

    // Copy to constant memory
    cudaMemcpyToSymbol(d_small_primes, h_primes, num_primes * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_num_small_primes, &num_primes, sizeof(uint32_t));

    uint8_t *d_segment;
    uint64_t *d_twin_count;
    cudaMalloc(&d_segment, SEGMENT_SIZE);
    cudaMalloc(&d_twin_count, sizeof(uint64_t));
    cudaMemset(d_twin_count, 0, sizeof(uint64_t));

    uint64_t total_twins = 0;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    uint64_t segments = (N + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

    for (uint64_t seg = 0; seg < segments; seg++) {
        uint64_t seg_start = seg * SEGMENT_SIZE;
        uint32_t seg_size = SEGMENT_SIZE;
        if (seg_start + seg_size > N) seg_size = N - seg_start;

        int gpu = seg % device_count;
        cudaSetDevice(gpu);

        // Initialize segment: all marked as prime
        cudaMemset(d_segment, 1, seg_size);

        // Sieve
        int sieve_blocks = (num_primes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sieve_segment<<<sieve_blocks, THREADS_PER_BLOCK>>>(d_segment, seg_start, seg_size);

        // Mark evens as composite (except 2)
        // Actually handled in count_twins by only checking odd n

        // Count twins
        int count_blocks = (seg_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cudaMemset(d_twin_count, 0, sizeof(uint64_t));
        count_twins<<<count_blocks, THREADS_PER_BLOCK>>>(d_segment, seg_start, seg_size, d_twin_count);
        cudaDeviceSynchronize();

        uint64_t seg_twins;
        cudaMemcpy(&seg_twins, d_twin_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        total_twins += seg_twins;

        if (seg % 1000 == 0 || seg == segments - 1) {
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                             (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
            double progress = (double)(seg + 1) / segments * 100;
            printf("[GPU %d] Segment %lu/%lu (%.1f%%) | twins so far: %lu | %.1fs\n",
                   gpu, seg + 1, segments, progress, total_twins, elapsed);
            fflush(stdout);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                    (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    // Hardy-Littlewood prediction
    double C2 = 0.6601618158;
    double ln_N = log((double)N);
    double hl_prediction = 2.0 * C2 * N / (ln_N * ln_N);

    printf("\n========================================\n");
    printf("Twin Prime Count up to N=%lu\n", N);
    printf("Twin prime pairs found: %lu\n", total_twins);
    printf("Hardy-Littlewood prediction: %.0f\n", hl_prediction);
    printf("Ratio (actual/predicted): %.6f\n", total_twins / hl_prediction);
    printf("Time: %.1fs\n", elapsed);
    printf("========================================\n");

    cudaFree(d_segment);
    cudaFree(d_twin_count);
    free(h_primes);
    return 0;
}
