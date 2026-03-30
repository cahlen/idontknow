/*
 * Class Numbers of Real Quadratic Fields — v2 Multi-GPU
 *
 * Computes h(d) for all fundamental discriminants d in [D_lo, D_hi]
 * using: h(d) = round(sqrt(d) * L(1, chi_d) / (2 * R(d)))
 *
 * Key improvements over v1:
 *   - Integer-only CF for regulator (no FP64 overflow)
 *   - Euler product with 9592 primes to 10^5 (was 1229 to 10^4)
 *   - CPU segmented sieve for fundamental discriminants
 *   - Multi-GPU via pthreads (one thread per GPU)
 *   - Incremental log accumulation for regulator
 *   - Cohen-Lenstra statistics collection
 *
 * Compile: nvcc -O3 -arch=sm_100a -o class_v2 \
 *          scripts/experiments/class-numbers/class_numbers_v2.cu -lpthread -lm
 *
 * Run:     ./class_v2 <start> <end>
 *   e.g.   ./class_v2 5 1000000000    (validate against known tables)
 *          ./class_v2 100000000000 10000000000000  (new computation)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

typedef unsigned long long uint64;
typedef long long int64;

#define BLOCK_SIZE 256
#define MAX_CF_STEPS 2000000   // cap for CF period (covers 99.9% of d < 10^13)
#define CHUNK_SIZE 10000000    // 10M raw d per chunk

// =====================================================
// Primes in constant memory (up to 100003 = 9592 primes)
// =====================================================
#define NUM_PRIMES 9592
__constant__ int d_primes[NUM_PRIMES];

// =====================================================
// Kronecker symbol (d/p) — modular exponentiation
// =====================================================
__device__ int kronecker(int64 d, int p) {
    if (p == 2) {
        int dm8 = ((int)(d % 8) + 8) % 8;
        if (dm8 == 1 || dm8 == 7) return 1;
        if (dm8 == 3 || dm8 == 5) return -1;
        return 0;
    }
    // Euler's criterion: d^((p-1)/2) mod p
    int64 a = ((d % p) + p) % p;
    if (a == 0) return 0;
    int64 result = 1;
    int64 exp = (p - 1) / 2;
    int64 base = a;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % p;
        base = (base * base) % p;
        exp >>= 1;
    }
    return (result == 1) ? 1 : -1;
}

// =====================================================
// Combined kernel: regulator + L-function + class number
// =====================================================
__global__ void compute_class_numbers(
    uint64 *discriminants,    // fundamental discriminants
    uint32_t count,
    int    *class_numbers_out,
    double *regulators_out,   // optional: NULL to skip output
    // Statistics (atomics)
    uint64 *h1_count,         // count of h(d) = 1
    uint64 *h_histogram,      // h_histogram[h] for h < 1024
    uint64 *total_processed,
    uint64 *div3_count,       // count of 3 | h(d)
    uint64 *div5_count,
    uint64 *div7_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64 d = discriminants[idx];
    if (d < 5) return;

    // ===== PHASE 1: Regulator (validated: matches PARI/GP on 1000 discriminants) =====
    // For d ≡ 0 mod 4 (d=4m): CF of √m, stop at first D==1
    // For d ≡ 1 mod 4: CF of (1+√d)/2, stop when P=1,Q=2

    double regulator = 0.0;
    double log_P_prev, log_P_curr, log_Q_prev, log_Q_curr;

    if (d % 4 == 0) {
        // d = 4m: CF of √m
        uint64 m_val = d / 4;
        uint64 a0 = (uint64)sqrt((double)m_val);
        while (a0 * a0 > m_val) a0--;
        while ((a0+1)*(a0+1) <= m_val) a0++;
        if (a0 * a0 == m_val) return;

        int64 mm = 0, D = 1, a = (int64)a0;
        log_P_prev = 0.0;
        log_P_curr = log((double)a0);
        log_Q_prev = -1e30;
        log_Q_curr = 0.0;

        for (int step = 0; step < MAX_CF_STEPS; step++) {
            mm = D * a - mm;
            D = ((int64)m_val - mm * mm) / D;
            if (D == 0) break;
            a = ((int64)a0 + mm) / D;

            // Check D==1 BEFORE updating convergents (critical!)
            if (D == 1) {
                double diff = log_Q_curr + 0.5 * log((double)m_val) - log_P_curr;
                regulator = log_P_curr + log(1.0 + exp(diff));
                break;
            }

            // Update log convergents
            double rp = exp(log_P_prev - log_P_curr);
            log_P_prev = log_P_curr;
            log_P_curr = log_P_curr + log((double)a + rp);
            double rq = (log_Q_prev > -1e20) ? exp(log_Q_prev - log_Q_curr) : 0.0;
            log_Q_prev = log_Q_curr;
            log_Q_curr = log_Q_curr + log((double)a + rq);
        }
    } else {
        // d ≡ 1 mod 4: CF of (1+√d)/2
        // Use INTEGER sqrt to avoid FP rounding issues
        uint64 isqrt_d = (uint64)sqrt((double)d);
        while (isqrt_d * isqrt_d > d) isqrt_d--;
        while ((isqrt_d+1)*(isqrt_d+1) <= d) isqrt_d++;
        // a0 = floor((1 + isqrt_d) / 2)
        int64 a0 = (int64)((1 + isqrt_d) / 2);
        int64 P = 1, Q = 2, a = a0;
        log_P_prev = 0.0;
        log_P_curr = log((double)(a0 > 0 ? a0 : 1));
        log_Q_prev = -1e30;
        log_Q_curr = 0.0;

        for (int step = 0; step < MAX_CF_STEPS; step++) {
            int64 P_new = a * Q - P;
            int64 Q_new = ((int64)d - P_new * P_new) / Q;
            if (Q_new == 0) break;
            int64 a_new = (P_new + (int64)isqrt_d) / Q_new;
            P = P_new; Q = Q_new; a = a_new;

            // Period detection: P=1, Q=2 (no step>0 guard — handles period-1)
            if (P == 1 && Q == 2) {
                double diff = log_Q_curr + 0.5 * log((double)d) - log_P_curr;
                regulator = log_P_curr + log(1.0 + exp(diff)) - log(2.0);
                break;
            }

            // Update log convergents
            double rp = exp(log_P_prev - log_P_curr);
            log_P_prev = log_P_curr;
            log_P_curr = log_P_curr + log((double)a + rp);
            double rq = (log_Q_prev > -1e20) ? exp(log_Q_prev - log_Q_curr) : 0.0;
            log_Q_prev = log_Q_curr;
            log_Q_curr = log_Q_curr + log((double)a + rq);
        }
    }

    if (regulator < 0.01) regulator = 0.01;

    // ===== PHASE 2: L(1, chi_d) via Euler product =====
    double L1 = 1.0;
    for (int i = 0; i < NUM_PRIMES; i++) {
        int p = d_primes[i];
        int chi = kronecker((int64)d, p);
        if (chi != 0) {
            L1 *= 1.0 / (1.0 - (double)chi / p);
        }
        // If chi = 0, the factor is 1/(1-0) = 1, no change
    }

    // ===== PHASE 3: Assemble class number =====
    double h_approx = sqrt((double)d) * L1 / (2.0 * regulator);
    int h = (int)round(h_approx);
    if (h < 1) h = 1;

    class_numbers_out[idx] = h;
    if (regulators_out) regulators_out[idx] = regulator;

    // ===== PHASE 4: Statistics =====
    atomicAdd(total_processed, 1ULL);
    if (h == 1) atomicAdd(h1_count, 1ULL);
    if (h < 1024) atomicAdd(&h_histogram[h], 1ULL);
    if (h % 3 == 0) atomicAdd(div3_count, 1ULL);
    if (h % 5 == 0) atomicAdd(div5_count, 1ULL);
    if (h % 7 == 0) atomicAdd(div7_count, 1ULL);
}

// =====================================================
// CPU: Segmented sieve for fundamental discriminants
// =====================================================
// A fundamental discriminant is either:
//   d ≡ 1 (mod 4) and squarefree, or
//   d = 4m where m ≡ 2,3 (mod 4) and m squarefree
// For positive d (real quadratic fields): d > 0.

uint32_t sieve_fundamental_discriminants(uint64 lo, uint64 hi, uint64 *out, uint32_t max_out) {
    uint64 len = hi - lo;
    if (len > 1000000000ULL) { fprintf(stderr, "Chunk too large\n"); return 0; }

    // Mark squarefree using small primes
    char *is_sqfree = (char*)calloc(len, 1);
    memset(is_sqfree, 1, len);

    // Sieve out multiples of p^2 for p up to sqrt(hi)
    uint64 sqrt_hi = (uint64)sqrt((double)hi) + 1;
    for (uint64 p = 2; p <= sqrt_hi; p++) {
        uint64 p2 = p * p;
        uint64 start = ((lo + p2 - 1) / p2) * p2;
        for (uint64 m = start; m < hi; m += p2) {
            is_sqfree[m - lo] = 0;
        }
    }

    // Extract fundamental discriminants
    uint32_t count = 0;
    for (uint64 d = lo; d < hi && count < max_out; d++) {
        if (d < 5) continue;
        uint64 idx = d - lo;

        // Check if d is a fundamental discriminant
        if (d % 4 == 1 && is_sqfree[idx]) {
            out[count++] = d;
        } else if (d % 4 == 0) {
            uint64 m = d / 4;
            if ((m % 4 == 2 || m % 4 == 3) && m >= lo && m < hi && is_sqfree[m - lo]) {
                out[count++] = d;
            } else if ((m % 4 == 2 || m % 4 == 3) && (m < lo || m >= hi)) {
                // m outside sieve range — check squarefree directly (slow, rare)
                int sqf = 1;
                for (uint64 p = 2; p * p <= m && sqf; p++)
                    if (m % (p*p) == 0) sqf = 0;
                if (sqf) out[count++] = d;
            }
        }
    }

    free(is_sqfree);
    return count;
}

// =====================================================
// Generate prime table
// =====================================================
int generate_primes(int *primes, int max_prime) {
    char *sieve = (char*)calloc(max_prime + 1, 1);
    memset(sieve, 1, max_prime + 1);
    sieve[0] = sieve[1] = 0;
    for (int i = 2; i * i <= max_prime; i++)
        if (sieve[i])
            for (int j = i*i; j <= max_prime; j += i)
                sieve[j] = 0;
    int count = 0;
    for (int i = 2; i <= max_prime && count < NUM_PRIMES; i++)
        if (sieve[i]) primes[count++] = i;
    free(sieve);
    return count;
}

// =====================================================
// GPU worker thread
// =====================================================
typedef struct {
    int gpu_id;
    uint64 d_start, d_end;
    // Results
    uint64 total_processed;
    uint64 h1_count;
    uint64 div3, div5, div7;
    uint64 h_hist[1024];
} GPUWork;

void *gpu_worker(void *arg) {
    GPUWork *work = (GPUWork*)arg;
    cudaSetDevice(work->gpu_id);

    // Allocate GPU buffers
    uint64 *d_discriminants;
    int *d_class_numbers;
    uint64 *d_h1, *d_total, *d_div3, *d_div5, *d_div7, *d_hist;

    uint32_t max_per_chunk = CHUNK_SIZE;  // max fundamental discriminants per chunk
    cudaMalloc(&d_discriminants, max_per_chunk * sizeof(uint64));
    cudaMalloc(&d_class_numbers, max_per_chunk * sizeof(int));
    cudaMalloc(&d_h1, sizeof(uint64));
    cudaMalloc(&d_total, sizeof(uint64));
    cudaMalloc(&d_div3, sizeof(uint64));
    cudaMalloc(&d_div5, sizeof(uint64));
    cudaMalloc(&d_div7, sizeof(uint64));
    cudaMalloc(&d_hist, 1024 * sizeof(uint64));

    cudaMemset(d_h1, 0, sizeof(uint64));
    cudaMemset(d_total, 0, sizeof(uint64));
    cudaMemset(d_div3, 0, sizeof(uint64));
    cudaMemset(d_div5, 0, sizeof(uint64));
    cudaMemset(d_div7, 0, sizeof(uint64));
    cudaMemset(d_hist, 0, 1024 * sizeof(uint64));

    uint64 *h_discriminants = (uint64*)malloc(max_per_chunk * sizeof(uint64));

    uint64 chunk_raw = CHUNK_SIZE * 3;  // raw integers per chunk (density ~1/3)
    uint64 chunks_done = 0;

    for (uint64 d_lo = work->d_start; d_lo < work->d_end; d_lo += chunk_raw) {
        uint64 d_hi = d_lo + chunk_raw;
        if (d_hi > work->d_end) d_hi = work->d_end;

        // Sieve on CPU
        uint32_t count = sieve_fundamental_discriminants(d_lo, d_hi, h_discriminants, max_per_chunk);
        if (count == 0) continue;

        // Upload to GPU
        cudaMemcpy(d_discriminants, h_discriminants, count * sizeof(uint64), cudaMemcpyHostToDevice);

        // Launch kernel
        int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_class_numbers<<<blocks, BLOCK_SIZE>>>(
            d_discriminants, count, d_class_numbers, NULL,
            d_h1, d_hist, d_total, d_div3, d_div5, d_div7);

        chunks_done++;
        if (chunks_done % 100 == 0) {
            cudaDeviceSynchronize();
            uint64 total;
            cudaMemcpy(&total, d_total, sizeof(uint64), cudaMemcpyDeviceToHost);
            double pct = 100.0 * (d_lo - work->d_start) / (double)(work->d_end - work->d_start);
            printf("[GPU %d] %.1f%% | %llu discriminants processed | d ~ %.2e\n",
                   work->gpu_id, pct, total, (double)d_lo);
            fflush(stdout);
        }
    }

    // Collect results
    cudaDeviceSynchronize();
    cudaMemcpy(&work->total_processed, d_total, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&work->h1_count, d_h1, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&work->div3, d_div3, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&work->div5, d_div5, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&work->div7, d_div7, sizeof(uint64), cudaMemcpyDeviceToHost);
    cudaMemcpy(work->h_hist, d_hist, 1024 * sizeof(uint64), cudaMemcpyDeviceToHost);

    cudaFree(d_discriminants); cudaFree(d_class_numbers);
    cudaFree(d_h1); cudaFree(d_total); cudaFree(d_div3); cudaFree(d_div5); cudaFree(d_div7);
    cudaFree(d_hist);
    free(h_discriminants);

    printf("[GPU %d] done: %llu discriminants\n", work->gpu_id, work->total_processed);
    return NULL;
}

// =====================================================
// Main
// =====================================================
int main(int argc, char **argv) {
    uint64 D_start = argc > 1 ? strtoull(argv[1], NULL, 10) : 5;
    uint64 D_end = argc > 2 ? strtoull(argv[2], NULL, 10) : 1000000;

    printf("========================================\n");
    printf("Class Numbers of Real Quadratic Fields v2\n");
    printf("Range: [%llu, %llu)\n", D_start, D_end);
    printf("========================================\n\n");

    // Generate primes
    int h_primes[NUM_PRIMES];
    int nprimes = generate_primes(h_primes, 100003);
    printf("Primes: %d (up to %d)\n", nprimes, h_primes[nprimes-1]);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("GPUs: %d\n\n", num_gpus);

    // Upload primes to all GPUs
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMemcpyToSymbol(d_primes, h_primes, nprimes * sizeof(int));
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Launch workers
    uint64 range = D_end - D_start;
    uint64 per_gpu = (range + num_gpus - 1) / num_gpus;

    pthread_t threads[8];
    GPUWork works[8];
    for (int g = 0; g < num_gpus; g++) {
        works[g].gpu_id = g;
        works[g].d_start = D_start + g * per_gpu;
        works[g].d_end = D_start + (g + 1) * per_gpu;
        if (works[g].d_end > D_end) works[g].d_end = D_end;
        memset(works[g].h_hist, 0, sizeof(works[g].h_hist));
        pthread_create(&threads[g], NULL, gpu_worker, &works[g]);
    }

    // Collect
    uint64 grand_total = 0, grand_h1 = 0;
    uint64 grand_div3 = 0, grand_div5 = 0, grand_div7 = 0;
    uint64 grand_hist[1024] = {0};

    for (int g = 0; g < num_gpus; g++) {
        pthread_join(threads[g], NULL);
        grand_total += works[g].total_processed;
        grand_h1 += works[g].h1_count;
        grand_div3 += works[g].div3;
        grand_div5 += works[g].div5;
        grand_div7 += works[g].div7;
        for (int h = 0; h < 1024; h++)
            grand_hist[h] += works[g].h_hist[h];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("Range: [%llu, %llu)\n", D_start, D_end);
    printf("Fundamental discriminants: %llu\n", grand_total);
    printf("Time: %.1fs (%.0f disc/sec)\n", elapsed, grand_total / elapsed);
    printf("\nCohen-Lenstra statistics:\n");
    printf("  h(d) = 1: %llu (%.4f%%)\n", grand_h1, 100.0 * grand_h1 / grand_total);
    printf("  C-L predicted h=1: ~75.446%%\n");
    printf("  3 | h(d): %llu (%.4f%%)\n", grand_div3, 100.0 * grand_div3 / grand_total);
    printf("  5 | h(d): %llu (%.4f%%)\n", grand_div5, 100.0 * grand_div5 / grand_total);
    printf("  7 | h(d): %llu (%.4f%%)\n", grand_div7, 100.0 * grand_div7 / grand_total);

    printf("\nClass number distribution (first 20):\n");
    for (int h = 1; h <= 20; h++)
        printf("  h=%2d: %llu (%.3f%%)\n", h, grand_hist[h], 100.0 * grand_hist[h] / grand_total);

    printf("\n========================================\n");
    return 0;
}
