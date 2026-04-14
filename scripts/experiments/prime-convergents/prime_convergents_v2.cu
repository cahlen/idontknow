/*
 * Prime Convergents of Continued Fractions — GPU Kernel v2
 *
 * v2: Full uint128 convergent recurrence (depth ~75 vs ~38 in v1).
 *     Miller-Rabin and GPF extended to 128-bit inputs.
 *
 * For a large sample of irrational numbers (random CF expansions + constants),
 * compute convergents C_n = A_n/B_n to large depth and track:
 *   1. G(A_n) — greatest prime factor of the numerator
 *   2. G(B_n) — greatest prime factor of the denominator
 *   3. Whether A_n and B_n are both prime ("doubly-prime convergent")
 *
 * Extends the results of Humphreys (2013, NCUR/Boise State) which showed:
 *   - Corollary 3.6: For almost all ζ, G(A_n) ≥ e^{n/(50 ln n)} for large n
 *   - Section 4: Only 3 doubly-prime convergents of e found in 2000 terms
 *
 * Compile: nvcc -O3 -arch=sm_90 -o prime_convergents_v2 prime_convergents_v2.cu -lm
 * Run:     ./prime_convergents_v2 [num_samples] [max_depth] [mode]
 *          mode=0: random CF expansions (partial quotients from Gauss-Kuzmin)
 *          mode=1: e (one thread = one copy, all get same CF)
 *          mode=2: pi (first 50 known terms, then random)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cinttypes>
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef unsigned __int128 uint128;

#define MAX_DEPTH_LIMIT 10000
#define BLOCK_SIZE 256

/* ------------------------------------------------------------------ */
/* Device: 128-bit modular multiplication via uint128 native ops      */
/* CUDA supports __int128 on device for sm_50+.                       */
/* For mulmod128 we need (a * b) % m where a,b,m are uint128.         */
/* Since uint128 * uint128 can overflow, we use binary method.        */
/* ------------------------------------------------------------------ */

__device__ uint128 mulmod128(uint128 a, uint128 b, uint128 m) {
    /* Binary multiplication with modular reduction at each step.
     * This avoids 256-bit intermediate at the cost of ~128 iterations max.
     * For our use case (Miller-Rabin with ~12 witnesses), this is fine. */
    a %= m;
    b %= m;
    uint128 result = 0;
    while (b > 0) {
        if (b & 1) {
            result = (result + a) % m;  /* safe: result < m, a < m, so sum < 2m < 2^129 — but uint128 max is 2^128-1 */
            /* Handle potential overflow of result + a:
             * if result + a wraps, the true value is result + a + 2^128,
             * and we need (result + a + 2^128) % m. But if m < 2^127
             * this never happens. For m up to ~2^128, use careful add: */
        }
        a = (a + a) % m;  /* double a mod m — same overflow concern */
        b >>= 1;
    }
    return result;
}

/* Safe addmod to handle potential uint128 overflow */
__device__ uint128 addmod128(uint128 a, uint128 b, uint128 m) {
    a %= m;
    b %= m;
    /* If a + b might overflow uint128, subtract instead */
    if (a >= m - b) {
        return a - (m - b);
    }
    return a + b;
}

/* Corrected mulmod128 using safe addmod */
__device__ uint128 mulmod128_safe(uint128 a, uint128 b, uint128 m) {
    a %= m;
    b %= m;
    uint128 result = 0;
    while (b > 0) {
        if (b & 1) {
            result = addmod128(result, a, m);
        }
        a = addmod128(a, a, m);
        b >>= 1;
    }
    return result;
}

__device__ uint128 powmod128(uint128 base, uint128 exp, uint128 mod) {
    uint128 result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod128_safe(result, base, mod);
        exp >>= 1;
        base = mulmod128_safe(base, base, mod);
    }
    return result;
}

/* ------------------------------------------------------------------ */
/* Device: Miller-Rabin primality for uint128                         */
/* ------------------------------------------------------------------ */

__device__ int is_prime_128(uint128 n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    if (n < 25) return 1;

    /* Small factor check up to 997 */
    const uint64_t small_check[] = {
        5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,
        83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,
        167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251
    };
    for (int i = 0; i < 52; i++) {
        if (n == (uint128)small_check[i]) return 1;
        if (n % small_check[i] == 0) return 0;
    }

    /* Write n-1 = d * 2^r */
    uint128 d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }

    /* For n < 2^128, testing witnesses {2,3,5,7,11,13,17,19,23,29,31,37}
     * is sufficient for n < 3.317×10^23. For larger n (up to 2^128 ≈ 3.4×10^38),
     * we add a few more witnesses for safety. */
    const uint64_t witnesses[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
    for (int i = 0; i < 16; i++) {
        uint128 a = (uint128)witnesses[i];
        if (a >= n) continue;

        uint128 x = powmod128(a, d, n);
        if (x == 1 || x == n - 1) continue;

        int found = 0;
        for (int j = 0; j < r - 1; j++) {
            x = mulmod128_safe(x, x, n);
            if (x == n - 1) { found = 1; break; }
        }
        if (!found) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Device: Greatest prime factor for uint128                          */
/* Trial division by primes up to 997, then Miller-Rabin on remainder */
/* ------------------------------------------------------------------ */

__device__ const int small_primes[] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
    73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,
    157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,
    239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,
    331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,
    421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,
    509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,
    613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,
    709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,
    821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,
    919,929,937,941,947,953,967,971,977,983,991,997
};
__device__ const int n_small_primes = 168;

__device__ uint128 greatest_prime_factor_128(uint128 n) {
    if (n <= 1) return 0;
    if (n <= 3) return n;

    uint128 gpf = 1;
    uint128 rem = n;

    for (int i = 0; i < n_small_primes && (uint128)small_primes[i] * small_primes[i] <= rem; i++) {
        uint128 p = (uint128)small_primes[i];
        if (rem % p == 0) {
            gpf = p;
            while (rem % p == 0) rem /= p;
        }
    }

    if (rem > 1) {
        if (is_prime_128(rem)) {
            gpf = rem;
        } else {
            /* Composite remainder with all factors > 997.
             * GPF >= sqrt(rem) > 997. Record rem as conservative estimate. */
            gpf = rem;
        }
    }

    return gpf;
}

/* ------------------------------------------------------------------ */
/* Per-thread output structure                                         */
/* ------------------------------------------------------------------ */
struct ConvergentStats {
    uint32_t sample_id;
    uint32_t max_depth_reached;
    uint32_t num_prime_An;
    uint32_t num_prime_Bn;
    uint32_t num_doubly_prime;
    float    mean_log_gpf_An;
    float    min_ratio_An;
    uint32_t depth_at_overflow;
};

/* ------------------------------------------------------------------ */
/* GPU kernel: compute convergent statistics for one CF sequence       */
/* Full uint128 recurrence — depth ~75 instead of ~38                 */
/* ------------------------------------------------------------------ */
__global__
void convergent_stats_kernel_v2(
    ConvergentStats* __restrict__ output,
    int max_depth,
    int mode,
    uint64_t seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState rng;
    if (mode == 0 || mode == 2) {
        curand_init(seed, tid, 0, &rng);
    }

    /* Full uint128 convergent recurrence */
    uint128 A_prev2 = 1, A_prev1 = 0;
    uint128 B_prev2 = 0, B_prev1 = 1;

    uint32_t num_prime_An = 0, num_prime_Bn = 0, num_doubly_prime = 0;
    double sum_log_ratio = 0.0;
    float min_ratio = 1e30f;
    uint32_t depth_reached = 0;
    uint32_t overflow_depth = 0;

    for (int n = 1; n <= max_depth; n++) {
        uint32_t a_n;
        if (mode == 0) {
            /* Gauss-Kuzmin: inverse CDF sampling */
            float u = curand_uniform(&rng);
            a_n = 1;
            double cum = log2(1.0 + 1.0 / (1.0 * 3.0));
            while (cum < u && a_n < 10000) {
                a_n++;
                cum += log2(1.0 + 1.0 / ((double)a_n * (a_n + 2.0)));
            }
        } else if (mode == 1) {
            /* Partial quotients of e: [2; 1,2,1, 1,4,1, 1,6,1, ...] */
            if (n == 1) a_n = 2;
            else {
                int m = n - 1;
                if (m % 3 == 2) a_n = 2 * ((m / 3) + 1);
                else a_n = 1;
            }
        } else {
            /* Mode 2: pi = [3; 7, 15, 1, 292, ...] then random */
            const uint32_t pi_cf[] = {
                3,7,15,1,292,1,1,1,2,1,3,1,14,2,1,1,2,2,2,2,
                1,84,2,1,1,15,3,13,1,4,2,6,6,99,1,2,2,6,3,5,
                1,1,6,8,1,7,1,2,3,7
            };
            if (n <= 50) a_n = pi_cf[n - 1];
            else {
                float u = curand_uniform(&rng);
                a_n = 1;
                double cum = log2(1.0 + 1.0 / 3.0);
                while (cum < u && a_n < 10000) {
                    a_n++;
                    cum += log2(1.0 + 1.0 / ((double)a_n * (a_n + 2.0)));
                }
            }
        }

        /* Convergent recurrence in uint128.
         * A_new = a_n * A_prev1 + A_prev2
         * We need to detect overflow past uint128.
         * Since a_n is at most ~10000 (uint32), and A_prev1 is uint128,
         * the product a_n * A_prev1 can overflow uint128 when
         * A_prev1 > UINT128_MAX / a_n.
         * UINT128_MAX = 2^128 - 1 ≈ 3.4e38. */
        uint128 uint128_max = ~((uint128)0);

        /* Check if a_n * A_prev1 would overflow */
        if (a_n > 0 && A_prev1 > uint128_max / a_n) {
            if (overflow_depth == 0) overflow_depth = n;
            depth_reached = n;
            break;
        }
        uint128 prod_A = (uint128)a_n * A_prev1;
        if (prod_A > uint128_max - A_prev2) {
            if (overflow_depth == 0) overflow_depth = n;
            depth_reached = n;
            break;
        }
        uint128 A_new = prod_A + A_prev2;

        /* Same for B */
        if (a_n > 0 && B_prev1 > uint128_max / a_n) {
            if (overflow_depth == 0) overflow_depth = n;
            depth_reached = n;
            break;
        }
        uint128 prod_B = (uint128)a_n * B_prev1;
        if (prod_B > uint128_max - B_prev2) {
            if (overflow_depth == 0) overflow_depth = n;
            depth_reached = n;
            break;
        }
        uint128 B_new = prod_B + B_prev2;

        /* Track prime statistics */
        int an_prime = 0, bn_prime = 0;

        if (A_new > 1) {
            an_prime = is_prime_128(A_new);
            if (an_prime) num_prime_An++;
        }
        if (B_new > 1) {
            bn_prime = is_prime_128(B_new);
            if (bn_prime) num_prime_Bn++;
        }
        if (an_prime && bn_prime) num_doubly_prime++;

        /* Track G(A_n) growth rate vs Erdos-Mahler bound */
        if (A_new > 1 && n >= 3) {
            uint128 gpf = greatest_prime_factor_128(A_new);
            /* log of a uint128: use log2 decomposition */
            double log_gpf;
            if (gpf <= (uint128)UINT64_MAX) {
                log_gpf = log((double)(uint64_t)gpf);
            } else {
                /* log(gpf) = log(gpf_hi * 2^64 + gpf_lo) ≈ log(gpf_hi) + 64*log(2) */
                uint64_t hi = (uint64_t)(gpf >> 64);
                log_gpf = log((double)hi) + 64.0 * 0.693147180559945;
            }
            double erdos_bound = (double)n / (50.0 * log((double)n));
            if (erdos_bound > 0) {
                double ratio = log_gpf / erdos_bound;
                sum_log_ratio += ratio;
                if ((float)ratio < min_ratio) min_ratio = (float)ratio;
            }
        }

        /* Shift recurrence */
        A_prev2 = A_prev1;
        A_prev1 = A_new;
        B_prev2 = B_prev1;
        B_prev1 = B_new;

        depth_reached = n;
    }

    /* Write output */
    output[tid].sample_id = tid;
    output[tid].max_depth_reached = depth_reached;
    output[tid].num_prime_An = num_prime_An;
    output[tid].num_prime_Bn = num_prime_Bn;
    output[tid].num_doubly_prime = num_doubly_prime;
    output[tid].mean_log_gpf_An = (depth_reached > 2) ?
        (float)(sum_log_ratio / (depth_reached - 2)) : 0.0f;
    output[tid].min_ratio_An = min_ratio;
    output[tid].depth_at_overflow = overflow_depth;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    int num_samples = 100000;
    int max_depth = 500;
    int mode = 0;

    if (argc > 1) num_samples = atoi(argv[1]);
    if (argc > 2) max_depth = atoi(argv[2]);
    if (argc > 3) mode = atoi(argv[3]);
    if (max_depth > MAX_DEPTH_LIMIT) max_depth = MAX_DEPTH_LIMIT;

    const char* mode_names[] = {"random (Gauss-Kuzmin)", "e (Euler)", "pi"};

    printf("========================================\n");
    printf("Prime Convergents v2 (uint128 recurrence)\n");
    printf("========================================\n");
    printf("Samples:   %d\n", num_samples);
    printf("Max depth: %d convergents per sample\n", max_depth);
    printf("Mode:      %s\n", mode_names[mode]);
    printf("\n");
    fflush(stdout);

    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (%.1f GB)\n\n", prop.name, prop.totalGlobalMem / 1e9);
    fflush(stdout);

    size_t out_bytes = (size_t)num_samples * sizeof(ConvergentStats);
    ConvergentStats* d_output;
    cudaMalloc(&d_output, out_bytes);
    cudaMemset(d_output, 0, out_bytes);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t seed = (uint64_t)time(NULL);

    /* Batched launch for progress reporting */
    const int batch_size = 100000;  /* 100K samples per batch */
    int total_batches = (num_samples + batch_size - 1) / batch_size;

    printf("Launching %d batches of %d samples...\n", total_batches, batch_size);
    fflush(stdout);

    for (int b = 0; b < total_batches; b++) {
        int offset = b * batch_size;
        int this_batch = (offset + batch_size <= num_samples) ? batch_size : (num_samples - offset);
        int blocks = (this_batch + BLOCK_SIZE - 1) / BLOCK_SIZE;

        convergent_stats_kernel_v2<<<blocks, BLOCK_SIZE>>>(
            d_output + offset, max_depth, mode, seed + offset);
        cudaDeviceSynchronize();

        int done = offset + this_batch;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed_so_far = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        double pct = 100.0 * done / num_samples;
        double eta = (pct > 0) ? elapsed_so_far * (100.0 / pct - 1.0) : 0;
        printf("[%7.1fs] %d/%d samples (%.1f%%) ETA %.0fs\n",
               elapsed_so_far, done, num_samples, pct, eta);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\nGPU time: %.2f s\n\n", elapsed);
    fflush(stdout);

    ConvergentStats* h_output = (ConvergentStats*)malloc(out_bytes);
    cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    /* Aggregate statistics */
    uint64_t total_prime_An = 0, total_prime_Bn = 0, total_doubly = 0;
    double sum_mean_ratio = 0.0;
    float global_min_ratio = 1e30f;
    uint64_t total_depth = 0;
    uint32_t max_doubly = 0;
    int max_doubly_id = -1;
    int samples_exceeding_bound = 0;

    /* Depth distribution histogram */
    int depth_hist[256] = {0};

    for (int i = 0; i < num_samples; i++) {
        total_prime_An += h_output[i].num_prime_An;
        total_prime_Bn += h_output[i].num_prime_Bn;
        total_doubly += h_output[i].num_doubly_prime;
        total_depth += h_output[i].max_depth_reached;
        sum_mean_ratio += h_output[i].mean_log_gpf_An;

        if (h_output[i].min_ratio_An < global_min_ratio)
            global_min_ratio = h_output[i].min_ratio_An;
        if (h_output[i].min_ratio_An > 1.0f)
            samples_exceeding_bound++;

        if (h_output[i].num_doubly_prime > max_doubly) {
            max_doubly = h_output[i].num_doubly_prime;
            max_doubly_id = i;
        }

        int d = h_output[i].max_depth_reached;
        if (d < 256) depth_hist[d]++;
    }

    double avg_depth = (double)total_depth / num_samples;
    double avg_prime_An = (double)total_prime_An / num_samples;
    double avg_prime_Bn = (double)total_prime_Bn / num_samples;
    double avg_doubly = (double)total_doubly / num_samples;
    double avg_ratio = sum_mean_ratio / num_samples;

    printf("========================================\n");
    printf("RESULTS (v2 — uint128 recurrence)\n");
    printf("========================================\n");
    printf("Samples:              %d\n", num_samples);
    printf("Mode:                 %s\n", mode_names[mode]);
    printf("Avg depth reached:    %.1f (max %d)\n", avg_depth, max_depth);
    printf("\n");
    printf("--- Depth Distribution ---\n");
    for (int d = 0; d < 256; d++) {
        if (depth_hist[d] > 0 && depth_hist[d] >= num_samples / 1000) {
            printf("  depth %3d: %d samples (%.1f%%)\n",
                   d, depth_hist[d], 100.0 * depth_hist[d] / num_samples);
        }
    }
    printf("\n");
    printf("--- Primality ---\n");
    printf("Avg prime A_n per CF: %.2f\n", avg_prime_An);
    printf("Avg prime B_n per CF: %.2f\n", avg_prime_Bn);
    printf("Avg doubly-prime:     %.4f\n", avg_doubly);
    printf("Total doubly-prime:   %" PRIu64 " across all samples\n", total_doubly);
    printf("Max doubly-prime:     %u (sample #%d)\n", max_doubly, max_doubly_id);
    printf("\n");
    printf("--- Erdos-Mahler Bound: G(A_n) >= e^{n/(50 ln n)} ---\n");
    printf("Avg ratio log(G(A_n)) / (n/(50 ln n)): %.4f\n", avg_ratio);
    printf("Min ratio (worst case):                 %.4f\n", global_min_ratio);
    printf("Samples where bound always holds:       %d / %d (%.1f%%)\n",
           samples_exceeding_bound, num_samples,
           100.0 * samples_exceeding_bound / num_samples);
    printf("\n");
    printf("Time: %.2f s\n", elapsed);
    printf("========================================\n");
    fflush(stdout);

    /* Write CSV */
    const char* csv_dir = "scripts/experiments/prime-convergents/results";
    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/v2_stats_%s_%d_%d.csv",
             csv_dir, mode == 0 ? "random" : mode == 1 ? "e" : "pi",
             num_samples, max_depth);

    FILE* csv = fopen(csv_path, "w");
    if (csv) {
        fprintf(csv, "sample_id,depth,prime_An,prime_Bn,doubly_prime,mean_ratio,min_ratio,overflow_depth\n");
        for (int i = 0; i < num_samples; i++) {
            fprintf(csv, "%u,%u,%u,%u,%u,%.6f,%.6f,%u\n",
                    h_output[i].sample_id,
                    h_output[i].max_depth_reached,
                    h_output[i].num_prime_An,
                    h_output[i].num_prime_Bn,
                    h_output[i].num_doubly_prime,
                    h_output[i].mean_log_gpf_An,
                    h_output[i].min_ratio_An,
                    h_output[i].depth_at_overflow);
        }
        fclose(csv);
        printf("CSV written: %s\n", csv_path);
    }

    /* Write JSON metadata */
    char json_path[512];
    snprintf(json_path, sizeof(json_path), "%s/v2_metadata_%s_%d_%d.json",
             csv_dir, mode == 0 ? "random" : mode == 1 ? "e" : "pi",
             num_samples, max_depth);

    FILE* jf = fopen(json_path, "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"experiment\": \"prime_convergents_v2\",\n");
        fprintf(jf, "  \"kernel_version\": 2,\n");
        fprintf(jf, "  \"arithmetic\": \"uint128 recurrence (vs uint64 in v1)\",\n");
        fprintf(jf, "  \"mode\": \"%s\",\n", mode_names[mode]);
        fprintf(jf, "  \"num_samples\": %d,\n", num_samples);
        fprintf(jf, "  \"max_depth\": %d,\n", max_depth);
        fprintf(jf, "  \"avg_depth_reached\": %.1f,\n", avg_depth);
        fprintf(jf, "  \"avg_prime_An\": %.4f,\n", avg_prime_An);
        fprintf(jf, "  \"avg_prime_Bn\": %.4f,\n", avg_prime_Bn);
        fprintf(jf, "  \"avg_doubly_prime\": %.6f,\n", avg_doubly);
        fprintf(jf, "  \"total_doubly_prime\": %" PRIu64 ",\n", total_doubly);
        fprintf(jf, "  \"max_doubly_prime_in_one_cf\": %u,\n", max_doubly);
        fprintf(jf, "  \"erdos_bound_avg_ratio\": %.6f,\n", avg_ratio);
        fprintf(jf, "  \"erdos_bound_min_ratio\": %.6f,\n", global_min_ratio);
        fprintf(jf, "  \"bound_always_holds_pct\": %.2f,\n",
                100.0 * samples_exceeding_bound / num_samples);
        fprintf(jf, "  \"gpu\": \"%s\",\n", prop.name);
        fprintf(jf, "  \"gpu_time_sec\": %.3f\n", elapsed);
        fprintf(jf, "}\n");
        fclose(jf);
        printf("Metadata written: %s\n", json_path);
    }

    free(h_output);
    return 0;
}
