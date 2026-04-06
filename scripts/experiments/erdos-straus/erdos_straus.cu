/*
 * Erdos-Straus Solution Counting Kernel
 *
 * For each prime p, counts all ordered triples (x, y, z) with x <= y <= z
 * satisfying 4/p = 1/x + 1/y + 1/z.
 *
 * Algorithm per prime p:
 *   For x in [ceil(p/4)+1, floor(3p/4)]:
 *     Let num = 4x - p, den = p*x
 *     For y in [ceil(den/num), floor(2*den/num)]:
 *       z_num = den * y
 *       z_den = num * y - den
 *       if z_den > 0 and z_num % z_den == 0: count++
 *
 * Compile:
 *   nvcc -O3 -arch=sm_90 -o erdos_straus erdos_straus.cu -lm
 *
 * Usage:
 *   ./erdos_straus [max_N_millions]    (default: 100 = 10^8)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cinttypes>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Error checking                                                      */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ------------------------------------------------------------------ */
/* CPU prime sieve (simple Eratosthenes, fine for N <= 10^8)           */
/* ------------------------------------------------------------------ */
static std::vector<uint64_t> sieve_primes(uint64_t max_n) {
    // Sieve of Eratosthenes with bit array
    size_t sz = (max_n / 2) + 1;
    std::vector<uint8_t> is_composite(sz, 0);

    for (uint64_t i = 3; i * i <= max_n; i += 2) {
        if (!is_composite[i / 2]) {
            for (uint64_t j = i * i; j <= max_n; j += 2 * i) {
                is_composite[j / 2] = 1;
            }
        }
    }

    std::vector<uint64_t> primes;
    primes.reserve((size_t)(max_n / (log((double)max_n) - 1.1)));
    if (max_n >= 2) primes.push_back(2);
    // Skip p=2 and p=3 for counting since conjecture trivially holds;
    // but we include them for completeness.
    for (uint64_t i = 3; i <= max_n; i += 2) {
        if (!is_composite[i / 2]) {
            primes.push_back(i);
        }
    }
    return primes;
}

/* ------------------------------------------------------------------ */
/* GPU kernel: count solutions for each prime                          */
/* ------------------------------------------------------------------ */
__global__
void count_solutions_kernel(const uint64_t* __restrict__ primes,
                            uint32_t* __restrict__ counts,
                            uint64_t n_primes)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    uint64_t p = primes[idx];

    // Special cases
    if (p == 2) {
        // 4/2 = 2 = 1/1 + 1/y + 1/z? No, 1/x+1/y+1/z <= 3, but = 2.
        // 1/1 + 1/y + 1/z = 2 => 1/y + 1/z = 1 => y=z=2 or y=2,z=inf...
        // Actually: (1,2,2) is the unique solution with x<=y<=z? No:
        // 1/1 + 1/2 + 1/2 = 2. Check: that's exactly 2 = 4/2. Yes.
        // Any others? Need 1/x >= 2/3, so x=1. Then 1/y+1/z=1.
        // y=2,z=2; y=3,z=6 (1/3+1/6=1/2 != 1)... Actually 1/2+1/2=1. Yes.
        // Also: 1/3+1/... hmm. We need 1/y+1/z=1, y<=z.
        // y=2: z=2. That's it (y=3: z=3/2 not int).
        // So f(2) = 1.
        counts[idx] = 1;
        return;
    }
    if (p == 3) {
        // 4/3 = 1/x+1/y+1/z with x<=y<=z
        // x >= ceil(3/4)+1 = 1+1 = 2? Wait: x > p/4 = 0.75, so x >= 1.
        // But also x <= 3p/4 = 2.25, so x in {1, 2}.
        // x=1: 1/y+1/z = 4/3-1 = 1/3. y<=z, y>=3, y<=6.
        //   y=3: z=inf (1/3+1/z=1/3 => z=inf). No.
        //   Actually 1/y+1/z=1/3. y>=ceil(3)=3, y<=floor(6)=6.
        //   y=3: 1/z=0. No.
        //   y=4: 1/z=1/3-1/4=1/12. z=12. Yes.
        //   y=5: 1/z=1/3-1/5=2/15. z=15/2. No.
        //   y=6: 1/z=1/3-1/6=1/6. z=6. Yes.
        // x=2: 1/y+1/z=4/3-1/2=5/6. y<=z, y>=ceil(6/5)=2, y<=floor(12/5)=2.
        //   y=2: 1/z=5/6-1/2=1/3. z=3. Yes. But check x<=y: 2<=2. OK.
        // So f(3)=3.
        // Let the algorithm handle it — but for p < 4 the ceil(p/4)+1 logic
        // might need care. Actually p=3: ceil(3/4)+1 = 1+1 = 2. floor(3*3/4)=2.
        // So x in {2}. That only finds the x=2 solution.
        // We need x=1 too. x > p/4 = 0.75 => x >= 1.
        // The bound should be x from ceil(p/4 + 1) but actually x > p/4.
        // For p=3: p/4 = 0.75, so x >= 1. But our loop starts at ceil(p/4)+1 = 2.
        // Bug: the formula ceil(p/4)+1 is wrong for small p.
        // Actually: x > p/4 means x >= floor(p/4) + 1 = ceil((p+1)/4) when p%4 != 0.
        // For p=3: floor(3/4)+1 = 0+1 = 1. Good.
        // And x <= floor(3p/4) = floor(9/4) = 2.
        // So the loop below should use x_min = p/4 + 1 (integer division gives floor).
        // Let me just let the general algorithm run for all primes.
        // Fall through to general case below.
    }

    uint32_t count = 0;

    // x ranges: x > p/4 and x <= 3p/4
    // x_min = floor(p/4) + 1
    // x_max = floor(3*p/4)  (but if 4 divides 3p exactly, 3p/4 yields x where num=0)
    uint64_t x_min = p / 4 + 1;
    uint64_t x_max = (3 * p) / 4;

    for (uint64_t x = x_min; x <= x_max; x++) {
        uint64_t num = 4 * x - p;   // numerator of remainder r = num / den
        uint64_t den = p * x;       // denominator

        if (num == 0) continue;

        // y ranges: y >= ceil(den/num) and y <= floor(2*den/num)
        // Also y >= x (since x <= y <= z)
        uint64_t y_min_r = (den + num - 1) / num;  // ceil(den/num)
        uint64_t y_min = (y_min_r > x) ? y_min_r : x;
        uint64_t y_max = (2 * den) / num;

        for (uint64_t y = y_min; y <= y_max; y++) {
            uint64_t z_num = den * y;
            uint64_t z_den = num * y - den;

            if (z_den == 0) continue;
            if (z_num % z_den != 0) continue;

            uint64_t z = z_num / z_den;
            if (z >= y) {
                count++;
            }
        }
    }

    counts[idx] = count;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const char* comma_fmt(uint64_t n) {
    static char buf[64];
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "%" PRIu64, n);
    int len = (int)strlen(tmp);
    int commas = (len - 1) / 3;
    int out_len = len + commas;
    buf[out_len] = '\0';
    int j = out_len - 1;
    for (int i = len - 1, c = 0; i >= 0; i--, c++) {
        if (c > 0 && c % 3 == 0) buf[j--] = ',';
        buf[j--] = tmp[i];
    }
    return buf;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    uint64_t max_millions = 100;
    if (argc > 1) {
        max_millions = (uint64_t)atoll(argv[1]);
        if (max_millions == 0) max_millions = 100;
    }
    uint64_t max_N = max_millions * 1000000ULL;

    printf("Erdos-Straus solution counting: f(p) for all primes p <= %s\n",
           comma_fmt(max_N));
    printf("=====================================================\n\n");

    /* ---- Device info ---- */
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (%.1f GB, SM %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / 1e9,
           prop.major, prop.minor);

    /* ---- Sieve primes ---- */
    printf("Sieving primes up to %s ... ", comma_fmt(max_N));
    fflush(stdout);
    double t0 = now_sec();
    std::vector<uint64_t> primes = sieve_primes(max_N);
    double t_sieve = now_sec() - t0;
    uint64_t n_primes = primes.size();
    printf("done. Found %s primes in %.2f s\n\n", comma_fmt(n_primes), t_sieve);

    /* ---- Allocate GPU memory ---- */
    uint64_t* d_primes = nullptr;
    uint32_t* d_counts = nullptr;
    size_t primes_bytes = n_primes * sizeof(uint64_t);
    size_t counts_bytes = n_primes * sizeof(uint32_t);

    printf("GPU memory: %.1f MB for primes + %.1f MB for counts\n\n",
           primes_bytes / 1e6, counts_bytes / 1e6);

    CUDA_CHECK(cudaMalloc(&d_primes, primes_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, counts_bytes));
    CUDA_CHECK(cudaMemcpy(d_primes, primes.data(), primes_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counts, 0, counts_bytes));

    /* ---- Launch kernel in batches with progress reporting ---- */
    const int threads_per_block = 256;
    const uint64_t batch_size = 50000;  // ~50K primes per batch for responsive progress
    uint64_t n_batches = (n_primes + batch_size - 1) / batch_size;

    printf("Launching kernel (%d threads/block, %" PRIu64 " batches of %" PRIu64 ") ...\n",
           threads_per_block, n_batches, batch_size);
    fflush(stdout);

    double t_gpu_start = now_sec();
    double last_report = t_gpu_start;
    uint64_t batch_num = 0;

    // Temporary host buffer for incremental min/max tracking
    std::vector<uint32_t> batch_counts;

    for (uint64_t offset = 0; offset < n_primes; offset += batch_size) {
        uint64_t this_batch = std::min(batch_size, n_primes - offset);
        int blocks = (int)((this_batch + threads_per_block - 1) / threads_per_block);

        count_solutions_kernel<<<blocks, threads_per_block>>>(
            d_primes + offset, d_counts + offset, this_batch);

        CUDA_CHECK(cudaDeviceSynchronize());

        batch_num++;
        uint64_t primes_done = offset + this_batch;
        double now = now_sec();
        double elapsed = now - t_gpu_start;

        // Report progress every batch or every 30 seconds, whichever is more frequent
        if (now - last_report >= 30.0 || batch_num == 1 || batch_num == n_batches ||
            (batch_num % 10 == 0)) {

            // Read back this batch to get min/max f values
            batch_counts.resize(this_batch);
            CUDA_CHECK(cudaMemcpy(batch_counts.data(), d_counts + offset,
                                  this_batch * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            uint32_t b_min = UINT32_MAX, b_max = 0;
            for (uint64_t i = 0; i < this_batch; i++) {
                if (batch_counts[i] < b_min) b_min = batch_counts[i];
                if (batch_counts[i] > b_max) b_max = batch_counts[i];
            }

            double pct = 100.0 * primes_done / n_primes;
            double eta = (pct > 0.0) ? elapsed * (100.0 / pct - 1.0) : 0.0;
            printf("[%.1fs] batch %" PRIu64 "/%" PRIu64 " (%.1f%%) %s primes done, "
                   "min_f=%u, max_f=%u, ETA %.0fs\n",
                   elapsed, batch_num, n_batches, pct,
                   comma_fmt(primes_done), b_min, b_max, eta);
            fflush(stdout);
            last_report = now;
        }
    }

    double t_gpu = now_sec() - t_gpu_start;
    printf("\nGPU time: %.2f s (%.0f primes/sec)\n\n",
           t_gpu, n_primes / t_gpu);
    fflush(stdout);

    /* ---- Copy results back ---- */
    std::vector<uint32_t> counts(n_primes);
    CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, counts_bytes,
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_primes));
    CUDA_CHECK(cudaFree(d_counts));

    /* ---- Compute statistics ---- */
    printf("Computing statistics ...\n\n");

    // Overall stats
    uint32_t global_min = UINT32_MAX, global_max = 0;
    uint64_t global_sum = 0;
    uint64_t min_prime = 0, max_prime = 0;
    uint64_t count_fp_1 = 0;  // "barely solvable"
    uint64_t count_fp_0 = 0;  // should be 0 if conjecture holds

    // Distribution: f(p) -> how many primes have that count
    std::vector<uint64_t> fp_distribution(1024, 0);
    uint32_t max_fp_for_dist = 0;

    // Per-decade stats
    struct DecadeStats {
        uint64_t decade_limit;
        uint64_t n_primes;
        uint64_t sum_fp;
        uint32_t min_fp;
        uint32_t max_fp;
        uint64_t min_prime;
        uint64_t max_prime;
    };

    int n_decades = (int)ceil(log10((double)max_N));
    std::vector<DecadeStats> decades(n_decades + 1);
    for (int d = 0; d <= n_decades; d++) {
        decades[d].decade_limit = (d == 0) ? 10 : (uint64_t)pow(10.0, d);
        decades[d].n_primes = 0;
        decades[d].sum_fp = 0;
        decades[d].min_fp = UINT32_MAX;
        decades[d].max_fp = 0;
        decades[d].min_prime = 0;
        decades[d].max_prime = 0;
    }

    for (uint64_t i = 0; i < n_primes; i++) {
        uint64_t p = primes[i];
        uint32_t fp = counts[i];

        global_sum += fp;
        if (fp < global_min) { global_min = fp; min_prime = p; }
        if (fp > global_max) { global_max = fp; max_prime = p; }
        if (fp == 1) count_fp_1++;
        if (fp == 0) count_fp_0++;

        if (fp < fp_distribution.size()) {
            fp_distribution[fp]++;
            if (fp > max_fp_for_dist) max_fp_for_dist = fp;
        }

        // Find decade
        int d = (p < 10) ? 1 : (int)floor(log10((double)p)) + 1;
        if (d <= n_decades) {
            decades[d].n_primes++;
            decades[d].sum_fp += fp;
            if (fp < decades[d].min_fp) { decades[d].min_fp = fp; decades[d].min_prime = p; }
            if (fp > decades[d].max_fp) { decades[d].max_fp = fp; decades[d].max_prime = p; }
        }
    }

    /* ---- Print summary ---- */
    printf("=== SUMMARY ===\n");
    printf("Primes processed:    %s\n", comma_fmt(n_primes));
    printf("Range:               [2, %s]\n", comma_fmt(primes.back()));
    printf("Global min f(p):     %u  (p = %s)\n", global_min, comma_fmt(min_prime));
    printf("Global max f(p):     %u  (p = %s)\n", global_max, comma_fmt(max_prime));
    printf("Mean f(p):           %.4f\n", (double)global_sum / n_primes);
    printf("Primes with f(p)=0:  %s%s\n", comma_fmt(count_fp_0),
           count_fp_0 > 0 ? " *** COUNTEREXAMPLE TO CONJECTURE ***" : " (conjecture holds)");
    printf("Primes with f(p)=1:  %s (barely solvable)\n", comma_fmt(count_fp_1));
    printf("\n");

    /* ---- Per-decade table ---- */
    printf("=== PER-DECADE STATISTICS ===\n");
    printf("%-12s %12s %8s %8s %10s %14s %14s\n",
           "Decade", "# Primes", "Min f", "Max f", "Mean f", "MinPrime", "MaxPrime");
    printf("%-12s %12s %8s %8s %10s %14s %14s\n",
           "------", "--------", "-----", "-----", "------", "--------", "--------");
    for (int d = 1; d <= n_decades; d++) {
        if (decades[d].n_primes == 0) continue;
        char label[32];
        snprintf(label, sizeof(label), "10^%d", d);
        printf("%-12s %12s %8u %8u %10.2f %14s",
               label,
               comma_fmt(decades[d].n_primes),
               decades[d].min_fp,
               decades[d].max_fp,
               (double)decades[d].sum_fp / decades[d].n_primes,
               comma_fmt(decades[d].min_prime));
        printf(" %14s\n", comma_fmt(decades[d].max_prime));
    }
    printf("\n");

    /* ---- Distribution table ---- */
    printf("=== f(p) DISTRIBUTION (top 30) ===\n");
    printf("%-8s %12s %10s\n", "f(p)", "# Primes", "%%");
    printf("%-8s %12s %10s\n", "----", "--------", "---");
    int shown = 0;
    for (uint32_t f = 0; f <= max_fp_for_dist && shown < 30; f++) {
        if (fp_distribution[f] > 0) {
            printf("%-8u %12s %9.4f%%\n", f, comma_fmt(fp_distribution[f]),
                   100.0 * fp_distribution[f] / n_primes);
            shown++;
        }
    }
    printf("\n");

    /* ---- Write CSV ---- */
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path),
             "scripts/experiments/erdos-straus/results/erdos_straus_1e%d.csv",
             (int)round(log10((double)max_N)));
    printf("Writing CSV to %s ... ", csv_path);
    fflush(stdout);
    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Error: cannot open %s for writing\n", csv_path);
        return 1;
    }
    fprintf(csv, "prime,f_count\n");
    for (uint64_t i = 0; i < n_primes; i++) {
        fprintf(csv, "%" PRIu64 ",%u\n", primes[i], counts[i]);
    }
    fclose(csv);
    printf("done.\n");

    /* ---- Write JSON metadata ---- */
    const char* json_path = "scripts/experiments/erdos-straus/results/metadata.json";
    printf("Writing metadata to %s ... ", json_path);
    fflush(stdout);
    FILE* jf = fopen(json_path, "w");
    if (!jf) {
        fprintf(stderr, "Error: cannot open %s for writing\n", json_path);
        return 1;
    }
    fprintf(jf, "{\n");
    fprintf(jf, "  \"experiment\": \"erdos_straus_solution_counting\",\n");
    fprintf(jf, "  \"max_N\": %" PRIu64 ",\n", max_N);
    fprintf(jf, "  \"n_primes\": %" PRIu64 ",\n", n_primes);
    fprintf(jf, "  \"largest_prime\": %" PRIu64 ",\n", primes.back());
    fprintf(jf, "  \"sieve_time_sec\": %.3f,\n", t_sieve);
    fprintf(jf, "  \"gpu_time_sec\": %.3f,\n", t_gpu);
    fprintf(jf, "  \"total_time_sec\": %.3f,\n", now_sec() - t0);
    fprintf(jf, "  \"gpu\": \"%s\",\n", prop.name);
    fprintf(jf, "  \"global_min_fp\": %u,\n", global_min);
    fprintf(jf, "  \"global_min_prime\": %" PRIu64 ",\n", min_prime);
    fprintf(jf, "  \"global_max_fp\": %u,\n", global_max);
    fprintf(jf, "  \"global_max_prime\": %" PRIu64 ",\n", max_prime);
    fprintf(jf, "  \"mean_fp\": %.6f,\n", (double)global_sum / n_primes);
    fprintf(jf, "  \"count_fp_0\": %" PRIu64 ",\n", count_fp_0);
    fprintf(jf, "  \"count_fp_1\": %" PRIu64 ",\n", count_fp_1);
    fprintf(jf, "  \"conjecture_holds\": %s\n", count_fp_0 == 0 ? "true" : "false");
    fprintf(jf, "}\n");
    fclose(jf);
    printf("done.\n\n");

    double total_time = now_sec() - t0;

    /* ---- RESULTS summary block ---- */
    printf("========================================================\n");
    printf("RESULTS: Erdos-Straus Solution Counting\n");
    printf("========================================================\n");
    printf("Range:               primes p <= %s\n", comma_fmt(max_N));
    printf("Primes processed:    %s\n", comma_fmt(n_primes));
    printf("Conjecture holds:    %s\n", count_fp_0 == 0 ? "YES (all f(p) >= 1)" : "NO — COUNTEREXAMPLE FOUND");
    if (count_fp_0 > 0) {
        printf("*** COUNTEREXAMPLES:   %s primes with f(p)=0 ***\n", comma_fmt(count_fp_0));
    }
    printf("Global min f(p):     %u  (at p = %s)\n", global_min, comma_fmt(min_prime));
    printf("Global max f(p):     %u  (at p = %s)\n", global_max, comma_fmt(max_prime));
    printf("Mean f(p):           %.4f\n", (double)global_sum / n_primes);
    printf("Barely solvable:     %s primes with f(p)=1\n", comma_fmt(count_fp_1));
    printf("GPU:                 %s\n", prop.name);
    printf("Sieve time:          %.2f s\n", t_sieve);
    printf("GPU time:            %.2f s (%.0f primes/sec)\n", t_gpu, n_primes / t_gpu);
    printf("Total wall time:     %.2f s\n", total_time);
    printf("CSV output:          %s\n", csv_path);
    printf("========================================================\n");
    fflush(stdout);

    return 0;
}
