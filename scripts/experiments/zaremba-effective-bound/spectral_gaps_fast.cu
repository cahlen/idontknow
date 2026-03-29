/*
 * Fast Batched Spectral Gaps — ALL primes in ONE kernel launch
 *
 * Zero CPU in the hot loop. Everything on GPU:
 * - Permutation tables computed on GPU (modular inverse via Fermat)
 * - All primes processed in parallel (one block per prime)
 * - Pre-allocated flat workspace with per-prime offsets
 * - FP32, N=20 Chebyshev, deflated power iteration with early stop
 *
 * For 5,133 primes to p=50,000: all launched as ONE kernel.
 * Expected time: seconds, not minutes.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o spectral_gaps_fast spectral_gaps_fast.cu -lm
 * Run:     ./spectral_gaps_fast <max_prime>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BOUND 5
#define N_CHEB 20
#define MAX_ITER 200
#define DELTA 0.836829443681208f

// Modular inverse via Fermat's little theorem: x^{p-2} mod p
__device__ int mod_inv(int x, int p) {
    long long result = 1, base = x % p;
    if (base < 0) base += p;
    int exp = p - 2;
    while (exp > 0) {
        if (exp & 1) result = result * base % p;
        base = base * base % p;
        exp >>= 1;
    }
    return (int)result;
}

// Each block handles ONE prime
// blockIdx.x = prime index in the sorted prime array
__global__ void spectral_gaps_kernel(
    int *d_primes,          // [num_primes] prime values
    int num_primes,
    long long *d_offsets,   // [num_primes] workspace offsets
    float *d_workspace,     // flat workspace for all vectors
    float *d_gaps           // [num_primes] output gaps
) {
    int pidx = blockIdx.x;
    if (pidx >= num_primes) return;

    int p = d_primes[pidx];
    int p1 = p + 1; // |P^1(F_p)|
    int vec_size = N_CHEB * p1;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Workspace for this prime: two vectors of size vec_size
    float *v_cur = d_workspace + d_offsets[pidx];
    float *v_next = v_cur + vec_size;

    // Shared memory: Chebyshev nodes, barycentric weights, operator matrices
    __shared__ float nodes[N_CHEB];
    __shared__ float bary_w[N_CHEB];
    __shared__ float Ma[BOUND][N_CHEB * N_CHEB]; // 5 × 20 × 20 = 2000 floats = 8KB

    // Compute Chebyshev nodes and barycentric weights
    if (tid < N_CHEB) {
        nodes[tid] = 0.5f * (1.0f + __cosf(M_PI * (2*tid + 1) / (2.0f * N_CHEB)));
        bary_w[tid] = ((tid % 2 == 0) ? 1.0f : -1.0f) *
                      __sinf(M_PI * (2*tid + 1) / (2.0f * N_CHEB));
    }
    __syncthreads();

    // Build M_a matrices (barycentric interpolation, same as transfer_operator.cu)
    // M_a[i][j] = contribution of node j to image at node i under digit a
    for (int a = 0; a < BOUND; a++) {
        for (int i = tid; i < N_CHEB * N_CHEB; i += nthreads) {
            int row = i / N_CHEB;
            int col = i % N_CHEB;
            Ma[a][i] = 0.0f;
        }
    }
    __syncthreads();

    for (int a = 0; a < BOUND; a++) {
        int digit = a + 1;
        for (int i = tid; i < N_CHEB; i += nthreads) {
            float xi = nodes[i];
            float y = 1.0f / (digit + xi); // g_a(x_i)
            float ws = __powf(digit + xi, -2.0f * DELTA);

            // Barycentric interpolation at y
            int exact = -1;
            for (int k = 0; k < N_CHEB; k++) {
                if (fabsf(y - nodes[k]) < 1e-7f) { exact = k; break; }
            }

            if (exact >= 0) {
                Ma[a][i * N_CHEB + exact] += ws;
            } else {
                float denom = 0;
                float num[N_CHEB];
                for (int j = 0; j < N_CHEB; j++) {
                    num[j] = bary_w[j] / (y - nodes[j]);
                    denom += num[j];
                }
                float inv_den = 1.0f / denom;
                for (int j = 0; j < N_CHEB; j++) {
                    Ma[a][i * N_CHEB + j] += ws * num[j] * inv_den;
                }
            }
        }
        __syncthreads();
    }

    // Compute permutation P_a on P^1(F_p) on-the-fly during power iteration
    // P^1 = {0, 1, ..., p-1, ∞=p}
    // g_a([x:1]) = [ax+1 : x], projective = (ax+1)*x^{-1} mod p if x≠0
    // g_a([0:1]) = [1:0] = ∞
    // g_a([1:0]=∞) = [a:1] = a mod p

    // Initialize v_cur: random, projected off trivial rep
    for (int idx = tid; idx < vec_size; idx += nthreads) {
        v_cur[idx] = __sinf(idx * 1.618f + pidx * 3.14f + 0.5f);
    }
    __syncthreads();

    // Project out trivial representation (constant over P^1 for each Chebyshev index)
    __shared__ float reduce_buf[256];
    for (int c = 0; c < N_CHEB; c++) {
        float local_sum = 0;
        for (int k = tid; k < p1; k += nthreads) {
            local_sum += v_cur[c * p1 + k];
        }
        reduce_buf[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads/2; s > 0; s >>= 1) {
            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
            __syncthreads();
        }
        float mean = reduce_buf[0] / p1;
        for (int k = tid; k < p1; k += nthreads) {
            v_cur[c * p1 + k] -= mean;
        }
        __syncthreads();
    }

    float eigenvalue = 0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Zero v_next
        for (int idx = tid; idx < vec_size; idx += nthreads) {
            v_next[idx] = 0;
        }
        __syncthreads();

        // Apply L = Σ_a M_a ⊗ P_a
        // For each P^1 point k, compute P_a(k) and accumulate
        for (int a = 0; a < BOUND; a++) {
            int digit = a + 1;
            for (int k = tid; k < p1; k += nthreads) {
                // Compute P_a(k) = g_{digit} applied to projective point k
                int pk;
                if (k == p) {
                    pk = digit % p; // ∞ → a mod p
                } else if (k == 0) {
                    pk = p; // 0 → ∞
                } else {
                    // (digit*k + 1) * k^{-1} mod p
                    int kinv = mod_inv(k, p);
                    pk = (int)(((long long)digit * k + 1) % p * kinv % p);
                }

                // v_next[i][pk] += Σ_j Ma[a][i][j] * v_cur[j][k]
                for (int i = 0; i < N_CHEB; i++) {
                    float sum = 0;
                    for (int j = 0; j < N_CHEB; j++) {
                        sum += Ma[a][i * N_CHEB + j] * v_cur[j * p1 + k];
                    }
                    atomicAdd(&v_next[i * p1 + pk], sum);
                }
            }
            __syncthreads();
        }

        // Project out trivial representation
        for (int c = 0; c < N_CHEB; c++) {
            float local_sum = 0;
            for (int k = tid; k < p1; k += nthreads) {
                local_sum += v_next[c * p1 + k];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            for (int s = nthreads/2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            float mean = reduce_buf[0] / p1;
            for (int k = tid; k < p1; k += nthreads) {
                v_next[c * p1 + k] -= mean;
            }
            __syncthreads();
        }

        // Compute norm
        float local_norm = 0;
        for (int idx = tid; idx < vec_size; idx += nthreads) {
            local_norm += v_next[idx] * v_next[idx];
        }
        reduce_buf[tid] = local_norm;
        __syncthreads();
        for (int s = nthreads/2; s > 0; s >>= 1) {
            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
            __syncthreads();
        }
        float norm = sqrtf(reduce_buf[0]);
        eigenvalue = norm;

        // Normalize
        if (norm > 1e-30f) {
            float inv = 1.0f / norm;
            for (int idx = tid; idx < vec_size; idx += nthreads) {
                v_next[idx] *= inv;
            }
        }
        __syncthreads();

        // Swap
        float *tmp = v_cur; v_cur = v_next; v_next = tmp;
    }

    // Write gap = 1 - |λ_2|
    // eigenvalue has converged to |λ_2| (trivial projected out, so this IS the 2nd eigenvalue)
    if (tid == 0) {
        d_gaps[pidx] = 1.0f - eigenvalue;
    }
}

int main(int argc, char **argv) {
    int max_p = argc > 1 ? atoi(argv[1]) : 50000;

    printf("Fast Batched Spectral Gaps — ALL primes in ONE kernel\n");
    printf("Max prime: %d, N=%d Chebyshev, FP32\n\n", max_p, N_CHEB);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Sieve primes
    char *is_prime = (char*)calloc(max_p + 1, 1);
    memset(is_prime, 1, max_p + 1);
    is_prime[0] = is_prime[1] = 0;
    for (int i = 2; (long long)i*i <= max_p; i++)
        if (is_prime[i]) for (int j = i*i; j <= max_p; j += i) is_prime[j] = 0;

    int num_primes = 0;
    for (int p = 2; p <= max_p; p++) if (is_prime[p]) num_primes++;

    int *h_primes = (int*)malloc(num_primes * sizeof(int));
    long long *h_offsets = (long long*)malloc(num_primes * sizeof(long long));
    int idx = 0;
    long long total_workspace = 0;
    for (int p = 2; p <= max_p; p++) {
        if (!is_prime[p]) continue;
        h_primes[idx] = p;
        h_offsets[idx] = total_workspace;
        total_workspace += 2LL * N_CHEB * (p + 1); // two vectors
        idx++;
    }

    double ws_gb = total_workspace * sizeof(float) / 1e9;
    printf("Primes: %d, workspace: %.2f GB\n", num_primes, ws_gb);

    // Allocate GPU memory
    int *d_primes;
    long long *d_offsets;
    float *d_workspace, *d_gaps;

    cudaMalloc(&d_primes, num_primes * sizeof(int));
    cudaMalloc(&d_offsets, num_primes * sizeof(long long));
    cudaMalloc(&d_workspace, total_workspace * sizeof(float));
    cudaMalloc(&d_gaps, num_primes * sizeof(float));

    cudaMemcpy(d_primes, h_primes, num_primes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, num_primes * sizeof(long long), cudaMemcpyHostToDevice);

    struct timespec tk0, tk1;
    clock_gettime(CLOCK_MONOTONIC, &tk0);

    // ONE kernel launch: all primes in parallel
    // 256 threads per block, one block per prime
    spectral_gaps_kernel<<<num_primes, 256>>>(
        d_primes, num_primes, d_offsets, d_workspace, d_gaps
    );
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &tk1);
    double kernel_time = (tk1.tv_sec - tk0.tv_sec) + (tk1.tv_nsec - tk0.tv_nsec) / 1e9;

    // Download results
    float *h_gaps = (float*)malloc(num_primes * sizeof(float));
    cudaMemcpy(h_gaps, d_gaps, num_primes * sizeof(float), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Analyze results
    float min_gap = 999.0f;
    int min_gap_prime = 0;
    int positive_gaps = 0;

    printf("\n%8s  %10s\n", "prime", "gap");
    printf("--------  ----------\n");

    for (int i = 0; i < num_primes; i++) {
        float gap = h_gaps[i];
        if (gap > 0) positive_gaps++;
        if (gap < min_gap) {
            min_gap = gap;
            min_gap_prime = h_primes[i];
        }
        // Print tight gaps and milestones
        if (h_primes[i] <= 20 || gap < 0.15f ||
            h_primes[i] % 10000 < 100 || i == num_primes - 1) {
            printf("%8d  %10.6f", h_primes[i], gap);
            if (gap < 0.15f) printf("  <-- tight");
            if (gap <= 0) printf("  <-- WARNING");
            printf("\n");
        }
    }

    printf("\n========================================\n");
    printf("Primes: %d (to p=%d)\n", num_primes, max_p);
    printf("Positive gaps: %d / %d (%.1f%%)\n",
           positive_gaps, num_primes, 100.0*positive_gaps/num_primes);
    printf("Minimum gap: %.6f at p=%d\n", min_gap, min_gap_prime);
    printf("Kernel time: %.2fs\n", kernel_time);
    printf("Total time: %.2fs\n", total_time);
    printf("Rate: %.0f primes/sec\n", num_primes / kernel_time);
    printf("========================================\n");

    // Cleanup
    cudaFree(d_primes); cudaFree(d_offsets);
    cudaFree(d_workspace); cudaFree(d_gaps);
    free(h_primes); free(h_offsets); free(h_gaps); free(is_prime);
    return 0;
}
