/*
 * Fast Spectral Gap Computation for ALL Primes to P_MAX
 *
 * For each prime p, compute the spectral gap of the congruence
 * transfer operator L_{δ,p} restricted to non-trivial representations.
 *
 * Key optimizations vs. the original transfer_operator.cu:
 * - FP32 (not FP64) — sufficient for gap ≥ 0.1
 * - N=20 Chebyshev nodes (not 40) — gap lower bound doesn't need high N
 * - Only PRIMES (property τ for composites follows from prime factors)
 * - Deflated power iteration with early termination
 * - Process one prime per GPU thread block (small p) or one per GPU (large p)
 *
 * The transfer operator for prime p acts on L^2([0,1]) ⊗ C^{p+1}
 * via implicit Kronecker: L = Σ_{a=1}^5 M_a ⊗ P_a
 * where M_a is the Chebyshev-discretized operator for digit a,
 * and P_a is the permutation on P^1(F_p) induced by g_a.
 *
 * Spectral gap = 1 - |λ_2/λ_1| where λ_1 = spectral radius ≈ 1
 * (evaluated at s = δ = 0.836829443681208).
 *
 * Compile: nvcc -O3 -arch=sm_100a -o spectral_gaps_primes spectral_gaps_primes.cu -lm
 * Run:     ./spectral_gaps_primes <max_prime>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BOUND 5
#define N_CHEB 20
#define MAX_POWER_ITER 200
#define EARLY_STOP_ITER 50
#define EARLY_STOP_GAP 0.10f  // stop if gap clearly > 0.10
#define DELTA 0.836829443681208

// Chebyshev nodes on [0,1]: x_j = (1 + cos(π(2j+1)/(2N))) / 2
__device__ __host__ float cheb_node(int j, int N) {
    return 0.5f * (1.0f + cosf(M_PI * (2*j + 1) / (2.0f * N)));
}

// One prime's spectral gap computation — runs on one thread block
// orbits: precomputed P^1(F_p) permutation tables for g_1,...,g_5
// p1_size = p + 1 (size of P^1(F_p))
__global__ void compute_gap_kernel(
    int *d_perm,        // [5][p1_size] permutation tables (flattened)
    int p, int p1_size,
    float *d_result,    // output: spectral gap
    float *d_workspace  // [2 * N_CHEB * p1_size] for vectors
) {
    // Shared Chebyshev data
    __shared__ float nodes[N_CHEB];
    __shared__ float Ma[BOUND][N_CHEB][N_CHEB]; // operator matrices for each digit

    int tid = threadIdx.x;

    // Compute nodes
    if (tid < N_CHEB) {
        nodes[tid] = cheb_node(tid, N_CHEB);
    }
    __syncthreads();

    // Precompute M_a[i][j] = (a + x_j)^{-2δ} * T_i(g_a(x_j)) * w_j
    // where g_a(x) = 1/(a+x), T_i are Chebyshev basis, w_j are bary weights
    // For power iteration, we just need the matrix-vector product.
    // M_a[i][j] = (a + x_j)^{-2δ} * cos(i * arccos(2*g_a(x_j) - 1)) * (2/N or 1/N)
    if (tid < BOUND) {
        int a = tid + 1;
        for (int i = 0; i < N_CHEB; i++) {
            for (int j = 0; j < N_CHEB; j++) {
                float xj = nodes[j];
                float ga = 1.0f / (a + xj);
                float weight = powf(a + xj, -2.0f * (float)DELTA);
                // Chebyshev basis on [0,1]: cos(i * arccos(2*ga - 1))
                float ti = cosf(i * acosf(fmaxf(-1.0f, fminf(1.0f, 2.0f*ga - 1.0f))));
                float wj = (j == 0 || j == N_CHEB-1) ? 1.0f/N_CHEB : 2.0f/N_CHEB;
                Ma[tid][i][j] = weight * ti * wj;
            }
        }
    }
    __syncthreads();

    // Power iteration on the FULL operator L = Σ_a M_a ⊗ P_a
    // Vector v has size N_CHEB * p1_size
    int vec_size = N_CHEB * p1_size;
    float *v_cur = d_workspace;
    float *v_next = d_workspace + vec_size;

    // Initialize with random-ish vector (orthogonal to trivial rep)
    // Trivial rep: same function on every P^1 point
    // Non-trivial: subtract the mean over P^1 points
    for (int idx = tid; idx < vec_size; idx += blockDim.x) {
        int cheb_idx = idx / p1_size;
        int p1_idx = idx % p1_size;
        // Use a simple deterministic "random" init
        v_cur[idx] = sinf(idx * 1.618f + 0.5f);
    }
    __syncthreads();

    // Project out trivial representation: subtract mean over P^1 for each Chebyshev index
    for (int c = 0; c < N_CHEB; c++) {
        float mean = 0;
        for (int k = tid; k < p1_size; k += blockDim.x) {
            mean += v_cur[c * p1_size + k];
        }
        // Block reduce
        __shared__ float smem[256];
        smem[tid] = mean;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        mean = smem[0] / p1_size;
        for (int k = tid; k < p1_size; k += blockDim.x) {
            v_cur[c * p1_size + k] -= mean;
        }
        __syncthreads();
    }

    float prev_norm = 0, cur_norm = 0;

    for (int iter = 0; iter < MAX_POWER_ITER; iter++) {
        // Zero next vector
        for (int idx = tid; idx < vec_size; idx += blockDim.x) {
            v_next[idx] = 0;
        }
        __syncthreads();

        // Apply L = Σ_a M_a ⊗ P_a
        for (int a = 0; a < BOUND; a++) {
            // For each P^1 point k, P_a maps k -> perm[a][k]
            // v_next[i][P_a(k)] += Σ_j M_a[i][j] * v_cur[j][k]
            for (int k = tid; k < p1_size; k += blockDim.x) {
                int pk = d_perm[a * p1_size + k];
                for (int i = 0; i < N_CHEB; i++) {
                    float sum = 0;
                    for (int j = 0; j < N_CHEB; j++) {
                        sum += Ma[a][i][j] * v_cur[j * p1_size + k];
                    }
                    atomicAdd(&v_next[i * p1_size + pk], sum);
                }
            }
            __syncthreads();
        }

        // Project out trivial representation
        for (int c = 0; c < N_CHEB; c++) {
            float mean = 0;
            for (int k = tid; k < p1_size; k += blockDim.x) {
                mean += v_next[c * p1_size + k];
            }
            __shared__ float smem2[256];
            smem2[tid] = mean;
            __syncthreads();
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) smem2[tid] += smem2[tid + s];
                __syncthreads();
            }
            mean = smem2[0] / p1_size;
            for (int k = tid; k < p1_size; k += blockDim.x) {
                v_next[c * p1_size + k] -= mean;
            }
            __syncthreads();
        }

        // Compute norm
        float local_norm = 0;
        for (int idx = tid; idx < vec_size; idx += blockDim.x) {
            local_norm += v_next[idx] * v_next[idx];
        }
        __shared__ float norm_smem[256];
        norm_smem[tid] = local_norm;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) norm_smem[tid] += norm_smem[tid + s];
            __syncthreads();
        }
        cur_norm = sqrtf(norm_smem[0]);

        // Normalize
        if (cur_norm > 1e-30f) {
            float inv = 1.0f / cur_norm;
            for (int idx = tid; idx < vec_size; idx += blockDim.x) {
                v_next[idx] *= inv;
            }
        }
        __syncthreads();

        // Swap
        float *tmp = v_cur; v_cur = v_next; v_next = tmp;

        // Early termination: if eigenvalue ratio is stable and gap > threshold
        if (iter >= EARLY_STOP_ITER && prev_norm > 0) {
            float ratio = cur_norm / prev_norm;
            // ratio converges to |λ_2| (since we deflated λ_1)
            // Actually ratio converges to |λ_2/λ_1| but λ_1 was projected out
            // So ratio → |λ_2| where λ_2 is the second eigenvalue of L_δ
            if (ratio < 1.0f - EARLY_STOP_GAP) {
                if (tid == 0) *d_result = 1.0f - ratio;
                return;
            }
        }
        prev_norm = cur_norm;
    }

    // Final gap estimate
    if (tid == 0) {
        // The eigenvalue ratio from last iterations
        *d_result = (prev_norm > 0) ? 1.0f - cur_norm : -1.0f;
    }
}

// Compute P^1(F_p) permutation tables on CPU
// P^1(F_p) = {0, 1, ..., p-1, ∞} where ∞ is index p
// g_a acts as: x → (a*x + 1)/(x) = a + 1/x on P^1
// More precisely: g_a = [[a,1],[1,0]], so g_a(x) = (a*x+1)/x for x ≠ 0,
// g_a(0) = ∞/0 = ∞... wait, g_a acts on column vectors:
// g_a * [x,1]^T = [ax+1, x]^T, projective point = (ax+1)/x = a + 1/x
// g_a * [1,0]^T (= ∞) = [a,1]^T = a
// g_a * [0,1]^T (= 0) = [1,0]^T = ∞
void compute_permutations(int p, int *perm) {
    // P^1 indices: 0..p-1 are finite, p is ∞
    int p1 = p + 1;
    for (int a = 1; a <= BOUND; a++) {
        for (int x = 0; x < p; x++) {
            // g_a([x,1]) = [ax+1, x]
            // If x = 0: result = [1, 0] = ∞
            if (x == 0) {
                perm[(a-1)*p1 + x] = p; // maps to ∞
            } else {
                // Projective: (ax+1)/x mod p
                // = (a + x^{-1}) mod p
                // Need modular inverse of x
                long long inv_x = 1;
                long long base = x, exp = p - 2, mod = p;
                while (exp > 0) {
                    if (exp & 1) inv_x = inv_x * base % mod;
                    base = base * base % mod;
                    exp >>= 1;
                }
                int result = (int)(((long long)a * x + 1) % p * inv_x % p);
                perm[(a-1)*p1 + x] = result;
            }
        }
        // g_a(∞) = [a,1] = a
        perm[(a-1)*p1 + p] = a % p;
    }
}

int main(int argc, char **argv) {
    int max_p = argc > 1 ? atoi(argv[1]) : 50000;
    int gpu_id = argc > 2 ? atoi(argv[2]) : 0;
    cudaSetDevice(gpu_id);

    printf("Spectral Gaps for Primes to %d (GPU %d)\n", max_p, gpu_id);
    printf("Chebyshev N=%d, FP32, deflated power iteration\n\n", N_CHEB);

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
    printf("Primes: %d\n\n", num_primes);

    printf("%8s  %8s  %10s\n", "prime", "gap", "time");
    printf("--------  --------  ----------\n");

    float min_gap = 999.0f;
    int min_gap_prime = 0;
    int primes_done = 0;

    for (int p = 2; p <= max_p; p++) {
        if (!is_prime[p]) continue;

        struct timespec tp0, tp1;
        clock_gettime(CLOCK_MONOTONIC, &tp0);

        int p1 = p + 1;
        int vec_size = N_CHEB * p1;

        // Compute permutations on CPU
        int *h_perm = (int*)malloc(BOUND * p1 * sizeof(int));
        compute_permutations(p, h_perm);

        // Allocate GPU memory
        int *d_perm;
        float *d_result, *d_workspace;
        cudaMalloc(&d_perm, BOUND * p1 * sizeof(int));
        cudaMalloc(&d_result, sizeof(float));
        cudaMalloc(&d_workspace, 2 * vec_size * sizeof(float));

        cudaMemcpy(d_perm, h_perm, BOUND * p1 * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel — one block, 256 threads
        int threads = 256;
        if (p1 < 256) threads = ((p1 + 31) / 32) * 32;
        if (threads < 32) threads = 32;

        compute_gap_kernel<<<1, threads>>>(d_perm, p, p1, d_result, d_workspace);
        cudaDeviceSynchronize();

        float gap;
        cudaMemcpy(&gap, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_perm);
        cudaFree(d_result);
        cudaFree(d_workspace);
        free(h_perm);

        clock_gettime(CLOCK_MONOTONIC, &tp1);
        double pt = (tp1.tv_sec - tp0.tv_sec) + (tp1.tv_nsec - tp0.tv_nsec) / 1e9;

        if (gap > 0 && gap < min_gap) {
            min_gap = gap;
            min_gap_prime = p;
        }

        primes_done++;
        if (p <= 100 || p % 1000 == 0 || p == max_p ||
            (gap > 0 && gap < 0.30f) || primes_done == num_primes) {
            printf("%8d  %8.4f  %8.3fs", p, gap, pt);
            if (gap > 0 && gap < 0.30f) printf("  <-- tight");
            printf("\n");
            fflush(stdout);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Primes computed: %d (to p=%d)\n", primes_done, max_p);
    printf("Minimum gap: %.4f at p=%d\n", min_gap, min_gap_prime);
    printf("Total time: %.1fs\n", total);
    printf("========================================\n");

    free(is_prime);
    return 0;
}
