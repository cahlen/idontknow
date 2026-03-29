/*
 * GPU BFS for Cayley Graph Diameter of Gamma_{1,...,5} in SL_2(Z/pZ)
 *
 * Each BFS level: one kernel launch expands ALL frontier nodes in parallel.
 * Each thread handles one frontier node, computes 10 neighbors (5 generators + inverses),
 * marks them in a visited bitset via atomicOr.
 *
 * The frontier is double-buffered: current frontier → next frontier.
 * Diameter = number of BFS levels until the frontier is empty.
 *
 * Group elements encoded as: index = a*p^3 + b*p^2 + c*p + d
 * where [[a,b],[c,d]] is the matrix mod p.
 * For p <= 200: index fits in uint32 (200^4 = 1.6B < 2^32).
 *
 * Visited set: bitset of size p^4/8 bytes.
 * For p=200: 1.6B bits = 200MB. Fits on one B200.
 * For p=500: 62.5B bits = 7.8GB. Still fits.
 *
 * Compile: nvcc -O3 -arch=sm_100a -o cayley_gpu scripts/experiments/zaremba-cayley-diameter/cayley_gpu.cu
 * Run:     ./cayley_gpu <max_prime>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define BOUND 5
#define BLOCK_SIZE 256
#define NUM_GENS 10

typedef unsigned int uint32;
typedef unsigned long long uint64;

// Generators stored in constant memory
__constant__ int d_gen[NUM_GENS][4]; // [g][0..3] = a,b,c,d of generator g

// BFS expand kernel: for each frontier node, compute 10 neighbors,
// mark in visited bitset, append to next frontier
__global__ void bfs_expand(
    uint32 *frontier, uint64 frontier_size,
    uint32 *next_frontier, unsigned long long *next_count,
    uint32 *visited, int p, uint64 max_next)
{
    uint64 idx = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    uint32 code = frontier[idx];
    int ma = code / (p*p*p);
    int mb = (code / (p*p)) % p;
    int mc = (code / p) % p;
    int md = code % p;

    for (int g = 0; g < NUM_GENS; g++) {
        int na = (ma * d_gen[g][0] + mb * d_gen[g][2]) % p;
        int nb = (ma * d_gen[g][1] + mb * d_gen[g][3]) % p;
        int nc = (mc * d_gen[g][0] + md * d_gen[g][2]) % p;
        int nd = (mc * d_gen[g][1] + md * d_gen[g][3]) % p;

        uint32 ncode = (uint32)na * p*p*p + (uint32)nb * p*p + (uint32)nc * p + (uint32)nd;

        // Check and set visited bit atomically
        uint32 word = ncode / 32;
        uint32 bit = 1u << (ncode % 32);
        uint32 old = atomicOr(&visited[word], bit);

        if (!(old & bit)) {
            // First time visiting — add to next frontier
            unsigned long long pos = atomicAdd(next_count, 1ULL);
            if (pos < max_next) {
                next_frontier[pos] = ncode;
            }
        }
    }
}

int cayley_diameter_gpu(int p, int gpu_id) {
    cudaSetDevice(gpu_id);

    uint64 p4 = (uint64)p * p * p * p;
    uint64 group_size = (uint64)p * (p*p - 1);
    uint64 bitset_words = (p4 + 31) / 32;
    uint64 bitset_bytes = bitset_words * sizeof(uint32);

    // Check memory
    double mem_gb = (bitset_bytes + group_size * 2 * sizeof(uint32)) / 1e9;
    if (mem_gb > 150) return -1; // too large for one GPU

    // Setup generators
    int h_gen[NUM_GENS][4];
    for (int a = 1; a <= BOUND; a++) {
        h_gen[a-1][0] = a; h_gen[a-1][1] = 1; h_gen[a-1][2] = 1; h_gen[a-1][3] = 0;
        h_gen[a+4][0] = 0; h_gen[a+4][1] = 1; h_gen[a+4][2] = 1; h_gen[a+4][3] = (p-a)%p;
    }
    cudaMemcpyToSymbol(d_gen, h_gen, sizeof(h_gen));

    // Allocate
    uint32 *d_visited;
    cudaMalloc(&d_visited, bitset_bytes);
    cudaMemset(d_visited, 0, bitset_bytes);

    uint64 max_frontier = group_size; // worst case
    if (max_frontier > 200000000ULL) max_frontier = 200000000ULL;

    uint32 *d_front_a, *d_front_b;
    cudaMalloc(&d_front_a, max_frontier * sizeof(uint32));
    cudaMalloc(&d_front_b, max_frontier * sizeof(uint32));

    unsigned long long *d_next_count;
    cudaMalloc(&d_next_count, sizeof(unsigned long long));

    // Start BFS from identity
    uint32 id_code = (uint32)1 * p*p*p + 0 * p*p + 0 * p + 1; // [[1,0],[0,1]]
    cudaMemcpy(d_front_a, &id_code, sizeof(uint32), cudaMemcpyHostToDevice);

    // Mark identity as visited
    uint32 id_word = id_code / 32;
    uint32 id_bit = 1u << (id_code % 32);
    uint32 h_word;
    cudaMemcpy(&h_word, d_visited + id_word, sizeof(uint32), cudaMemcpyDeviceToHost);
    h_word |= id_bit;
    cudaMemcpy(d_visited + id_word, &h_word, sizeof(uint32), cudaMemcpyHostToDevice);

    uint64 frontier_size = 1;
    uint64 total_visited = 1;
    int diameter = 0;

    while (frontier_size > 0 && total_visited < group_size) {
        cudaMemset(d_next_count, 0, sizeof(unsigned long long));

        int blocks = (int)((frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (blocks > 2147483647) blocks = 2147483647;

        bfs_expand<<<blocks, BLOCK_SIZE>>>(
            d_front_a, frontier_size,
            d_front_b, d_next_count,
            d_visited, p, max_frontier
        );
        cudaDeviceSynchronize();

        unsigned long long h_next;
        cudaMemcpy(&h_next, d_next_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        frontier_size = h_next < max_frontier ? h_next : max_frontier;
        total_visited += h_next;

        if (h_next > 0) diameter++;

        // Swap
        uint32 *tmp = d_front_a; d_front_a = d_front_b; d_front_b = tmp;
    }

    cudaFree(d_visited);
    cudaFree(d_front_a);
    cudaFree(d_front_b);
    cudaFree(d_next_count);

    return diameter;
}

int main(int argc, char **argv) {
    int max_p = argc > 1 ? atoi(argv[1]) : 200;

    printf("GPU Cayley Diameters: Gamma_{1,...,5} in SL_2(Z/pZ)\n");
    printf("Max prime: %d\n\n", max_p);

    int ngpus;
    cudaGetDeviceCount(&ngpus);
    printf("GPUs: %d\n\n", ngpus);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    printf("%6s  %12s  %8s  %8s  %10s  %6s\n",
           "p", "|SL_2|", "diameter", "log(p)", "diam/logp", "time");
    printf("------  ------------  --------  --------  ----------  ------\n");

    // Sieve
    char *is_p = (char*)calloc(max_p+1, 1);
    memset(is_p, 1, max_p+1); is_p[0]=is_p[1]=0;
    for (int i=2; (long long)i*i<=max_p; i++)
        if (is_p[i]) for (int j=i*i; j<=max_p; j+=i) is_p[j]=0;

    for (int p = 2; p <= max_p; p++) {
        if (!is_p[p]) continue;

        struct timespec tp0, tp1;
        clock_gettime(CLOCK_MONOTONIC, &tp0);

        int diam = cayley_diameter_gpu(p, 0);

        clock_gettime(CLOCK_MONOTONIC, &tp1);
        double pt = (tp1.tv_sec-tp0.tv_sec)+(tp1.tv_nsec-tp0.tv_nsec)/1e9;

        uint64 gs = (uint64)p * (p*p-1);
        double logp = log((double)p);

        if (diam >= 0)
            printf("%6d  %12llu  %8d  %8.2f  %10.4f  %5.1fs\n",
                   p, (unsigned long long)gs, diam, logp, diam/logp, pt);
        else
            printf("%6d  %12llu  (too large)\n", p, (unsigned long long)gs);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("\nTotal: %.1fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9);
    free(is_p);
    return 0;
}
