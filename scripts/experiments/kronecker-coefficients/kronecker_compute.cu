/*
 * Kronecker coefficient computation via Murnaghan-Nakayama rule
 *
 * g(λ,μ,ν) = Σ_{ρ⊢n} (1/z_ρ) χ^λ(ρ) χ^μ(ρ) χ^ν(ρ)
 *
 * Phase 1: CPU builds full character table via MN rule
 * Phase 2: GPU computes all Kronecker triples in parallel
 *
 * For n≤50: full table (all partitions, all triples)
 * For n>50: height-bounded partitions only
 *
 * Compile: nvcc -O3 -arch=sm_100a -o kronecker kronecker_compute.cu -lm
 * Run:     ./kronecker <n> [max_height]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_N 200
#define MAX_PARTS 64
#define BLOCK_SIZE 256

typedef struct {
    int parts[MAX_PARTS];  // descending order
    int len;               // number of nonzero parts
    int n;                 // sum
} Partition;

/* ── Partition generation ────────────────────────────────── */

// Generate all partitions of n (optionally bounded by max_height parts)
// Returns count. Partitions stored in out[].
int generate_partitions(int n, int max_height, Partition *out, int max_out) {
    if (n == 0) {
        out[0].n = 0; out[0].len = 0;
        memset(out[0].parts, 0, sizeof(out[0].parts));
        return 1;
    }

    int count = 0;
    int a[MAX_PARTS];
    memset(a, 0, sizeof(a));
    a[0] = n;
    int num_parts = 1;

    while (1) {
        if (num_parts <= max_height && count < max_out) {
            Partition p;
            p.n = n; p.len = num_parts;
            memset(p.parts, 0, sizeof(p.parts));
            for (int i = 0; i < num_parts; i++) p.parts[i] = a[i];
            out[count++] = p;
        }

        // Find rightmost part > 1
        int idx = num_parts - 1;
        while (idx >= 0 && a[idx] == 1) idx--;
        if (idx < 0) break;

        a[idx]--;
        int remainder = num_parts - idx - 1 + 1;
        int fill_val = a[idx];
        int pos = idx + 1;
        while (remainder > 0) {
            int val = (remainder >= fill_val) ? fill_val : remainder;
            a[pos] = val;
            remainder -= val;
            pos++;
        }
        num_parts = pos;
    }
    return count;
}

/* ── Young diagram operations ────────────────────────────── */

// Convert partition to row-lengths array (same as parts, but we work with it)
// The "diagram" is just the partition itself: row i has parts[i] cells.

// Check if removing cells from rows r_start..r_end (inclusive) of the border
// gives a valid border strip of size k.
// A border strip: connected, no 2x2 square, size k.
// We use the column-based approach: find removable border strips.

// For MN: we need to enumerate all border strips of size k in partition lambda.
// A border strip of size k is removed from the SE boundary.
// It can be described by: starting column c, and which rows it spans.

// Simpler approach: use the recursive rim-hook removal.
// A rim hook (= border strip) of size k starting at row r:
// Remove cells from the rim of the diagram, starting from row r's rightmost cell,
// going down and left along the boundary, total k cells.

// We represent the partition as an array of row lengths.
// The rim goes: from (r, lambda[r]-1) stepping to (r+1, ...) etc.

// For efficiency, enumerate border strips by their bottom row and top row.
// A border strip occupying rows r_top..r_bot has:
//   - In row r_top: cells from some column to lambda[r_top]-1
//   - In row r_bot: cells from lambda[r_bot+1] (or 0) to some column
//   - In between: exactly lambda[i] - lambda[i+1] cells removed from row i
// Total size = sum of cells removed.

// The sign is (-1)^(r_bot - r_top) = (-1)^height.

// Recursive MN: χ^λ(ρ_1, ρ_2, ..., ρ_m) =
//   Σ over border strips B of size ρ_1 in λ:
//     (-1)^height(B) * χ^{λ\B}(ρ_2, ..., ρ_m)

// Implementation: for each removable border strip of size k in lambda,
// compute the residual partition and recurse.

// Find all border strips of size k in partition lambda.
// Store results as (residual partition, sign) pairs.
typedef struct {
    Partition residual;
    int sign;  // +1 or -1
} BorderStripResult;

// Recursive helper: extend a border strip from row r downward,
// having already removed 'used' cells from rows above.
// new_parts is modified in-place (caller must save/restore).
static void find_strips_recursive(
    int *new_parts, int n_total, int k_remaining, int r_top, int r_current,
    BorderStripResult *results, int *count, int max_results)
{
    if (*count >= max_results) return;

    if (k_remaining == 0) {
        // Found a valid strip. Check partition validity.
        int ok = 1;
        for (int i = 0; i < MAX_PARTS - 1; i++) {
            if (new_parts[i] == 0) break;
            if (new_parts[i] < new_parts[i + 1]) { ok = 0; break; }
        }
        if (r_top > 0 && new_parts[r_top] > new_parts[r_top - 1]) ok = 0;

        if (ok) {
            BorderStripResult *res = &results[*count];
            res->residual.n = n_total - 0;  // will be set by caller
            memcpy(res->residual.parts, new_parts, sizeof(int) * MAX_PARTS);
            res->residual.len = 0;
            for (int i = 0; i < MAX_PARTS && new_parts[i] > 0; i++)
                res->residual.len = i + 1;
            res->sign = ((r_current - 1 - r_top) % 2 == 0) ? 1 : -1;
            (*count)++;
        }
        return;
    }

    if (r_current >= MAX_PARTS || new_parts[r_current] == 0) return;

    int next_row_len = (r_current + 1 < MAX_PARTS) ? new_parts[r_current + 1] : 0;
    int max_remove = new_parts[r_current] - next_row_len;  // overhang

    if (max_remove <= 0) return;  // no cells to remove in this row

    // Option A: remove some cells from this row and STOP here (1..min(max_remove, k_remaining))
    int can_remove = (max_remove < k_remaining) ? max_remove : k_remaining;
    for (int remove = 1; remove <= can_remove; remove++) {
        int saved = new_parts[r_current];
        new_parts[r_current] -= remove;

        if (remove == k_remaining) {
            // Strip ends here
            find_strips_recursive(new_parts, n_total, 0, r_top, r_current + 1,
                                  results, count, max_results);
        }

        new_parts[r_current] = saved;
    }

    // Option B: remove the FULL overhang and continue to next row
    if (max_remove < k_remaining) {
        int saved = new_parts[r_current];
        new_parts[r_current] = next_row_len;

        find_strips_recursive(new_parts, n_total, k_remaining - max_remove,
                              r_top, r_current + 1, results, count, max_results);

        new_parts[r_current] = saved;
    }
}

int find_border_strips(const Partition *lambda, int k, BorderStripResult *results, int max_results) {
    int count = 0;
    int new_parts[MAX_PARTS];

    for (int r_top = 0; r_top < lambda->len; r_top++) {
        memcpy(new_parts, lambda->parts, sizeof(int) * MAX_PARTS);
        find_strips_recursive(new_parts, lambda->n, k, r_top, r_top,
                              results, &count, max_results);
    }

    // Set residual n
    for (int i = 0; i < count; i++)
        results[i].residual.n = lambda->n - k;

    return count;
}

/* ── Murnaghan-Nakayama character computation ────────────── */

// Compute χ^λ(ρ) recursively via MN rule
// rho is given as cycle lengths rho[0] >= rho[1] >= ... >= rho[rho_len-1]
int64_t mn_character(const Partition *lambda, const int *rho, int rho_len) {
    // Base case: empty partition, empty cycle type
    if (rho_len == 0) {
        return (lambda->n == 0) ? 1 : 0;
    }
    if (lambda->n == 0) return 0;

    int k = rho[0];  // largest cycle
    BorderStripResult strips[1024];
    int num_strips = find_border_strips(lambda, k, strips, 1024);

    int64_t result = 0;
    for (int i = 0; i < num_strips; i++) {
        int64_t sub = mn_character(&strips[i].residual, rho + 1, rho_len - 1);
        result += strips[i].sign * sub;
    }
    return result;
}

/* ── Centralizer order ───────────────────────────────────── */

// z_ρ = Π_i i^{m_i} * m_i!  where m_i = multiplicity of i in ρ
double compute_z_inv(const Partition *rho) {
    int mult[MAX_N + 1];
    memset(mult, 0, sizeof(mult));
    for (int i = 0; i < rho->len; i++) {
        if (rho->parts[i] > 0 && rho->parts[i] <= MAX_N)
            mult[rho->parts[i]]++;
    }

    double log_z = 0.0;
    for (int i = 1; i <= MAX_N; i++) {
        if (mult[i] > 0) {
            log_z += mult[i] * log((double)i);
            for (int j = 2; j <= mult[i]; j++)
                log_z += log((double)j);  // log(m_i!)
        }
    }
    return exp(-log_z);
}

/* ── GPU kernel: Kronecker triple sum ────────────────────── */

// Character table is stored as: char_table[lambda_idx * num_classes + rho_idx]
// GPU kernel: one thread per triple (i, j, k) with i <= j <= k
__global__ void kronecker_kernel(
    const int64_t *char_table,   // [num_parts x num_classes]
    const double *z_inv,         // [num_classes]
    int num_parts,               // number of partitions (= rows)
    int num_classes,             // number of conjugacy classes (= cols)
    int64_t *kronecker_out,      // output: g(lambda_i, lambda_j, lambda_k)
    uint64_t num_triples)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= num_triples) return;

    // Decode triple index (i, j, k) with i <= j <= k
    // Use the combinatorial number system
    // For simplicity, use flat indexing: triple = i * np^2 + j * np + k
    int np = num_parts;
    int i = tid / ((uint64_t)np * np);
    int j = (tid / np) % np;
    int k = tid % np;

    // Only compute i <= j <= k (symmetry)
    if (i > j || j > k) { kronecker_out[tid] = 0; return; }

    // g(λ_i, λ_j, λ_k) = Σ_ρ (1/z_ρ) χ^λ_i(ρ) χ^λ_j(ρ) χ^λ_k(ρ)
    double sum = 0.0;
    for (int c = 0; c < num_classes; c++) {
        double chi_i = (double)char_table[(uint64_t)i * num_classes + c];
        double chi_j = (double)char_table[(uint64_t)j * num_classes + c];
        double chi_k = (double)char_table[(uint64_t)k * num_classes + c];
        sum += z_inv[c] * chi_i * chi_j * chi_k;
    }

    // Kronecker coefficients are integers — round
    kronecker_out[tid] = (int64_t)round(sum);
}

/* ── Main ────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n> [max_height]\n", argv[0]);
        fprintf(stderr, "  n: symmetric group S_n\n");
        fprintf(stderr, "  max_height: max partition height (default: n)\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int max_height = (argc > 2) ? atoi(argv[2]) : n;

    struct timespec t_start, t_char, t_gpu, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("========================================\n");
    printf("Kronecker Coefficients for S_%d\n", n);
    if (max_height < n)
        printf("Height bound: %d\n", max_height);
    printf("========================================\n\n");

    // Generate partitions
    int max_alloc = 50000000;  // 50M partitions max
    Partition *partitions = (Partition *)malloc(max_alloc * sizeof(Partition));
    if (!partitions) { fprintf(stderr, "malloc failed\n"); return 1; }

    int num_parts = generate_partitions(n, max_height, partitions, max_alloc);
    printf("Partitions of %d (height <= %d): %d\n", n, max_height, num_parts);

    // Conjugacy classes = ALL partitions of n (cycle types)
    Partition *classes = (Partition *)malloc(max_alloc * sizeof(Partition));
    int num_classes = generate_partitions(n, n, classes, max_alloc);
    printf("Conjugacy classes: %d\n", num_classes);

    uint64_t num_triples = (uint64_t)num_parts * num_parts * num_parts;
    uint64_t unique_triples = 0;
    for (uint64_t i = 0; i < (uint64_t)num_parts; i++)
        for (uint64_t j = i; j < (uint64_t)num_parts; j++)
            for (uint64_t k = j; k < (uint64_t)num_parts; k++)
                unique_triples++;

    printf("Unique triples (i<=j<=k): %lu\n", unique_triples);
    printf("Character table: %d x %d = %lu entries\n\n",
           num_parts, num_classes, (uint64_t)num_parts * num_classes);

    // Phase 1: Build character table on CPU via MN rule
    printf("Phase 1: Computing character table via Murnaghan-Nakayama...\n");
    fflush(stdout);

    uint64_t table_size = (uint64_t)num_parts * num_classes;
    int64_t *char_table = (int64_t *)calloc(table_size, sizeof(int64_t));
    double *z_inv = (double *)malloc(num_classes * sizeof(double));

    // Compute z_inv for each conjugacy class
    for (int c = 0; c < num_classes; c++) {
        z_inv[c] = compute_z_inv(&classes[c]);
    }

    // Compute character values
    int progress_step = (num_parts * num_classes > 1000) ?
                        (num_parts * num_classes / 20) : 1;
    int computed = 0;

    for (int i = 0; i < num_parts; i++) {
        for (int c = 0; c < num_classes; c++) {
            char_table[(uint64_t)i * num_classes + c] =
                mn_character(&partitions[i], classes[c].parts, classes[c].len);

            computed++;
            if (computed % progress_step == 0) {
                printf("  Character table: %d / %lu (%.0f%%)\n",
                       computed, table_size,
                       100.0 * computed / table_size);
                fflush(stdout);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_char);
    double char_time = (t_char.tv_sec - t_start.tv_sec) +
                       (t_char.tv_nsec - t_start.tv_nsec) / 1e9;
    printf("Character table: %.2f seconds\n\n", char_time);

    // Validation: χ^(n)(ρ) = 1 for all ρ (trivial representation)
    // The trivial rep is the partition (n), which should be index 0
    printf("Validation:\n");
    printf("  χ^(%d)(any ρ) should be 1 (trivial rep): ", n);
    int trivial_ok = 1;
    for (int c = 0; c < num_classes && c < 5; c++) {
        int64_t val = char_table[0 * num_classes + c];  // partition (n) = index 0
        printf("%ld ", val);
        if (val != 1) trivial_ok = 0;
    }
    printf("%s\n", trivial_ok ? "OK" : "FAIL");

    // χ^(1^n)(ρ) = sign(ρ) = (-1)^(n - len(ρ)) (sign representation)
    // The sign rep is partition (1,1,...,1) = last partition
    printf("  χ^(1^%d)(ρ) should be sign(ρ): ", n);
    int sign_ok = 1;
    for (int c = 0; c < num_classes && c < 5; c++) {
        int64_t val = char_table[(uint64_t)(num_parts - 1) * num_classes + c];
        int expected_sign = ((n - classes[c].len) % 2 == 0) ? 1 : -1;
        printf("%ld(exp %d) ", val, expected_sign);
        if (val != expected_sign) sign_ok = 0;
    }
    printf("%s\n", sign_ok ? "OK" : "FAIL");

    // Column orthogonality: Σ_λ χ^λ(id)^2 = n! (where id = (1,1,...,1))
    // Find the identity class (cycle type (1^n))
    int id_class = -1;
    for (int c = 0; c < num_classes; c++) {
        if (classes[c].len == n && classes[c].parts[0] == 1) { id_class = c; break; }
    }
    if (id_class >= 0 && max_height >= n) {
        int64_t dim_sum = 0;
        for (int i = 0; i < num_parts; i++) {
            int64_t d = char_table[(uint64_t)i * num_classes + id_class];
            dim_sum += d * d;
        }
        // Should equal n!
        int64_t nfact = 1;
        for (int i = 2; i <= n && i <= 20; i++) nfact *= i;
        if (n <= 20)
            printf("  Σ dim(λ)² = %ld (expected %ld = %d!): %s\n",
                   dim_sum, nfact, n, dim_sum == nfact ? "OK" : "FAIL");
    }
    printf("\n");

    // Phase 2: GPU Kronecker coefficient computation
    printf("Phase 2: Computing Kronecker coefficients on GPU...\n");
    fflush(stdout);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("GPUs available: %d\n", num_gpus);

    // For small n, compute on single GPU
    int gpu_id = 0;
    cudaSetDevice(gpu_id);

    int64_t *d_char_table;
    double *d_z_inv;
    int64_t *d_kronecker;

    cudaMalloc(&d_char_table, table_size * sizeof(int64_t));
    cudaMalloc(&d_z_inv, num_classes * sizeof(double));
    cudaMalloc(&d_kronecker, num_triples * sizeof(int64_t));

    cudaMemcpy(d_char_table, char_table, table_size * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z_inv, z_inv, num_classes * sizeof(double), cudaMemcpyHostToDevice);

    int blocks = (num_triples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kronecker_kernel<<<blocks, BLOCK_SIZE>>>(
        d_char_table, d_z_inv, num_parts, num_classes,
        d_kronecker, num_triples);
    cudaDeviceSynchronize();

    // Copy back
    int64_t *kronecker = (int64_t *)calloc(num_triples, sizeof(int64_t));
    cudaMemcpy(kronecker, d_kronecker, num_triples * sizeof(int64_t), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &t_gpu);
    double gpu_time = (t_gpu.tv_sec - t_char.tv_sec) +
                      (t_gpu.tv_nsec - t_char.tv_nsec) / 1e9;
    printf("GPU Kronecker computation: %.2f seconds\n\n", gpu_time);

    // Statistics
    uint64_t nonzero = 0, total_checked = 0;
    int64_t max_val = 0;
    for (uint64_t i = 0; i < (uint64_t)num_parts; i++) {
        for (uint64_t j = i; j < (uint64_t)num_parts; j++) {
            for (uint64_t k = j; k < (uint64_t)num_parts; k++) {
                int64_t g = kronecker[i * num_parts * num_parts + j * num_parts + k];
                total_checked++;
                if (g != 0) nonzero++;
                if (g > max_val) max_val = g;
            }
        }
    }

    // Output CSV
    char csv_path[256];
    snprintf(csv_path, 256,
             "scripts/experiments/kronecker-coefficients/results/kronecker_n%d%s.csv",
             n, max_height < n ? "_bounded" : "");

    // Ensure results directory exists
    system("mkdir -p scripts/experiments/kronecker-coefficients/results");

    FILE *csv = fopen(csv_path, "w");
    if (csv) {
        fprintf(csv, "lambda,mu,nu,g\n");
        for (int i = 0; i < num_parts; i++) {
            for (int j = i; j < num_parts; j++) {
                for (int k = j; k < num_parts; k++) {
                    int64_t g = kronecker[(uint64_t)i * num_parts * num_parts +
                                          j * num_parts + k];
                    if (g != 0) {
                        // Format partitions
                        fprintf(csv, "\"(");
                        for (int p = 0; p < partitions[i].len; p++)
                            fprintf(csv, "%s%d", p?",":"", partitions[i].parts[p]);
                        fprintf(csv, ")\",\"(");
                        for (int p = 0; p < partitions[j].len; p++)
                            fprintf(csv, "%s%d", p?",":"", partitions[j].parts[p]);
                        fprintf(csv, ")\",\"(");
                        for (int p = 0; p < partitions[k].len; p++)
                            fprintf(csv, "%s%d", p?",":"", partitions[k].parts[p]);
                        fprintf(csv, ")\",%ld\n", g);
                    }
                }
            }
        }
        fclose(csv);
        printf("Output: %s\n", csv_path);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_time = (t_end.tv_sec - t_start.tv_sec) +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("Kronecker Coefficients for S_%d\n", n);
    printf("Partitions: %d (height <= %d)\n", num_parts, max_height);
    printf("Conjugacy classes: %d\n", num_classes);
    printf("Unique triples: %lu\n", unique_triples);
    printf("Nonzero coefficients: %lu (%.1f%%)\n",
           nonzero, 100.0 * nonzero / total_checked);
    printf("Max coefficient: %ld\n", max_val);
    printf("Character table time: %.2f sec\n", char_time);
    printf("GPU triple-sum time: %.2f sec\n", gpu_time);
    printf("Total time: %.2f sec\n", total_time);
    printf("========================================\n");

    // Cleanup
    free(char_table); free(z_inv); free(kronecker);
    free(partitions); free(classes);
    cudaFree(d_char_table); cudaFree(d_z_inv); cudaFree(d_kronecker);

    return 0;
}
