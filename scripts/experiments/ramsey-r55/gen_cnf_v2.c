// Generate DIMACS CNF for Ramsey R(5,5) on K_n with enhanced symmetry breaking
// For n=43: 903 variables (edges), ~1.9M clauses + symmetry constraints
//
// Symmetry breaking strategy:
// 1. Fix edge(0,1) = red (standard)
// 2. Fix vertex 0's neighborhood: first k red, rest blue (lex-leader on vertex 0)
// 3. This eliminates 2^(n-2) symmetric configurations
//
// Output: /tmp/ramsey_k{n}.cnf (or argv[2] if provided)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gN;

int edge_var(int i, int j) {
    if (i > j) { int t = i; i = j; j = t; }
    return i * gN - i * (i + 1) / 2 + j - i;  // 1-indexed
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 43;
    gN = n;
    int num_edges = n * (n - 1) / 2;

    // Count Ramsey clauses: C(n,5) * 2
    long long num_5subsets = 1;
    for (int i = 0; i < 5; i++) num_5subsets *= (n - i);
    for (int i = 1; i <= 5; i++) num_5subsets /= i;
    long long ramsey_clauses = num_5subsets * 2;

    // Symmetry breaking clauses:
    // Fix edge(0,1) = red: 1 clause
    // Lex-leader on vertex 0: edges to vertex 0 must be non-increasing
    // edge(0,1) >= edge(0,2) >= ... >= edge(0,n-1)
    // This means: if edge(0,k) is blue (0), then edge(0,k+1) must be blue too
    // Encoded as: NOT edge(0,k+1) OR edge(0,k) for k=1..n-3
    int sym_clauses = 1 + (n - 3);  // fix(0,1) + lex-leader chain
    long long total_clauses = ramsey_clauses + sym_clauses;

    char path[256];
    if (argc > 2)
        strncpy(path, argv[2], 255);
    else
        snprintf(path, 256, "/tmp/ramsey_k%d.cnf", n);

    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); return 1; }

    fprintf(f, "c Ramsey R(5,5) on K_%d\n", n);
    fprintf(f, "c %d variables (edges), %lld clauses\n", num_edges, total_clauses);
    fprintf(f, "c Symmetry breaking: fix edge(0,1)=red + lex-leader on vertex 0\n");
    fprintf(f, "p cnf %d %lld\n", num_edges, total_clauses);

    // Ramsey clauses: for each 5-subset, forbid mono-red and mono-blue K5
    for (int a = 0; a < n; a++)
    for (int b = a + 1; b < n; b++)
    for (int c = b + 1; c < n; c++)
    for (int d = c + 1; d < n; d++)
    for (int e = d + 1; e < n; e++) {
        int v[5] = {a, b, c, d, e};
        // No all-red K5
        for (int i = 0; i < 5; i++)
            for (int j = i + 1; j < 5; j++)
                fprintf(f, "%d ", -edge_var(v[i], v[j]));
        fprintf(f, "0\n");
        // No all-blue K5
        for (int i = 0; i < 5; i++)
            for (int j = i + 1; j < 5; j++)
                fprintf(f, "%d ", edge_var(v[i], v[j]));
        fprintf(f, "0\n");
    }

    // Symmetry breaking: edge(0,1) = red (true)
    fprintf(f, "%d 0\n", edge_var(0, 1));

    // Lex-leader on vertex 0's neighborhood:
    // edge(0,k) >= edge(0,k+1) for k=1..n-3
    // Encoded as: NOT edge(0,k+1) OR edge(0,k)
    // i.e., edge(0,k+1) => edge(0,k)
    for (int k = 1; k <= n - 3; k++) {
        fprintf(f, "%d %d 0\n", edge_var(0, k), -edge_var(0, k + 1));
    }

    fclose(f);
    fprintf(stderr, "n=%d: %d vars, %lld clauses (incl %d symmetry-breaking)\n",
            n, num_edges, total_clauses, sym_clauses);
    fprintf(stderr, "Output: %s\n", path);
    return 0;
}
