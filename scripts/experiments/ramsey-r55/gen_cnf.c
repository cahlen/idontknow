// Generate DIMACS CNF for Ramsey R(5,5) on K_n
// For n=43: 903 variables (edges), ~1.9M 10-clauses
#include <stdio.h>
#include <stdlib.h>

static int gN;

int edge_var(int i, int j) {
    if (i > j) { int t=i; i=j; j=t; }
    return i * gN - i * (i+1) / 2 + j - i;  // 1-indexed: edge(0,1)=1
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 43;
    gN = n;
    int num_edges = n * (n-1) / 2;
    int num_clauses = 0;
    for (int a=0;a<n;a++) for (int b=a+1;b<n;b++) for (int c=b+1;c<n;c++)
        for (int d=c+1;d<n;d++) for (int e=d+1;e<n;e++) num_clauses += 2;
    num_clauses += 1; // symmetry breaking

    char path[256];
    snprintf(path, 256, "/tmp/ramsey_k%d.cnf", n);
    FILE *f = fopen(path, "w");
    fprintf(f, "p cnf %d %d\n", num_edges, num_clauses);

    for (int a=0;a<n;a++) for (int b=a+1;b<n;b++) for (int c=b+1;c<n;c++)
    for (int d=c+1;d<n;d++) for (int e=d+1;e<n;e++) {
        int v[5]={a,b,c,d,e};
        for (int i=0;i<5;i++) for (int j=i+1;j<5;j++) fprintf(f,"%d ",-edge_var(v[i],v[j]));
        fprintf(f,"0\n");
        for (int i=0;i<5;i++) for (int j=i+1;j<5;j++) fprintf(f,"%d ",edge_var(v[i],v[j]));
        fprintf(f,"0\n");
    }
    fprintf(f, "%d 0\n", edge_var(0, 1)); // symmetry: edge(0,1) red
    fclose(f);
    fprintf(stderr, "n=%d: %d vars, %d clauses\n", n, num_edges, num_clauses);
    return 0;
}
