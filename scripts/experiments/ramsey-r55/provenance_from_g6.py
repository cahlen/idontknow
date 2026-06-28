#!/usr/bin/env python3
"""
Full-provenance verification from the ORIGINAL McKay-Radziszowski source data.

Unlike verify_independent.py (which trusts the precomputed K4 masks in
mckay_k42_all.bin), this script starts from the raw graph6 file published by
Brendan McKay and re-derives everything itself:

  1. Parse r55_42some.g6  -> 328 Ramsey(5,5,42) adjacency matrices
  2. Form each graph's complement -> 656 colorings total
  3. Enumerate red/blue K4s directly from the adjacency matrices
  4. Encode + solve the single-vertex extension as SAT (kissat)
  5. Cross-check the re-derived K4 counts against mckay_k42_all.bin

This removes all trust in this repo's CUDA preprocessing: the only inputs are
McKay's published graphs and an external SAT solver.

Source: https://users.cecs.anu.edu.au/~bdm/data/r55_42some.g6
  ("Exoo found several Ramsey(5,5,42)-graphs in 1989; McKay and Radziszowski
   expanded this to 656 graphs and conjectured there are none larger.")

Scope/limits are the same as verify_independent.py: confirms the known graphs
do not extend by one vertex; does NOT prove R(5,5)=43. See VERIFICATION.md.

Usage:
  python3 provenance_from_g6.py r55_42some.g6 [--bin mckay_k42_all.bin]
"""
import argparse, os, struct, subprocess, sys, tempfile, time

N = 42
FULL = (1 << N) - 1


def parse_g6(line):
    data = [ord(c) - 63 for c in line.strip()]
    n = data[0]
    bits = []
    for b in data[1:]:
        for k in range(5, -1, -1):
            bits.append((b >> k) & 1)
    adj = [0] * n
    idx = 0
    for j in range(1, n):           # graph6 edge order: column j, rows i<j
        for i in range(j):
            if bits[idx]:
                adj[i] |= (1 << j); adj[j] |= (1 << i)
            idx += 1
    return n, adj


def four_cliques(adj, n):
    """All 4-cliques as vertex bitmasks, each enumerated once (a<b<c<d)."""
    res = []
    for a in range(n):
        bb = adj[a] & ~((1 << (a + 1)) - 1)
        while bb:
            b = (bb & -bb).bit_length() - 1; bb &= bb - 1
            cb = adj[a] & adj[b] & ~((1 << (b + 1)) - 1); cc = cb
            while cc:
                c = (cc & -cc).bit_length() - 1; cc &= cc - 1
                cd = cb & adj[c] & ~((1 << (c + 1)) - 1); dd = cd
                while dd:
                    d = (dd & -dd).bit_length() - 1; dd &= dd - 1
                    res.append((1 << a) | (1 << b) | (1 << c) | (1 << d))
    return res


def complement(adj, n):
    return [(~adj[v]) & FULL & ~(1 << v) for v in range(n)]


def cnf(red, blue):
    L = []
    for m in red:
        L.append(" ".join(str(-(v + 1)) for v in range(N) if (m >> v) & 1) + " 0")
    for m in blue:
        L.append(" ".join(str(v + 1) for v in range(N) if (m >> v) & 1) + " 0")
    return f"p cnf {N} {len(L)}\n" + "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("g6", help="path to r55_42some.g6")
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "mckay_k42_all.bin"))
    args = ap.parse_args()

    graphs = [parse_g6(l)[1] for l in open(args.g6) if l.strip()]
    print(f"Parsed {len(graphs)} base graphs (+complements = {2 * len(graphs)} colorings)")

    t0 = time.time()
    unsat = sat = other = 0
    sat_ids = []
    derived = []
    cid = 0
    for adj in graphs:
        for A in (adj, complement(adj, N)):
            red = four_cliques(A, N)
            blue = four_cliques(complement(A, N), N)
            derived.append((len(red), len(blue)))
            with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as tf:
                tf.write(cnf(red, blue)); path = tf.name
            rc = subprocess.run(["kissat", "-q", path], capture_output=True).returncode
            os.unlink(path)
            if rc == 20:
                unsat += 1
            elif rc == 10:
                sat += 1; sat_ids.append(cid)
            else:
                other += 1
            cid += 1
    dt = time.time() - t0

    # cross-check re-derived K4 counts against the repo's binary
    binset = []
    with open(args.bin, "rb") as f:
        nb_ = struct.unpack("<I", f.read(4))[0]
        for _ in range(nb_):
            nr, nbb = struct.unpack("<II", f.read(8)); f.read(8 * nr); f.read(8 * nbb)
            binset.append((nr, nbb))

    print("\n============= FULL PROVENANCE VERIFICATION =============")
    print(f"Colorings (raw graphs + complements): {cid}")
    print(f"UNSAT (no 1-vertex extension)       : {unsat}")
    print(f"SAT  (would mean R(5,5)>=44)         : {sat} {sat_ids if sat_ids else ''}")
    print(f"OTHER/error                          : {other}")
    print(f"Re-derived (nr,nb) == repo .bin?     : {sorted(derived) == sorted(binset)}")
    print(f"Total red K4 (mine / repo)           : {sum(r for r,_ in derived)} / {sum(r for r,_ in binset)}")
    print(f"Time                                 : {dt:.1f}s")
    ok = (unsat == cid == 656 and sat == 0 and other == 0 and sorted(derived) == sorted(binset))
    print(f"VERDICT                              : {'CONFIRMED from source' if ok else 'DISCREPANCY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
