#!/usr/bin/env python3
"""
Enumerate R(4,5)-good graphs for the R(5,5) ≤ 45 structural attack.

A graph G on n vertices is R(4,5)-good if:
  - omega(G) < 4  (no clique of size 4, i.e., K4-free)
  - alpha(G) < 5  (no independent set of size 5)

For proving R(5,5) ≤ 45 via Angeltveit-McKay's excess identity:
  - Vertex neighborhoods must be R(4,5)-good
  - Edge count must exceed a threshold e_0 (from excess identity)

This script uses SMS (SAT Modulo Symmetries) to enumerate all such
graphs up to isomorphism.

Usage:
  python3 enumerate_r45.py <n> <min_edges> [max_edges]

Example:
  python3 enumerate_r45.py 23 118 122   # R(4,5,23) with 118 ≤ e ≤ 122
  python3 enumerate_r45.py 21 90 107    # R(4,5,21) with 90 ≤ e ≤ 107
"""

import sys
import os
import time
import subprocess

def count_graphs_sms(n, min_edges, max_edges, timeout=3600):
    """Use SMS to enumerate R(4,5,n) graphs with min_edges ≤ e ≤ max_edges."""

    results = []
    total = 0

    for e in range(min_edges, max_edges + 1):
        cmd = [
            "python3", "-m", "pysms.graph_builder",
            "--vertices", str(n),
            "--omega-upp", "4",     # K4-free (clique < 4)
            "--alpha-upp", "5",     # independent set < 5
            "--num-edges-low", str(e),
            "--num-edges-upp", str(e),
            "--all-graphs",
            "--hide-graphs",
        ]

        print(f"  e={e}: ", end="", flush=True)
        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            output = proc.stdout + proc.stderr

            # Count graphs: each graph output line starts with [
            graph_lines = [l for l in output.split('\n') if l.startswith('[')]
            count = len(graph_lines)

            # Get time
            time_match = [l for l in output.split('\n') if 'Total time:' in l]
            sms_time = time_match[-1].split(':')[-1].strip() if time_match else "?"

            elapsed = time.time() - t0
            print(f"{count} graphs ({elapsed:.1f}s)")
            results.append((e, count))
            total += count

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"TIMEOUT after {elapsed:.0f}s")
            results.append((e, -1))

    return results, total


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <n> <min_edges> [max_edges]")
        print(f"  Enumerates R(4,5,n) graphs with edge count in [min, max]")
        sys.exit(1)

    n = int(sys.argv[1])
    min_e = int(sys.argv[2])
    max_e = int(sys.argv[3]) if len(sys.argv) > 3 else min_e
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 7200

    max_possible = n * (n - 1) // 2

    print(f"========================================")
    print(f"R(4,5,{n}) Graph Enumeration")
    print(f"Edge range: [{min_e}, {max_e}]")
    print(f"Max possible edges: {max_possible}")
    print(f"Timeout per edge count: {timeout}s")
    print(f"========================================\n")

    # Quick feasibility check
    # R(4,5) = 25, so n must be ≤ 24
    if n >= 25:
        print(f"ERROR: R(4,5)=25, no R(4,5)-good graphs on {n} ≥ 25 vertices")
        sys.exit(1)

    t0 = time.time()
    results, total = count_graphs_sms(n, min_e, max_e, timeout)
    elapsed = time.time() - t0

    print(f"\n========================================")
    print(f"RESULTS: R(4,5,{n}) graphs")
    print(f"========================================")
    print(f"{'Edges':>6} {'Count':>10}")
    print(f"{'-'*6} {'-'*10}")
    for e, count in results:
        status = str(count) if count >= 0 else "TIMEOUT"
        print(f"{e:>6} {status:>10}")
    print(f"{'-'*6} {'-'*10}")
    print(f"{'TOTAL':>6} {total:>10}")
    print(f"\nTime: {elapsed:.1f}s")
    print(f"========================================")

    # Save results
    os.makedirs("scripts/experiments/ramsey-r55/results", exist_ok=True)
    outpath = f"scripts/experiments/ramsey-r55/results/R45_{n}_{min_e}_{max_e}.csv"
    with open(outpath, 'w') as f:
        f.write("edges,count\n")
        for e, count in results:
            f.write(f"{e},{count}\n")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
