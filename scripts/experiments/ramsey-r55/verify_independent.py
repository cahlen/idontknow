#!/usr/bin/env python3
"""
Independent, GPU-free verification of the "656/656 K42 colorings UNSAT" result.

This reproduces the single-vertex extension result WITHOUT the CUDA kernels:
it decodes mckay_k42_all.bin, re-encodes each extension problem as SAT, and
solves it with an external solver (kissat). With --drat, each UNSAT result is
emitted as a DRAT proof and machine-checked with drat-trim, upgrading the
result from "a solver reported UNSAT" to a formally verified refutation.

What this establishes:
  For each of the 656 known K42 Ramsey(5,5) colorings, there is NO way to add a
  43rd vertex without creating a monochromatic K5. (Each instance is a tiny
  42-variable SAT: one 4-literal clause per red/blue K4 in the existing graph.)

What this does NOT establish:
  R(5,5) = 43. This only shows the *known* K42 graphs do not extend by one
  vertex. It is consistent with R(5,5) = 43 but is not a proof: it depends on
  the (unproven) completeness of the 656-graph set, and the direct K43 search
  remains intractable. See VERIFICATION.md.

Requires: kissat on PATH (`brew install kissat`). Optional: drat-trim for --drat.
Runs in a few seconds on a laptop CPU.

Usage:
  python3 verify_independent.py [--drat PATH_TO_DRAT_TRIM] [--bin mckay_k42_all.bin]
"""
import argparse, os, struct, subprocess, sys, tempfile, time

N = 42  # vertices in the base coloring; the 43rd is the one we try to add


def cnf(red_k4, blue_k4):
    """Extension SAT: var v+1 == 'edge from new vertex to v is red'.
    A red K4 in the base graph must not be all-red to the new vertex (else red K5);
    a blue K4 must not be all-blue (else blue K5)."""
    lines = []
    for m in red_k4:
        lines.append(" ".join(str(-(v + 1)) for v in range(N) if (m >> v) & 1) + " 0")
    for m in blue_k4:
        lines.append(" ".join(str(v + 1) for v in range(N) if (m >> v) & 1) + " 0")
    return f"p cnf {N} {len(lines)}\n" + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "mckay_k42_all.bin"))
    ap.add_argument("--drat", default=None, help="path to drat-trim to machine-check every UNSAT proof")
    args = ap.parse_args()

    t0 = time.time()
    unsat = sat = other = cert_ok = cert_run = 0
    sat_ids = []
    with open(args.bin, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        for i in range(n):
            nr, nb = struct.unpack("<II", f.read(8))
            red = struct.unpack("<%dQ" % nr, f.read(8 * nr))
            blue = struct.unpack("<%dQ" % nb, f.read(8 * nb))
            with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as tf:
                tf.write(cnf(red, blue))
                path = tf.name
            if args.drat:
                proof = path + ".drat"
                rc = subprocess.run(["kissat", "-q", "--no-binary", path, proof],
                                    capture_output=True).returncode
                if rc == 20:
                    chk = subprocess.run([args.drat, path, proof], capture_output=True, text=True)
                    cert_run += 1
                    if "s VERIFIED" in chk.stdout:
                        cert_ok += 1
                os.unlink(proof)
            else:
                rc = subprocess.run(["kissat", "-q", path], capture_output=True).returncode
            os.unlink(path)
            if rc == 20:
                unsat += 1
            elif rc == 10:
                sat += 1; sat_ids.append(i)
            else:
                other += 1
            if (i + 1) % 100 == 0:
                print(f"  ...{i + 1}/{n}", flush=True)

    dt = time.time() - t0
    print("\n=============== INDEPENDENT VERIFICATION ===============")
    print(f"Colorings checked        : {n}")
    print(f"UNSAT (no 1-vtx extension): {unsat}")
    print(f"SAT  (would beat R55>=43) : {sat} {sat_ids if sat_ids else ''}")
    print(f"OTHER/error              : {other}")
    if args.drat:
        print(f"DRAT certs machine-checked: {cert_ok}/{cert_run} VERIFIED")
    print(f"Solver                   : kissat (independent of the CUDA kernels)")
    print(f"Time                     : {dt:.1f}s")
    ok = (unsat == n and n == 656 and other == 0 and sat == 0)
    print(f"VERDICT                  : {'CONFIRMED 656/656 UNSAT' if ok else 'DISCREPANCY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
