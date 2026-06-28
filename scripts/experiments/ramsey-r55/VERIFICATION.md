# Independent Verification — Ramsey R(5,5) single-vertex extension

This documents an independent, GPU-free reproduction of the
"656/656 K₄₂ colorings UNSAT" result, plus a careful statement of what that
result does and does not establish.

## TL;DR

The computational claim **reproduces exactly** and is now **formally certified**.
But it confirms a logically *narrow* fact — the known K₄₂ graphs do not extend by
one vertex — which is **consistent with, but not a proof of, R(5,5) = 43**.

## What was verified (three independent layers)

| Layer | Removes trust in | Tool | Result |
|---|---|---|---|
| 1. Re-solve from `mckay_k42_all.bin` | the CUDA subset-checker | `kissat` | 656/656 UNSAT |
| 2. Machine-check every refutation | the SAT solver itself | `drat-trim` | 656/656 VERIFIED |
| 3. Re-derive from raw McKay graphs | this repo's preprocessing | `parse g6` + `kissat` | 656/656 UNSAT |

Layer 3 also cross-checks the re-derived K₄ structure against the committed
binary: the per-coloring `(red_K4, blue_K4)` multiset matches exactly, with
**758,044** red K₄ masks on each side. So the data in `mckay_k42_all.bin` is
provably faithful to McKay's published `r55_42some.g6`.

Each extension instance is a 42-variable SAT problem: one 4-literal clause per
red K₄ (forbidding an all-red extension → red K₅) and per blue K₄ (forbidding
all-blue → blue K₅). These are trivial; the entire run is a few seconds on a
laptop CPU — confirming the 8×B200 hardware was never required for this step.

## Reproduce it

```bash
brew install kissat                          # the only hard dependency

# Layer 1 (fast, self-contained, ~4s):
python3 verify_independent.py

# Layers 1+2 (machine-checked certificates; needs drat-trim built):
python3 verify_independent.py --drat /path/to/drat-trim

# Layer 3 (full provenance from McKay's source graphs):
#   curl -fsSLO https://users.cecs.anu.edu.au/~bdm/data/r55_42some.g6
python3 provenance_from_g6.py r55_42some.g6
```

## What this establishes

For each of the **656 known** Ramsey(5,5,42) colorings, there is **no** way to
add a 43rd vertex without creating a monochromatic K₅.

## What this does NOT establish

This is **not** a proof that R(5,5) = 43. Two gaps:

1. **Completeness is unproven.** "656/656 don't extend" only implies R(5,5)=43
   if the 656 are *all* K₄₂ Ramsey(5,5) graphs. That set is conjectured but not
   proven complete (Lieby–McKay, unpublished 2014, give supporting evidence).
2. **Single-vertex extension is a weak test.** A 43-vertex Ramsey graph need not
   arise by adding one vertex to a *known* 42-vertex graph. A real proof
   requires showing no valid K₄₃ coloring exists at all — the direct K₄₃ SAT,
   which remains **intractable** here (903 vars, ~1.9M clauses; see
   `../../../data/ramsey-r55/README.txt` and `DESIGN.md`).

The honest framing: this is **strong corroboration of the long-standing
conjecture** R(5,5) = 43, fully reproducible and certified — not new evidence
that materially narrows the open problem, and not a proof. Current published
bounds remain **43 ≤ R(5,5) ≤ 46** (Angeltveit–McKay, arXiv:2409.15709, 2024).
