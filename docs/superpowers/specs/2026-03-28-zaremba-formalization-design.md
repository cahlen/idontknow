# Zaremba's Conjecture Formalization — Pipeline-First Design

## Goal

Formalize Zaremba's Conjecture in Lean 4, starting with computationally verifiable small cases that validate the LLM proving pipeline, then growing into a structured mathematical library.

## Background

**Zaremba's Conjecture (1972):** There exists an absolute constant A such that for every positive integer d, there exists a positive integer a with gcd(a, d) = 1 whose continued fraction expansion a/d = [0; a₁, a₂, ..., aₖ] has all partial quotients aᵢ ≤ A. The conjectured optimal bound is A = 5.

**Known results:**
- Bourgain-Kontorovich (2014): True for a density-1 subset of positive integers (with A = 50)
- Huang (2015): Improved the density bound
- Computationally verified for all d up to large bounds with A = 5
- For small d (1-20), explicit witnesses are easily found

## Architecture

Single file: `lean4-proving/conjectures/zaremba.lean`

Four layers, built bottom-up:

### Layer 1 — Definitions

```
cfExpansion : ℕ → ℕ → List ℕ
```
Compute the continued fraction partial quotients of a/d using the Euclidean algorithm. For a/d where a < d, this gives [0; a₁, a₂, ...] — we store only the aᵢ (dropping the leading 0).

```
AllBoundedBy : List ℕ → ℕ → Prop
```
Every element of the list is ≤ the bound.

```
ZarembaWitness (d A : ℕ) : Prop
```
∃ a, 0 < a ∧ Nat.Coprime a d ∧ AllBoundedBy (cfExpansion a d) A

All definitions are computable so `decide`/`native_decide` can discharge proofs for concrete values.

### Layer 2 — Small cases (pipeline targets)

For each d ∈ {1, 2, ..., 20}, a theorem:
```
theorem zaremba_d<N> : ZarembaWitness <N> 5 := by sorry
```

These are the targets the LLM prover attacks. Each has a concrete witness a that can be verified by computation. Examples:
- d=1: a=1 (trivial, CF is empty)
- d=2: a=1 (CF of 1/2 = [2], all ≤ 5)
- d=7: a=3 (CF of 3/7 = [2, 3], all ≤ 5)

The prover should fill each `sorry` with something like:
```
exact ⟨witness, by decide, by decide, by decide⟩
```
or use `native_decide` for larger d.

### Layer 3 — The conjecture

```
theorem zaremba_conjecture : ∀ d : ℕ, d > 0 → ZarembaWitness d 5 := by sorry
```

North star statement. Not expected to be proved by the pipeline — this is the open conjecture.

### Layer 4 — Growth hooks (for future work)

Stated with `sorry`, not targeted by the pipeline yet:

```
theorem bourgain_kontorovich :
  -- For all but finitely many d, ZarembaWitness d 50
  sorry
```

```
-- Future: connect cfExpansion to Mathlib's GenContFract
-- Future: convergent recurrence relations
-- Future: prove d=1..1000 programmatically
```

## Dependencies

Mathlib imports (lightweight):
- `Mathlib.Data.Nat.GCD.Basic` — for `Nat.Coprime`
- `Mathlib.Data.List.Basic` — for list operations
- `Mathlib.Tactic` — for `decide`, `omega`, etc.

No heavy Mathlib CF machinery in Layer 1-3. Layer 4 will introduce `Mathlib.Algebra.ContinuedFractions.*` when connecting to the existing library.

## Success criteria

1. All 20 small-case theorems compile when `sorry` is replaced with valid proofs
2. The LLM prover (`prover.py`) can fill at least some of the sorries automatically
3. `zaremba_conjecture` is correctly stated (even though it remains `sorry`)
4. File builds cleanly against Mathlib

## Growth path to structured library (Phase B)

Once pipeline is validated:
1. Extend small cases to d=1..100, then d=1..1000
2. Formalize the Bourgain-Kontorovich density argument
3. Connect `cfExpansion` to Mathlib's `GenContFract` via equivalence lemma
4. Add convergent recurrence properties
5. Explore computational approaches to larger d values using the cluster
