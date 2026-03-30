# Regulator Computation Notes

## Status: Formula correct, regulator algorithm needs fix

The class number formula `h(d) = round(sqrt(d) * L(1,chi_d) / (2*R(d)))` gives
**exact results** when using PARI's `quadregulator()` values. Tested on 1000
fundamental discriminants: 0 errors.

The Euler product with 9592 primes (up to 100003) gives L(1,chi_d) accurately
enough for rounding to the correct integer h(d).

## The Regulator Bug

The continued fraction of √d gives a unit ε^k where k varies (1-6× depending
on d). The CF finds the fundamental solution to the Pell equation x²-dy²=±1,
but the fundamental UNIT of Q(√d) can be smaller.

Specifically:
- For d ≡ 1 mod 4: O_K = Z[(1+√d)/2], and the fundamental unit may be
  (x+y√d)/2 rather than x+y√d
- For d = 4m: the CF of √m gives the Pell solution, but the fundamental
  unit may come from x²-my²=-1 (half-period) when it exists
- The ratio R_cf / R_true varies: 1, 2, 3, 4, 5, or 6 depending on d

## Correct Algorithm

PARI uses the **infrastructure algorithm** (Shanks):
1. Represent the ideal class as (a, b+√d)
2. Track the reduced ideal through the CF
3. Detect when the ideal returns to the principal class
4. The regulator is the accumulated log at that point

This is equivalent to computing in the **class group** of the real quadratic
order, not just solving Pell's equation.

Reference: Cohen, "A Course in Computational Algebraic Number Theory", §5.7.

## GPU Implementation Plan

The infrastructure algorithm has the same O(√d) complexity as the CF.
Each step is integer arithmetic (no FP needed until the final log).
The key difference from naive CF: track (a, b) of the ideal, not just
(m, D) of the CF state.

The algorithm is equally parallelizable: one thread per discriminant,
same warp divergence characteristics.
