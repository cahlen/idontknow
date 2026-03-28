/-
  Continued Fractions — starter theorems for LLM-assisted proving.

  These are basic properties of simple continued fractions that
  can serve as test cases for the proving pipeline.
-/

import Mathlib.Data.Rat.Basic
import Mathlib.Tactic

/-- A simple continued fraction represented as a list of natural numbers -/
def SimpleCF := List Nat

/-- Evaluate a finite simple continued fraction to a rational number.
    [a₀, a₁, a₂, ...] = a₀ + 1/(a₁ + 1/(a₂ + ...)) -/
def SimpleCF.eval : SimpleCF → ℚ
  | [] => 0
  | [a] => a
  | a :: rest => a + 1 / (SimpleCF.eval rest)

/-- The convergents p_n/q_n of a continued fraction satisfy the recurrence:
    p_n = a_n * p_{n-1} + p_{n-2}
    q_n = a_n * q_{n-1} + q_{n-2}

    This is a fundamental property we want to verify. -/
theorem cf_single_eval : SimpleCF.eval [3] = 3 := by
  sorry

theorem cf_two_eval : SimpleCF.eval [1, 2] = (3 : ℚ) / 2 := by
  sorry

/-- For any continued fraction [a₀], eval returns a₀ -/
theorem cf_singleton (a : Nat) : SimpleCF.eval [a] = (a : ℚ) := by
  sorry

/-- The golden ratio φ has the simplest possible infinite continued fraction:
    φ = [1; 1, 1, 1, ...]

    We can verify finite approximations. -/
theorem golden_ratio_approx_1 : SimpleCF.eval [1, 1] = 2 := by
  sorry

theorem golden_ratio_approx_2 : SimpleCF.eval [1, 1, 1] = (3 : ℚ) / 2 := by
  sorry

/-- √2 = [1; 2, 2, 2, ...]
    Verify finite approximations. -/
theorem sqrt2_approx_1 : SimpleCF.eval [1, 2] = (3 : ℚ) / 2 := by
  sorry

theorem sqrt2_approx_2 : SimpleCF.eval [1, 2, 2] = (7 : ℚ) / 5 := by
  sorry

/-- e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]
    Verify the first convergent. -/
theorem e_approx_1 : SimpleCF.eval [2, 1] = 3 := by
  sorry

theorem e_approx_2 : SimpleCF.eval [2, 1, 2] = (8 : ℚ) / 3 := by
  sorry
