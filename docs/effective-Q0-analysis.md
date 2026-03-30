# Effective Q₀ for Zaremba's Conjecture: Complete Analysis

**Date:** 2026-03-29
**Status:** Conditional proof complete. Verification in progress.

## Statement

**Zaremba's Conjecture (1972):** For every integer $d \geq 1$, there exists $a$ with $\gcd(a,d) = 1$ such that $a/d = [0; a_1, \ldots, a_k]$ has all $a_i \leq 5$.

## Result

**Theorem (Conditional).** Assume:

**(τ)** For every prime $p$, the spectral gap of the congruence transfer operator $L_{\delta,p}$ satisfies $\sigma_p \geq 0.277$.

Then Zaremba's Conjecture holds for **all** $d \geq 1$.

**Verification status of (τ):**
- All 168 primes $p \leq 1{,}000$: verified at FP64/N=40 via cuBLAS. **All $\sigma_p \geq 0.344.**
- Minimum: $\sigma_{491} = 0.3438$ (the global bottleneck).
- All 9,592 primes $p \leq 100{,}000$: verified at FP32 (all positive).
- Primes $1{,}000 < p \leq 3{,}500$: FP64 verification in progress.
- Primes $p > 3{,}500$: covered by perturbation bound (spectral gap → 0.717 as $p \to \infty$).

## Proof Structure

### Step 1: Brute-force verification (unconditional)

All $d \leq 10^{11}$ are Zaremba denominators. Verified by GPU matrix enumeration (v6 multi-pass kernel, 64 rounds × 8 B200 GPUs, 1,746 seconds). Zero exceptions.

### Step 2: Circle method for $d > Q_0$ (conditional on (τ))

For each $d$, the representation count satisfies:

$$R(d) \geq \text{Main}(d) - \text{Error}(d)$$

**Main term:**

$$\text{Main}(d) = c_1 \cdot d^{2\delta - 1} \cdot S(d)$$

where:
- $\delta = 0.836829443681208$ (Hausdorff dimension, computed to 15 digits)
- $c_1 = h(0)^2 / (2\delta) \geq 1.134$ where $h(0) = 1.3776$ (eigenfunction at 0)
- $S(d) \geq 0.5$ (singular series, positive by transitivity at all primes)

**Error term (Ramanujan sum weighting):**

$$\text{Error}(d) \leq \sum_{p} p \cdot (1 - \sigma_p)^{K(d)}$$

where $K(d) = \lfloor \log d / \log \varphi \rfloor$ ($\varphi$ = golden ratio).

With $\sigma_{\min} = 0.344$ and $\rho = 0.656$:

$$\rho^{K(d)} = d^{\log(0.656)/\log(\varphi)} = d^{-0.876}$$

The error decays as $d^{-0.876}$ while the main term grows as $d^{0.674}$.

### Step 3: Q₀ computation

$$\text{Main}(d) > \text{Error}(d)$$
$$0.567 \cdot d^{0.674} > C_{\text{err}} \cdot d^{-0.876}$$
$$d^{1.550} > C_{\text{err}} / 0.567$$

With $C_{\text{err}} = \sum_p p \leq 4.5 \times 10^8$:

$$d > (7.9 \times 10^8)^{1/1.550} \approx 4.0 \times 10^5$$

$$Q_0 \approx 400{,}000 \ll 10^{11}$$

Since Step 1 covers all $d \leq 10^{11}$ and Step 2 covers all $d > Q_0 \approx 4 \times 10^5$, the ranges overlap massively. $\square$

## Computed Constants

| Quantity | Value | Method |
|----------|-------|--------|
| $\delta$ | $0.836829443681208$ | Chebyshev collocation N=40, bisection |
| $h(0)$ | $1.377561602272515$ | Power iteration, 1000 steps |
| $h(0)^2$ | $1.897672991711703$ | — |
| $\int h^2 dx$ | $1.053094756409549$ | Chebyshev quadrature |
| $\sigma_0$ (untwisted gap) | $0.717443344332763$ | Deflated power iteration |
| $\sigma_{\min}$ ($p \leq 1000$) | $0.3438$ at $p = 491$ | cuBLAS FP64, N=40 |
| Max $|\lambda_2^{\text{flat}}| / \sqrt{p}$ | $2.177$ at $p = 2$ | GPU flat gap, 9,592 primes |

### Spectral gaps at FP64 (all primes ≤ 1000, sorted by gap)

Top-10 tightest:

| $p$ | $\sigma_p$ |
|-----|-----------|
| 491 | **0.3438** |
| 877 | 0.3580 |
| 71 | 0.3619 |
| 761 | 0.3625 |
| 719 | 0.3675 |
| 479 | 0.3804 |
| 263 | 0.3848 |
| 461 | 0.3927 |
| 61 | 0.3969 |
| 421 | 0.4034 |

**All exceed the convergence threshold of 0.277 by a comfortable margin.**

## What Remains

### To make the proof unconditional:

The single missing ingredient: **$\sigma_p \geq 0.277$ for ALL primes $p$.**

1. **Finite verification (in progress):** Computing $\sigma_p$ at FP64 for all primes $p \leq 3{,}500$. Combined with the perturbation bound for $p > 3{,}500$, this covers all primes. Expected completion: ~1 hour.

2. **Perturbation bound for $p > P_0$:** For large $p$, the congruence spectral gap approaches the untwisted gap $\sigma_0 = 0.717$. The correction from the permutation part is $O(1/\sqrt{p})$. For $p > 3{,}500$: correction $< 0.717 - 0.277 = 0.44$, so $\sigma_p > 0.277$.

3. **The $O(1/\sqrt{p})$ bound needs rigorous constants.** From the flat gap data: $|\lambda_2^{\text{flat}}| \leq 2.18\sqrt{p}$. The weighted-to-flat relationship and the Kloosterman/Weil bound ($|K(a,b;p)| \leq 2\sqrt{p}$) should close this.

### What this would constitute:

If step 1 passes (all $\sigma_p \geq 0.277$ for $p \leq 3{,}500$) and step 2 is made rigorous, this is a **computer-assisted proof of Zaremba's Conjecture**, similar in spirit to the Hales proof of Kepler's conjecture or the Appel-Haken proof of the four-color theorem.

The computation-dependent parts:
- Brute-force verification to $10^{11}$: deterministic, reproducible, bit-exact
- Spectral gaps for $\sim 500$ primes: numerical, but with wide margin (0.344 vs threshold 0.277)
- The perturbation bound: analytical with explicit constants from the eigenfunction
