# Zaremba's Conjecture for All Primes: The Exact Circle Method Argument

## Statement

**Goal:** For every prime p, there exists a with gcd(a,p) = 1 such that all CF partial quotients of a/p are at most 5.

## Available Data

| Quantity | Value | Source |
|----------|-------|--------|
| delta = dim_H(E_5) | 0.836829443681208 | Transfer operator (15 digits) |
| 2*delta | 1.673658887362416 | |
| 2*delta - 1 | 0.673658887362416 | The critical exponent |
| Lyapunov exponent lambda | 0.763 | MC + Ruelle formula h_mu/(2*delta) |
| PS entropy h_mu | 1.280 | From branch probabilities |
| Spectral gap (untwisted) | 0.717 | lambda_1/lambda_0 = 0.283 |
| Spectral gap sigma_p (primes p <= 1999) | >= 0.237 | Computed |
| Transitivity | Gamma_{1,...,5} acts on P^1(F_p) transitively for ALL p | Proved (Dickson) |
| Brute-force | All primes up to 10^7 are Zaremba | v4 kernel |
| R(d) empirical | R(d) >= 1 for all d <= 10^6 | Representation counter |

### Patterson-Sullivan branch probabilities for A = {1,...,5}

| Branch a | Weight (PS measure) |
|----------|-------------------|
| a = 1 | 0.5323 |
| a = 2 | 0.2138 |
| a = 3 | 0.1199 |
| a = 4 | 0.0783 |
| a = 5 | 0.0558 |

Entropy: h_mu = 1.280. Lyapunov exponent: lambda = h_mu / (2*delta) = 0.765 (Ruelle), confirmed by Monte Carlo at 0.763.

---

## 1. The Counting Function R_N(p) for Prime d = p

Define:

    R_N(p) = #{gamma in Gamma_A : ||gamma|| <= N, q(gamma) = p}

where q(gamma) is the bottom-right entry of the matrix gamma = product of [a_i, 1; 1, 0] generators with a_i in {1,...,5}. If R_N(p) > 0 for any N, then p is a Zaremba denominator.

The CF depth for matrices with ||gamma|| <= N is K ~ log(N) / lambda where lambda = 0.763 is the Lyapunov exponent.

### Why primes simplify the circle method

For general d, the Farey dissection at level Q involves all moduli q <= Q, and the major arcs are at all q dividing d. For composite d with many divisors, this creates a complex web of major arcs.

For d = p prime, the only divisors are 1 and p itself. So:
- **q = 1:** One major arc around alpha = 0
- **q = p:** Arcs around a/p for a = 1, ..., p-1

There are no intermediate moduli. This eliminates the combinatorial complexity of the general BK argument.

---

## 2. Explicit Main Term: Main(p)

By the circle method (following BK14, Section 4):

    R_N(p) = Main(p) + Error_major(p) + Error_minor(p)

### Main(p) explicitly

    Main(p) = C_delta * N^{2*delta} / (p + 1) * S(p)

where:

**C_delta** is the Patterson-Sullivan constant, determined by the leading eigenfunction of L_delta. It depends only on the alphabet {1,...,5} and satisfies C_delta > 0. From the transfer operator, C_delta is of order 1.

**N^{2*delta}** counts the total semigroup elements: |{gamma in Gamma_A : ||gamma|| <= N}| ~ C * N^{2*delta}.

**1/(p+1)** comes from equidistribution on P^1(F_p): each of the p+1 projective points receives an equal share. The condition q(gamma) = p corresponds to the point [0:1] in P^1(F_p) (the "cusp at infinity"), which is one out of p+1 points.

**S(p)** is the singular series, correcting for the deviation from perfect equidistribution:

    S(p) = p^2 / (p^2 - 1)

This follows from: Gamma_{1,...,5} acts transitively on P^1(F_p) (proved for all primes), so the local factor is:

    sigma_p(local) = |orbit of cusp| / |expected size| = (p+1)/(p+1) * p^2/(p^2-1) = p^2/(p^2-1)

For small primes: S(2) = 4/3 = 1.333, S(3) = 9/8 = 1.125, S(5) = 25/24 = 1.042, S(101) = 1.000098.

**With N = p (natural scale):**

    Main(p) = C_delta * p^{2*delta} / (p+1) * p^2/(p^2-1)
            ~ C_delta * p^{2*delta - 1}
            = C_delta * p^{0.6737}

**This grows with p.** For C_delta ~ 1, Main(p) ~ p^{0.674}. For p = 100, Main ~ 22. For p = 10^6, Main ~ 2800.

---

## 3. Error_major(p): The Major Arc Error

The error from q = p comes from nontrivial representations of SL_2(F_p) in the spectral decomposition of the transfer operator.

### The spectral decomposition

The Hecke operator T_K on L^2(P^1(F_p)) acts as:

    T_K f(x) = sum_{gamma: length K} f(gamma.x)

It decomposes as T_K = lambda_0^K * P_triv + Error, where P_triv projects onto constant functions and the nontrivial part has operator norm bounded by the spectral gap:

    ||T_K - lambda_0^K * P_triv||_{op} <= lambda_0^K * (1 - sigma_p)^K

### Pointwise bound via Cauchy-Schwarz

The main term comes from P_triv applied to the indicator of [0:1]:

    P_triv(delta_{[0:1]}) = 1/(p+1)

The error on the indicator function satisfies:

    |Error at [0:1]| <= ||T_K - lambda_0^K * P_triv||_{op} * ||delta_{[0:1]}||_2

Since ||delta_{[0:1]}||_2 = sqrt(1/(p+1)) (in the L^2 norm on P^1(F_p) with counting measure normalized by 1/(p+1)):

Actually, more carefully: ||delta_{[0:1]}||_2 = 1 in unnormalized L^2, and the operator norm gives:

    |Error at [0:1]| <= sqrt(p+1) * lambda_0^K * (1-sigma_p)^K

(The sqrt(p+1) comes from the L^2 -> L^inf bound on P^1(F_p).)

### Error/Main ratio

    Error_major(p) / Main(p) = sqrt(p+1) * (1-sigma_p)^K / S(p)
                              ~ sqrt(p) * (1-sigma_p)^K

With K = log(N)/lambda and N = p^alpha:

    Error_major / Main ~ p^{1/2 + alpha * log(1-sigma_p) / lambda}
                       = p^{1/2 - alpha * 0.2705 / 0.763}
                       = p^{1/2 - 0.3547 * alpha}

**For alpha = 1 (N = p):**

    Exponent = 0.5 - 0.355 = 0.145 > 0  --> Error_major grows (FAILS)

**For alpha = 2 (N = p^2):**

    Exponent = 0.5 - 0.709 = -0.209 < 0  --> Error_major shrinks (PASSES)

**Critical alpha:**

    alpha_crit = 0.5 / 0.3547 = 1.41

So for N = p^{1.41} or larger, the major arc error is smaller than the main term.

### Numerical evaluation (alpha = 2, N = p^2)

| p | K = 2*log(p)/0.763 | Main ~ p^{2.347} | Err_major ~ sqrt(p)*(0.763)^K | Ratio |
|---|---------------------|-------------------|-------------------------------|-------|
| 5 | 4.2 | 36 | 0.78 | 0.022 |
| 101 | 12.1 | 5.0e4 | 0.38 | 7.6e-6 |
| 1009 | 18.1 | 1.1e7 | 0.24 | 2.1e-8 |
| 10007 | 24.1 | 2.5e9 | 0.15 | 5.9e-11 |
| 100003 | 30.2 | 5.5e11 | 0.09 | 1.6e-13 |

**The major arc error is negligible for all p >= 5.**

---

## 4. Error_minor(p): The Minor Arc Problem

This is where the argument becomes incomplete. For N = p^alpha with alpha > 1, the Farey dissection at level Q = N = p^alpha includes all moduli q <= p^alpha. The major arcs are at q = 1 and q = p. Everything else is "minor arc" (in the divisor-of-d sense).

### What the minor arc contributes

For q coprime to p (i.e., every q in {2, 3, ..., p-1, p+1, ..., p^alpha}), the exponential sum at frequency a/q is:

    S_q(a) = sum_{gamma: ||gamma|| <= N} e(a * q(gamma) / q)

The spectral gap at modulus q gives:

    |S_q(a)| <= C * N^{2*delta} * (1-sigma_q)^K

### The total minor arc contribution to R(p)

    |Error_minor(p)| = |sum_{q=2, q != p}^{Q} sum_{gcd(a,q)=1} S_q(a) * e(-a*p/q)|

**Naive bound (no cancellation in q):**

    |Error_minor| <= sum_{q=2}^{Q} phi(q) * |S_q| / N^{2*delta}  (relative to main)
                  <= sum_{q=2}^{p^alpha} phi(q) * (1-sigma_min)^K

With sigma_min = 0.237, K = 2*alpha*log(p)/0.763, and sum phi(q) ~ 3*Q^2/pi^2:

    |Error_minor| <= (3/pi^2) * p^{2*alpha} * p^{-0.709*alpha}
                   = O(p^{1.291*alpha})

The main term is p^{2*delta*alpha - 1} = p^{1.674*alpha - 1}. For error < main:

    1.291*alpha < 1.674*alpha - 1
    0.383*alpha > 1
    alpha > 2.61

**With alpha = 3 (N = p^3):** the naive minor arc bound gives error exponent 3.873 vs main exponent 4.021. Marginal.

### Better bound (with cancellation in a)

The sum over a in sum_{gcd(a,q)=1} e(-a*p/q) is the Ramanujan sum c_q(p):

    c_q(p) = sum_{gcd(a,q)=1} e(a*p/q) = mu(q/gcd(q,p)) * phi(q) / phi(q/gcd(q,p))

For q coprime to p: c_q(p) = mu(q) * phi(q)/phi(q). Wait, for gcd(q,p) = 1:

    c_q(p) = mu(q)

if q is squarefree, and 0 otherwise. So the Ramanujan sum provides ENORMOUS cancellation: instead of phi(q) terms of size 1, we get |c_q(p)| <= 1 for squarefree q and 0 for non-squarefree q.

With this cancellation:

    |Error_minor(p)| <= sum_{q=2, squarefree}^{Q} |S_q| * 1 / N^{2*delta}
                     <= Q * (1-sigma_min)^K
                     = p^alpha * p^{-0.3547*alpha}
                     = p^{0.6453*alpha}

For error < main (p^{1.674*alpha - 1}):

    0.6453*alpha < 1.674*alpha - 1
    1.029*alpha > 1
    alpha > 0.972

**With Ramanujan sum cancellation, alpha > 1 suffices!** Even N = p^{1.1} would work.

### The catch

The Ramanujan sum argument requires that the spectral contribution S_q(a) at each (q,a) pair is bounded independently, and that the Ramanujan sum c_q(p) = mu(q) correctly captures the phase. This is essentially what BK14 proves — but making it rigorous requires:

1. A UNIFORM bound on S_q(a) for all q up to Q (not just the spectral gap at each q, but the full transfer operator bound at all q simultaneously)
2. Handling the cross-terms between different q values
3. Bounding the tail where q > some Q_0 where we have spectral data

Point 1 is Property (tau) — which you have numerically but not with an explicit constant.

---

## 5. What Spectral Gap sigma_p Do We Need?

### Summary of thresholds

| Argument | alpha (N = p^alpha) | sigma needed | Your sigma | Status |
|----------|-----------------------|--------------|------------|--------|
| Single scale, major arc only | 1 | > 0.319 | 0.237 | FAIL |
| Double scale, major arc only | 2 | > 0.175 | 0.237 | **PASS** |
| Multi-scale with Ramanujan cancellation | 1.01 | > 0.175 | 0.237 | **PASS** |
| Multi-scale, naive minor arc | 3 | > 0.175 | 0.237 | **PASS** (marginal) |

**The key threshold is sigma_p > 1 - exp(-lambda/(2*alpha)) where lambda = 0.763.**

For alpha = 2: sigma > 1 - exp(-0.763/4) = 0.175. **Your gap of 0.237 exceeds this comfortably.**

The problem is not the spectral gap — it is making the minor arc bound rigorous.

---

## 6. For Which Primes Can We Prove R_N(p) > 0 Rigorously?

### Currently provable

**All primes up to 10^7:** By brute-force computation (your v4 kernel).

### Provable with additional work

**All primes p, conditional on:**

(a) Property (tau) with an explicit constant: i.e., sigma_q >= c for all squarefree q, where c > 0 is effective. Your data shows c >= 0.237 for q with prime factors <= 1999, but you need this for ALL q.

(b) The Ramanujan sum cancellation argument made rigorous: this is the substance of BK14 Sections 5-7. The key input is the sum-product theorem, which provides cancellation in the exponential sums beyond what the spectral gap alone gives.

If (a) and (b) can be made effective, the argument gives: R(p) > 0 for all p > P_0 where P_0 depends on the effective constants.

**Combining with brute force:** If P_0 < 10^7, then Zaremba holds for ALL primes. If P_0 is larger, extend the brute force.

---

## 7. The Literature: What Is Already Proved

### Density-1 (Bourgain-Kontorovich 2014 + Huang 2015)

**Theorem:** The set of positive integers d for which Zaremba's conjecture fails has density 0. The alphabet A = {1,...,5} suffices.

This implies: almost all primes are Zaremba denominators.

### Partial quotients O(sqrt(log q)) for primes (Kowalski-Maynard direction)

The most recent results in this direction (as of early 2026) establish that for all sufficiently large primes q, there exists a with CF partial quotients bounded by O(sqrt(log q)). This uses sieve methods and moment computations for Kloosterman sums, NOT the spectral gap / transfer operator approach.

**Can we sharpen O(sqrt(log q)) to a constant 5?**

No, not by those methods. The sieve approach inherently loses a growing factor because:
1. It counts CF fractions by their partial quotient statistics, averaging over many choices
2. The bound comes from variance estimates that scale with the number of CF digits
3. A fixed bound of 5 requires EXACT control of the CF structure, not just averaged bounds

The transfer operator / spectral gap approach IS the right method for a constant bound, but requires effective minor arc estimates.

### The Magee-Oh-Winter spectral gap (2019)

[MOW19] proves explicit spectral gaps for congruence quotients of thin groups, but the constants depend on the group and are not sharp enough for Zaremba.

### What would be a new result

If you could show:

    R(p) > 0 for all primes p > P_0 (with P_0 effective and < 10^7)

this would be a genuine theorem, combining:
- The BK circle method framework
- Your computed spectral gaps (making Property tau effective for primes)
- The Ramanujan sum cancellation (making the minor arc rigorous for prime moduli)
- Brute-force verification below P_0

The key new input would be an effective bound on the minor arc for prime d, possibly using the specific structure of P^1(F_p) that makes the representation theory cleaner.

---

## 8. Concrete Next Steps

### (1) Compute sigma_p for all primes up to 10,000

Your current data covers p <= 1999. Extending to p <= 10,000 (about 1,229 primes) would:
- Confirm sigma_p >= 0.237 persists (or find the true minimum)
- Provide enough data to extrapolate to all primes

### (2) Compute the Lyapunov exponent to high precision

The value lambda = 0.763 comes from Monte Carlo. You can compute it exactly from:

    lambda = -d/ds log(rho(L_s))|_{s=delta} = h_mu / (2*delta)

Using your Chebyshev collocation code, compute rho(L_s) for s near delta and differentiate numerically. This gives lambda to 15 digits.

### (3) Try the direct exponential sum for moderate primes

For primes p in [10^7, 10^8], use your GPU to directly compute:

    F_N(a/p) = sum_{gamma: ||gamma|| <= N} e(a * q(gamma) / p)

for all a = 1, ..., p-1, and check that sum_a |F_N(a/p)|^2 is small compared to the main term. This would bound the minor arc NUMERICALLY for each specific p.

### (4) Write down the prime-specific Ramanujan cancellation

For d = p prime, the minor arc simplifies because c_q(p) = mu(q) for all squarefree q coprime to p. Write out the full argument:

    Error_minor(p) = sum_{q squarefree, gcd(q,p)=1} mu(q) * S_q(p/q or whatever)

and show this has better cancellation than the general case. This could be a new result for prime moduli specifically.

### (5) Extend brute force to 10^9 or beyond

Your v4 kernel can handle this. Every prime verified directly is one fewer that needs a theoretical argument.
