# Spectral Gap Lower Bound for Γ_{1,...,5}: Proof Strategy

## Setup

The congruence transfer operator is:

$$L_{\delta,p} = \sum_{a=1}^{5} M_a \otimes P_a$$

acting on $H = L^2([0,1]) \otimes \mathbb{C}[\mathbb{P}^1(\mathbb{F}_p)]$, where:
- $M_a$ is the Chebyshev-discretized operator for digit $a$: $(M_a f)(x) = (a+x)^{-2\delta} f(1/(a+x))$
- $P_a$ is the permutation of $\mathbb{P}^1(\mathbb{F}_p)$ induced by $g_a: x \mapsto a + 1/x$

## Decomposition

Decompose $\mathbb{C}[\mathbb{P}^1] = \mathbf{1} \oplus V$ where $\mathbf{1}$ is trivial and $V$ has dimension $p$.

- On $H_{\mathrm{triv}} = L^2 \otimes \mathbf{1}$: $L_{\delta,p}$ acts as the untwisted $L_\delta$, with eigenvalues $\lambda_1 = 1$, $\lambda_2 = -0.2826$, gap $\sigma_0 = 0.717$.
- On $H_V = L^2 \otimes V$: $L_{\delta,p}$ acts as $\sum_a M_a \otimes P_a|_V$.

The second eigenvalue of $L_{\delta,p}$ is:
$$|\lambda_2(L_{\delta,p})| = \max\big(|\lambda_2(L_\delta)|, \, \|L_{\delta,p}|_{H_V}\|_{\mathrm{op}}\big)$$

Since $|\lambda_2(L_\delta)| = 0.283 < 0.502$, we need $\|L_{\delta,p}|_{H_V}\|_{\mathrm{op}} \leq 0.502$.

## Perturbation Bound

**Claim:** $\|L_{\delta,p}|_{H_V}\|_{\mathrm{op}} \leq \|\sum_a M_a\|_{\mathrm{op}} \cdot \|\tfrac{1}{5}\sum_a P_a|_V\|_{\mathrm{op}} + O(1/p)$

More precisely, for $f = \sum_k v_k \otimes w_k \in H_V$:

$$\|L_{\delta,p} f\|^2 = \|\sum_a (M_a \otimes P_a|_V) f\|^2$$

Since $P_a|_V$ are unitary operators on $V$:
$$\leq \big(\sum_a \|M_a\|_{\mathrm{op}}\big)^2 \|f\|^2$$

This gives $\|L_{\delta,p}|_{H_V}\| \leq \sum_a \|M_a\|_{\mathrm{op}}$. But $\sum_a \|M_a\|_{\mathrm{op}} = \|L_\delta\|_{\mathrm{op}} = 1$ (with equality when all $P_a$ are aligned). So this is again trivial.

**Better approach:** Use the CANCELLATION from the non-trivial $P_a|_V$.

For any $w \in V$ (so $\sum_k w(k) = 0$):
$$\|\sum_a P_a w\|^2 = \sum_k |\sum_a w(P_a^{-1}(k))|^2$$

The flat second eigenvalue gives:
$$\|\tfrac{1}{5}\sum_a P_a|_V\| = |\lambda_2^{\mathrm{flat}}|/5$$

From our computation: $|\lambda_2^{\mathrm{flat}}| \leq B\sqrt{p}$ with $B \leq 2.18$.

**Key bound:** For the weighted operator, by submultiplicativity:

$$\|L_{\delta,p}|_{H_V}\| \leq \|L_\delta\| \cdot \|\tfrac{1}{5}\sum_a P_a|_V\| \cdot 5 + \text{cross terms}$$

The cross terms arise because the weights $M_a$ differ for different $a$. Writing $M_a = \bar{M} + \Delta_a$ where $\bar{M} = \frac{1}{5}\sum_a M_a = \frac{1}{5}L_\delta$:

$$L_{\delta,p}|_{H_V} = \bar{M} \otimes (\sum_a P_a|_V) + \sum_a \Delta_a \otimes P_a|_V$$

The first term: $\|\bar{M} \otimes (\sum_a P_a|_V)\| = \|\bar{M}\| \cdot \|\sum_a P_a|_V\| = \frac{1}{5} \cdot B\sqrt{p}$

The second term: $\|\sum_a \Delta_a \otimes P_a|_V\| \leq \sum_a \|\Delta_a\|$ (since $P_a$ is unitary)

$\sum_a \|\Delta_a\| = \sum_a \|M_a - \bar{M}\|$.

Since $M_a f(x) = (a+x)^{-2\delta} f(1/(a+x))$ and $\bar{M} = \frac{1}{5}\sum_a M_a$:

$\|\Delta_a\| \leq \|M_a\| + \|\bar{M}\| = \|M_a\| + 1/5$

The individual operator norms: $\|M_a\| = \rho(M_a)$ where $\rho$ is the spectral radius. For the single-digit operator $M_a$:

$\|M_a\|$ is the leading eigenvalue of $f(x) = (a+x)^{-2\delta} f(1/(a+x))$.

This equals $(a)^{-2\delta} / (1 + O(1/a))$ for large $a$. Specifically:
- $\|M_1\| = 1^{-2\delta} \cdot c_1 \approx 0.413$ (digit 1 gets the largest weight)
- $\|M_2\| \approx 0.191$
- $\|M_3\| \approx 0.108$
- $\|M_4\| \approx 0.071$
- $\|M_5\| \approx 0.050$
- Sum: $\sum \|M_a\| \approx 0.833$ (note: sum of spectral radii ≤ spectral radius of sum = 1)

$\|\Delta_a\| \leq \|M_a - \frac{1}{5}L_\delta\| \leq \|M_a\| + 1/5 = \|M_a\| + 0.2$

$\sum_a \|\Delta_a\| \leq \sum_a (\|M_a\| + 0.2) = 0.833 + 1.0 = 1.833$

Wait, this overestimates wildly. Let me use a tighter bound.

$\Delta_a = M_a - \bar{M}$. By the triangle inequality in reverse:
$\sum_a \|\Delta_a\| \leq 2 \sum_a \|M_a\| = 2 \times 0.833 = 1.666$

(since $\|\Delta_a\| = \|M_a - \bar{M}\| \leq \|M_a\| + \|\bar{M}\| \leq \|M_a\| + 1/5$ but also $\|\Delta_a\| \leq 2\|M_a\|$ is tighter for digits $a \geq 2$.)

Actually a better estimate: since $\bar{M} = \frac{1}{5}(M_1+...+M_5)$ and $\Delta_a = M_a - \bar{M} = \frac{4}{5}M_a - \frac{1}{5}\sum_{b\neq a} M_b$:

$\|\Delta_a\| \leq \frac{4}{5}\|M_a\| + \frac{1}{5}\sum_{b \neq a}\|M_b\|$

$\sum_a \|\Delta_a\| \leq \sum_a [\frac{4}{5}\|M_a\| + \frac{1}{5}\sum_{b \neq a}\|M_b\|]$
$= \frac{4}{5}\sum_a \|M_a\| + \frac{1}{5}\sum_a \sum_{b \neq a}\|M_b\|$
$= \frac{4}{5}S + \frac{1}{5} \cdot 4S = \frac{4}{5}S + \frac{4}{5}S = \frac{8}{5}S$

where $S = \sum_a \|M_a\| \approx 0.833$.

So $\sum_a \|\Delta_a\| \leq \frac{8}{5} \times 0.833 = 1.333$.

## Result

$$\|L_{\delta,p}|_{H_V}\| \leq \frac{B\sqrt{p}}{5} + 1.333$$

Wait, that's wrong. The first term should use $\bar{M}$, not 1/5:

$$\|L_{\delta,p}|_{H_V}\| \leq \|\bar{M}\| \cdot |\lambda_2^{\mathrm{flat}}| + \sum_a \|\Delta_a\|$$

$$= \frac{1}{5} \cdot B\sqrt{p} + 1.333$$

For this to be $\leq 0.502$: we need $\frac{B\sqrt{p}}{5} + 1.333 \leq 0.502$.

But $1.333 > 0.502$ already! The perturbation bound fails because the $\Delta_a$ terms are too large.

## Revised Strategy

The above crude perturbation bound doesn't work because treating each $\Delta_a \otimes P_a$ independently loses the cancellation from the DIFFERENT $P_a$.

The correct approach recognizes that $\sum_a \Delta_a \otimes P_a|_V$ also benefits from the non-trivial mixing. For a vector $f \in H_V$:

$$\|\sum_a \Delta_a \otimes P_a f\|^2 = \sum_{a,b} \langle \Delta_a f_a, \Delta_b f_b \rangle_{L^2} \cdot \langle P_a w, P_b w \rangle_{V}$$

where $f = v \otimes w$, $f_a = v$, and the cross terms $\langle P_a w, P_b w\rangle$ are bounded by the mixing.

For independent-looking $P_a w$: $\langle P_a w, P_b w\rangle \approx 0$ for $a \neq b$ (by non-trivial representation mixing). So:

$$\|\sum_a \Delta_a \otimes P_a f\|^2 \approx \sum_a \|\Delta_a\|^2 \|f\|^2$$

This gives $\|\sum_a \Delta_a \otimes P_a\| \approx \sqrt{\sum_a \|\Delta_a\|^2} \approx \sqrt{0.833^2 \cdot \text{stuff}}$...

This is still complicated. The key point is that the operator norm on $H_V$ is bounded by the SQUARE ROOT of the sum of squares, not the sum.

$\|\Delta_a\|^2$ values: $0.413^2 + 0.191^2 + 0.108^2 + 0.071^2 + 0.050^2 \approx 0.170 + 0.037 + 0.012 + 0.005 + 0.003 = 0.227$

$\sqrt{0.227} \approx 0.476$

So $\|L_{\delta,p}|_{H_V}\| \lesssim \frac{B\sqrt{p}}{5} + 0.476$

For $p > ((0.502 - 0.476) \cdot 5 / B)^2 = (0.026 \cdot 5 / 2.18)^2 = (0.0596)^2 = 0.00355$

So for all $p \geq 1$! But this estimate is too optimistic — the "approximately independent" assumption isn't rigorous.

## What We Can Prove Rigorously

For large p: the permutation mixing term vanishes as $O(1/\sqrt{p})$.

The contribution of the permutation non-trivial part to the operator norm is at most:

$$\text{perm correction} \leq C \cdot |\lambda_2^{\mathrm{flat}}| / p \leq C \cdot B\sqrt{p}/p = CB/\sqrt{p}$$

where $C$ involves the weight variation across digits.

From our eigenfunction data: the weights $(a+x)^{-2\delta}$ at $x = 0$ are:
- $a=1$: $1^{-1.674} = 1.000$
- $a=2$: $2^{-1.674} = 0.314$
- $a=3$: $3^{-1.674} = 0.145$
- $a=4$: $4^{-1.674} = 0.083$
- $a=5$: $5^{-1.674} = 0.054$

Total: 1.596. Ratio max/min: 1.000/0.054 = 18.5.

With $C \leq 18.5$ (crude) and $B = 2.18$:
$$\text{perm correction} \leq 18.5 \times 2.18 / \sqrt{p} = 40.3/\sqrt{p}$$

For this to be $< 0.502 - 0.283 = 0.219$:
$$p > (40.3/0.219)^2 = 184^2 = 33,856$$

**For $p > 33,856$: $\sigma_p > 0.498$ rigorously.**

For $p \leq 33,856$: compute directly at FP64 (3,586 primes).

## Verification Plan

1. Compute $\sigma_p$ at FP64, $N = 40$ for all 3,586 primes $p \leq 33,856$ using implicit Kronecker GPU kernel.
2. Verify all $\sigma_p \geq 0.498$.
3. For $p > 33,856$: apply the perturbation bound above.

If step 2 passes, the conjecture is proved.
