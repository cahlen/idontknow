---
license: cc-by-4.0
task_categories:
  - tabular-classification
tags:
  - number-theory
  - class-numbers
  - real-quadratic-fields
  - cohen-lenstra
  - gpu-computation
  - mathematics
pretty_name: Class Numbers of Real Quadratic Fields
size_categories:
  - 1B<n<10B
---

# Class Numbers of Real Quadratic Fields

## Dataset Description

Class numbers $h(d)$ of real quadratic fields $\mathbb{Q}(\sqrt{d})$ for all fundamental discriminants $d$ in large ranges, computed on an 8× NVIDIA B200 DGX cluster.

**This is the raw output of a GPU-accelerated computation — every fundamental discriminant in the range has its class number computed and stored.**

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `discriminant` | `uint64` | Fundamental discriminant $d$ |
| `class_number` | `int32` | Class number $h(d)$ of $\mathbb{Q}(\sqrt{d})$ |

## Computation Method

For each fundamental discriminant $d$:

$$h(d) = \text{round}\left(\frac{\sqrt{d} \cdot L(1, \chi_d)}{2 \cdot R(d)}\right)$$

where:
- $R(d) = \log \varepsilon_d$ is the regulator (continued fraction expansion)
- $L(1, \chi_d) = \prod_p (1 - \chi_d(p)/p)^{-1}$ via Euler product (9,592 primes up to 99,991)
- $\chi_d(n) = \left(\frac{d}{n}\right)$ is the Kronecker symbol

### Regulator Algorithm
- $d \equiv 0 \pmod{4}$: CF of $\sqrt{d/4}$, first $D=1$ detection in log-space
- $d \equiv 1 \pmod{4}$: CF of $(1+\sqrt{d})/2$ with reduced-state cycle detection

### Validation
- Exact match with PARI/GP `qfbclassno()` on 1,000 test discriminants
- 42.13% $h=1$ rate for $d < 10^4$ matches PARI exactly

## Hardware

- 8× NVIDIA B200 (183 GB VRAM each, 1.43 TB total)
- NVLink 5 full mesh
- Throughput: 1.5M discriminants/sec

## Ranges Computed

| Range | Discriminants | Time | h=1 rate |
|-------|--------------|------|----------|
| $[10^9, 10^{10})$ | 2,735,671,820 | 30 min | 16.70% |

## Cohen-Lenstra Heuristics

Cohen and Lenstra (1984) predict $\text{Prob}(h(d) = 1) \approx 75.446\%$ asymptotically. At $d \sim 10^9$, the observed rate is 16.7% — convergence is very slow and non-monotone.

## Source Code

- **Computation kernel**: [class_numbers_v2.cu](https://github.com/cahlen/idontknow/blob/main/scripts/experiments/class-numbers/class_numbers_v2.cu)
- **Full repository**: [github.com/cahlen/idontknow](https://github.com/cahlen/idontknow)
- **Results website**: [bigcompute.science](https://bigcompute.science)

## Citation

```bibtex
@dataset{humphreys2026classnumbers,
  title={Class Numbers of Real Quadratic Fields: GPU-Accelerated Computation},
  author={Humphreys, Cahlen},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/cahlen/class-numbers-real-quadratic}
}
```

## References

- Cohen, H. and Lenstra, H.W. Jr. (1984). "Heuristics on class groups of number fields." *Number Theory Noordwijkerhout 1983*, Lecture Notes in Mathematics 1068, pp. 33–62.
- Jacobson, M.J. Jr., Ramachandran, S., and Williams, H.C. (2006). "Numerical results on class groups of imaginary quadratic fields." *Mathematics of Computation*, 75(254), pp. 1003–1024.

## License

CC BY 4.0 — free to use with attribution.
